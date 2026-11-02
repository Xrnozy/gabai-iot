"""A 3D hand and a 3D held object that move together.

HAND. MediaPipe gives 21 keypoints in the picture plus a relative depth for each. Stereo gives a measured depth at some of them. solve_hand_3d()
combines the two and says, per keypoint, which one it used:
  stereo                       median trusted stereo depth in a 5x5 window (measured, a model prediction of disparity, left-right checked)
  model (no stereo depth)      the keypoint has no trusted depth: palm depth plus MediaPipe's relative depth, scaled to metres (a model estimate)
  model (stereo disagreed)     stereo gave a depth more than DISAGREE_M from the hand's own shape (background or a flying pixel): the model's value is used
  flat (no relative depth)     the hand has no relative depth at all: the palm depth is used, so the fingers are flat
The whole hand is refused when its size is not a human hand (wrist to middle knuckle outside 5.5-14 cm): that means the depth or the detection is wrong,
and a wrong 3D hand is worse than none. That test only applies when depth is metric (a calibrated stereo pair); for plain video with assumed focal length
and baseline it is skipped and the factor that would make the hand 9.5 cm is reported instead.

HAND FRAME. hand_frame() makes a rigid frame from the palm (origin at the palm centre, x from the little-finger knuckle to the index knuckle, z = palm
normal). It moves and turns with the hand, so anything fixed in it moves and turns with the hand.

HELD OBJECT. HeldObjectModel stores the object's 3D points in the HAND frame. Each time the object is seen (mask minus hand, inside the object's depth band),
its points are put into the hand frame and fused into 1 cm voxels, so more views make it more complete. Drawn back with the current hand frame, the object
sits in the hand every frame, also when the hand covers it or the detector loses it. What was never seen is not invented: the model is only as
complete as the views, and the box is a fit to the points seen.
"""
import numpy as np

from gabai.cloud import VoxelMap, backproject

PALM = (0, 5, 9, 13, 17)
WINDOW = 2
MIN_WINDOW_PX = 5
DISAGREE_M = 0.06
MAX_OFF_FLAT_M = 0.25
HAND_SIZE_M = (0.055, 0.14)            # wrist to middle-finger knuckle, human hands
SMOOTH_ALPHA = 0.55
JUMP_M = 0.10
BAND_M = 0.12


def _stereo_z(hand, disp, f, B):
    d = np.asarray(disp)
    h, w = d.shape
    z = np.full(21, np.nan)
    for i, (u, v) in enumerate(np.asarray(hand.kp2d, float)):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            continue
        win = d[max(vi - WINDOW, 0):vi + WINDOW + 1, max(ui - WINDOW, 0):ui + WINDOW + 1]
        ok = win[np.isfinite(win) & (win > 0.5)]
        if ok.size >= MIN_WINDOW_PX:
            z[i] = f * B / float(np.median(ok))
    return z


def solve_hand_3d(hand, disp, f, B, cx, cy, metric=True):
    """Camera-frame xyz (21, 3) in metres, a per-keypoint source label, and whether the result can be trusted."""
    h, w = np.asarray(disp).shape
    z_s = _stereo_z(hand, disp, f, B)
    out = {"ok": False, "reason": "", "xyz": np.full((21, 3), np.nan), "source": np.array(["none"] * 21, dtype=object), "palm_depth_m": None,
           "scale": {"wrist_to_middle_knuckle_m": None, "plausible": None}, "n_stereo": 0, "n_model": 0, "stereo_valid_fraction": float(np.isfinite(z_s).mean())}
    palm = z_s[list(PALM)]
    palm = palm[np.isfinite(palm)]
    pool = palm if len(palm) else z_s[np.isfinite(z_s)]
    if len(pool) == 0:
        out["reason"] = "no trusted stereo depth at any hand keypoint, so the hand cannot be placed in depth"
        return out
    zp = float(np.median(pool))
    out["palm_depth_m"] = zp
    zr = None if hand.z_rel is None else np.asarray(hand.z_rel, float)
    if zr is not None:
        dz = (zr - zr[list(PALM)].mean()) * w * zp / f                # MediaPipe's z is in about the units of x (a fraction of the image width)
        z_m = zp + dz
    z = np.empty(21)
    for i in range(21):
        if np.isfinite(z_s[i]):
            if zr is not None and abs(z_s[i] - z_m[i]) > DISAGREE_M:
                z[i], out["source"][i] = z_m[i], "model (stereo disagreed)"
            elif zr is None and abs(z_s[i] - zp) > MAX_OFF_FLAT_M:
                z[i], out["source"][i] = zp, "flat (stereo disagreed)"
            else:
                z[i], out["source"][i] = z_s[i], "stereo"
        elif zr is not None:
            z[i], out["source"][i] = z_m[i], "model (no stereo depth)"
        else:
            z[i], out["source"][i] = zp, "flat (no relative depth)"
    uv = np.asarray(hand.kp2d, float)
    xyz = np.stack([(uv[:, 0] - cx) / f * z, (uv[:, 1] - cy) / f * z, z], 1)
    size = float(np.linalg.norm(xyz[0] - xyz[9]))
    ok_size = HAND_SIZE_M[0] <= size <= HAND_SIZE_M[1]
    out["scale"] = {"wrist_to_middle_knuckle_m": size, "plausible": bool(ok_size)}
    if not metric:
        # Depth from assumed focal length and baseline is only proportional to metres, so a human-size test cannot judge it. The hand is kept, and the size
        # it would need to have a real hand is reported as the factor by which this depth scale is off (an observation, not a calibration).
        out["scale"] = {"wrist_to_middle_knuckle_m": size, "plausible": None, "depth_not_metric": True, "scale_factor_if_hand_is_9p5cm": round(0.095 / max(size, 1e-6), 2)}
        ok_size = True
    out["n_stereo"] = int((out["source"] == "stereo").sum())
    out["n_model"] = int(sum(str(s).startswith("model") for s in out["source"]))
    if not ok_size:
        out["reason"] = (f"hand size {size * 100:.0f} cm (wrist to middle knuckle) is not a human hand ({HAND_SIZE_M[0] * 100:.1f}-{HAND_SIZE_M[1] * 100:.0f} cm): "
                         "the depth at the hand or the detection is wrong, so no 3D hand is made")
        return out
    out.update(ok=True, xyz=xyz, reason="ok")
    return out


def hand_frame(xyz):
    """(R, t): columns of R are the hand's x, y, z axes in the frame of xyz; t is the palm centre. None when the palm is not measured well enough."""
    X = np.asarray(xyz, float)
    ok = np.isfinite(X[list(PALM), 2])
    if ok.sum() < 4 or not (np.isfinite(X[0, 2]) and np.isfinite(X[5, 2]) and np.isfinite(X[17, 2])):
        return None
    pts = X[list(PALM)][ok]
    t = pts.mean(0)
    x = X[5] - X[17]
    y0 = X[[5, 9, 13, 17]][np.isfinite(X[[5, 9, 13, 17], 2])].mean(0) - X[0]
    x = x / np.linalg.norm(x)
    z = np.cross(x, y0)
    nz = np.linalg.norm(z)
    if nz < 1e-9:
        return None
    z = z / nz
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1), t


class HandSmoother:
    """Exponential smoothing of a hand's world keypoints. A move of the palm of more than JUMP_M in one frame is followed at once, not smoothed."""

    def __init__(self, alpha=SMOOTH_ALPHA):
        self.alpha, self.prev = alpha, {}

    def drop(self, hid):
        self.prev.pop(hid, None)

    def update(self, hid, X):
        X = np.asarray(X, float)
        p = self.prev.get(hid)
        if p is None:
            out = X.copy()
        else:
            palm_new, palm_old = np.nanmean(X[list(PALM)], axis=0), np.nanmean(p[list(PALM)], axis=0)
            a = 1.0 if (not np.all(np.isfinite(palm_new)) or not np.all(np.isfinite(palm_old)) or np.linalg.norm(palm_new - palm_old) > JUMP_M) else self.alpha
            out = np.where(np.isfinite(p) & np.isfinite(X), a * X + (1 - a) * p, X)
        self.prev[hid] = out
        return out.copy()


def extract_object_points(disp, trusted, mask, hand_px, f, B, cx, cy, z_ref, band=BAND_M, stride=2):
    """Camera-frame points of the object: inside its mask, outside the hand, trusted, and within `band` metres of the object's own depth."""
    d = np.asarray(disp)
    valid = np.isfinite(d) & (d > 0.5)
    m = np.asarray(mask, bool)
    info = {"mask_px": int((m & valid)[::stride, ::stride].sum())}
    info["dropped_hand"] = int((m & valid & np.asarray(hand_px, bool))[::stride, ::stride].sum())
    m = m & ~np.asarray(hand_px, bool)
    info["dropped_untrusted"] = int((m & valid & ~np.asarray(trusted, bool))[::stride, ::stride].sum())
    m = m & np.asarray(trusted, bool)
    pts, _ = backproject(d, f, B, cx, cy, mask=m, stride=stride)
    near = np.abs(pts[:, 2] - z_ref) <= band
    info["dropped_background"] = int((~near).sum())
    pts = pts[near]
    info["kept"] = int(len(pts))
    return pts, info


def oriented_box(points_local, min_points=30, trim=1.0):
    """Principal-axis box around the points (trimmed at the 1st and 99th percentile): {center, R, half_sizes}. A fit to what was seen, not the true shape."""
    P = np.asarray(points_local, float)
    if len(P) < min_points:
        return None
    c0 = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c0, full_matrices=False)
    Rb = Vt.T
    if np.linalg.det(Rb) < 0:
        Rb[:, 2] *= -1
    q = (P - c0) @ Rb
    lo, hi = np.percentile(q, trim, axis=0), np.percentile(q, 100 - trim, axis=0)
    center = c0 + Rb @ ((lo + hi) / 2)
    return {"center": center, "R": Rb, "half_sizes": (hi - lo) / 2}


class HeldObjectModel:
    def __init__(self, key, label, voxel=0.01):
        self.key, self.label = key, label
        self.vm = VoxelMap(voxel)
        self.n_views = 0
        self.last_seen_t = None

    def add_world_points(self, pts_world, R_wh, t_wh):
        if len(pts_world) == 0:
            return
        local = (np.asarray(pts_world, float) - np.asarray(t_wh, float)) @ np.asarray(R_wh, float)
        self.vm.integrate(local)
        self.n_views += 1

    def points_local(self, min_obs=2, core=0.0):
        return self.vm.points(min_obs=min_obs, core=core)

    def points_world(self, R_wh, t_wh, min_obs=2, core=0.0):
        return self.points_local(min_obs, core) @ np.asarray(R_wh, float).T + np.asarray(t_wh, float)

    def box_local(self):
        return oriented_box(self.points_local(min_obs=2))


class HandIdentity:
    """Stable ids for the hands seen from frame to frame.

    MediaPipe's own left/right label is NOT an identity: it flips from frame to frame, and two hands can carry the same label. Using it as the id made
    a new 3D hand appear under a new name whenever the label changed, and the old one stayed in the scene. Here an id belongs to a TRACK: each detection is
    matched to the nearest existing track by palm position in 3D (or by picture position when a palm has no depth), inside a gate that grows a little while
    the track is unseen. Unmatched detections start a new track; a track unseen for more than `lost_frames` frames is retired and its id is returned so
    the caller can remove everything that belongs to it."""

    def __init__(self, lost_frames=8, gate_m=0.30, gate_px=180.0):
        self.lost_frames, self.gate_m, self.gate_px = lost_frames, gate_m, gate_px
        self.tracks, self._next = {}, 1

    def _cost(self, tr, d):
        grow = 1.0 + 0.5 * tr["age"]
        if tr["palm"] is not None and d["palm"] is not None:
            c = float(np.linalg.norm(tr["palm"] - d["palm"])) / (self.gate_m * grow)
        else:
            c = float(np.hypot(tr["px"][0] - d["px"][0], tr["px"][1] - d["px"][1])) / (self.gate_px * grow)
        return c if c <= 1.0 else None

    def assign(self, dets):
        """dets: [{"palm": xyz or None, "px": (u, v)}]. Returns (ids in the same order, ids retired this call)."""
        pairs = []
        for tid, tr in self.tracks.items():
            for j, d in enumerate(dets):
                c = self._cost(tr, d)
                if c is not None:
                    pairs.append((c, tid, j))
        pairs.sort()
        ids, used = [None] * len(dets), set()
        for c, tid, j in pairs:
            if tid in used or ids[j] is not None:
                continue
            ids[j] = tid
            used.add(tid)
        for j, d in enumerate(dets):
            if ids[j] is None:
                ids[j] = f"hand{self._next}"
                self._next += 1
                self.tracks[ids[j]] = {"palm": None, "px": d["px"], "age": 0}
        for j, d in enumerate(dets):
            tr = self.tracks[ids[j]]
            tr["px"], tr["age"] = d["px"], 0
            if d["palm"] is not None:
                tr["palm"] = np.asarray(d["palm"], float)
        retired = []
        for tid, tr in list(self.tracks.items()):
            if tid not in ids:
                tr["age"] += 1
                if tr["age"] > self.lost_frames:
                    del self.tracks[tid]
                    retired.append(tid)
        return ids, retired
