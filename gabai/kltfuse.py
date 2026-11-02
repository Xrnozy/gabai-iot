"""The Pi's KLT tracks (sender --klt) as a second measurement of the camera's TRANSLATION for stereo_imu, trusted more than the PC's own picture tracking.

Two pieces.
  PointMap   puts a point the Pi found in the raw sent picture where the same spot is in the pipeline's left picture (turned by the calibration's rotation, rectified, shrunk,
             and swapped / turned 180 deg when the pipeline does that). The sender's picture and the pipeline's picture are not the same picture.
  KltFusion  keeps the 3D points of a keyframe (depth of the Pi's tracked points at the frame the keyframe was made) and finds each later frame's translation from where the Pi's
             tracks have moved, with the rotation FIXED to the one KeyframeVO already has (the IMU's wherever the gyro covers the frame). The KLT never changes the rotation. Its
             translation is blended with KeyframeVO's (weights sum to 1: one estimate of one quantity, never added as a second constraint), a difference larger than max_step_m is
             closed over several frames, and when the PC tracker lost the frame the KLT translation holds the position. The result is written back into KeyframeVO.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from gabai.diag import log as dlog
from gabai.kfvo import _t_only
from gabai.ptmap import RectPoints, turn_points  # noqa: F401  (turn_points: tests and other callers import it from here)

log = dlog.get("pose")
Z_MIN, Z_MAX = 0.3, 6.0                      # landmark depths outside this are too near to trust the stereo match or too far to have a useful depth


@dataclass
class KltObs:
    """The Pi's points of one frame in the pipeline's left picture (pixels, OpenCV convention). ids/uv are the live points (tracked and new); lost ones are kept apart."""
    ids: np.ndarray                          # (n,) int64, the Pi's feature ids
    uv: np.ndarray                           # (n, 2) float32
    quality: object = None                   # (n,) uint8, LK patch error x 8 (255 worst), None = not sent
    new: object = None                       # (n,) bool, found in this frame
    lost_ids: object = None
    lost_uv: object = None                   # last known place of the points lost in this frame
    frame_seq: object = None
    klt_ms: float = 0.0
    right_uv: object = None                  # (n, 2) the same points in the pipeline's RIGHT picture, NaN where the Pi did not find them there (needs sender --klt-stereo)
    motion: object = None                    # the Pi's own frame-to-frame motion estimate (sender --klt-motion), see protocol.pack_klt


class PointMap:
    """Raw sent-picture coordinates -> coordinates in the pipeline's pictures. Mirrors PairPrep.frame (gabai/pipeline/sources.py) step by step."""

    def __init__(self, calib_path, sent_size, rotation, k=1.0, flip180=False, swapped=False):
        self.rp = RectPoints(calib_path, sent_size)
        self.rp.rotation = tuple(rotation)                                   # the rotation in use now, which may differ from the one stored in the calibration
        self.k, self.flip, self.swapped = float(k), bool(flip180), bool(swapped)
        self.out = (int(round(self.rp.size[0] * self.k)), int(round(self.rp.size[1] * self.k)))

    def _map(self, xy_left, xy_right, slot):
        from_right = (slot == "L") == self.swapped                             # a swapped layout feeds the other camera's picture into each slot
        xy = xy_right if from_right else xy_left
        if xy is None:
            return None
        u = self.rp(xy, slot, self.swapped, "R" if from_right else "L")
        if self.k != 1.0:
            u = (u + 0.5) * self.k - 0.5                                       # INTER_AREA shrink: pixel centres sit at (p + 0.5) * k - 0.5
        if self.flip:
            u = np.stack([self.out[0] - 1 - u[:, 0], self.out[1] - 1 - u[:, 1]], 1)
        return u.astype(np.float32)

    def __call__(self, xy_left, xy_right=None):
        """Points of the pipeline's LEFT picture as (n, 2) float32, NaN where they cannot be given; None when right-picture points are needed and were not sent.
        (The 180 deg turn hands the rectified RIGHT picture over as the left one, hence slot R then.)"""
        return self._map(xy_left, xy_right, "R" if self.flip else "L")

    def right(self, xy_left, xy_right=None):
        """The same for the pipeline's RIGHT picture."""
        return self._map(xy_left, xy_right, "L" if self.flip else "R")

    def motion(self, m):
        """The Pi's motion estimate (rectified left camera) in the pipeline's left camera frame, or None. A swapped layout gives none (the Pi's left camera is not the pipeline's);
        the 180 deg turn is a rotation of the camera about its optical axis by pi: both vectors change sign in x and y (the 4 cm to the other camera is ignored)."""
        if m is None or self.swapped:
            return None
        if not self.flip:
            return m
        D = np.array([-1.0, -1.0, 1.0])
        return dict(m, rvec=np.asarray(m["rvec"]) * D, t=np.asarray(m["t"]) * D)

    def obs(self, kp):
        """KltObs from the receiver's KltPoints, or None. Points that fall outside the picture are dropped."""
        from gabai.net.protocol import KLT_LOST, KLT_NEW
        uv = self(kp.xy, kp.xy_right)
        if uv is None:
            return None
        inside = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] <= self.out[0] - 1) & (uv[:, 1] <= self.out[1] - 1)
        live, lost = inside & (kp.status != KLT_LOST), inside & (kp.status == KLT_LOST)
        ruv = self.right(kp.xy, kp.xy_right)
        return KltObs(kp.ids[live].astype(np.int64), uv[live], kp.quality[live], kp.status[live] == KLT_NEW, kp.ids[lost].astype(np.int64), uv[lost], kp.frame_seq, kp.klt_ms,
                      None if ruv is None else ruv[live], self.motion(getattr(kp, "motion", None)))


def _angle_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


class KltFusion:
    def __init__(self, cal, weight=0.75, min_inliers=15, max_reproj_px=1.5, agree_m=0.08, max_step_m=0.02, reseed_frac=0.4, min_landmarks=30):
        """weight: share of the KLT translation while the two agree (0.5 = equal, 1 = KLT only). min_inliers / max_reproj_px: a KLT translation needs this many points
        agreeing within this many pixels (RMS) to count. agree_m: the translations agree within this; farther apart, the pose moves toward KLT by at most max_step_m a frame.
        reseed_frac / min_landmarks: take new keyframe points when fewer than this share (or number) of them are still tracked."""
        self.cal, self.w = cal, float(np.clip(weight, 0.0, 1.0))
        self.K = np.array([[cal["f"], 0, cal["cx"]], [0, cal["f"], cal["cy"]], [0, 0, 1.0]])
        self.min_in, self.max_rms, self.agree_m, self.max_step = int(min_inliers), float(max_reproj_px), float(agree_m), float(max_step_m)
        self.reseed_frac, self.min_lm = float(reseed_frac), int(min_landmarks)
        self.kf = None                                                     # dict: ids, X (keyframe camera frame), node (KeyframeVO's keyframe id, or None), R, t (odometry frame)
        self.counts = {"blend": 0, "limited": 0, "rescued": 0, "vo_only": 0, "none": 0}

    # -- keyframe points ----------------------------------------------------------------------------------------------------------------------------
    def _seed(self, obs, disp, trusted, R, t, node):
        h, w = disp.shape[:2]
        u, v = np.clip(np.rint(obs.uv[:, 0]).astype(int), 0, w - 1), np.clip(np.rint(obs.uv[:, 1]).astype(int), 0, h - 1)
        d = disp[v, u].astype(np.float64)
        ok = np.isfinite(d) & (d > 0.5) & np.asarray(trusted, bool)[v, u]
        z = np.where(ok, self.cal["f"] * self.cal["B"] / np.where(ok, d, 1.0), np.nan)
        ok &= (z >= Z_MIN) & (z <= Z_MAX)
        if ok.sum() < self.min_in:
            self.kf = None
            return
        zz = z[ok]
        X = np.stack([(obs.uv[ok, 0] - self.cal["cx"]) * zz / self.cal["f"], (obs.uv[ok, 1] - self.cal["cy"]) * zz / self.cal["f"], zz], 1)
        order = np.argsort(obs.ids[ok])
        self.kf = {"ids": obs.ids[ok][order], "X": X[order], "node": node, "R": R.copy(), "t": t.copy()}

    def _kf_pose(self, vo):
        node = self.kf["node"]
        n = vo.node_pose(node) if node is not None else None            # the pose graph / bundle adjustment may have moved a keyframe the PC tracker made
        return n if n is not None else (self.kf["R"], self.kf["t"])

    # -- translation from the tracks ----------------------------------------------------------------------------------------------------------------
    def _measure(self, obs, vo, Rv):
        """dict with matches and, when it worked, t (odometry frame), inliers, rms: the camera translation that best explains the tracks given the rotation Rv."""
        if self.kf is None or obs is None or len(obs.ids) == 0:
            return None
        _, ik, io = np.intersect1d(self.kf["ids"], obs.ids, return_indices=True)
        info = {"matches": len(ik)}
        if len(ik) < self.min_in:
            return info
        X, uv = self.kf["X"][ik], obs.uv[io].astype(np.float64)
        R_k, t_k = self._kf_pose(vo)
        Rm = (R_k.T @ Rv).T                                                # X_cur = Rm X_kf for the rotation we are given (KeyframeVO's convention)
        res = _t_only(X, uv, self.K, Rm)
        if res is None or len(res[1]) < self.min_in:
            return info
        tvec, s = res[0], res[1]
        Xc = X[s] @ Rm.T + tvec
        e = np.stack([self.K[0, 0] * Xc[:, 0] / Xc[:, 2] + self.K[0, 2], self.K[1, 1] * Xc[:, 1] / Xc[:, 2] + self.K[1, 2]], 1) - uv[s]
        info.update(t=t_k + R_k @ (-Rm.T @ tvec), inliers=len(s), rms=float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))))
        return info

    def _good(self, m):
        return m is not None and "t" in m and m["inliers"] >= self.min_in and m["rms"] <= self.max_rms

    def apply(self, vo, obs, disp, trusted, Rv, tv, info):
        """Combine KeyframeVO's translation of this frame with the KLT translation, write the result into vo and info, return (Rv, t). Call right after vo.step(). Rv is never changed."""
        m = self._measure(obs, vo, Rv)
        vo_ok, klt_ok = bool(info["ok"]), self._good(m)
        out = {"klt_matches": 0 if m is None else m["matches"], "klt_inliers": None, "klt_rms_px": None, "klt_used": "none", "klt_weight": 0.0, "klt_vs_vo_m": None}
        t = tv
        if klt_ok:
            diff = m["t"] - tv
            out.update(klt_inliers=m["inliers"], klt_rms_px=round(m["rms"], 2), klt_vs_vo_m=round(float(np.linalg.norm(diff)), 4))
            if not vo_ok:                                                  # the PC tracker lost the frame: its pose is a held one, the KLT translation replaces it
                t, how, wt = m["t"], "rescued", 1.0
            else:
                wt = self.w * min(1.0, m["inliers"] / 30.0)                # fewer agreeing points = less weight, even though KLT is the favoured source
                step = wt * diff
                n = float(np.linalg.norm(step))
                how = "blend"
                if n > self.max_step:                                      # no sudden correction: the rest of the difference is closed over the next frames
                    step, how = step * (self.max_step / n), "limited"
                t = tv + step
            out.update(klt_used=how, klt_weight=round(wt, 3))
        elif vo_ok:
            out["klt_used"] = "vo_only"
        self.counts[out["klt_used"]] += 1
        if out["klt_used"] in ("blend", "limited", "rescued"):
            if info.get("keyframe"):                                       # a keyframe was just made at the PC tracker's pose: move it to the fused one
                vo.correct(Rv, t)
            else:
                vo.t_wc = np.array(t, float)
            if out["klt_used"] == "rescued":
                info.update(ok=True, reason="position from the KLT tracks (the PC tracker lost this frame; rotation from the IMU)")
                vo.lost_n = 0
        info.update(out)
        self._keyframe(vo, obs, disp, trusted, info, Rv, t, m)
        return Rv, t

    def _keyframe(self, vo, obs, disp, trusted, info, R, t, m):
        """New keyframe points when the PC tracker made a keyframe, or when too few of ours are still tracked."""
        if obs is None or not (info.get("ok") or info.get("keyframe")):
            return
        n_kf = 0 if self.kf is None else len(self.kf["ids"])
        left = 0 if m is None else m["matches"]
        if info.get("keyframe") and getattr(vo, "node", None) is not None:
            self._seed(obs, disp, trusted, vo.R_wk, vo.t_wk, vo.node["id"])
        elif (self.kf is None or left < self.reseed_frac * n_kf or left < self.min_lm) and info.get("ok"):
            self._seed(obs, disp, trusted, np.asarray(R, float), np.asarray(t, float), None)
