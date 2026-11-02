"""Keyframe stereo visual odometry, camera only. Same interface as gabai.tools.video_cloud_view.StereoVO, less drift, and it survives rough motion.

StereoVO re-seeds its features every frame and chains frame-to-frame motions, so every frame adds its own error to the path. Here features are
seeded once per KEYFRAME with a 3D position from the stereo depth, and each new frame is located against that keyframe. Path error then grows with
the number of keyframes, not the number of frames.

Per frame (the techniques are the ones described in docs/visual-slam-research.md):
  1. PREDICT the pose from the last motion (constant velocity), or from a rotation handed in by the caller (a gyro later). Project the keyframe's 3D
     points into the picture with it. This is the "guided search" of ORB-SLAM, and is what keeps tracking alive when the camera turns fast: the search
     starts where the feature should be, not where it was.
  2. TRACK with pyramidal Lucas-Kanade from the keyframe picture, started at the predicted positions. If that finds too few agreeing features, start again from
     the previous positions (a wrong prediction after a sudden change of direction must not cost the frame).
  3. SOLVE the pose with RANSAC PnP on the keyframe's 3D points, then REFINE it with Levenberg-Marquardt on the inliers (motion-only bundle adjustment).
  4. Average each feature's stereo depth over the frames that see it, so one frame's depth noise does not stay in the keyframe.
  5. A new KEYFRAME is made from the current frame when too few features survive, or the camera moved or turned enough.
  If tracking fails (no pose at all) for reloc_after frames in a row: RELOCALISE. ORB features of the current picture are matched against the stored keyframes and the pose comes from PnP
  against the best one, so a jerk that blurs or shifts the picture does not leave a permanent offset in the path. If that also fails the pose is held
  and the reason is returned; after reanchor_after lost frames a new keyframe is made at the held pose.
  Each keyframe also stores ORB descriptors with 3D positions. With loop_closure on, a new keyframe is checked against OLD keyframes that were seen
  at least loop_min_path_m of travel earlier and lie within loop_radius_m of where the camera is believed to be. A verified match gives the drift since
  then (reported in info["loop"]); with loop_apply the current pose is moved to it (a forward-only correction: the path before it stays as it was).

What is NOT here: bundle adjustment over many keyframes, a pose graph that spreads a loop correction back over the path, appearance-based place
recognition (a loop is only found when the drift is smaller than loop_radius_m), photometric or rolling-shutter modelling, and any inertial term.

Seam for an inertial sensor (no IMU is read here):
  step(..., dR_prior=R)   rotation of the camera between the previous and this frame, from a gyro. It replaces the constant-velocity rotation in step 1
                          (so the search starts in the right place even through a sudden spin), and it also gates the result: when the visual rotation
                          differs by more than prior_gate_deg the pose is held and the frame flagged. It never enters the optimisation itself.
  correct(R_wc, t_wc)     replace the current pose (the map refiner, a loop correction, or a fused estimator). The keyframe moves with it, so the
                          following frames continue from the corrected pose instead of drifting back.
A tightly coupled filter would replace step 3 with an optimisation that has the inertial residual in it; steps 1, 2, 4, 5 and the relocaliser stay as they are.
"""
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from gabai import localba, posegraph as pg
from gabai.sim import look_pose

MIN_INLIERS = 15
MIN_FEATURES = 25
DEPTH_RANGE = (0.3, 8.0)
DISP_SIGMA_PX = 0.3         # assumed disparity noise of the stereo network, for weighting depth measurements
FUSE_W_CAP = 30.0           # a landmark's weight never exceeds this many measurements, so it can still follow a real change
ORB_FEATURES = 600
RELOC_MIN_INLIERS = 30
RELOC_MAX_JUMP_M = 3.0


def _zncc(img_a, pa, img_b, pb, r=4):
    """Zero-mean normalised cross-correlation of the (2r+1)^2 patches around pa in img_a and pb in img_b, per point. A patch with no texture (a blank or washed-out picture) gets -1: it matches
    nothing, however well it sits on a prediction."""
    d = np.arange(-r, r + 1)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    out = []
    for img, pts in ((img_a, pa), (img_b, pb)):
        h, w = img.shape[:2]
        x = np.clip(np.rint(pts[:, 0]).astype(int), r, w - 1 - r)
        y = np.clip(np.rint(pts[:, 1]).astype(int), r, h - 1 - r)
        out.append(img[(y[:, None] + dy.ravel()), (x[:, None] + dx.ravel())].astype(np.float32))
    a, b = out[0] - out[0].mean(1, keepdims=True), out[1] - out[1].mean(1, keepdims=True)
    na, nb = np.sqrt((a * a).sum(1)), np.sqrt((b * b).sum(1))
    z = (a * b).sum(1) / np.maximum(na * nb, 1e-6)
    return np.where((na < 1.0) | (nb < 1.0), -1.0, z)


def _pnp(obj, img, K, iters, err):
    """RANSAC PnP that reports a degenerate set of points (all the same place, almost no spread: OpenCV raises) as a failure, never as a crash."""
    try:
        return cv2.solvePnPRansac(obj, img, K, None, iterationsCount=iters, reprojectionError=err, confidence=0.99, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return False, None, None, None


def _t_only(obj, img, K, Rm, iters=150, err=2.5, seed=0):
    """Camera translation when the rotation is known (from a gyro): X_cur = Rm X + t. Every correspondence gives two equations linear in t, so two points make a hypothesis
    (RANSAC over 2-point samples, far more forgiving than the 4 to 6 points of a full pose when most tracks are wrong). Returns (t, inlier indices) or None."""
    Y = obj @ Rm.T
    a, b = (img[:, 0] - K[0, 2]) / K[0, 0], (img[:, 1] - K[1, 2]) / K[1, 1]
    n = len(obj)
    A = np.zeros((2 * n, 3))
    A[0::2, 0], A[0::2, 2], A[1::2, 1], A[1::2, 2] = 1.0, -a, 1.0, -b
    r = np.zeros(2 * n)
    r[0::2], r[1::2] = a * Y[:, 2] - Y[:, 0], b * Y[:, 2] - Y[:, 1]
    rng = np.random.default_rng(seed)
    best_in = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        rows = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        t = np.linalg.lstsq(A[rows], r[rows], rcond=None)[0]
        Xc = Y + t
        z = Xc[:, 2]
        e = np.hypot(K[0, 0] * Xc[:, 0] / np.maximum(z, 1e-3) + K[0, 2] - img[:, 0], K[1, 1] * Xc[:, 1] / np.maximum(z, 1e-3) + K[1, 2] - img[:, 1])
        inl = np.nonzero((e < err) & (z > 0.1))[0]
        if best_in is None or len(inl) > len(best_in):
            best_in = inl
    if best_in is None or len(best_in) < 3:
        return None
    rows = np.sort(np.r_[2 * best_in, 2 * best_in + 1])
    return np.linalg.lstsq(A[rows], r[rows], rcond=None)[0], best_in


ZNCC_MIN = 0.3                                                        # a tracked patch must correlate with the keyframe's patch at least this well (blur lowers it; nothing at all gives -1)
LK_MAX_ERR = 25.0                                                     # mean absolute grey difference per pixel of a tracked patch (0-255); a good match is under 10, a smeared one under 20
LM_MIN_RESID_CM, LM_GAIN_FRAC = 3.0, 0.85                                  # an object solve is applied only when the objects' median residual was at least 3 cm and falls to 85% or less
FIX_KEEP_FRAC = 0.8                                                   # with a supplied rotation (gyro) its translation-only fit is kept when it has at least this share of the free pose's inliers
WIDE_RETRY = True                                                     # look again with a wider search when a frame would otherwise be lost
SMEAR_FRAC, KF_DEFER_MAX = 0.6, 4                                     # a frame sharper than this share of the recent median may start a keyframe; at most this many frames are waited out
RELOC_GYRO_GATE_DEG = 20.0                                          # a relocalisation that turns the camera this far from where the gyro says it is looking is a wrong match, not a find
GAP_S = 0.4                                                          # a pair this long after the previous one: constant-velocity and gyro predictions are dropped, a lost frame is relocalised at once


def _angle_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


class KeyframeVO:
    def __init__(self, fx, fy, cx, cy, max_corners=500, kf_min_inliers=60, kf_keep_frac=0.5, kf_dist_m=0.12, kf_rot_deg=12.0, kf_max_frames=15,
                 prior_gate_deg=6.0, fuse_depth=True, carry=False, max_carry=350, window_ba=False, ba_window=5, ba_every=1, ba_max_landmarks=150, max_step_m=0.5, max_step_deg=45.0, predict=True, refine=True, relocalize=True, reloc_after=2, reanchor_after=5,
                 loop_closure=False, loop_apply=False, loop_radius_m=1.0, loop_min_path_m=2.0, max_keyframes=300, reloc_keyframes=6, loop_apply_inliers=60, graph_async=True, initial_pitch_deg=0.0,
                 loop_appearance=True, loop_every=2, loop_max_drift_m=2.0):
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.fx, self.fy, self.cx, self.cy, self.max_corners = fx, fy, cx, cy, max_corners
        self.kf_min_inliers, self.kf_keep_frac, self.kf_dist_m, self.kf_rot_deg, self.kf_max_frames = kf_min_inliers, kf_keep_frac, kf_dist_m, kf_rot_deg, kf_max_frames
        self.prior_gate_deg, self.fuse_depth, self.carry, self.max_carry = prior_gate_deg, fuse_depth, carry or window_ba, max_carry     # the bundle adjustment needs landmarks that live across keyframes
        self.window_ba, self.ba_window, self.ba_every, self.ba_max_landmarks = window_ba, ba_window, ba_every, ba_max_landmarks
        self._next_tid, self.kf_tid, self._bthread, self._bout = 0, None, None, None
        self.max_step_m, self.max_step_deg = max_step_m, max_step_deg      # one frame cannot move the camera farther than this; a result that does is a tracking error
        self.predict, self.refine, self.relocalize, self.reloc_after, self.reanchor_after = predict, refine, relocalize, reloc_after, reanchor_after
        self.loop_closure, self.loop_apply, self.loop_radius_m, self.loop_min_path_m = loop_closure, loop_apply, loop_radius_m, loop_min_path_m
        self.max_keyframes, self.reloc_keyframes, self.loop_apply_inliers, self.graph_async = max_keyframes, reloc_keyframes, loop_apply_inliers, graph_async
        self.R_wc, self.t_wc = look_pose(np.zeros(3), 0.0, initial_pitch_deg)   # world z is up; the camera starts looking initial_pitch_deg below the horizon (0 = level)
        self.R_wk, self.t_wk = self.R_wc.copy(), self.t_wc.copy()       # keyframe pose in the world
        self.kf_gray = self.kf2d = self.kf3d = self.cur2d = None        # keyframe picture, its features, their 3D (keyframe camera frame), current positions
        self.alive = None
        self.kf_age, self.n, self.n_keyframes = 0, 0, 0
        self.last_flow = None
        self.dR, self.dt, self.has_vel = np.eye(3), np.zeros(3), False  # last motion, in the previous camera's frame: the constant-velocity model
        self.vel_s = None                                               # seconds that last motion took, so a frame that comes late is predicted farther (frame_s of step)
        self.path_m, self.lost_n = 0.0, 0
        self.recs, self.rec = [], None                                  # keyframes that have ORB descriptors with 3D, and the current one if it has
        self.kfs, self.kf_ids, self.node = {}, [], None                 # the keyframe graph: id -> node {R, t, path, odo edge}; the current node
        self.loops, self.frame_log = [], []                             # verified loop edges; per frame (keyframe id, pose in that keyframe) so the path can be rebuilt after a correction
        self.graph_version, self.last_graph, self.last_ba = 0, None, None
        self.loop_appearance, self.loop_every, self.loop_max_drift_m = loop_appearance, max(int(loop_every), 1), loop_max_drift_m
        self.loop_version, self._loop_prev = 0, None                      # loop_version counts pose-graph corrections that moved keyframes: whoever holds points placed by keyframes re-lays them when it changes
        self._gthread, self._gout, self._gdirty = None, None, False
        self._sharp, self._defer = [], 0                                    # sharpness of the last frames (variance of the Laplacian), keyframe deferrals in a row
        self.landmark_source = None                                       # callable -> [(landmark id, keyframe id, p_k, sigma)]: objects seen from keyframes (gabai/objects3d.py); they become extra factors of the pose graph
        self.lm_pos, self.lm_every, self.lm_min_gap, self.lm_max_shift_m, self.lm_max_rot_deg, self._lm_kf = {}, 8, 6, 0.5, 15.0, -99
        self.lm_rebuild_m, self.lm_rebuild_deg = 0.03, 0.5
        self.last_correction = (np.eye(3), np.zeros(3))
        self._lm_n, self.lm_new_obs = 0, 4
        self.stats = {"landmark_runs": 0, "landmark_applied": 0, "landmark_rejected": 0, "landmark_unhelpful": 0, "relocalized": 0, "loops_found": 0, "loops_applied": 0, "graph_runs": 0, "ba_runs": 0, "ba_applied": 0, "ba_skipped_busy": 0, "predicted_ok": 0, "fallback_init": 0, "wide_ok": 0, "fixrot_used": 0, "free_rot_used": 0, "reloc_gyro_rejected": 0, "kf_deferred": 0}
        self._orb = self._bf = None
        self._B = 0.0507

    def set_baseline(self, B):
        self._B = B

    # ---- keyframes and the ORB database --------------------------------------------------------------------------------------------------------------
    def _seed(self, gray, disp, trusted, taken=None):
        mask = None
        if taken is not None and len(taken):
            mask = np.full(gray.shape[:2], 255, np.uint8)
            for x, y in taken:
                cv2.circle(mask, (int(x), int(y)), 10, 0, -1)                    # no new corner next to a landmark that is carried over
        c = cv2.goodFeaturesToTrack(gray, max(self.max_corners - (0 if taken is None else len(taken)), 50), 0.01, 10, mask=mask)
        if c is None:
            return None, None
        c = c.reshape(-1, 2)
        u, v = np.clip(c[:, 0].astype(int), 0, disp.shape[1] - 1), np.clip(c[:, 1].astype(int), 0, disp.shape[0] - 1)
        d = disp[v, u]
        ok = np.isfinite(d) & (d > 0.5) & np.asarray(trusted, bool)[v, u]
        z = np.where(ok, self.fx * self._B / np.where(ok, d, 1), np.nan)
        ok &= (z > DEPTH_RANGE[0]) & (z < DEPTH_RANGE[1])
        c, z = c[ok], z[ok]
        return c.astype(np.float32), np.stack([(c[:, 0] - self.cx) / self.fx * z, (c[:, 1] - self.cy) / self.fy * z, z], 1)

    def _weight(self, z):
        return (self.fx * self._B / np.maximum(z, 0.1) ** 2 / DISP_SIGMA_PX) ** 2          # 1 / depth variance of a stereo depth with DISP_SIGMA_PX of disparity noise

    def _fuse_depth(self, gi, cur, disp, trusted, R_kc, t_kc):
        """Every frame sees each tracked feature again with its own stereo depth. Average them (inverse variance) in the keyframe frame, so the noise of one
        frame's depth does not stay in the keyframe for its whole life. A measurement far from the stored point is a wrong depth or a moving thing: ignored."""
        u, v = np.clip(cur[:, 0].astype(int), 0, disp.shape[1] - 1), np.clip(cur[:, 1].astype(int), 0, disp.shape[0] - 1)
        d = disp[v, u]
        ok = np.isfinite(d) & (d > 0.5) & np.asarray(trusted, bool)[v, u]
        if not ok.any():
            return
        gi, u, v, d = gi[ok], cur[ok, 0], cur[ok, 1], d[ok]
        z = self.fx * self._B / d
        p_kf = (np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], 1)) @ R_kc.T + t_kc
        near = np.linalg.norm(p_kf - self.kf3d[gi], axis=1) < 0.05 + 0.04 * z
        gi, p_kf, z = gi[near], p_kf[near], z[near]
        w_m = self._weight(z)
        w = self.kf_w[gi]
        self.kf3d[gi] = (self.kf3d[gi] * w[:, None] + p_kf * w_m[:, None]) / (w + w_m)[:, None]
        self.kf_w[gi] = np.minimum(w + w_m, FUSE_W_CAP * w_m)

    @property
    def _need_orb(self):
        return self.relocalize or self.loop_closure

    def _orb_features(self, gray, disp=None, trusted=None):
        """ORB keypoints and descriptors; with a depth map, only those with a trusted depth, plus their 3D position in the camera frame."""
        if self._orb is None:
            self._orb = cv2.ORB_create(ORB_FEATURES, fastThreshold=10)
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        kps, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or not len(kps):
            return None
        uv = np.array([k.pt for k in kps], np.float32)
        if disp is None:
            return desc, uv, None
        u, v = np.clip(uv[:, 0].astype(int), 0, disp.shape[1] - 1), np.clip(uv[:, 1].astype(int), 0, disp.shape[0] - 1)
        d = disp[v, u]
        ok = np.isfinite(d) & (d > 0.5) & np.asarray(trusted, bool)[v, u]
        z = np.where(ok, self.fx * self._B / np.where(ok, d, 1), np.nan)
        ok &= (z > DEPTH_RANGE[0]) & (z < DEPTH_RANGE[1])
        uv, z, desc = uv[ok], z[ok], desc[ok]
        return desc, uv, np.stack([(uv[:, 0] - self.cx) / self.fx * z, (uv[:, 1] - self.cy) / self.fy * z, z], 1)

    def _carried(self):
        """The landmarks that the old keyframe tracked into this frame and that are still agreeing: their 2D positions here, their 3D positions in THIS camera's
        frame, and the weight they have collected. Carrying them over means a landmark's depth keeps being averaged across keyframes, and the new keyframe's
        pose is not chained to the old keyframe's through a fresh, single-frame depth."""
        if not self.carry or self.kf3d is None or self.alive is None:
            return None
        a = np.nonzero(self.alive)[0]
        if len(a) < 10:
            return None
        Xn = (self.kf3d[a] @ self.R_wk.T + self.t_wk - self.t_wc) @ self.R_wc
        uv = self.cur2d[a]
        ok = (Xn[:, 2] > DEPTH_RANGE[0]) & (Xn[:, 2] < DEPTH_RANGE[1])
        a, Xn, uv = a[ok], Xn[ok], uv[ok]
        if len(a) > self.max_carry:                                              # keep the heaviest (best measured) ones
            top = np.argsort(-self.kf_w[a])[:self.max_carry]
            a, Xn, uv = a[top], Xn[top], uv[top]
        return uv.astype(np.float32), Xn, self.kf_w[a], self.kf_tid[a]

    def _new_keyframe(self, gray, disp, trusted, carry=False):
        old = self._carried() if carry else None
        p2, p3 = self._seed(gray, disp, trusted, None if old is None else old[0])
        w = None if p2 is None else self._weight(p3[:, 2])
        n_new = 0 if p2 is None else len(p2)
        tid = np.arange(self._next_tid, self._next_tid + n_new)
        if old is not None:
            p2 = old[0] if p2 is None else np.concatenate([old[0], p2])
            p3 = old[1] if p3 is None or not len(p3) else np.concatenate([old[1], p3])
            w = old[2] if w is None else np.concatenate([old[2], w])
            tid = np.concatenate([old[3], tid])
        self._next_tid += n_new
        self.kf_tid = None if p2 is None else tid
        self.kf_gray, self.kf2d, self.kf3d, self.cur2d = gray, p2, p3, None if p2 is None else p2.copy()
        self.alive = None if p2 is None else np.ones(len(p2), bool)
        self.kf_w = w
        self.R_wk, self.t_wk, self.kf_age = self.R_wc.copy(), self.t_wc.copy(), 0
        self.n_keyframes += 1
        node = {"id": self.n_keyframes, "R": self.R_wk, "t": self.t_wk, "path": self.path_m}
        if self.node is not None:
            node["odo"] = pg.odometry_edge(self.node["id"], node["id"], self.node["R"], self.node["t"], node["R"], node["t"])     # the motion the odometry measured since the last keyframe
        if self.window_ba and self.kf2d is not None and len(self.kf2d):
            u, v = np.clip(self.kf2d[:, 0].astype(int), 0, disp.shape[1] - 1), np.clip(self.kf2d[:, 1].astype(int), 0, disp.shape[0] - 1)
            d = disp[v, u].astype(np.float64)
            node["obs"] = (self.kf_tid.copy(), self.kf2d.astype(np.float64).copy(), np.where(np.isfinite(d) & (d > 0.5) & np.asarray(trusted, bool)[v, u], d, np.nan))
        self.kfs[node["id"]] = node
        self.kf_ids.append(node["id"])
        while len(self.kf_ids) > self.max_keyframes:
            del self.kfs[self.kf_ids.pop(0)]
        self.node, self.rec = node, None
        if self._need_orb:
            f = self._orb_features(gray, disp, trusted)
            if f is not None and len(f[0]) >= RELOC_MIN_INLIERS:
                node.update(desc=f[0], uv=f[1], xyz=f[2])
                self.rec = node
                self.recs.append(node)
                del self.recs[:-self.max_keyframes]
        return p2 is not None and len(p2) >= MIN_FEATURES

    def correct(self, R_wc, t_wc):
        """The pose was refined from outside. Move the keyframe by the same rigid change so later frames keep the correction."""
        Rc = np.asarray(R_wc) @ self.R_wc.T
        self.t_wk = Rc @ self.t_wk + (np.asarray(t_wc) - Rc @ self.t_wc)
        self.R_wk = Rc @ self.R_wk
        self.R_wc, self.t_wc = np.array(R_wc, float), np.array(t_wc, float)
        if self.node is not None:
            self.node["R"], self.node["t"] = self.R_wk, self.t_wk                # the graph node of the current keyframe follows it

    def _locate(self, rec, desc_q, uv_q, min_inliers=RELOC_MIN_INLIERS):
        """World pose of a camera that sees (desc_q, uv_q), from the 3D points stored with keyframe `rec`. None if too few features agree."""
        if len(desc_q) < min_inliers or len(rec["desc"]) < min_inliers:
            return None
        pairs = self._bf.knnMatch(desc_q, rec["desc"], 2)
        good = [(m.queryIdx, m.trainIdx) for p in pairs if len(p) == 2 for m, n in [p] if m.distance < 0.8 * n.distance]
        if len(good) < min_inliers:
            return None
        qi, ti = np.array(good).T
        obj, img = rec["xyz"][ti].astype(np.float64), uv_q[qi].astype(np.float64)
        ok, rvec, tvec, inl = _pnp(obj, img, self.K, 300, 3.0)
        if not ok or inl is None or len(inl) < min_inliers:
            return None
        sel = inl.reshape(-1)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj[sel], img[sel], self.K, None, rvec, tvec)
        except cv2.error:
            return None
        Rm, _ = cv2.Rodrigues(rvec)
        R_rc, t_rc = Rm.T, -Rm.T @ tvec.reshape(3)
        return rec["R"] @ R_rc, rec["t"] + rec["R"] @ t_rc, len(sel), R_rc, t_rc

    def _relocalize(self, gray):
        f = self._orb_features(gray)
        if f is None:
            return None
        for rec in reversed(self.recs[-self.reloc_keyframes:]):
            r = self._locate(rec, f[0], f[1])
            if r is not None and np.linalg.norm(r[1] - self.t_wc) < RELOC_MAX_JUMP_M:
                return r[0], r[1], r[2], rec["id"]
        return None

    def _similar(self, cur, old, top=3, min_votes=25):
        """Place recognition by appearance: every ORB descriptor of the new keyframe votes for the old keyframe whose descriptor it matches best (ratio test), so a place
        seen before is found however far the odometry has drifted from it. The votes only choose WHICH keyframes to verify; they prove nothing (see _locate)."""
        desc = np.concatenate([r["desc"] for r in old])
        owner = np.repeat(np.arange(len(old)), [len(r["desc"]) for r in old])
        votes = np.zeros(len(old), int)
        for p in self._bf.knnMatch(cur["desc"], desc, 2):
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance:
                votes[owner[p[0].trainIdx]] += 1
        return [old[i] for i in np.argsort(-votes)[:top] if votes[i] >= min_votes]

    def _loop_check(self, info):
        """Compare the new keyframe with old keyframes: those near where we believe we are, and (loop_appearance) those that look like this place wherever the
        odometry thinks they are. A match is VERIFIED by PnP on the old keyframe's 3D points (RANSAC, at least 40 inliers) and must not disagree with the odometry by
        more than loop_max_drift_m or a share of the path walked since: a loop that would need a larger jump is a look-alike, not a revisit. With loop_apply it must
        also be CONFIRMED by the next check (the same old place, the same drift vector within 25 cm) before it becomes an edge of the pose graph, because one wrong
        edge bends the whole path. The graph is then optimised (off the tracking thread when graph_async) and the poses of all keyframes move."""
        cur = self.rec
        if cur is None or self.n_keyframes % self.loop_every:
            return
        old = [r for r in self.recs[:-1] if self.path_m - r["path"] >= self.loop_min_path_m]
        cand = sorted([r for r in old if np.linalg.norm(r["t"] - self.t_wc) < self.loop_radius_m], key=lambda r: np.linalg.norm(r["t"] - self.t_wc))[:3]
        if self.loop_appearance and old:
            cand += [r for r in self._similar(cur, old) if all(r["id"] != c["id"] for c in cand)]
        for rec in cand:
            r = self._locate(rec, cur["desc"], cur["uv"], min_inliers=40)
            if r is None:
                continue
            R_l, t_l, n_in, R_rc, t_rc = r
            drift_m, drift_deg = float(np.linalg.norm(t_l - self.t_wc)), _angle_deg(self.R_wc, R_l)
            max_m = min(self.loop_max_drift_m, 0.3 + 0.15 * (self.path_m - rec["path"]))
            plausible = n_in >= self.loop_apply_inliers and drift_m <= max_m and drift_deg <= 25.0
            vec = t_l - self.t_wc
            prev, self._loop_prev = self._loop_prev, (self.n_keyframes, rec["id"], vec)
            confirmed = prev is not None and self.n_keyframes - prev[0] <= 3 * self.loop_every and abs(rec["id"] - prev[1]) <= 4 and float(np.linalg.norm(vec - prev[2])) < 0.25
            apply = bool(self.loop_apply and plausible and confirmed)
            info["loop"] = {"keyframe": rec["id"], "inliers": n_in, "drift_cm": drift_m * 100, "drift_deg": drift_deg, "applied": apply, "plausible": bool(plausible), "confirmed": bool(confirmed)}
            self.stats["loops_found"] += 1
            if apply:
                self.loops.append(pg.loop_edge(rec["id"], cur["id"], R_rc, t_rc))
                del self.loops[:-50]
                self.stats["loops_applied"] += 1
                self._start_graph()
            return

    # ---- pose graph ----------------------------------------------------------------------------------------------------------------------------------
    def _lm_collect(self, index):
        """Landmark factors for the graph: only objects seen from at least two keyframes at least lm_min_gap keyframes apart (one look ties nothing to anything). Returns None when there are none,
        else (obs [(node, slot, p_k, sigma)], X0 (L,3), landmark ids): start positions are the last solution, or the mean of where the looks put it with the poses as they are now."""
        if self.landmark_source is None:
            return None
        by = {}
        for lid, kf, p_k, sg in self.landmark_source():
            if kf in index:
                by.setdefault(lid, []).append((kf, np.asarray(p_k, float), float(sg)))
        by = {l: o for l, o in by.items() if len({k for k, _, _ in o}) >= 2 and max(k for k, _, _ in o) - min(k for k, _, _ in o) >= self.lm_min_gap}
        if not by:
            return None
        lids = sorted(by)
        obs, X0 = [], []
        for slot, lid in enumerate(lids):
            guess = [self.kfs[k]["R"] @ p + self.kfs[k]["t"] for k, p, _ in by[lid]]
            X0.append(self.lm_pos.get(lid, np.mean(guess, axis=0)))
            obs += [(index[k], slot, p, sg) for k, p, sg in by[lid]]
        return obs, np.array(X0), lids

    def _snapshot(self):
        ids = list(self.kf_ids)
        index = {k: i for i, k in enumerate(ids)}
        edges = []
        for k in ids:
            e = self.kfs[k].get("odo")
            if e is not None and e[0] in index:
                edges.append((index[e[0]], index[e[1]]) + tuple(e[2:]))
        edges += [(index[e[0]], index[e[1]]) + tuple(e[2:]) for e in self.loops if e[0] in index and e[1] in index]
        return ids, np.stack([self.kfs[k]["R"] for k in ids]), np.stack([self.kfs[k]["t"] for k in ids]), edges, self._lm_collect(index)

    def _solve(self, snap):
        ids, R, t, edges, lm = snap
        if lm is None:
            R2, t2, info = pg.optimize(R, t, edges)
        else:
            R2, t2, info = pg.optimize(R, t, edges, lm_obs=lm[0], lm_x=lm[1])
            info["lm_ids"] = lm[2]
        self._gout = (ids, R, t, R2, t2, info)

    def _landmark_check(self):
        """At a new keyframe: when an object has been seen again from a keyframe well apart from its earlier ones, solve the graph with the objects as factors (off the tracking thread)."""
        if self.landmark_source is None or self.n_keyframes - self._lm_kf < self.lm_every:
            return
        lm = self._lm_collect({k: i for i, k in enumerate(self.kf_ids)})
        if lm is None or len(lm[0]) - self._lm_n < self.lm_new_obs:      # nothing new to learn from since the last solve: it would give the same answer
            return
        self._lm_kf, self._lm_n = self.n_keyframes, len(lm[0])
        self.stats["landmark_runs"] += 1
        self._start_graph()

    def _start_graph(self):
        if self._gthread is not None and self._gthread.is_alive():
            self._gdirty = True                                          # a run is going; the new loop edge is included in the next one
            return
        snap = self._snapshot()
        self.stats["graph_runs"] += 1
        if self.graph_async:
            self._gthread = threading.Thread(target=self._solve, args=(snap,), name="pose-graph", daemon=True)
            self._gthread.start()
        else:
            self._solve(snap)
            self._apply_graph()

    def flush(self):
        """Wait for a running window adjustment and graph optimisation and apply their results (used at the end of a run and in tests)."""
        self.flush_ba()
        while self._gthread is not None and self._gthread.is_alive():
            self._gthread.join(10)
        self._apply_graph()

    def _apply_graph(self):
        """Take over a finished optimisation. Nodes that existed when it started get the optimised poses. Nodes made since then, the current keyframe and the
        current pose are moved by the same rigid change as the newest optimised node, so the path stays continuous."""
        out, self._gout = self._gout, None
        if out is None:
            if self._gdirty and not (self._gthread is not None and self._gthread.is_alive()):
                self._gdirty = False
                self._start_graph()
            return
        ids, R_old, t_old, R_new, t_new, info = out
        live = [k for k in ids if k in self.kfs]
        if not live:
            return
        if info.get("lm_ids") is not None and (info["max_shift_m"] > self.lm_max_shift_m or info["max_rot_deg"] > self.lm_max_rot_deg):
            self.stats["landmark_rejected"] += 1                          # the objects ask for a bigger change than any drift we believe in: a wrong association, not a correction
            self.last_graph = dict(info, rejected=True, keyframes=len(ids))
            return
        if info.get("lm_ids") is not None and not (info["lm_cm_before"] >= LM_MIN_RESID_CM and info["lm_cm_after"] <= LM_GAIN_FRAC * info["lm_cm_before"]):
            self.stats["landmark_unhelpful"] += 1                         # the objects were already consistent with the path (or the fit does not make them more so): moving keyframes would only add noise and a whole-map rebuild
            self.last_graph = dict(info, skipped=True, keyframes=len(ids))
            return
        last = ids.index(live[-1])
        Rc = R_new[last] @ R_old[last].T
        tc = t_new[last] - Rc @ t_old[last]
        self.last_correction = (Rc, tc)                                   # what moved the current pose: the caller can show it gradually instead of as a jump
        for k, Rn, tn in zip(ids, R_new, t_new):
            if k in self.kfs:
                self.kfs[k]["R"], self.kfs[k]["t"] = Rn, tn
        for k in self.kf_ids:
            if k > ids[-1]:
                n = self.kfs[k]
                n["R"], n["t"] = Rc @ n["R"], Rc @ n["t"] + tc
        self.R_wc, self.t_wc = Rc @ self.R_wc, Rc @ self.t_wc + tc
        self.R_wk, self.t_wk = self.node["R"], self.node["t"]
        if info.get("lm_ids") is not None:
            self.lm_pos.update({lid: x for lid, x in zip(info["lm_ids"], info["landmarks"])})
            self.stats["landmark_applied"] += 1
        self.graph_version += 1
        if info.get("lm_ids") is None or info["max_shift_m"] >= self.lm_rebuild_m or info["max_rot_deg"] >= self.lm_rebuild_deg:
            self.loop_version += 1                                        # whoever holds points placed by keyframes re-lays them (a whole-map rebuild: not for a correction of a few millimetres)
        self.last_graph = dict(info, correction_cm=float(np.linalg.norm(tc)) * 100, keyframes=len(ids))

    # ---- sliding-window bundle adjustment ------------------------------------------------------------------------------------------------------------
    def _start_ba(self):
        ids = self.kf_ids[-self.ba_window:]
        if len(ids) < 4:
            return
        if self._bthread is not None and self._bthread.is_alive():
            self.stats["ba_skipped_busy"] += 1                               # the last adjustment is still running: skip this keyframe rather than queue work
            return
        nodes = [self.kfs[k] for k in ids]
        if any("obs" not in n for n in nodes):
            return
        count = {}
        for n in nodes:
            for tid in n["obs"][0]:
                count[int(tid)] = count.get(int(tid), 0) + 1
        keep = [tid for tid, c in count.items() if c >= 3]
        if len(keep) < 20:
            return
        keep.sort(key=lambda tid: -count[tid])
        keep = keep[:self.ba_max_landmarks]
        lm_index = {tid: i for i, tid in enumerate(keep)}
        on, ol, ouv, od = [], [], [], []
        first = {}
        for j, n in enumerate(nodes):
            tids, uv, d = n["obs"]
            for k, tid in enumerate(tids):
                i = lm_index.get(int(tid))
                if i is None:
                    continue
                on.append(j), ol.append(i), ouv.append(uv[k]), od.append(d[k])
                if i not in first and np.isfinite(d[k]):
                    first[i] = (j, uv[k], d[k])                              # the earliest valid stereo depth gives the starting position
        X0 = np.full((len(keep), 3), np.nan)
        for i, (j, uv, d) in first.items():
            z = self.fx * self._B / d
            Xc = np.array([(uv[0] - self.cx) / self.fx * z, (uv[1] - self.cy) / self.fy * z, z])
            X0[i] = nodes[j]["R"] @ Xc + nodes[j]["t"]
        ok_l = np.isfinite(X0[:, 0])
        sel = ok_l[np.array(ol)]
        if sel.sum() < 60:
            return
        job = (ids, np.stack([n["R"] for n in nodes]), np.stack([n["t"] for n in nodes]), np.array(on)[sel], np.array(ol)[sel], np.array(ouv)[sel], np.array(od)[sel],
               np.where(ok_l[:, None], X0, 0.0), keep, self.K.copy(), self._B)
        self.stats["ba_runs"] += 1
        if self.graph_async:
            self._bthread = threading.Thread(target=self._ba_solve, args=(job,), name="window-ba", daemon=True)
            self._bthread.start()
        else:
            self._ba_solve(job)
            self._apply_ba()

    def _ba_solve(self, job):
        ids, R, t, on, ol, ouv, od, X0, tids, K, B = job
        t_start = time.perf_counter()
        R2, t2, X2, info = localba.optimize(K, B, R, t, 2, on, ol, ouv, od, X0)
        info["ms"] = (time.perf_counter() - t_start) * 1000
        self._bout = (ids, R, t, R2, t2, X2, tids, info)

    def _apply_ba(self):
        """Take over a finished window adjustment: the window's keyframes get the refined poses; later keyframes, the current pose and the current keyframe are
        moved by the same rigid change as the newest window keyframe; the odometry edges inside the window are re-measured from the refined poses (they are the
        better measurements the pose graph should use); and the current keyframe's landmarks that the adjustment also refined take the refined positions."""
        out, self._bout = self._bout, None
        if out is None:
            return
        ids, R_old, t_old, R_new, t_new, X, tids, info = out
        live = [k for k in ids if k in self.kfs]
        if len(live) < len(ids):
            return
        Rc = R_new[-1] @ R_old[-1].T
        tc = t_new[-1] - Rc @ t_old[-1]
        for j, k in enumerate(ids):
            self.kfs[k]["R"], self.kfs[k]["t"] = R_new[j], t_new[j]
        for j in range(1, len(ids)):
            a, b = self.kfs[ids[j - 1]], self.kfs[ids[j]]
            b["odo"] = pg.odometry_edge(a["id"], b["id"], a["R"], a["t"], b["R"], b["t"])
        for k in self.kf_ids:
            if k > ids[-1]:
                n = self.kfs[k]
                n["R"], n["t"] = Rc @ n["R"], Rc @ n["t"] + tc
        self.R_wc, self.t_wc = Rc @ self.R_wc, Rc @ self.t_wc + tc
        self.last_correction = (Rc, tc)                                     # shown gradually by the caller, like a graph correction
        self.R_wk, self.t_wk = self.node["R"], self.node["t"]
        if self.kf_tid is not None and self.kf3d is not None:               # the current keyframe may be a newer one than the window's: its pose was moved by the same change, so the formula holds
            pos = {int(tid): i for i, tid in enumerate(tids)}
            moved = 0
            for a, tid in enumerate(self.kf_tid):
                i = pos.get(int(tid))
                if i is None or not self.alive[a]:
                    continue
                Xk = (X[i] - self.t_wk) @ self.R_wk
                if DEPTH_RANGE[0] < Xk[2] < DEPTH_RANGE[1] and np.linalg.norm(Xk - self.kf3d[a]) < 0.3:
                    self.kf3d[a] = Xk
                    self.kf_w[a] = max(self.kf_w[a], FUSE_W_CAP * self._weight(Xk[2]) * 0.5)          # a landmark seen from several keyframes is better known than one stereo frame says
                    moved += 1
            info["landmarks_updated"] = moved
        self.stats["ba_applied"] += 1
        self.last_ba = dict(info, correction_cm=float(np.linalg.norm(tc)) * 100)
        self.graph_version += 1

    def flush_ba(self):
        while self._bthread is not None and self._bthread.is_alive():
            self._bthread.join(10)
        self._apply_ba()

    def node_pose(self, kf_id):
        """(R_wk, t_wk) of keyframe kf_id as the pose graph has it now, or None when that keyframe has been forgotten."""
        n = self.kfs.get(kf_id)
        return None if n is None else (n["R"], n["t"])

    def keyframe_poses(self):
        return {k: (self.kfs[k]["R"], self.kfs[k]["t"]) for k in self.kf_ids}

    def trajectory(self):
        """Every frame's pose rebuilt from the current keyframe poses: the path as it is NOW, after any loop correction. Frames whose keyframe has been dropped are skipped."""
        R, t = [], []
        for k, R_kc, t_kc in self.frame_log:
            n = self.kfs.get(k)
            if n is not None:
                R.append(n["R"] @ R_kc)
                t.append(n["t"] + n["R"] @ t_kc)
        return np.array(R), np.array(t)

    # ---- tracking ------------------------------------------------------------------------------------------------------------------------------------
    def _predict(self, dR_prior, frame_s=None):
        """Constant velocity in time, not per frame: a pair that arrives after a longer wait than the last one is predicted proportionally farther (up to 3x, down to 0.3x).
        A rotation handed in by the caller (a gyro) is already for this interval and is used as it is."""
        if dR_prior is None and not (self.predict and self.has_vel):
            return None
        k = 1.0 if not (frame_s and self.vel_s) else float(np.clip(frame_s / self.vel_s, 0.3, 3.0))
        if dR_prior is not None:
            dR = np.asarray(dR_prior, float)
        elif k == 1.0:
            dR = self.dR
        else:
            dR = Rotation.from_rotvec(Rotation.from_matrix(self.dR).as_rotvec() * k).as_matrix()
        dt = self.dt * k if self.has_vel else np.zeros(3)
        return self.R_wc @ dR, self.t_wc + self.R_wc @ dt

    def _project(self, idx, R_pred, t_pred):
        """Where the keyframe's features should be in the current picture if the camera has the predicted pose; features that would fall outside keep their last position."""
        R_kc, t_kc = self.R_wk.T @ R_pred, self.R_wk.T @ (t_pred - self.t_wk)
        Xc = (self.kf3d[idx] - t_kc) @ R_kc
        z = np.maximum(Xc[:, 2], 1e-3)
        uv = np.stack([self.fx * Xc[:, 0] / z + self.cx, self.fy * Xc[:, 1] / z + self.cy], 1).astype(np.float32)
        h, w = self.kf_gray.shape[:2]
        bad = (Xc[:, 2] < 0.1) | (uv[:, 0] < -20) | (uv[:, 0] > w + 20) | (uv[:, 1] < -20) | (uv[:, 1] > h + 20)
        uv[bad] = self.cur2d[idx][bad]
        return uv

    def _lk_pnp(self, gray, idx, init, win=21, levels=3):
        nxt, st, err = cv2.calcOpticalFlowPyrLK(self.kf_gray, gray, self.kf2d[idx].reshape(-1, 1, 2), init.reshape(-1, 1, 2).copy(), winSize=(win, win), maxLevel=levels,
                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        good = st.reshape(-1).astype(bool) & (err.reshape(-1) < LK_MAX_ERR)
        if good.any():                                              # a patch that does not look like the keyframe's is not a match, however well it sits on the prediction: on a picture with nothing in it LK returns the predicted spot and the "tracking" would only hand the prediction back as a measurement
            gi_ = np.nonzero(good)[0]
            good[gi_[_zncc(self.kf_gray, self.kf2d[idx][gi_], gray, nxt.reshape(-1, 2)[gi_]) < ZNCC_MIN]] = False
        out = {"tracked": int(good.sum()), "n_in": 0, "ok": False, "reason": f"only {int(good.sum())} features survived tracking (need {MIN_FEATURES})"}
        out["dead"] = idx[~good]
        if good.sum() < MIN_FEATURES:
            return out
        gi, cur = idx[good], nxt.reshape(-1, 2)[good]
        obj, img = self.kf3d[gi].astype(np.float64), cur.astype(np.float64)
        ok, rvec, tvec, inl = _pnp(obj, img, self.K, 200, 2.5)
        n_in = 0 if inl is None else len(inl)
        if self._fixR is not None:
            Rm = (self.R_wk.T @ self._fixR).T                                      # X_cur = Rm X_kf for the supplied rotation
            res = _t_only(obj, img, self.K, Rm)
            if res is not None and len(res[1]) >= MIN_INLIERS and (not ok or n_in < MIN_INLIERS or len(res[1]) >= FIX_KEEP_FRAC * n_in):     # the gyro's rotation wins unless the full pose has clearly more support
                rvec, tvec, inl, ok, n_in = cv2.Rodrigues(Rm)[0], res[0].reshape(3, 1), res[1].reshape(-1, 1), True, len(res[1])
                self.stats["fixrot_used"] += 1
            elif ok:
                self.stats["free_rot_used"] += 1                                  # the pictures' own rotation went into the pose this frame
        if self._fixR is not None and ok and n_in > 0:                              # what the pictures alone say about the rotation, against the gyro's: the caller learns the gyro's bias from it and warns when they disagree for long
            R_free = self.R_wk @ cv2.Rodrigues(rvec)[0].T
            out["free_rot_err"] = Rotation.from_matrix(R_free.T @ self._fixR).as_rotvec()       # (free -> gyro), camera frame, rad: since the keyframe
            out["free_inl"], out["free_vs_prior_deg"] = n_in, _angle_deg(R_free, self._fixR)
        out["n_in"], out["reason"] = n_in, f"only {n_in} tracked features agree on a camera motion (need {MIN_INLIERS})"
        if not ok or n_in < MIN_INLIERS:
            return out
        sel = inl.reshape(-1)
        if self.refine:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(obj[sel], img[sel], self.K, None, rvec, tvec)    # motion-only bundle adjustment on the inliers
            except cv2.error:
                pass                                                                              # keep the RANSAC pose
        keep = np.zeros(len(gi), bool)
        keep[sel] = True
        out.update(ok=True, gi=gi, cur=cur, keep=keep, rvec=rvec, tvec=tvec)
        return out

    def _track(self, gray, pred):
        idx = np.nonzero(self.alive)[0]
        if len(idx) < MIN_FEATURES:
            return {"ok": False, "tracked": 0, "n_in": 0, "reason": f"only {len(idx)} keyframe features left (need {MIN_FEATURES})", "dead": idx[:0]}
        inits = ([("predicted", self._project(idx, *pred))] if pred is not None else []) + [("previous", self.cur2d[idx])]
        best = None
        for k, (name, init) in enumerate(inits):
            r = self._lk_pnp(gray, idx, init)
            r["init"] = name
            if best is None or r["n_in"] > best["n_in"]:
                best = r
            if r["ok"] and r["n_in"] >= 0.5 * len(idx):
                break
            if k + 1 < len(inits):
                self.stats["fallback_init"] += 1
        if not best["ok"] and WIDE_RETRY:                           # a stall, a spin or a blur took every feature farther than the normal search reaches: look again with a wider window and two more pyramid levels
            for name, init in inits:                                # (capture range about 10 px * 2^levels: 80 px -> 320 px); only paid for on a frame that would otherwise be lost
                r = self._lk_pnp(gray, idx, init, win=31, levels=5)
                r["init"] = name + "+wide"
                if r["n_in"] > best["n_in"]:
                    best = r
                if r["ok"]:
                    self.stats["wide_ok"] += 1
                    break
        if best["init"] == "predicted" and best["ok"]:
            self.stats["predicted_ok"] += 1
        return best

    def step(self, gray, disp, trusted, dR_prior=None, frame_s=None, prior_over_gap=False, fix_rot=False):
        """Returns (R_wc, t_wc, info). info["ok"] False = the pose was held, with the reason. frame_s: seconds since the previous frame, when known."""
        self._apply_ba()
        self._apply_graph()
        gap = bool(frame_s and frame_s > GAP_S)
        self._fixR = None if (not fix_rot or dR_prior is None) else self.R_wc @ np.asarray(dR_prior, float)     # fix_rot: the rotation is the caller's (a gyro), only the translation is searched
        if gap:                                                   # the link stalled: the last velocity and a gyro that missed the gap say nothing about where the camera is now
            self.has_vel = False
            if not prior_over_gap:                                # unless the caller had gyro samples for the whole stall
                dR_prior = None
        R, t, info = self._step(gray, disp, trusted, dR_prior, frame_s, gap)
        info["loop_version"] = self.loop_version
        if self.node is not None:
            info["anchor"] = (self.node["id"], self.R_wk, self.t_wk)           # the keyframe this frame is placed relative to, and that keyframe's pose now
            self.frame_log.append((self.node["id"], self.R_wk.T @ self.R_wc, self.R_wk.T @ (self.t_wc - self.t_wk)))
            del self.frame_log[:-20000]
        return R, t, info

    def _step(self, gray, disp, trusted, dR_prior=None, frame_s=None, gap=False):
        self.n += 1
        self.last_flow = None                                        # only a frame that tracked features has any to show
        info = {"ok": False, "reason": "", "inliers": 0, "tracked": 0, "keyframe": False, "reproj_px": None, "relocalized": False, "loop": None, "init": None}
        if self.kf_gray is None:
            if self._new_keyframe(gray, disp, trusted):
                info.update(ok=True, reason="first frame", keyframe=True)
            else:
                info["reason"] = "no features with trusted depth to start from"
            return self.R_wc.copy(), self.t_wc.copy(), info
        self.kf_age += 1
        if self.kf2d is None or len(self.kf2d) < MIN_FEATURES:
            info["reason"] = "no features with trusted depth in the keyframe"
            return self._lost(gray, disp, trusted, info, no_pose=True, dead=True, dR_prior=dR_prior)
        res = self._track(gray, self._predict(dR_prior, frame_s))
        info["tracked"], info["inliers"], info["init"] = res["tracked"], res["n_in"], res.get("init")
        for k in ("free_rot_err", "free_inl", "free_vs_prior_deg"):
            if k in res:
                info[k] = res[k]
        self.alive[res["dead"]] = False
        no_pose = not res["ok"]
        if res["ok"]:
            gi, cur, keep, rvec, tvec = res["gi"], res["cur"], res["keep"], res["rvec"], res["tvec"]
            self.alive[gi[~keep]] = False                          # a feature that disagreed once is not trusted again from this keyframe
            self.cur2d[gi[keep]] = cur[keep]
            Rm, _ = cv2.Rodrigues(rvec)                           # X_cur = Rm X_kf
            R_kc, t_kc = Rm.T, -Rm.T @ tvec.reshape(3)            # current camera in the keyframe camera frame
            R_new, t_new = self.R_wk @ R_kc, self.t_wk + self.R_wk @ t_kc
            proj, _ = cv2.projectPoints(self.kf3d[gi[keep]].astype(np.float64), rvec, tvec, self.K, None)
            info["reproj_px"] = float(np.median(np.linalg.norm(proj.reshape(-1, 2) - cur[keep], axis=1)))
            bad_prior = False
            if dR_prior is not None:
                info["prior_disagreement_deg"] = _angle_deg(self.R_wc.T @ R_new, np.asarray(dR_prior, float))
                bad_prior = info["prior_disagreement_deg"] > self.prior_gate_deg
            step_m, step_deg = float(np.linalg.norm(t_new - self.t_wc)), _angle_deg(self.R_wc, R_new)
            if self.max_step_m and (step_m > self.max_step_m or step_deg > self.max_step_deg) and self.n > 2:
                info["reason"] = f"pose jump of {step_m:.2f} m / {step_deg:.0f} deg in one frame: not trusted, pose held"
            elif bad_prior:
                info["reason"] = f"visual rotation disagrees with the supplied rotation prior by {info['prior_disagreement_deg']:.1f} deg"
            else:
                if self.fuse_depth:
                    self._fuse_depth(gi[keep], cur[keep], disp, trusted, R_kc, t_kc)
                self.last_flow = (self.kf2d[gi[keep]].copy(), cur[keep].copy())
                self.dR, self.dt, self.has_vel, self.vel_s = self.R_wc.T @ R_new, self.R_wc.T @ (t_new - self.t_wc), True, frame_s
                self.path_m += float(np.linalg.norm(t_new - self.t_wc))
                self.R_wc, self.t_wc = R_new, t_new
                self.lost_n = 0
                info["ok"] = True
        else:
            info["reason"] = res["reason"]
        if info["ok"]:
            live = int(self.alive.sum())
            moved = float(np.linalg.norm(self.t_wc - self.t_wk))
            sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            smeared = len(self._sharp) >= 8 and sharp < SMEAR_FRAC * float(np.median(self._sharp))
            self._sharp.append(sharp)
            del self._sharp[:-30]
            due = (live < max(self.kf_min_inliers, self.kf_keep_frac * len(self.kf2d)) or moved > self.kf_dist_m or _angle_deg(self.R_wk, self.R_wc) > self.kf_rot_deg
                   or self.kf_age >= self.kf_max_frames)
            if due and smeared and self._defer < KF_DEFER_MAX and live >= 2 * self.kf_min_inliers:        # a keyframe taken on a smeared picture has smeared patches: every later frame is matched against them. Wait for a sharper one while the old keyframe still holds
                self._defer += 1
                self.stats["kf_deferred"] += 1
                due = False
            if due:
                self._defer = 0
                info["keyframe"] = self._new_keyframe(gray, disp, trusted, carry=True)
                if self.window_ba and self.n_keyframes % self.ba_every == 0:
                    self._start_ba()
                if self.loop_closure:
                    self._loop_check(info)
                if info["keyframe"]:
                    self._landmark_check()
            return self.R_wc.copy(), self.t_wc.copy(), info
        return self._lost(gray, disp, trusted, info, no_pose, dR_prior=dR_prior, gap=gap)

    def _lost(self, gray, disp, trusted, info, no_pose, dead=False, dR_prior=None, gap=False):
        """No usable pose this frame. Try the ORB relocaliser (at once when the keyframe has no features left, else after reloc_after lost frames in a row:
        one lost frame is usually a blur that tracking recovers from by itself, and that keeps the more exact keyframe pose). Otherwise hold the pose, and
        after reanchor_after frames start a new keyframe at the held pose."""
        self.lost_n += 1
        self.has_vel = False
        if dR_prior is not None:                                  # the pose is held, but the camera is not: turn it with the gyro so the next search and a new keyframe start at the right orientation (translation stays)
            self.R_wc = self.R_wc @ np.asarray(dR_prior, float)
            info["held_rotated_deg"] = _angle_deg(np.eye(3), np.asarray(dR_prior, float))
        if no_pose and self.relocalize and self.recs and (dead or gap or self.lost_n >= self.reloc_after):
            r = self._relocalize(gray)
            if r is not None and dR_prior is not None and _angle_deg(r[0], self.R_wc) > RELOC_GYRO_GATE_DEG:      # self.R_wc was turned by the gyro just above: the match must agree with it
                self.stats["reloc_gyro_rejected"] += 1
                r = None
            if r is not None:
                self.R_wc, self.t_wc = r[0], r[1]
                self.stats["relocalized"] += 1
                info.update(ok=True, relocalized=True, reason=f"relocalised against keyframe {r[3]} ({r[2]} features)")
                info["keyframe"] = self._new_keyframe(gray, disp, trusted)
                self.lost_n = 0
                return self.R_wc.copy(), self.t_wc.copy(), info
        if self.lost_n >= self.reanchor_after or (not dead and int(self.alive.sum()) < MIN_FEATURES):
            info["keyframe"] = self._new_keyframe(gray, disp, trusted)       # start over from this frame at the held pose
            self.lost_n = 0
        return self.R_wc.copy(), self.t_wc.copy(), info
