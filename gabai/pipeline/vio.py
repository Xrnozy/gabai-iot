"""stereo_imu: the stereo pictures and the BMI160 IMU fused in Python, no Basalt.

Pictures give the position and the rotation (the same KeyframeVO as stereo_only, with the same depth); the IMU adds three things the pictures cannot:
  1. the gyro rotation between two frames starts the feature search and carries the pose through a lost-tracking stretch (bias removed, estimated while the rig is at rest);
  2. the accelerometer fixes the world: z is up from the first frame on, and roll/pitch drift of the pictures is pulled back to gravity a little every quiet frame
     (yaw is not observable from gravity, it stays what the pictures say);
  3. while the IMU says the rig is at rest the reported pose is frozen, so a rig on a table does not wander.
IMU samples come in IMU axes; R_ic (camera axes in IMU axes, from the calibration) turns them into camera axes. Loop closure and map refinement are off: both would move the
pose by a transform that the gravity alignment here does not know about.
"""
import collections

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation

from gabai.diag import log as dlog
from gabai.pipeline.poses import PoseResult

log = dlog.get("pose")
G = 9.80665
WARMUP = 50                       # IMU samples before the first pose (0.5 s at 100 Hz)
GRAV_TAU_S, GRAV_MAX_DPS, GRAV_REJECT_DEG = 3.0, 5.0, 45.0     # tilt is pulled to the accelerometer's gravity with a time constant of 3 s and at most 5 deg/s: in TIME, not per frame (a per-frame gain was 5x too slow when the Pi delivers 3 pairs a second)
BIAS_MIN_INLIERS, BIAS_GAIN, BIAS_MAX_DPS = 40, 0.0, 3.0   # learn the gyro's bias from how far its rotation since the keyframe ends up from the pictures' own (0.05 = on). OFF: on the stutter recording it gave no gain beyond run-to-run noise (20 vs 23 deg) and was worse with blur (73 vs 46-57 deg): the blurred pictures' rotation is the thing that is wrong there
COAST_S = 2.0                     # after the pictures are lost the gyro may turn the held pose this long; past it the pose stands still (an unaided gyro drifts, and that would be invented motion)
USE_GYRO_ACROSS_STALLS = True   # False = the behaviour before 2026-11-01 (scripts/evaluate_stutter.py --baseline compares the two)
SMOOTH_TAU_S = 1.0                  # a pose-graph correction is shown by easing the pose to it over about this long, not as a jump (the keyframes, the map and the objects take the corrected values at once)
STALL_S = 0.15                    # a pair this long after the previous one: the pictures moved far between the frames, so the gyro's rotation is used for the pose and only the translation is searched
MAX_SPEED_MPS, STEP_SLACK_M = 3.0, 0.05      # a hand-held rig does not move faster; a picture pose that does is a tracking error, not motion


def tilt_to_z(up):
    """Smallest rotation that turns the unit vector `up` into world +z."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    v = np.cross(up, [0.0, 0.0, 1.0])
    s, c = np.linalg.norm(v), float(up[2])
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    return R.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


class VisualInertialProvider:
    name, uses_imu = "stereo_imu", True

    @property
    def wants_klt(self):
        """The pipeline hands over the Pi's tracks (and motion) only to a provider that uses them."""
        return self.fusion is not None or self.ekf is not None
    world = "gravity-aligned: z up (accelerometer, first frames), heading = the first camera's"

    def __init__(self, cal, R_ic, fuse_depth=True, klt=None, filt=None, **kfvo_options):
        """klt: a KltSection (gabai/pipeline/config.py) to fuse the Pi's feature tracks into the pose (gabai/kltfuse.py); None = the pictures and the IMU only, as before."""
        from gabai.kfvo import KeyframeVO
        self.ekf, self._ekf_out, self._motion_seen, self._n_frames = None, None, False, 0
        if filt is not None and filt.enabled:
            from gabai.vioekf import VioEkf
            self._max_dev = (filt.max_dev_cm / 100.0, np.radians(filt.max_dev_deg))
            self.ekf = VioEkf(filt.pos_sigma_m, filt.att_sigma_deg, filt.klt_sigma_m, filt.klt_sigma_deg, filt.gyro_noise, filt.acc_noise, gate_chi2=filt.gate_chi2)
        self.fusion = None
        if klt is not None and klt.enabled:
            from gabai.kltfuse import KltFusion
            self.fusion = KltFusion(cal, klt.weight, klt.min_inliers, klt.max_reproj_px, klt.agree_m, klt.max_step_m)
        self.R_ic = np.asarray(R_ic, float)
        opts = dict(kfvo_options, loop_closure=False, loop_apply=False, prior_gate_deg=25.0)      # the gyro only steers the search; a gross disagreement (wrong axes) holds the pose
        self.vo = KeyframeVO(cal["f"], cal["f"], cal["cx"], cal["cy"], fuse_depth=fuse_depth, **opts)
        self.vo.set_baseline(cal["B"])
        self.win = collections.deque()                         # (t, gyro, accel) in camera axes, last 0.4 s
        self.bias, self.bias_known = np.zeros(3), False
        self.g_c, self.n_imu, self.rest_now = None, 0, False  # accelerometer low-pass (camera axes, points up), samples seen
        self._dR, self._n_frame_imu, self._t_imu, self._t_prev = R.identity(), 0, None, None
        self.R_w, self.R_c = None, np.eye(3)                   # world alignment (set once), slow gravity correction
        self.held, self.Rh, self.th = None, np.eye(3), np.zeros(3)
        self._S, self._gv = (np.eye(3), np.zeros(3)), 0                          # offset still to be eased out (rotation, translation, odometry frame), graph version seen
        self.disagree, self._kf_t = collections.deque(maxlen=60), None                        # gyro-vs-pictures disagreement (deg) of the last frames; time of the current keyframe
        self.tilt_corr_deg = 0.0
        self._cover_s = 0.0                                    # seconds of gyro integrated since the last frame
        self.last_good, self.lost_since = None, None           # last accepted odometry pose (its own world), time the pictures were lost

    # --- IMU -----------------------------------------------------------------------------------------------------------------------------------------
    def _at_rest(self):
        if len(self.win) < 10 or self.win[-1][0] - self.win[0][0] < 0.36:
            return False
        g = np.array([w[1] for w in self.win])
        a = np.array([np.linalg.norm(w[2]) for w in self.win])
        m = g.mean(axis=0)
        return bool(np.linalg.norm(m) < np.radians(3.0) and np.linalg.norm(g - m, axis=1).max() < np.radians(1.5)
                    and a.max() - a.min() < 0.5 and abs(a.mean() - G) < 0.8)

    def feed_imu(self, samples):
        """(t s, gyro rad/s, accel m/s^2) rows in IMU axes, oldest first."""
        for t, gyr, acc in samples:
            g_c, a_c = self.R_ic.T @ np.asarray(gyr, float), self.R_ic.T @ np.asarray(acc, float)
            self.win.append((float(t), g_c, a_c))
            if self.ekf is not None and self.ekf.ready:
                self.ekf.predict([(float(t), g_c, a_c)])
            while t - self.win[0][0] > 0.4:
                self.win.popleft()
            self.rest_now = self._at_rest()
            if self.rest_now:
                mean = np.mean([w[1] for w in self.win], axis=0)
                self.bias = self.bias + 0.02 * (mean - self.bias) if self.bias_known else mean
                self.bias_known = True
            if abs(np.linalg.norm(a_c) - G) < 0.8:
                self.g_c = a_c.copy() if self.g_c is None else 0.98 * self.g_c + 0.02 * a_c
            if self._t_imu is not None and 0 < t - self._t_imu < 0.1:        # rotation lost across a longer gap is unknown: that frame's prior is short, the pictures' gate decides
                self._dR = self._dR * R.from_rotvec((g_c - self.bias) * (t - self._t_imu))
                self._n_frame_imu += 1
                self._cover_s += t - self._t_imu
            self._t_imu = float(t)
            self.n_imu += 1

    @property
    def ready(self):
        return self.n_imu >= WARMUP and self.g_c is not None

    def waiting_lines(self):
        return [f"waiting for IMU data: {self.n_imu} samples (need {WARMUP}), accelerometer {'read' if self.g_c is not None else 'no usable reading yet'}"]

    # --- pose ----------------------------------------------------------------------------------------------------------------------------------------
    def _align(self):
        up_world_vo = self.vo.R_wc @ self.g_c                  # the true up, in the odometry's world (which assumes the first camera is level)
        self.R_w = tilt_to_z(up_world_vo)
        tilt = np.degrees(np.arccos(np.clip(up_world_vo[2] / np.linalg.norm(up_world_vo), -1, 1)))
        if tilt > 90:
            log.warning("pose.imu_orientation", tilt_from_level_deg=round(float(tilt), 1), hint="gravity says the camera is upside down, so the world (map) comes out turned 180 deg. Unless the rig really is upside down, "
                        "the camera-IMU rotation in the imu_sync file is off by 180 deg about the optical axis: redo gabai.tools.sync_imu_camera and check_extrinsics --write")
        log.info("pose.imu_aligned", tilt_from_level_deg=round(float(tilt), 1), gyro_bias_dps=[round(float(v), 3) for v in np.degrees(self.bias)], bias_known=self.bias_known,
                 hint="z is up from here; tilt = how far the first camera was from level")

    def push(self, t_ns, gray_left, gray_right, disp, trusted, klt=None):
        """klt: this frame's KltObs (the Pi's tracked points in the left picture) when klt.enabled."""
        if not self.ready:
            return []
        if self.R_w is None:
            self._align()
        kw = {"frame_s": None if self._t_prev is None else (t_ns - self._t_prev) * 1e-9}
        coasting_over = self.lost_since is not None and (t_ns - self.lost_since) * 1e-9 > COAST_S
        if self._n_frame_imu and not coasting_over:
            kw["dR_prior"] = self._dR.as_matrix()
        else:
            log.warning_every("imu_silent", 5.0, "pose.imu_silent", hint="no gyro samples since the last frame: the pictures alone steer the tracking and a lost pose cannot turn; "
                              "on the Pi look at pi.stats imu_hz / i2c_errors")
        covered = kw["frame_s"] is not None and self._cover_s >= 0.8 * kw["frame_s"]
        if covered and USE_GYRO_ACROSS_STALLS:                 # the IMU delivered the whole stall: the gyro rotation is good over it, the constant-velocity guess is not
            kw["prior_over_gap"] = True
            kw["fix_rot"] = True                               # not only after stalls: the picture-to-picture rotation chain drifts (measured 13-33 deg over 5-30 s) while the gyro stays near 5 deg
        self._dR, self._n_frame_imu, self._t_prev, self._cover_s = R.identity(), 0, t_ns, 0.0
        Rv, tv, info = self.vo.step(gray_left, disp, trusted, **kw)
        if self.fusion is not None:                            # the Pi's KLT tracks as a second, favoured pose; the result also goes back into the tracker
            Rv, tv = self.fusion.apply(self.vo, klt, disp, trusted, Rv, tv, info)
        if info.get("free_vs_prior_deg") is not None:
            self.disagree.append(float(info["free_vs_prior_deg"]))
            if len(self.disagree) >= 30 and float(np.median(self.disagree)) > 10.0:
                log.warning_every("gyro_vs_pictures", 10.0, "pose.gyro_vs_pictures", median_deg=round(float(np.median(self.disagree)), 1),
                                  hint="the gyro's rotation and the pictures' disagree by more than 10 deg frame after frame: the camera-IMU rotation in the imu_sync file is probably wrong (redo sync_imu_camera and check_extrinsics)")
        if info.get("free_rot_err") is not None and info.get("free_inl", 0) >= BIAS_MIN_INLIERS and self._kf_t is not None:
            span = (t_ns - self._kf_t) * 1e-9
            if span > 0.3:
                e = np.asarray(info["free_rot_err"], float) / span                         # rad/s: the gyro turned this much too fast over the time since the keyframe
                if np.linalg.norm(e) < np.radians(BIAS_MAX_DPS):
                    self.bias, self.bias_known = self.bias + BIAS_GAIN * e, True
        if info.get("keyframe"):
            self._kf_t = t_ns
        if getattr(self.vo, "graph_version", 0) != self._gv:   # the pose graph moved the keyframes and the current pose: keep showing the old pose and ease to the new one
            self._gv = self.vo.graph_version
            self._smooth_in(*self.vo.last_correction)
        if info["ok"] and self.last_good is not None and kw["frame_s"] is not None:
            step = float(np.linalg.norm(tv - self.last_good[1]))
            if step > MAX_SPEED_MPS * min(kw["frame_s"], 0.5) + STEP_SLACK_M:
                info.update(ok=False, reason=f"picture pose jumped {step:.2f} m in {kw['frame_s']:.2f} s (limit {MAX_SPEED_MPS} m/s): not trusted, pose held")
                self.vo.correct(*self.last_good)               # the odometry continues from the last good pose, not from the jump
                Rv, tv = self.last_good
        if info["ok"]:
            self.last_good, self.lost_since = (Rv.copy(), tv.copy()), None
        else:
            self.lost_since = self.lost_since or t_ns
            log.warning_every("vo_lost", 5.0, "pose.vo_held", reason=info["reason"], inliers=info["inliers"], tracked=info["tracked"],
                              gyro_only=coasting_over, hint="position held; rotation follows the gyro for " + f"{COAST_S:.0f} s, then the pose stands still")
        if info["ok"] and self.win and abs(np.linalg.norm(self.win[-1][2]) - G) < 0.5:
            self._pull_to_gravity(Rv, kw["frame_s"] if kw["frame_s"] else 1 / 15)
        Rs, ts = self._smooth_out(kw["frame_s"] if kw["frame_s"] else 1 / 15)
        R_wc, t_wc = self.R_w @ self.R_c @ (Rs @ Rv), self.R_w @ self.R_c @ (Rs @ tv + ts)
        if self.ekf is not None:                               # Kalman filter: this frame's pose from the pictures is a measurement, the IMU has already predicted it
            R_wc, t_wc = self._filter(t_ns, R_wc, t_wc, info, klt)
        R_wc, t_wc, info["at_rest"] = self._hold(R_wc, t_wc)
        if info.get("anchor") is not None:                     # the keyframe this frame hangs on, in the frame the poses are reported in (the odometry's own frame is turned and held, see _out)
            kf, R_k, t_k = info["anchor"]
            M, th = self._out()
            info["anchor"] = (kf, M @ R_k, M @ t_k + th)
        if self.bias_known:
            info["gyro_bias_dps"] = float(np.degrees(np.linalg.norm(self.bias)))
        info["tilt_correction_deg"] = self.tilt_corr_deg
        return [PoseResult(int(t_ns), R_wc, t_wc, info["ok"], self.name, info)]

    def _filter(self, t_ns, R_wc, t_wc, info, klt):
        """The reported pose with the Kalman filter on: the filter's, as long as it is healthy and stays within max_dev of the raw picture pose; otherwise the raw pose."""
        e, ok = self.ekf, bool(info["ok"])
        self._n_frames += 1
        if not e.ready:
            if ok:
                e.init(R_wc, t_wc, self._t_imu if self._t_imu is not None else t_ns * 1e-9, self.bias if self.bias_known else None)
            return R_wc, t_wc
        if not e.healthy():                                    # blown up (a corrupted IMU read, a wrong timestamp): start again from the pictures, report the raw pose meanwhile
            log.warning_every("ekf_diverged", 5.0, "pose.ekf_diverged", diverged=e.stats["diverged"], hint="the filter left what a hand-held rig can do and was restarted from the pictures; the raw pose is reported. See pi.stats i2c_errors and filter/ekf_imu_rejected")
            e.init(R_wc if ok else e.R, t_wc if ok else e.p, e.t, np.clip(np.nan_to_num(e.bg), -0.05, 0.05))
            info.update(e.diag(), ekf_using_filter=0)
            return R_wc, t_wc
        if ok:
            e.update_pose(R_wc, t_wc, info.get("inliers"))
        mo = getattr(klt, "motion", None)
        if mo is not None:
            self._motion_seen = True
            e.update_klt(mo["rvec"], mo["t"], mo["dt"], mo["n_inl"], mo["rms"])
        elif self._n_frames > 90 and not self._motion_seen:
            log.warning_every("ekf_no_motion", 30.0, "pose.klt_motion_missing", hint="the filter runs without the Pi's own motion estimate: start the sender with --klt-motion (and --klt) and the pipeline with klt points enabled")
        if self.rest_now:
            e.update_rest()
        dev_m = float(np.linalg.norm(e.p - t_wc)) if ok else 0.0
        dev_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(e.R.T @ R_wc).as_rotvec()))) if ok else 0.0
        refused = ok and not e.last.get("pos_applied", True)  # the filter just refused the picture position as an outlier: that is its job, so it may stand farther from the raw pose (up to twice max_dev, and at most 5 frames in a row before it resets)
        use = e.healthy() and (not ok and e.since_pose < 0.5 or ok and dev_m <= self._max_dev[0] * (2 if refused else 1) and np.radians(dev_deg) <= self._max_dev[1])
        info.update(e.diag(), ekf_dev_cm=100 * dev_m, ekf_dev_deg=dev_deg, ekf_using_filter=int(use))
        if use:                                                # (without pictures the filter coasts on the IMU for half a second, then the pose holds)
            self._ekf_out = (e.R.copy(), e.p.copy())
            return self._ekf_out
        if ok:
            self._ekf_out = None
            return R_wc, t_wc                                  # the filter has drifted from the pictures: report them as they are
        return self._ekf_out if self._ekf_out is not None else (R_wc, t_wc)

    def _smooth_in(self, Rc, tc):
        """A correction C (the pose was P, now C P): show S P with S = S_old C^-1, which is still the old pose at this frame."""
        SR, St = self._S
        self._S = (SR @ Rc.T, SR @ (-Rc.T @ tc) + St)

    def _smooth_out(self, dt):
        """The offset to put on this frame's pose, then let it fade: exp(-dt / SMOOTH_TAU_S) of it is left for the next frame."""
        SR, St = self._S
        out = (SR.copy(), St.copy())
        f = float(np.exp(-min(dt, 5.0) / SMOOTH_TAU_S))
        rv = R.from_matrix(SR).as_rotvec() * f
        St = St * f
        self._S = (np.eye(3), np.zeros(3)) if np.linalg.norm(rv) < 1e-4 and np.linalg.norm(St) < 1e-4 else (R.from_rotvec(rv).as_matrix(), St)
        return out

    def _out(self):
        """(M, th): a point x of the odometry's frame is reported at M x + th (gravity-aligned world, then the rest hold)."""
        return self.Rh @ self.R_w @ self.R_c, self.th

    def node_pose(self, kf_id):
        """Keyframe pose as the pose graph has it now, in the reported frame (what cloud.rebuild and the objects need after a correction), or None."""
        n = self.vo.node_pose(kf_id)
        if n is None or self.R_w is None:
            return None
        M, th = self._out()
        return M @ n[0], M @ n[1] + th

    def landmark_positions(self):
        """Object positions as the last pose-graph solution put them, in the reported frame."""
        M, th = self._out()
        return {lid: M @ x + th for lid, x in self.vo.lm_pos.items()}

    def _pull_to_gravity(self, Rv, dt):
        """Turn the world a little so the camera's predicted 'up' meets the accelerometer's: removes roll/pitch drift, leaves yaw alone."""
        meas = self.g_c / np.linalg.norm(self.g_c)
        pred = (self.R_w @ self.R_c @ Rv).T @ np.array([0.0, 0.0, 1.0])
        r = np.cross(meas, pred)                              # camera-frame rotation that moves pred onto meas
        ang = np.degrees(np.linalg.norm(r))
        if ang > GRAV_REJECT_DEG:
            return                                            # an accelerometer that disagrees this much is shaken, not showing gravity
        r = r * min(1.0 - np.exp(-min(dt, 5.0) / GRAV_TAU_S), np.radians(GRAV_MAX_DPS) * min(dt, 5.0) / max(np.linalg.norm(r), 1e-9))
        self.R_c = self.R_c @ R.from_rotvec(Rv @ r).as_matrix()
        self.tilt_corr_deg = float(ang)

    def _hold(self, R_wc, t_wc):
        """Frozen pose while the IMU says rest; afterwards the pictures' motion continues from the frozen pose, so there is no jump."""
        R_r, t_r = self.Rh @ R_wc, self.Rh @ t_wc + self.th
        if not self.rest_now:
            self.held = None
            return R_r, t_r, False
        if self.held is None:
            self.held = (R_r, t_r)
        self.Rh = self.held[0] @ R_wc.T
        self.th = self.held[1] - self.Rh @ t_wc
        return self.held[0], self.held[1], True

    def correct(self, R_wc, t_wc):
        pass                                                  # map refinement is off in this mode: nothing moves the pose from outside (tried 2026-11-01: scan-to-map ICP here held 38 poses and did not reduce the rejected frames)

    def close(self):
        pass
