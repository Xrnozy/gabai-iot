"""Error-state Kalman filter for the stereo_imu pose, in the gravity-aligned world (z up, metres).

State: position p, velocity v, orientation R (camera to world), gyro bias b_g, accelerometer bias b_a (15 error states: dp dv dtheta dbg dba).
  predict   every IMU sample (camera axes): R from the bias-corrected gyro, v and p from the bias-corrected specific force plus gravity. The IMU is the primary source of rotation:
            its process noise is small, so a rotation measurement can nudge it but never override it.
  pose      KeyframeVO's pose of the frame: position (sigma_pos, widened when few features agreed) and orientation (sigma_att). A measurement farther than the chi-square gate from the
            prediction is refused (an outlier, not a correction); many in a row mean the filter, not the measurement, is wrong, and it is reset to the measurement.
  klt       the Pi's own motion estimate between two sent pairs (gabai/pi/klt.py MotionEstimator): its translation over dt is the mean velocity of that interval, its rotation over dt is
            the mean angular rate, which measures the gyro bias. It does not measure orientation or position directly, so it cannot pull the rotation away from the IMU.
  rest      the rig is at rest (the provider's detector): velocity 0 and angular rate 0, which pins the gyro bias.
Smooth by construction: every correction is a Kalman gain times an innovation, never a replacement.

The KLT motion and the VO pose come from the same pixels, so their errors are not independent. The filter treats them as two measurements and the defaults already widen both, but a
smaller klt_sigma_m than pos_sigma_m (KLT trusted more) makes the filter lean on a source that was derived from the same pictures: judge it by the innovation plots, not by the covariance.
"""
import numpy as np

G = 9.80665
GRAV = np.array([0.0, 0.0, -G])
GAP_S = 0.1                               # a longer hole between two IMU samples: no rotation or acceleration is integrated across it, only the velocity carries on and the covariance grows
COAST_S = 1.0                             # without any pose measurement for this long the position is not extrapolated any further by the caller (it holds)
RESET_AFTER = 5                           # this many refused pose measurements in a row: the filter is wrong, not the measurements
ACC_MAX, GYRO_MAX = 2.2 * G, 9.0          # the BMI160 reads +-2 g and +-500 deg/s (8.7 rad/s): a sample beyond that is a bus error, not motion, and one such sample is enough to wreck the velocity
BA_MAX, BG_MAX, V_MAX = 1.0, 0.15, 4.0    # bounds that hold the state to what a hand-held rig can be: accelerometer bias (m/s^2), gyro bias (rad/s), speed (m/s)


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def expm(w):
    """Rotation matrix of the rotation vector w."""
    th = float(np.linalg.norm(w))
    if th < 1e-9:
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * K @ K


def logm(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    return th / (2.0 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class VioEkf:
    def __init__(self, pos_sigma_m=0.02, att_sigma_deg=1.0, klt_sigma_m=0.012, klt_sigma_deg=0.3, gyro_noise=0.004, acc_noise=0.1, gyro_walk=2e-5, acc_walk=2e-3, gate_chi2=16.27):
        """pos_sigma_m / att_sigma_deg: noise of one VO pose. klt_sigma_m / klt_sigma_deg: noise of the Pi's motion over one pair interval. gyro_noise (rad/s) and acc_noise (m/s^2):
        IMU noise per sample at 100 Hz. gyro_walk / acc_walk: how fast the biases may drift (per sqrt(s)). gate_chi2: refuse a 3-dof measurement past this (16.27 = 99.9%)."""
        self.s_pos, self.s_att = float(pos_sigma_m), np.radians(att_sigma_deg)
        self.s_kt, self.s_kr = float(klt_sigma_m), np.radians(klt_sigma_deg)
        self.q_g, self.q_a, self.q_bg, self.q_ba, self.gate = float(gyro_noise), float(acc_noise), float(gyro_walk), float(acc_walk), float(gate_chi2)
        self.ready, self.t = False, None
        self.stats = {"pose_rejected": 0, "att_rejected": 0, "klt_rejected": 0, "resets": 0, "updates": 0, "imu_rejected": 0, "diverged": 0}

    # -- setup --------------------------------------------------------------------------------------------------------------------------------------
    def init(self, R, p, t, bg=None):
        self.R, self.p, self.v = np.array(R, float), np.array(p, float), np.zeros(3)
        self.bg, self.ba = np.zeros(3) if bg is None else np.array(bg, float), np.zeros(3)
        self.P = np.diag(np.r_[np.full(3, 0.05 ** 2), np.full(3, 0.3 ** 2), np.full(3, np.radians(2.0) ** 2), np.full(3, 0.02 ** 2), np.full(3, 0.1 ** 2)])
        self.t, self.ready, self.since_pose, self.refused_p, self.refused_a = float(t), True, 0.0, 0, 0
        self.gyro = []                                                    # (t, raw gyro) of the last second, for the mean rate over a KLT interval
        self.last = {}

    # -- IMU ----------------------------------------------------------------------------------------------------------------------------------------
    def predict(self, samples):
        """samples: rows (t, gyro rad/s, accel m/s^2) in camera axes, oldest first. Rows not newer than the filter's time are skipped."""
        for t, g, a in samples:
            if t <= self.t:
                continue
            dt = float(t - self.t)
            g, a = np.asarray(g, float), np.asarray(a, float)
            if not (np.isfinite(g).all() and np.isfinite(a).all()) or np.linalg.norm(a) > ACC_MAX or np.abs(g).max() > GYRO_MAX:
                self.stats["imu_rejected"] += 1                                # a corrupted read (I2C error on a starved Pi): skipped, the next good sample carries on over the longer dt
                continue
            self.gyro.append((float(t), g))
            if dt > GAP_S:                                                # a hole in the IMU record: the rotation across it is unknown, velocity carries on, uncertainty grows
                self.p = self.p + self.v * dt
                self.P[0:3, 0:3] += np.eye(3) * (0.5 * self.q_a * dt * dt) ** 2
                self.P[3:6, 3:6] += np.eye(3) * (self.q_a * dt) ** 2
                self.P[6:9, 6:9] += np.eye(3) * (0.5 * dt) ** 2
                self.t, self.since_pose = float(t), self.since_pose + dt
                continue
            w, f = g - self.bg, a - self.ba
            Rw = self.R @ expm(w * dt)
            a_w = self.R @ f + GRAV
            F = np.eye(15)
            F[0:3, 3:6] = np.eye(3) * dt
            F[3:6, 6:9] = -self.R @ skew(f) * dt
            F[3:6, 12:15] = -self.R * dt
            F[6:9, 6:9] = expm(-w * dt)
            F[6:9, 9:12] = -np.eye(3) * dt
            Q = np.zeros((15, 15))
            Q[3:6, 3:6] = np.eye(3) * self.q_a ** 2 * dt
            Q[6:9, 6:9] = np.eye(3) * self.q_g ** 2 * dt
            Q[9:12, 9:12] = np.eye(3) * self.q_bg ** 2 * dt
            Q[12:15, 12:15] = np.eye(3) * self.q_ba ** 2 * dt
            self.P = F @ self.P @ F.T + Q
            self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
            self.v = self.v + a_w * dt
            self.R = Rw
            self.t, self.since_pose = float(t), self.since_pose + dt
        while self.gyro and self.t - self.gyro[0][0] > 1.0:
            self.gyro.pop(0)

    def _bound(self):
        """The biases cannot be larger than a real sensor's: whatever the filter wants beyond that is model error (wrong axes, a bad timestamp), not bias."""
        for name, lim in (("bg", BG_MAX), ("ba", BA_MAX)):
            v = getattr(self, name)
            n = float(np.linalg.norm(v))
            if n > lim:
                setattr(self, name, v * (lim / n))

    def healthy(self):
        """False when the state is not finite or has left what a hand-held rig can do: the caller restarts the filter from the pictures."""
        ok = bool(np.isfinite(self.P).all() and np.isfinite(self.p).all() and np.isfinite(self.v).all() and np.isfinite(self.R).all() and np.linalg.norm(self.v) <= V_MAX)
        if not ok:
            self.stats["diverged"] += 1
        return ok

    # -- measurements -------------------------------------------------------------------------------------------------------------------------------
    def _update(self, y, H, Rm, gate=True):
        """One Kalman update; returns (applied, mahalanobis^2)."""
        S = H @ self.P @ H.T + Rm
        maha = float(y @ np.linalg.solve(S, y))
        if gate and maha > self.gate:
            return False, maha
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T                    # Joseph form: stays symmetric and positive
        self.P = 0.5 * (self.P + self.P.T)
        self.p, self.v = self.p + dx[0:3], self.v + dx[3:6]
        self.R = self.R @ expm(dx[6:9])
        self.bg, self.ba = self.bg + dx[9:12], self.ba + dx[12:15]
        self._bound()
        self.stats["updates"] += 1
        return True, maha

    def update_pose(self, R_meas, p_meas, n_inliers=None):
        """KeyframeVO's pose of this frame. Returns a dict for the log: innovations, whether each part was applied."""
        wide = 1.0 if not n_inliers else float(np.clip(40.0 / n_inliers, 1.0, 4.0))             # fewer agreeing features = a less certain pose
        H = np.zeros((3, 15))
        yp = np.asarray(p_meas, float) - self.p
        H[:, 0:3] = np.eye(3)
        ok_p, m_p = self._update(yp, H.copy(), np.eye(3) * (self.s_pos * wide) ** 2)
        H = np.zeros((3, 15))
        ya = logm(self.R.T @ np.asarray(R_meas, float))
        H[:, 6:9] = np.eye(3)
        ok_a, m_a = self._update(ya, H, np.eye(3) * (self.s_att * wide) ** 2)
        if ok_p or ok_a:
            self.since_pose = 0.0
        self.refused_p = 0 if ok_p else self.refused_p + 1
        self.refused_a = 0 if ok_a else self.refused_a + 1
        self.stats["pose_rejected"] += not ok_p
        self.stats["att_rejected"] += not ok_a
        if self.refused_p >= RESET_AFTER:                                  # the position has drifted from what the pictures keep saying (or an outlier was let in): believe them, with a wide covariance
            self.p = np.asarray(p_meas, float).copy()
            self.P[0:3, :] = 0.0
            self.P[:, 0:3] = 0.0
            self.P[0:3, 0:3] = np.eye(3) * 0.05 ** 2
            self.P[3:6, 3:6] += np.eye(3) * 0.3 ** 2
            self.refused_p, self.since_pose = 0, 0.0
            self.stats["resets"] += 1
        if self.refused_a >= RESET_AFTER:
            self.R = np.array(R_meas, float)
            self.P[6:9, :] = 0.0
            self.P[:, 6:9] = 0.0
            self.P[6:9, 6:9] = np.eye(3) * np.radians(3.0) ** 2
            self.refused_a = 0
            self.stats["resets"] += 1
        self.last.update(innov_pos_cm=100 * float(np.linalg.norm(yp)), innov_att_deg=float(np.degrees(np.linalg.norm(ya))), pos_applied=ok_p, att_applied=ok_a)
        return dict(self.last)

    def mean_gyro(self, t0, t1):
        g = [w for t, w in self.gyro if t0 < t <= t1]
        return np.mean(g, axis=0) if g else None

    def update_klt(self, rvec, t_pc, dt, n_inliers=None, rms=None):
        """The Pi's motion of the left camera between the previous pair and this one: rvec / t_pc = this camera in the previous camera's frame, dt seconds apart."""
        if dt is None or dt <= 1e-3 or dt > 0.5:
            return None
        wide = 1.0 if not n_inliers else float(np.clip(40.0 / n_inliers, 1.0, 4.0))
        if rms:
            wide *= float(np.clip(rms / 0.7, 1.0, 3.0))
        R_pc = expm(np.asarray(rvec, float))
        v_meas = self.R @ R_pc.T @ np.asarray(t_pc, float) / dt           # the previous camera's orientation is this one's times R_pc^T
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        ok_v, _ = self._update(v_meas - self.v, H, np.eye(3) * (self.s_kt * wide / dt) ** 2)
        gm, ok_w = self.mean_gyro(self.t - dt, self.t), False
        if gm is not None:
            H = np.zeros((3, 15))
            H[:, 9:12] = -np.eye(3)                                        # omega_klt - (gyro - bg) = -(bias error) + noise
            ok_w, _ = self._update(np.asarray(rvec, float) / dt - (gm - self.bg), H, np.eye(3) * (self.s_kr * wide / dt) ** 2)
        if not (ok_v or ok_w):
            self.stats["klt_rejected"] += 1
        self.last.update(klt_vel_applied=ok_v, klt_rate_applied=ok_w)
        return ok_v, ok_w

    def update_rest(self, dt=0.4):
        """The rig is at rest: velocity and angular rate are zero. Pins the velocity and, from the mean gyro, the gyro bias."""
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        self._update(-self.v, H, np.eye(3) * 0.02 ** 2, gate=False)
        gm = self.mean_gyro(self.t - dt, self.t)
        if gm is not None:
            H = np.zeros((3, 15))
            H[:, 9:12] = -np.eye(3)
            self._update(-(gm - self.bg), H, np.eye(3) * 0.003 ** 2, gate=False)

    def diag(self):
        return {"ekf_speed_mps": float(np.linalg.norm(self.v)), "ekf_gyro_bias_dps": float(np.degrees(np.linalg.norm(self.bg))), "ekf_acc_bias": float(np.linalg.norm(self.ba)),
                "ekf_sigma_pos_cm": 100 * float(np.sqrt(np.trace(self.P[0:3, 0:3]) / 3)), "ekf_sigma_att_deg": float(np.degrees(np.sqrt(np.trace(self.P[6:9, 6:9]) / 3))),
                "ekf_since_pose_s": float(self.since_pose), **{k: v for k, v in self.last.items()}, "ekf_rejected": self.stats["pose_rejected"], "ekf_resets": self.stats["resets"], "ekf_imu_rejected": self.stats["imu_rejected"], "ekf_diverged": self.stats["diverged"]}
