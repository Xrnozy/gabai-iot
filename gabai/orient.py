"""Orientation from gyro + accelerometer (Madgwick filter, no magnetometer, so yaw drifts slowly) and a timeline
that can be asked "what was the orientation at time t" so it can be matched to a camera frame's capture time.

Frame convention: world z up. A sensor lying flat reads accel (0, 0, +1) g. Quaternions are (w, x, y, z) and
rotate sensor-frame vectors into the world frame.
"""
import math
from bisect import bisect_left
from collections import deque

import numpy as np


def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def q_norm(q):
    return q / np.linalg.norm(q)


def q_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2)], axis * math.sin(angle / 2)])


def q_to_R(q):
    w, x, y, z = q_norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def q_slerp(a, b, t):
    a, b = q_norm(a), q_norm(b)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return q_norm(a + t * (b - a))
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def euler_deg(q):
    """(roll, pitch, yaw) in degrees, z-up convention."""
    w, x, y, z = q_norm(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def q_from_accel(accel):
    """Orientation (yaw 0) that puts the measured 'up' (the accelerometer at rest) on world z. Starts the filter already level instead of
    letting it creep towards gravity for many seconds from the identity, which is wrong by up to 180 degrees on a board that does not start flat."""
    a = np.asarray(accel, float)
    n = np.linalg.norm(a)
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = a / n
    axis = np.cross(a, [0.0, 0.0, 1.0])
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if a[2] > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    return q_from_axis_angle(axis, math.atan2(s, a[2]))


def q_from_up_north(up, north):
    """Absolute orientation from two body-frame vectors: where 'up' (accelerometer) and magnetic north (compass, horizontal part) point. World x = north,
    y = west, z = up, as the filter's compass update uses. Returns None when the field is (nearly) parallel to up (no horizontal part)."""
    from scipy.spatial.transform import Rotation
    u = np.asarray(up, float)
    u = u / np.linalg.norm(u)
    h = np.asarray(north, float) - (np.asarray(north, float) @ u) * u
    n = np.linalg.norm(h)
    if n < 1e-9:
        return None
    h = h / n
    R_wb = np.array([h, np.cross(u, h), u])                       # rows: north, west, up in body axes = world <- body
    x, y, z, w = Rotation.from_matrix(R_wb).as_quat()             # q_to_R(q) is this matrix: it rotates body vectors into the world frame
    return np.array([w, x, y, z])


def accel_trust(norm_g, tol=0.03, zero_at=0.25):
    """How far the accelerometer can be believed to measure gravity alone: 1 while its length is within `tol` of 1 g, falling linearly to 0 at `zero_at`.
    A hand that shakes or turns adds acceleration to gravity and makes the length differ from 1 g; that reading says nothing about tilt."""
    return float(np.clip(1.0 - (abs(norm_g - 1.0) - tol) / (zero_at - tol), 0.0, 1.0))


class Madgwick:
    """Gyro + gravity (+ compass heading) orientation filter. Name and update/update_marg interface are the old Madgwick's; the correction is not.

    Why not the gradient step of the original Madgwick filter: it is normalised, so the correction always has length beta, however small the error. Measured on
    your recordings (scripts/evaluate_orientation.py) beta 0.05 pushed the orientation at 4.9 deg/s all the time, resting or turning, in a direction that followed
    the accelerometer's noise and every hand shake. Here the correction is proportional to the error, so it dies out at rest instead of chattering.
      - tilt: rate = kp_tilt * trust * (a_measured x a_predicted) in the body frame, kp_tilt = 20 * beta per second (beta 0.05: e-fold time 1 s). trust falls to 0
        when the accelerometer length is not 1 g and when it has been varying over the last 0.15 s (a hand that shakes adds acceleration to gravity). The rate is capped
        at max_tilt_dps: a gyro holds the tilt to a fraction of a degree a minute, so a gross, sudden disagreement is a push, not a tilt error, and must not be followed
        quickly. The cap is lifted at the start and after the disagreement has lasted resync_s (a gyro that saturated really can be far off).
      - heading: the compass corrects ONLY a rotation about the vertical (its error in the horizontal plane), kp_yaw = 6 * beta per second (beta 0.05: 3.3 s),
        so a disturbed or lagging compass can never tilt the horizon. The compass weight falls with the turn rate (a stale reading is worst while turning), and the
        rate is capped at 3 deg/s + 0.2 x the turn rate once the heading is known (the gyro says how fast it can really change). A small integral term (critically
        damped, clamped to max_yaw_bias_dps) takes out a steady drift about the vertical that a proportional loop alone would leave as a constant heading offset.
    The gyro bias on the other axes is not estimated here (gabai.orient.GyroBias does that)."""

    def __init__(self, beta=0.05, max_yaw_bias_dps=3.0, max_tilt_dps=4.0, resync_s=2.0):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.tilt_trust = self.yaw_trust = 0.0                  # last weights, for the logs
        self.yaw_bias = 0.0                                     # rad/s about the world vertical, from the compass
        self.yaw_synced = False                                 # until the first agreement with the compass, the heading may be turned freely onto it
        self.max_yaw_bias = math.radians(max_yaw_bias_dps)
        self.max_tilt, self.resync_s = math.radians(max_tilt_dps), resync_s
        self.tilt_synced, self._apart_s = False, 0.0            # same for the tilt: free until gravity and the estimate agree once
        self._a_mean, self._a_var = None, 0.0                   # running mean and variance of the accelerometer (g), time constant 0.15 s
        self.tilt_error_deg = 0.0

    @property
    def kp_tilt(self):
        return 20.0 * self.beta

    @property
    def kp_yaw(self):
        return 6.0 * self.beta

    def _shaking(self, a, dt):
        """1 for a steady accelerometer, towards 0 as it varies (sigma 0.05 g halves it)."""
        k = min(1.0, dt / 0.15)
        self._a_mean = a.copy() if self._a_mean is None else self._a_mean + k * (a - self._a_mean)
        self._a_var += k * (float(np.sum((a - self._a_mean) ** 2)) - self._a_var)
        return 1.0 / (1.0 + self._a_var / 0.05 ** 2)

    def _tilt_rate(self, accel, dt):
        """Body-frame angular rate (rad/s) that turns the predicted gravity towards the measured one, weighted by how much the accelerometer is believed."""
        a = np.asarray(accel, float)
        n = np.linalg.norm(a)
        if n < 1e-6:
            self.tilt_trust = 0.0
            return np.zeros(3)
        steady = self._shaking(a, dt)
        self.tilt_trust = accel_trust(n) * steady
        q1, q2, q3, q4 = self.q
        g_pred = np.array([2 * (q2 * q4 - q1 * q3), 2 * (q1 * q2 + q3 * q4), 1 - 2 * (q2 * q2 + q3 * q3)])
        e = np.cross(a / n, g_pred)
        self.tilt_error_deg = math.degrees(math.asin(min(1.0, float(np.linalg.norm(e)))))
        if self.tilt_trust > 0.5 and self.tilt_error_deg > 10.0:
            self._apart_s += dt                                 # a steady 1 g reading that disagrees with the estimate for long: the estimate is what is wrong
            if self._apart_s > self.resync_s:
                self.tilt_synced = False
        elif self.tilt_error_deg < 3.0:
            self._apart_s, self.tilt_synced = 0.0, True
        if self.tilt_trust == 0.0:
            return np.zeros(3)
        w = self.kp_tilt * self.tilt_trust * e
        cap = self.max_tilt if self.tilt_synced else math.radians(90.0)
        nw = float(np.linalg.norm(w))
        return w if nw <= cap else w * (cap / nw)

    def _yaw_rate(self, mag, gyro_rad, dt):
        """World-frame rate (rad/s) about the vertical that turns the field's horizontal part towards world +x (north). Zero when the field is (nearly) vertical."""
        m = np.asarray(mag, float)
        nm = np.linalg.norm(m)
        if nm < 1e-6:
            self.yaw_trust = 0.0
            return 0.0
        h = q_to_R(self.q) @ (m / nm)
        horiz = math.hypot(h[0], h[1])
        spin = float(np.linalg.norm(gyro_rad))
        self.yaw_trust = min(1.0, horiz / 0.3) / (1.0 + (spin / math.radians(90.0)) ** 2)
        err = -math.atan2(h[1], h[0])                                       # rotation about the vertical that would bring the field's heading to north
        kp = self.kp_yaw
        self.yaw_bias = float(np.clip(self.yaw_bias + 0.25 * kp * kp * self.yaw_trust * err * dt, -self.max_yaw_bias, self.max_yaw_bias))
        if abs(err) < math.radians(5.0):
            self.yaw_synced = True
        limit = math.radians(3.0) + 0.2 * spin if self.yaw_synced else math.radians(90.0)     # the gyro says how fast the heading can really change: once the heading is known the compass
        return float(np.clip(kp * self.yaw_trust * err + self.yaw_bias, -limit, limit))       # only trims it, so a still sensor next to a magnet is not swung round at a rate drift cannot explain

    def update(self, gyro_rad, accel, dt):
        w = np.asarray(gyro_rad, float) + self._tilt_rate(accel, dt)
        self.q = q_norm(self.q + 0.5 * q_mul(self.q, np.array([0.0, *w])) * dt)
        return self.q

    def update_marg(self, gyro_rad, accel, mag, dt):
        w = np.asarray(gyro_rad, float) + self._tilt_rate(accel, dt)
        qdot = 0.5 * q_mul(self.q, np.array([0.0, *w]))
        wz = self._yaw_rate(mag, gyro_rad, dt)
        qdot = qdot + 0.5 * q_mul(np.array([0.0, 0.0, 0.0, wz]), self.q)           # a rotation about the world vertical multiplies from the left
        self.q = q_norm(self.q + qdot * dt)
        return self.q


class OrientationTrack:
    """Time-ordered (t, q) history with slerp lookup, so a camera frame can be paired with the orientation at the
    moment it was captured instead of the latest one."""

    def __init__(self, maxlen=800):
        self.t = deque(maxlen=maxlen)
        self.q = deque(maxlen=maxlen)

    def add(self, t, q):
        self.t.append(t)
        self.q.append(np.array(q, float))

    def at(self, t):
        if not self.t:
            return np.array([1.0, 0.0, 0.0, 0.0])
        ts = list(self.t)
        if t <= ts[0]:
            return self.q[0]
        if t >= ts[-1]:
            return self.q[-1]
        i = bisect_left(ts, t)
        t0, t1 = ts[i - 1], ts[i]
        return q_slerp(self.q[i - 1], self.q[i], (t - t0) / (t1 - t0))


# ---- heading that does not break when the board is on its side, and a gyro bias that follows the sensor ---------------------------------------------------
def wrap_deg(a):
    """Angle in degrees wrapped to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


class HeadingTracker:
    """Heading of one body axis (the 'forward' axis) around the vertical, as a wrapped angle AND a continuous one.

    Why not euler_deg: its yaw is the heading of the body x axis, and it is undefined when x points straight up or down (pitch +-90, gimbal lock). A board
    mounted on its side has gravity on x, so a few thousandths of noise sent the 'yaw' through all of +-180 every few seconds. Here the heading is the azimuth
    of a body axis that is NOT near vertical: with forward=None the axis most horizontal at the first sample is chosen and then kept. If that axis comes
    within `min_horizontal` of vertical the heading is not defined and the last value is held (valid=False).

    unwrapped continues past +-180 (it keeps counting, 179, 180, 181 ...), so a plot of it has no jump; wrapped is the same angle folded into [-180, 180)."""

    def __init__(self, forward=None, min_horizontal=math.sin(math.radians(20))):
        self.forward = None if forward is None else np.asarray(forward, float)
        self.min_horizontal = min_horizontal
        self.unwrapped, self.wrapped, self._last, self.valid = 0.0, 0.0, None, False
        self.axis_name = None

    def _pick(self, R):
        up_in_body = R.T @ np.array([0.0, 0.0, 1.0])
        k = int(np.argmin(np.abs(up_in_body)))                      # the body axis with the least vertical component
        self.forward = np.eye(3)[k]
        self.axis_name = "xyz"[k]

    def update(self, q):
        R = q_to_R(q)
        if self.forward is None:
            self._pick(R)
        elif self.axis_name is None:
            self.axis_name = "custom"
        v = R @ self.forward
        h = math.hypot(v[0], v[1])
        self.valid = h >= self.min_horizontal
        if self.valid:
            ang = math.degrees(math.atan2(v[1], v[0]))
            if self._last is None:
                self.unwrapped = ang
            else:
                self.unwrapped += wrap_deg(ang - self._last)        # the smallest signed step: crossing +-180 is a step of a degree, not 360
            self._last, self.wrapped = ang, wrap_deg(ang)
        return self.wrapped, self.unwrapped, self.valid


class GyroBias:
    """Residual gyro bias estimated while the board is still (a zero-rate update), subtracted from every sample.

    The Pi already subtracts a bias measured in the first 1.5 s. That number is only as good as that second and a half (it was taken while the rig moved,
    in your run) and the sensor's bias also changes with temperature. This follows it: when a one-second window shows no motion (low spread of the gyro on every
    axis, accelerometer steady, mean rate below max_rate_dps) the mean of the window pulls the estimate with time constant tau_s.

    Limit: a rotation that is slower than max_rate_dps AND perfectly steady for a whole second looks like rest and is partly absorbed (at tau 8 s, a 0.5 deg/s
    turn would be cancelled after about half a minute). Real turns are faster than that, and anything with changing rate is not taken as rest."""

    def __init__(self, window_s=1.0, tau_s=8.0, max_rate_dps=1.5, spread_dps=0.25, accel_spread_g=0.02, max_bias_dps=3.0):
        self.window_s, self.tau_s, self.max_rate, self.spread, self.accel_spread, self.max_bias = window_s, tau_s, max_rate_dps, spread_dps, accel_spread_g, max_bias_dps
        self.bias = np.zeros(3)
        self._w = deque()                                            # (t, gyro_after_bias_removal... stored raw), accel
        self._t_last_eval, self.still, self.updates, self.still_s = None, False, 0, 0.0

    def feed(self, t, gyro_dps, accel):
        """Returns the gyro (deg/s) with the current bias estimate removed. gyro_dps is what arrived, whatever correction the sender already applied."""
        g = np.asarray(gyro_dps, float)
        self._w.append((t, g, np.asarray(accel, float)))
        while self._w and t - self._w[0][0] > self.window_s:
            self._w.popleft()
        if self._t_last_eval is None or t - self._t_last_eval >= 0.1:
            dt = 0.1 if self._t_last_eval is None else t - self._t_last_eval
            self._t_last_eval = t
            self._evaluate(dt)
        return g - self.bias

    def _evaluate(self, dt):
        if len(self._w) < 10 or self._w[-1][0] - self._w[0][0] < 0.9 * self.window_s:
            self.still = False
            return
        G = np.array([x[1] for x in self._w])
        A = np.array([x[2] for x in self._w])
        mean, sd = G.mean(0), G.std(0)
        a_sd = np.linalg.norm(A.std(0))
        self.still = bool(np.all(sd < self.spread) and np.linalg.norm(mean - self.bias) < self.max_rate and a_sd < self.accel_spread)
        if self.still:
            # a running mean over the first seconds of rest (so the first estimate arrives quickly), settling to an exponential average with time constant tau_s
            self.still_s += dt
            self.bias = np.clip(self.bias + (mean - self.bias) * min(1.0, dt / min(self.tau_s, self.still_s + 1.0)), -self.max_bias, self.max_bias)
            self.updates += 1
