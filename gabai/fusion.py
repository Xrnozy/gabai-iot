"""Gyro + accelerometer + compass fused into one orientation, with every sensor's time stamp put on the same footing.

All three sensors are stamped on the Pi's monotonic clock when read. Each has its own acquisition delay, measured
by gabai.tools.sync_sensors: the camera frame is stamped late by `delay` (handled when looking up the orientation for
a frame), and the compass is stamped late by `mag delay` (handled here). The gyro is the reference.
"""
import math
from collections import deque

import numpy as np

from gabai.diag import log as dlog
from gabai.mag import MagGate, apply_cal, tilt_heading_deg
from gabai.orient import (GyroBias, HeadingTracker, Madgwick, OrientationTrack, q_from_accel, q_from_up_north, q_mul, q_norm,
                          wrap_deg)

log = dlog.get("fusion")
STALE_MAG_S = 0.3   # if the compass has gone quiet for this long, fall back to gyro+gravity only
G = 9.80665
GATE_EVERY = 4      # the compass gate is evaluated on every 4th IMU sample (it is slow-changing; the decision is kept in between)


class SensorFusion:
    def __init__(self, mag_cfg=None, beta=0.05, auto_bias=True, log_every_s=1.0, accel_cal=None, mag_gate=True, frames=None):
        """mag_cfg: None (no compass) or {offset, scale or W, R_im, delay_s, radius_counts}: corrected = W @ (raw - offset) (or (raw - offset) * scale),
        then mapped into the IMU frame with R_im; delay_s is how late the compass stamps are. radius_counts (the calibrated field strength) switches on the gate
        that ignores disturbed readings. accel_cal: None or {bias, scale} from data/accel_calib.json (m/s^2), applied to every accelerometer sample."""
        self.cfg = mag_cfg
        self.accel_cal = accel_cal
        self.frames = frames                                     # gabai.frames.Frames: the heading is then that of the DEVICE's forward axis (camera z), not a guessed axis
        if frames is not None:
            self.heading_forward_fixed = frames.forward_in_imu()
        self.mount, self._mount_acc = None, []                   # (name, details) from gravity once the device has been still for a second, e.g. chest_upright
        self.gate = MagGate(mag_cfg["radius_counts"]) if (mag_cfg is not None and mag_gate and mag_cfg.get("radius_counts")) else None
        self.mag_ok, self._gate_n, self.mag_heading, self.mag_field_ratio = True, 0, None, None
        self.mag_used_n = self.mag_gated_n = 0
        self.filt = Madgwick(beta=beta)
        self.track = OrientationTrack(maxlen=1200)
        self._imu = deque()
        self._mag = deque()        # (t_true, vector in IMU frame)
        self._last_t = None
        self.used_compass = False
        self.bias = GyroBias() if auto_bias else None            # follows the gyro's residual bias while the board is still (gabai.orient.GyroBias)
        self.heading = HeadingTracker(forward=getattr(self, "heading_forward_fixed", None))                          # wrapped + continuous heading of the most horizontal body axis, for logs and plots
        self._shadow = np.array([1.0, 0.0, 0.0, 0.0])            # gyro-only orientation (no gravity, no compass): the difference to the filter is the correction they apply
        self._shadow_heading = HeadingTracker()
        self.log_every_s, self._t_log, self._raw_sum, self._raw_n = log_every_s, None, np.zeros(3), 0

    def add_imu(self, samples):
        self._imu.extend(samples)

    def add_mag(self, samples):
        if self.cfg is None:
            return
        for (t, x, y, z) in samples:
            v = apply_cal(self.cfg, [x, y, z])
            self._mag.append((t - self.cfg["delay_s"], self.cfg["R_im"] @ v))

    def _mag_at(self, ts):
        """Compass vector at time ts (linear interpolation), or None if the buffer does not reach ts yet."""
        m = self._mag
        if not m or m[-1][0] < ts:
            return None
        if m[0][0] > ts or len(m) == 1:
            return m[0][1]
        for i in range(len(m) - 1, 0, -1):
            if m[i - 1][0] <= ts <= m[i][0]:
                t0, t1 = m[i - 1][0], m[i][0]
                a = 0.0 if t1 == t0 else (ts - t0) / (t1 - t0)
                return (1 - a) * m[i - 1][1] + a * m[i][1]
        return m[-1][1]

    def _mount_check(self, ts, raw_gyro, accel):
        """Recognise the pose (chest_upright, flat_front_up, ...) from gravity in the device frame, once, after one second of rest."""
        if self.frames is None or self.mount is not None:
            return
        self._mount_acc.append((ts, raw_gyro, accel))
        while self._mount_acc and ts - self._mount_acc[0][0] > 1.0:
            self._mount_acc.pop(0)
        if len(self._mount_acc) < 20 or ts - self._mount_acc[0][0] < 0.9:
            return
        g = np.array([x[1] for x in self._mount_acc])
        a = np.array([x[2] for x in self._mount_acc])
        if np.linalg.norm(g.mean(0)) < 3.0 and np.linalg.norm(a.std(0)) < 0.03:
            self.mount = self.frames.mount(a.mean(0))
            log.info("fusion.mount", pose=self.mount[0], **{k: round(v, 1) for k, v in self.mount[1].items()},
                     hint="chest_upright is the intended worn pose; pitch_down_deg is how far the camera looks below the horizon (the SLAM world assumes tracking.initial_pitch_deg)")

    def _debug(self, ts, gyro):
        """One line a second: where a yaw error comes from. raw_gyro = what arrived (mean over the second); bias = the estimate removed; heading =
        the filter's final heading; gyro_only = the heading from the corrected gyro alone; correction = how far gravity and the compass have moved the
        filter away from the gyro alone (a steadily growing correction means the gyro is biased)."""
        if self._t_log is None:
            self._t_log = ts
            self._shadow = self.filt.q.copy()
            self._shadow_heading.forward = self.heading.forward
            return
        if ts - self._t_log < self.log_every_s or not self._raw_n:
            return
        self._shadow_heading.update(self._shadow)
        h, hs = self.heading, self._shadow_heading
        log.debug("fusion.yaw", raw_gyro_mean_dps=[round(v, 3) for v in self._raw_sum / self._raw_n], bias_dps=None if self.bias is None else [round(v, 3) for v in self.bias.bias],
                  still=None if self.bias is None else self.bias.still, axis=h.axis_name, heading_deg=round(h.wrapped, 2), heading_unwrapped_deg=round(h.unwrapped, 2),
                  heading_valid=h.valid, gyro_only_heading_deg=round(hs.wrapped, 2), correction_deg=round(wrap_deg(h.wrapped - hs.wrapped), 2),
                  tilt_trust=round(self.filt.tilt_trust, 2), compass_weight=round(self.filt.yaw_trust, 2), compass_yaw_bias_dps=round(math.degrees(self.filt.yaw_bias), 3),
                  compass=self.used_compass, compass_gated=self.mag_gated_n, compass_used=self.mag_used_n, compass_ok=self.mag_ok,
                  compass_reason=None if self.gate is None else self.gate.reason, field_vs_calibrated=None if self.mag_field_ratio is None else round(self.mag_field_ratio, 3),
                  compass_heading_deg=None if self.mag_heading is None else round(self.mag_heading, 2),
                  compass_minus_filter_deg=None if self.mag_heading is None else round(wrap_deg(self.mag_heading - h.wrapped), 2),
                  visual="not used here: this orientation is gyro + gravity (+ compass); camera pose from vision is a separate estimate")
        self._t_log, self._raw_sum, self._raw_n = ts, np.zeros(3), 0

    def step(self):
        while self._imu:
            ts, gx, gy, gz, ax, ay, az = self._imu[0]
            mag = None
            if self.cfg is not None:
                mag = self._mag_at(ts)
                if mag is None:
                    newest = self._mag[-1][0] if self._mag else None
                    if newest is None or ts - newest < STALE_MAG_S:
                        break                      # wait for the compass sample that covers this moment
            self._imu.popleft()
            raw = np.array([gx, gy, gz], float)
            if self.accel_cal is not None:
                ax, ay, az = ((np.array([ax, ay, az]) * G * (1.0 + np.asarray(self.accel_cal["scale"])) - np.asarray(self.accel_cal["bias"])) / G).tolist()
            if mag is not None:
                self._gate_n += 1
                if self.gate is not None and self._gate_n % GATE_EVERY == 1:
                    gravity_only = abs(float(np.linalg.norm((ax, ay, az))) - 1.0) < 0.05 and float(np.linalg.norm(raw)) < 150.0     # the angle to gravity means something only then
                    self.mag_ok = self.gate.check(ts, mag, (ax, ay, az), trusted=gravity_only)
                self.mag_field_ratio = float(np.linalg.norm(mag)) / self.gate.radius if self.gate is not None else None
                if not self.mag_ok:
                    self.mag_gated_n += 1
                    mag_use = None                                   # a disturbed reading must not pull the heading
                else:
                    mag_use = mag
                    hd, _ = tilt_heading_deg(mag, (ax, ay, az), self.heading.forward) if self.heading.forward is not None else (None, 0)
                    self.mag_heading = hd
            else:
                mag_use = None
            if self._last_t is None:
                q0 = q_from_up_north((ax, ay, az), mag_use) if mag_use is not None else None
                self.filt.q = q0 if q0 is not None else q_from_accel((ax, ay, az))    # start level, and pointing at magnetic north when the compass is good
                self._shadow = self.filt.q.copy()
            gyro = raw if self.bias is None else self.bias.feed(ts, raw, (ax, ay, az))
            self._mount_check(ts, raw, (ax, ay, az))
            self._raw_sum += raw
            self._raw_n += 1
            if self._last_t is not None and 0 < ts - self._last_t < 0.2:
                dt = ts - self._last_t
                if mag_use is not None:
                    self.filt.update_marg(np.radians(gyro), (ax, ay, az), mag_use, dt)
                    self.used_compass = True
                    self.mag_used_n += 1
                else:
                    self.filt.update(np.radians(gyro), (ax, ay, az), dt)
                    self.used_compass = False
                self.track.add(ts, self.filt.q)
                self._shadow = q_norm(self._shadow + 0.5 * q_mul(self._shadow, np.array([0.0, *np.radians(gyro)])) * dt)
                self.heading.update(self.filt.q)
                self._debug(ts, gyro)
            self._last_t = ts
        while self._mag and len(self._mag) > 600:
            self._mag.popleft()
