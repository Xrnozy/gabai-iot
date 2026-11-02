"""Hold Basalt back until the rig has been still for a moment.

Basalt starts from "at rest, gravity along the accelerometer" and sets its first orientation from the IMU. Started while the
rig is moving, its result depends on thread timing: the same recording gave 20 m of path on one run and 8.8 km on another.
So the tool waits for `window_s` of stillness (gyro under `gyro_max_dps` and the accelerometer reading gravity), then hands
over those samples so Basalt starts from a clean, known rest.
"""
from collections import deque

import numpy as np

G = 9.80665


class StillGate:
    def __init__(self, window_s=1.0, gyro_max_dps=3.0, accel_tol=0.8):
        self.window_ns = int(window_s * 1e9)
        self.gyro_max = np.radians(gyro_max_dps)
        self.accel_tol = accel_tol
        self.buf = deque()
        self.ready = False
        self.still_since = None
        self.last_gyro_dps = None      # what the last sample read, so a gate that never opens can say why
        self.last_accel_err = None

    @property
    def progress(self):
        """0 to 1: how much of the required still time has passed."""
        if self.ready:
            return 1.0
        if not self.buf or self.still_since is None:
            return 0.0
        return min((self.buf[-1][0] - self.still_since) / self.window_ns, 1.0)

    def feed(self, t_ns, accel, gyro):
        """Give one IMU sample (accel m/s^2, gyro rad/s). Returns the samples to push to Basalt: nothing while waiting, the
        buffered still window the moment it is ready, then each sample as it comes."""
        sample = (int(t_ns), np.asarray(accel, float), np.asarray(gyro, float))
        if self.ready:
            return [sample]
        self.last_gyro_dps = float(np.degrees(np.linalg.norm(sample[2])))
        self.last_accel_err = float(abs(np.linalg.norm(sample[1]) - G))
        moving = np.linalg.norm(sample[2]) > self.gyro_max or abs(np.linalg.norm(sample[1]) - G) > self.accel_tol
        if moving:
            self.buf.clear()
            self.still_since = None
            return []
        if self.still_since is None:
            self.still_since = sample[0]
        self.buf.append(sample)
        if sample[0] - self.still_since >= self.window_ns:
            self.ready = True
            out, self.buf = list(self.buf), deque()
            return out
        return []
