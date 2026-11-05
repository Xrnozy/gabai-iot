"""A recorded stereo pair folder (left.avi, right.avi, pairs.csv, meta.json), read-only, with random access and the exact preparation live.py's replay gives the pairs.

The raw pairs are what the receiver handed over (already turned by the calibration's rotation, not yet rectified). prepare() does what gabai.pipeline.sources.PairPrep does for a replay
(layout swap if the cameras came out in the other order, rectify, shrink to live_width) and what rtabmap_lab/live.py adds on top (the 180 deg turn when the accelerometer says the
pictures are upside down). Left and right are read together from files of the same pair index, so they cannot drift apart. Nothing is ever written into the recording's folder.
"""
import csv
import json
import os
import threading

import cv2
import numpy as np

FILES = ("left.avi", "right.avi", "pairs.csv", "meta.json")


class Recording:
    def __init__(self, folder, calib_params=None, live_width=640):
        self.folder = os.path.abspath(folder)
        self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
        self.rows = list(csv.DictReader(open(os.path.join(self.folder, "pairs.csv"))))
        self._cap = [cv2.VideoCapture(os.path.join(self.folder, f)) for f in ("left.avi", "right.avi")]
        frames = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in self._cap))
        self.n = min(len(self.rows), frames) if frames > 0 else len(self.rows)
        self.size = (int(self._cap[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.t = np.array([float(r["t_capture"]) for r in self.rows[:self.n]])
        self.dt_ms = [None if r["dt_ms"] == "" else float(r["dt_ms"]) for r in self.rows[:self.n]]
        self.k = min(1.0, live_width / self.size[0])
        ip = os.path.join(self.folder, "imu.csv")
        self.imu = np.loadtxt(ip, delimiter=",", skiprows=1) if os.path.exists(ip) else None          # t (same clock as t_capture), gyro deg/s, accel g: used only to judge motion, never by RTAB-Map
        self._lock, self._pos, self._cache = threading.Lock(), 0, {}
        self.swapped, self.flip, self.flip_reason, self.layout_note = False, False, "", ""
        self.R_ci, self.imu_delay = None, 0.0
        self.shutter = None                                                  # set by decide(): the rig's rolling-shutter correction, if measured (data/rolling_shutter.json)
        self.stamp = self.fingerprint()

    # -- files -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def fingerprint(self):
        """Size and modification time of the recording's files: an experiment stores it, and a run checks it is unchanged afterwards (the recording is the fixed input)."""
        out = {}
        for f in FILES:
            s = os.stat(os.path.join(self.folder, f))
            out[f] = [s.st_size, int(s.st_mtime)]
        return out

    def unchanged(self):
        return self.fingerprint() == self.stamp

    # -- raw pairs ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def raw(self, idx):
        """(left, right) BGR as recorded (before layout swap)."""
        idx = int(np.clip(idx, 0, self.n - 1))
        with self._lock:
            if idx in self._cache:
                return self._cache[idx]
            if idx != self._pos:
                for c in self._cap:
                    c.set(cv2.CAP_PROP_POS_FRAMES, idx)
            okl, l = self._cap[0].read()
            okr, r = self._cap[1].read()
            if not (okl and okr):
                raise IndexError(f"pair {idx} cannot be read from {self.folder}")
            self._pos = idx + 1
            self._cache[idx] = (l, r)
            if len(self._cache) > 24:
                self._cache.pop(next(iter(self._cache)))
            return l, r

    def calibrated_pair(self, idx):
        """(left, right, dt_ms) RAW pictures in the orientation the calibration expects: the layout swap applied, nothing else (this is what auto-calibration matches)."""
        l, r = self.raw(idx)
        if self.swapped:
            l, r = cv2.rotate(r, cv2.ROTATE_180), cv2.rotate(l, cv2.ROTATE_180)
        dt = self.dt_ms[int(np.clip(idx, 0, self.n - 1))]
        return l, r, (-dt if self.swapped and dt is not None else dt)

    def omega_dps(self, idx):
        """Angular speed of the rig at pair idx from the recorded gyro (deg/s), or None without an IMU file."""
        if self.imu is None:
            return None
        t = self.t[int(np.clip(idx, 0, self.n - 1))]
        k = np.abs(self.imu[:, 0] - t) < 0.02
        return float(np.linalg.norm(self.imu[k, 1:4], axis=1).mean()) if k.any() else None

    def gyro_rotvec(self, a, b):
        """Rotation of the camera from pair a to pair b from the recorded gyro, as a rotation vector (rad) in the optical frame RTAB-Map gets, or None without an IMU.
        Only a flip (the rig upside down) turns that frame about the optical axis; a layout swap undoes a device-number mix-up and gives back the calibrated cameras, gyro axes unchanged
        (vio_20261101_014854, swapped: the rolling-shutter fit has the walk's signs only without the turn)."""
        from rtabmap_lab.orientation import gyro_rotvec
        if self.imu is None or self.R_ci is None or b <= a:
            return None
        t0, t1 = self.t[int(a)], self.t[int(min(b, self.n - 1))]
        ts = self.imu[:, 0] + self.imu_delay
        m = (ts >= t0) & (ts < t1)
        return gyro_rotvec(ts[m], np.radians(self.imu[m, 1:4]), self.R_ci.T, self.flip, t1)

    def smear_px(self, idx, f):
        """How far the left and right pictures of pair idx disagree vertically-or-sideways just because the cameras were not exposed at the same instant while the rig turned: f * omega * dt.
        This is NOT a calibration error; judging row alignment on a pair where it is large blames the calibration for the motion."""
        w, dt = self.omega_dps(idx), self.dt_ms[int(np.clip(idx, 0, self.n - 1))]
        return None if w is None or dt is None else float(f * np.radians(w) * abs(dt) / 1000.0)

    # -- decisions made once, with the ORIGINAL calibration, so they cannot change when a parameter is edited -------------------------------------------------------------------------------
    def decide(self, rect0, imu_sync="data/imu_sync_corrected.json", shutter_file=None):
        from gabai.pipeline.sources import layout_needs_swap
        from rtabmap_lab import orientation
        swap = None
        for n in range(10):
            l, r = self.raw(n * 3)
            swap, a, b = layout_needs_swap(rect0, l, r, n)
            if swap is not None:
                break
        self.swapped = bool(swap)
        from rtabmap_lab.rolling_shutter import ShutterCorrection
        self.shutter = ShutterCorrection.load(shutter_file) if shutter_file else None          # opt-in: gyro-based, see rolling_shutter.py
        try:
            sync = json.load(open(imu_sync))
            self.R_ci, self.imu_delay = np.array(sync["R_ci"], float), float(sync.get("delay_s", 0.0))
        except (OSError, KeyError, ValueError):
            self.R_ci, self.imu_delay = None, 0.0
        self.layout_note = (f"cameras swapped (row error {a:.2f} px as recorded, {b:.2f} px swapped)" if self.swapped else f"layout as calibrated (row error {a:.2f} px)") if np.isfinite(a) else "layout could not be measured, as calibrated"
        try:
            v = orientation.from_recording(self.folder, imu_sync=imu_sync)
            self.flip, self.flip_reason = v["decision"] == "flip", v["reason"]
        except Exception as e:                                               # no IMU file: pictures as they are
            self.flip, self.flip_reason = False, f"orientation not checked ({type(e).__name__})"
        return self

    # -- the pair RTAB-Map gets --------------------------------------------------------------------------------------------------------------------------------------------------------
    def prepare(self, idx, rect, shutter="default"):
        """(left, right) rectified exactly as live.py replay would, with the CURRENT calibration (rect = CalibParams.rect()).
        shutter: a rolling_shutter.ShutterCorrection (the right picture resampled at the left picture's time, row by row, from the gyro), None for none, "default" for self.shutter."""
        shutter = self.shutter if shutter == "default" else shutter
        l, r, dt = self.calibrated_pair(idx)
        l, r = rect.apply(l, r)
        if self.k < 1:
            size = (int(round(self.size[0] * self.k)), int(round(self.size[1] * self.k)))
            l, r = cv2.resize(l, size, interpolation=cv2.INTER_AREA), cv2.resize(r, size, interpolation=cv2.INTER_AREA)
        if self.flip:
            l, r = cv2.rotate(r, cv2.ROTATE_180), cv2.rotate(l, cv2.ROTATE_180)
        if shutter is not None and self.imu is not None and self.R_ci is not None:
            from rtabmap_lab.rolling_shutter import gyro_rate
            w_imu = gyro_rate(self.imu, self.t[int(idx)], self.imu_delay)
            if w_imu is not None:
                h, w = l.shape[:2]
                F = np.diag([-1.0, -1.0, 1.0]) if self.flip else np.eye(3)
                cx, cy = rect.cx * self.k, rect.cy * self.k
                if self.flip:
                    cx, cy = (w - 1) - cx, (h - 1) - cy
                r = shutter.correct(r, F @ self.R_ci @ w_imu, rect.f * self.k, cx, cy, self.flip, None if dt is None else (-dt if self.flip else dt))
        return l, r

    def info(self):
        return {"folder": self.folder, "pairs": self.n, "size": list(self.size), "duration_s": float(self.t[-1] - self.t[0]) if self.n > 1 else 0.0, "swapped": self.swapped, "flip": self.flip,
                "layout": self.layout_note, "orientation": self.flip_reason, "created": self.meta.get("created"), "recorded_with": (self.meta.get("calibration") or {}).get("file")}

    def close(self):
        for c in self._cap:
            c.release()
