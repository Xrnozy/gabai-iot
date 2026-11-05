"""Live stereo from the Raspberry Pi as the mapper's input, shaped like a calib_lab Recording so mapper.run_offline runs on it unchanged.

  Pi:   python3 -m gabai.pi.sender --host <this PC> --left /dev/video2 --right /dev/video0 --width 640 --height 360 --fps 15 --capture-fps 30 --exposure auto
  PC:   py -m mapper.run_offline --pi --out data/mapper/pi --semantic --live

The pairs come from the existing receiver and pairing code (gabai.pipeline.sources.live_source via rtabmap_lab.live.frames, read-only use): layout and rectification are decided there, with the
calibration in --calib. Pairs the mapper has not got to when the next one arrives are dropped by that source's own lag limit, so the map follows the camera instead of falling behind it.
Stop with Ctrl-C: the run then writes its maps like a finished recording. The last RING pairs are kept (a node's depth is computed a few frames after the pair was sent).
"""
import argparse
import signal
import threading
import time

import cv2
import numpy as np

RING = 300


class _Stamps:
    def __init__(self):
        self.d = {}

    def __getitem__(self, i):
        return self.d[int(i)]

    def __len__(self):
        return len(self.d)


class LiveRect:
    """Stands in for the rectifier where the run asks for the camera model: the model of the live pairs (already rectified and scaled by the source)."""

    def __init__(self, cal):
        self.cal = cal

    def camera(self, flip=False, k=1.0):
        return self.cal


class PiRecording:
    k = 1.0
    imu = None

    def __init__(self, port=5600, calib=None, flip=False, seconds=0, rotate=(-1, -1), pipeline_config="config/pipeline.toml"):
        from rtabmap_lab import live as livemod
        ap = livemod.add_args(argparse.ArgumentParser())
        ns = ap.parse_args(["--source", "live", "--port", str(port), "--pipeline-config", pipeline_config] + (["--calib", calib] if calib else []))
        self.cfg, self._source = livemod.frames(ns)
        self.cfg.source.rotate_left, self.cfg.source.rotate_right = rotate          # the source reads these when its first pair is asked for; -1 = as the calibration was made (180 / 0 for the current one)
        self.flip, self.seconds = flip, seconds
        self.t, self._ring, self._lock = _Stamps(), {}, threading.Lock()
        self.n = 0
        self.rect = None
        self._first, self._stop = None, False

    def _on_sigint(self, *_):
        self._stop = True                                  # finish the pair in hand, then end the stream so the maps are written; a second Ctrl-C interrupts a wait for the Pi
        signal.signal(signal.SIGINT, signal.default_int_handler)

    def wait_first(self):
        """Blocks until the Pi's first pair arrived; returns the camera model of the pairs."""
        self._first = next(self._source)
        fr = self._first
        h, w = fr.left.shape[:2]
        c = fr.cal
        cx, cy = (c["cx"], c["cy"]) if not self.flip else ((w - 1) - c["cx"], (h - 1) - c["cy"])
        self.rect = LiveRect({"f": float(c["f"]), "B": float(c["B"]), "cx": float(cx), "cy": float(cy), "size": (w, h), "metric": bool(c.get("metric", True))})
        return self.rect.cal

    def indices(self):
        """Yields the index of each new pair as it arrives, until Ctrl-C or the stream ends."""
        fr, end = self._first, (time.monotonic() + self.seconds if self.seconds else None)
        signal.signal(signal.SIGINT, self._on_sigint)
        try:
            while not self._stop and (end is None or time.monotonic() < end):
                if fr is None:
                    fr = next(self._source)
                self._store(fr)
                yield self.n - 1
                fr = None
        except (KeyboardInterrupt, StopIteration):
            return

    def _store(self, fr):
        l, r = fr.left, fr.right
        if self.flip:
            l, r = cv2.rotate(r, cv2.ROTATE_180), cv2.rotate(l, cv2.ROTATE_180)
        i = self.n
        with self._lock:
            self._ring[i] = (l, r)
            self._ring.pop(i - RING, None)
        self.t.d[i] = float(fr.t)
        self.n += 1

    def prepare(self, idx, rect=None, shutter=None):
        return self._ring[int(idx)]                       # KeyError once a pair is older than RING

    def gyro_rotvec(self, last, idx):
        return None
