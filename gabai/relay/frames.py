"""Latest Raspberry Pi camera pair, JPEG-encoded on demand so an idle relay spends no CPU on video."""
import os
import threading
import time

import cv2

SAVE_EVERY_S = 5.0


class FrameStore:
    def __init__(self, save_dir=None, clock=time.time):
        self.now = clock
        self.save_dir = save_dir
        self._lock = threading.Lock()
        self._pair = None            # (seq, left ndarray, right ndarray)
        self._t = None               # server time the pair arrived
        self._cache = {}             # side -> (seq, jpeg)
        self._saved_at = 0.0
        self._disk = {}              # side -> (jpeg, mtime) loaded from the previous run

    def put(self, seq, left, right):
        with self._lock:
            if self._pair is None or seq != self._pair[0]:
                self._pair, self._t = (seq, left, right), self.now()

    def age_s(self):
        with self._lock:
            return None if self._t is None else self.now() - self._t

    def seq(self):
        with self._lock:
            return None if self._pair is None else self._pair[0]

    def jpeg(self, side, quality=70):
        """(jpeg bytes, age in seconds) of the newest frame; falls back to the last frame saved to disk, with its real age."""
        with self._lock:
            pair, t = self._pair, self._t
            cached = self._cache.get(side)
        if pair is None:
            return self._from_disk(side)
        if cached and cached[0] == pair[0]:
            return cached[1], self.now() - t
        img = pair[1] if side == "left" else pair[2]
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None, None
        data = buf.tobytes()
        with self._lock:
            self._cache[side] = (pair[0], data)
        return data, self.now() - t

    def _from_disk(self, side):
        if not self.save_dir:
            return None, None
        path = os.path.join(self.save_dir, f"relay_last_{side}.jpg")
        try:
            with open(path, "rb") as f:
                data = f.read()
            return data, self.now() - os.path.getmtime(path)
        except OSError:
            return None, None

    def persist(self):
        """Keep the last good frame on disk so it survives a relay restart; called from the bridge loop."""
        if not self.save_dir or self.now() - self._saved_at < SAVE_EVERY_S or self._pair is None:
            return
        self._saved_at = self.now()
        os.makedirs(self.save_dir, exist_ok=True)
        for side in ("left", "right"):
            data, _ = self.jpeg(side)
            if data:
                tmp = os.path.join(self.save_dir, f"relay_last_{side}.tmp")
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, os.path.join(self.save_dir, f"relay_last_{side}.jpg"))
