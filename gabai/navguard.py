"""Navigation first. The navigation viewer (gabai.spike.demo_viewer) writes a heartbeat file while it runs; other tools on the same PC that are not part of
the navigation (the Rerun pipeline, hand and object analysis) read it and slow themselves down while it is alive.

Why a file: the two programs are separate processes, and the GPU cannot be given to one of them by priority. Measured on this PC, a Rerun pipeline running
flat out next to the navigation viewer raised the viewer's per-frame time from about 42 ms to about 58 ms (median); a capped pipeline raised it to about
47 ms. With this heartbeat the pipeline caps itself and switches the hand/object work off.

nav_active() is True only when the file is fresh (written within max_age_s), names a process that is still alive, and that process is not the caller.
A crashed viewer stops refreshing the file, so the pipeline goes back to full speed within a few seconds. Anything unreadable counts as "not active".
"""
import json
import os
import threading
import time

HEARTBEAT = os.path.join("data", "nav_active.json")


class NavHeartbeat:
    def __init__(self, path=HEARTBEAT, what="demo_viewer", period_s=1.0):
        self.path, self.what, self.period = path, what, period_s
        self._stop = threading.Event()
        self._th = None

    def _write(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + f".{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "t": time.time(), "what": self.what}, f)
            os.replace(tmp, self.path)
        except (OSError, ValueError):
            pass                                                  # the heartbeat is advice for other tools; a write failure must never disturb the viewer

    def start(self):
        self._write()
        self._th = threading.Thread(target=self._run, name="nav-heartbeat", daemon=True)
        self._th.start()
        return self

    def _run(self):
        while not self._stop.wait(self.period):
            self._write()

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(2.0)
        try:
            os.remove(self.path)
        except OSError:
            pass


def nav_active(path=HEARTBEAT, max_age_s=3.0, now=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if (now if now is not None else time.time()) - float(d["t"]) > max_age_s:
            return False
        pid = int(d["pid"])
        if pid == os.getpid():
            return False
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return False
