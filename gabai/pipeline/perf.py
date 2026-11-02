"""Keeping the visual pipeline from getting in the way.

The things that must never wait for drawing are the depth, the pose and, in the navigation viewer, the obstacle alert. This module gives the Rerun
pipeline four tools:

  Stopwatch      time every stage of a frame; report median / mean / 95th percentile per stage (the summary and the log show where the time goes)
  AsyncViz       run all Rerun drawing on its own thread behind a bounded queue. The caller never waits. When the queue is long, the heavy picture calls
                 are DROPPED (counted, and said so in the summary); scalars, events and timeline positions are cheap and always kept. Order is preserved.
  lower_priority put this process below normal priority, so that if the navigation viewer runs on the same PC it gets the CPU first
  limit_threads  cap OpenCV / torch CPU threads so that the pipeline does not take every core

What this does not do: it cannot give the navigation viewer priority on the GPU. Two processes using the GPU share it; the pipeline's own rate caps
(fps_cap, hands_every, detector_every) are the lever for that.
"""
import os
import queue
import threading
import time
from collections import Counter

import numpy as np

from gabai.diag import log as dlog

log = dlog.get("perf")
DROPPABLE = {"stereo", "rejects", "annotate", "features", "pose", "cloud", "log", "hands3d", "image", "fusion"}
SIDE_DROPPABLE = {"pov_mesh"}                                   # a slow extra view: drawn by its own thread, at most one waiting, so it never delays the pose and cloud calls


class Stopwatch:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        return _Timer(self, name)

    def add(self, name, ms):
        self.t.setdefault(name, []).append(ms)

    def report(self):
        return {k: {"n": len(v), "median_ms": round(float(np.median(v)), 2), "mean_ms": round(float(np.mean(v)), 2), "p95_ms": round(float(np.percentile(v, 95)), 2),
                    "total_s": round(float(np.sum(v)) / 1000, 2)} for k, v in self.t.items()}


class _Timer:
    def __init__(self, sw, name):
        self.sw, self.name = sw, name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *a):
        self.sw.add(self.name, (time.perf_counter() - self.t0) * 1000)


class AsyncViz:
    def __init__(self, inner, max_queue=6):
        self.inner, self.max_queue = inner, max_queue
        self.q, self.side_q = queue.Queue(), queue.Queue()
        self.dropped, self.errors, self.done = Counter(), 0, 0
        self._lags = []
        self._busy = {"heavy": 0, "pov_mesh": 0}                 # droppable calls waiting or running
        self._lock = threading.Lock()
        self._tl = threading.local()                             # each submitting thread has its own current frame position (stage A and stage B work on different frames)
        self._th = threading.Thread(target=self._work, args=(self.q,), name="viz-worker", daemon=True)
        self._side_th = threading.Thread(target=self._work, args=(self.side_q,), name="viz-side-worker", daemon=True)     # the timeline position is per thread in Rerun, so each worker sets its own
        self._th.start()
        self._side_th.start()

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if not callable(attr) or name.startswith("_"):
            return attr
        key = name if name in SIDE_DROPPABLE else "heavy" if name in DROPPABLE else None

        def call(*a, **k):
            if name == "at":
                self._tl.pos = (a, k)                            # remembered for this thread; attached to every later call it makes
                return
            pos = getattr(self._tl, "pos", None)
            if key:
                with self._lock:
                    if self._busy[key] >= (self.max_queue if key == "heavy" else 1):
                        self.dropped[name] += 1
                        return
                    self._busy[key] += 1
            (self.side_q if name in SIDE_DROPPABLE else self.q).put((name, a, k, time.perf_counter(), key, pos))
        return call

    def _work(self, q):
        applied = None
        while True:
            item = q.get()
            if item is None:
                return
            name, a, k, t_enq, key, pos = item
            try:
                if pos is not None and pos is not applied:
                    self.inner.at(*pos[0], **pos[1])             # put the timeline where the calling thread said it was, only when that changed
                    applied = pos
                getattr(self.inner, name)(*a, **k)
            except Exception:
                self.errors += 1
                log.error_every(f"viz:{name}", 10.0, "viz.error", call=name, exc=True) if hasattr(log, "error_every") else log.error("viz.error", call=name, exc=True)
            finally:
                if key:
                    with self._lock:
                        self._busy[key] -= 1
                self.done += 1
                self._lags.append((time.perf_counter() - t_enq) * 1000)

    def close(self, timeout=10.0):
        self.q.put(None)
        self.side_q.put(None)
        self._th.join(timeout)
        self._side_th.join(timeout)
        if self._th.is_alive():
            log.warning("viz.close_timeout", pending=self.q.qsize(), hint="the viewer thread did not finish: the last frames may be missing from the recording")

    def stats(self):
        lag = self._lags
        return {"done": self.done, "dropped": dict(self.dropped), "errors": self.errors,
                "lag_ms_median": round(float(np.median(lag)), 1) if lag else None, "lag_ms_max": round(float(max(lag)), 1) if lag else None}


def lower_priority():
    """Below-normal priority for this process. Returns a sentence saying what happened."""
    try:
        import psutil
        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return "process priority set to below normal (Windows)"
        p.nice(10)
        return "process niceness set to 10"
    except Exception as e:
        return f"could not lower the priority: {type(e).__name__}: {e}"


def limit_threads(n):
    """Cap the CPU threads used by OpenCV and torch. n = 0 leaves them alone."""
    if not n:
        return "thread limits left at their defaults"
    done = []
    try:
        import cv2
        cv2.setNumThreads(n)
        done.append(f"OpenCV {n}")
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(n)
        done.append(f"torch {n}")
    except Exception:
        pass
    return "CPU threads capped: " + ", ".join(done)


_END = object()


class Prefetch:
    """Read the source (video decode, resize, row-alignment) on its own thread so the compute loop never waits for it."""

    def __init__(self, iterable, depth=4):
        self.q, self.exc, self._stop = queue.Queue(maxsize=depth), None, threading.Event()
        self._th = threading.Thread(target=self._run, args=(iter(iterable),), name="prefetch", daemon=True)
        self._th.start()

    def _run(self, it):
        try:
            for item in it:
                while not self._stop.is_set():
                    try:
                        self.q.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
        except Exception as e:
            self.exc = e
        finally:
            while True:
                try:
                    self.q.put(_END, timeout=0.2)
                    return
                except queue.Full:
                    if self._stop.is_set():
                        return

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = self.q.get(timeout=0.2)          # not a bare get(): on Windows a wait without a timeout cannot be interrupted by Ctrl-C, so a source that went quiet (the Pi stopped) made the run unstoppable
                break
            except queue.Empty:
                continue
        if item is _END:
            if self.exc is not None:
                raise self.exc
            raise StopIteration
        return item

    def close(self):
        self._stop.set()


class StageWorker:
    """Run fn(item) for each submitted item on its own thread, in order. With drop_when_busy a new item is refused (and counted) while the worker is
    still busy and its small queue is full: for a live stream the newest frame matters more than finishing an old one. Without it submit() waits,
    so a recording is processed completely. An exception in fn is kept and re-raised by check() / close() on the caller's thread."""

    def __init__(self, fn, maxsize=2, drop_when_busy=False, name="stage-b"):
        self.fn, self.drop_when_busy = fn, drop_when_busy
        self.q = queue.Queue(maxsize=maxsize)
        self.exc, self.dropped, self.done = None, 0, 0
        self._th = threading.Thread(target=self._run, name=name, daemon=True)
        self._th.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is _END:
                return
            if self.exc is not None:
                continue                                          # after an error, drain without working
            try:
                self.fn(item)
                self.done += 1
            except Exception as e:
                self.exc = e

    def check(self):
        if self.exc is not None:
            raise self.exc

    def submit(self, item):
        self.check()
        if self.drop_when_busy:
            try:
                self.q.put_nowait(item)
                return True
            except queue.Full:
                self.dropped += 1
                return False
        while True:
            try:
                self.q.put(item, timeout=0.2)
                return True
            except queue.Full:
                self.check()

    def close(self, timeout=10.0):
        self.q.put(_END)
        waited = 0.0
        while self._th.is_alive() and waited < timeout:     # short joins: a bare join() cannot be interrupted by Ctrl-C on Windows
            self._th.join(0.2)
            waited += 0.2
        if self._th.is_alive():
            log.warning("pipeline.stage_close_timeout", hint="the pose/map stage did not finish within %.0f s of the stop request; leaving it (it is a daemon thread)" % timeout)
            return
        self.check()
