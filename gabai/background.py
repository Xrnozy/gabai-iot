"""Background jobs that must never slow the live lanes (spec phase 6: "runs only when idle and never raises a lane over 100 ms").

BackgroundRunner.tick(settings, lane_ms) is called from the main loop every frame. It never blocks. A queued job starts only when the
governor says background work is allowed (settings["background_ok"]: rig still, GPU cool, tier normal) AND no lane is over LANE_SLOW_MS.
When either stops being true, the running job is asked to stop (its `stop()` callable turns True) and goes back to the queue to run again
later from the start. A job must check stop() often. A job that does not stop within `stop_grace_s` is reported as ignoring the stop request
(status()["ignoring_stop"]) rather than waited for: the runner cannot kill a thread, and it says so.
"""
import threading
import time

from gabai.diag import log as dlog

LANE_SLOW_MS = 100.0


class BackgroundRunner:
    def __init__(self, stop_grace_s=2.0):
        self.log = dlog.get("background")
        self._queue, self._results = [], {}
        self._run = None            # dict(name, fn, thread, stop (Event), asked_t)
        self._lock = threading.Lock()
        self.grace = stop_grace_s

    def submit(self, name, fn):
        with self._lock:
            self._queue.append((name, fn))
        self.log.info("background.queued", name=name)

    def result(self, name):
        return self._results.get(name)

    def status(self):
        with self._lock:
            r = self._run
            ignoring = r["name"] if (r and r["asked_t"] is not None and time.monotonic() - r["asked_t"] > self.grace and r["thread"].is_alive()) else None
            return {"running": r["name"] if r else None, "queued": [n for n, _ in self._queue], "done": sorted(self._results), "ignoring_stop": ignoring}

    def _target(self, run):
        name, fn, ev = run["name"], run["fn"], run["stop"]
        t0 = time.monotonic()
        try:
            v = fn(ev.is_set)
            res = {"ok": True, "value": v, "stopped": ev.is_set()}
        except Exception as e:                                    # a failing background job must never reach the live loop
            res = {"ok": False, "error": f"{type(e).__name__}: {e}", "stopped": ev.is_set()}
            self.log.error("background.failed", name=name, exc=True)
        with self._lock:
            if res.get("stopped") and res["ok"] and res["value"] in (None, "stopped"):
                self._queue.insert(0, (name, fn))                 # interrupted: try again later from the start
                self.log.info("background.interrupted", name=name, ran_s=round(time.monotonic() - t0, 2))
            else:
                self._results[name] = res
                self.log.info("background.done", name=name, ok=res["ok"], ran_s=round(time.monotonic() - t0, 2))
            self._run = None

    def tick(self, settings, lane_ms=None):
        slow = [k for k, v in (lane_ms or {}).items() if v is not None and v > LANE_SLOW_MS]
        allowed = bool(settings.get("background_ok")) and not slow
        with self._lock:
            run = self._run
            if run is not None:
                if not allowed and run["asked_t"] is None:
                    run["stop"].set()
                    run["asked_t"] = time.monotonic()
                    self.log.info("background.stop_requested", name=run["name"], reason=("lane %s over %.0f ms" % (slow[0], LANE_SLOW_MS)) if slow else "not idle")
                elif run["asked_t"] is not None and time.monotonic() - run["asked_t"] > self.grace and run["thread"].is_alive():
                    self.log.warning_every("deaf:" + run["name"], 10.0, "background.ignoring_stop", name=run["name"],
                                           hint="the job does not check stop(); it keeps using the machine until it finishes")
                return
            if not allowed or not self._queue:
                return
            name, fn = self._queue.pop(0)
            run = {"name": name, "fn": fn, "stop": threading.Event(), "asked_t": None}
            run["thread"] = threading.Thread(target=self._target, args=(run,), daemon=True, name=f"bg-{name}")
            self._run = run
        run["thread"].start()
        self.log.info("background.started", name=name)
