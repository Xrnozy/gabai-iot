"""Structured, non-blocking, rotating diagnostic log for every Gabai component (PC and Pi).

One JSON object per line in <log_dir>/<run>/gabai.jsonl:
    {"ts": "2026-10-26T18:12:09.123", "mono": 12345.678, "run": "...", "comp": "depth", "lvl": "INFO",
     "ev": "depth.done", "fid": 380179347, "dur_ms": 12.3, ...}
`ev` is a short dotted event name; every other key is evidence. `fid` is the frame correlation id (see gabai.diag.trace).

Settings (arguments of configure(), or environment variables):
    GABAI_LOG_LEVEL        default INFO         DEBUG for everything, or per component with GABAI_LOG_COMPONENTS
    GABAI_LOG_COMPONENTS   e.g. "depth=DEBUG,slam=DEBUG"
    GABAI_LOG_CONSOLE      default WARNING      level shown on the screen (stderr)
    GABAI_LOG_DIR          default logs         where run folders go
    GABAI_LOG_MAX_MB       default 10           size of one file before it rotates
    GABAI_LOG_BACKUPS      default 3            rotated files kept per run
    GABAI_LOG_KEEP_RUNS    default 10           run folders kept; older ones are deleted
    GABAI_LOG_QUEUE        default 20000        records waiting for the writer thread

The caller never waits for the disk: records go into a bounded queue and a writer thread saves them. If the queue is full
the record is dropped and COUNTED (a "log.dropped" line is written as soon as possible and at shutdown): a gap in the log
is never silent. Before configure() is called, only WARNING and above are printed to the screen.
"""
import atexit
import json
import logging
import logging.handlers
import math
import os
import platform
import queue
import shutil
import sys
import threading
import time
import traceback

LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_NAMES = {v: k for k, v in LEVELS.items()}
_lock = threading.RLock()
_loggers = {}
_resume = threading.Event()
_resume.set()
_S = dict(configured=False, listener=None, queue=None, handler=None, fh=None, default=20, console=30, levels={}, run=None,
          run_dir=None, path=None, dropped=0, reported=0)


def _lvl(x, fallback):
    if x is None:
        return fallback
    if isinstance(x, int):
        return x
    return LEVELS.get(str(x).upper(), fallback)


def _clean(v, depth=0):
    """Anything -> JSON-safe, without ever failing and without dumping big arrays."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if callable(v):
        return "<callable>"
    if isinstance(v, bytes):
        return {"bytes": len(v)}
    mod = type(v).__module__
    if mod == "numpy" or mod.startswith("numpy."):
        if hasattr(v, "shape") and getattr(v, "ndim", 0) > 0:
            if v.size <= 8:
                return [_clean(x, depth + 1) for x in v.ravel().tolist()]
            return {"shape": list(v.shape), "dtype": str(v.dtype)}
        return _clean(v.item(), depth + 1)
    if depth > 3:
        return repr(v)[:80]
    if isinstance(v, dict):
        return {str(k): _clean(x, depth + 1) for k, x in list(v.items())[:60]}
    if isinstance(v, (list, tuple, set)):
        items = list(v)
        return [_clean(x, depth + 1) for x in items[:40]] + ([f"...+{len(items) - 40}"] if len(items) > 40 else [])
    if isinstance(v, BaseException):
        return f"{type(v).__name__}: {v}"
    return repr(v)[:200]


class _DropQueueHandler(logging.handlers.QueueHandler):
    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            _S["dropped"] += 1


class _PausableRotating(logging.handlers.RotatingFileHandler):
    def emit(self, record):
        _resume.wait()
        super().emit(record)


class StructLogger:
    def __init__(self, comp):
        self.comp = comp
        self._last = {}
        self._swallowed = {}

    def enabled(self, level):
        return _lvl(level, 20) >= _S["levels"].get(self.comp, _S["default"])

    def _emit(self, lvl, ev, fields):
        n = LEVELS[lvl]
        configured = _S["configured"]
        to_file = configured and n >= _S["levels"].get(self.comp, _S["default"])
        to_screen = n >= (_S["console"] if configured else 30)
        if not (to_file or to_screen):
            return
        exc = fields.pop("exc", None)
        now = time.time()
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}",
               "mono": round(time.monotonic(), 4), "run": _S["run"], "comp": self.comp, "lvl": lvl, "ev": ev}
        for k, v in fields.items():
            if callable(v):
                v = v()
            rec[k] = _clean(v)
        if exc:
            rec["exc"] = traceback.format_exc(limit=8) if exc is True else _clean(exc)
        if to_screen:
            extra = " ".join(f"{k}={rec[k]}" for k in rec if k not in ("ts", "mono", "run", "comp", "lvl", "ev", "exc"))
            sys.stderr.write(f"{rec['ts'][11:]} {lvl:<7} {self.comp}: {ev} {extra}\n")
            if "exc" in rec:
                sys.stderr.write(rec["exc"])
        if to_file:
            q = _S["queue"]
            lost = _S["dropped"] - _S["reported"]
            if lost > 0 and q.qsize() < q.maxsize // 2:
                _S["reported"] += lost
                self._push(dict(rec, lvl="WARNING", ev="log.dropped", comp="log", count=lost,
                                note="records lost because the writer fell behind"))
            self._push(rec)

    @staticmethod
    def _push(rec):
        handler = _S["handler"]
        if handler is None:
            return
        handler.handle(logging.LogRecord("gabai", 20, "", 0, json.dumps(rec, separators=(",", ":"), allow_nan=False), None, None))

    def debug(self, ev, **f):
        self._emit("DEBUG", ev, f)

    def info(self, ev, **f):
        self._emit("INFO", ev, f)

    def warning(self, ev, **f):
        self._emit("WARNING", ev, f)

    def error(self, ev, **f):
        self._emit("ERROR", ev, f)

    def info_every(self, rl_key, period_s, ev, /, **f):
        """info(), at most once per period_s for this key (see warning_every)."""
        self._every("INFO", rl_key, period_s, ev, f)

    def warning_every(self, rl_key, period_s, ev, /, **f):
        """warning(), at most once per period_s for this key; the count of swallowed repeats is added to the next line."""
        self._every("WARNING", rl_key, period_s, ev, f)

    def _every(self, lvl, rl_key, period_s, ev, f):
        now = time.monotonic()
        if now - self._last.get(rl_key, -1e18) < period_s:
            self._swallowed[rl_key] = self._swallowed.get(rl_key, 0) + 1
            return
        self._last[rl_key] = now
        n = self._swallowed.pop(rl_key, 0)
        if n:
            f["suppressed"] = n
        self._emit(lvl, ev, f)


def get(comp):
    with _lock:
        if comp not in _loggers:
            _loggers[comp] = StructLogger(comp)
        return _loggers[comp]


def configure(log_dir=None, run_name=None, level=None, console_level=None, component_levels=None, max_mb=None, backups=None,
              keep_runs=None, queue_size=None):
    """Start (or restart) logging. Returns the run folder. Safe to call again: the previous run is closed first."""
    with _lock:
        shutdown()
        for lg in _loggers.values():                 # rate limits belong to one run: an earlier run must not silence this one
            lg._last.clear()
            lg._swallowed.clear()
        env = os.environ.get
        log_dir = log_dir or env("GABAI_LOG_DIR", "logs")
        run = (run_name + "_" if run_name else "") + time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(log_dir, run)
        os.makedirs(run_dir, exist_ok=True)
        comps = dict(component_levels or {})
        for item in filter(None, env("GABAI_LOG_COMPONENTS", "").split(",")):
            if "=" in item:
                k, v = item.split("=", 1)
                comps.setdefault(k.strip(), v.strip())
        _S["levels"] = {k: _lvl(v, 20) for k, v in comps.items()}
        _S["default"] = _lvl(level or env("GABAI_LOG_LEVEL"), 20)
        _S["console"] = _lvl(console_level or env("GABAI_LOG_CONSOLE"), 30)
        q = queue.Queue(maxsize=int(queue_size or env("GABAI_LOG_QUEUE", 20000)))
        path = os.path.join(run_dir, "gabai.jsonl")
        fh = _PausableRotating(path, maxBytes=int(float(max_mb or env("GABAI_LOG_MAX_MB", 10)) * 1e6),
                               backupCount=int(backups if backups is not None else env("GABAI_LOG_BACKUPS", 3)), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        listener = logging.handlers.QueueListener(q, fh, respect_handler_level=False)
        listener.start()
        qh = _DropQueueHandler(q)
        qh.setFormatter(logging.Formatter("%(message)s"))
        _S.update(configured=True, listener=listener, queue=q, handler=qh, fh=fh, run=run, run_dir=run_dir, path=path,
                  dropped=0, reported=0)
        meta = {"run": run, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv, "python": sys.version.split()[0],
                "platform": platform.platform(), "pid": os.getpid(), "cwd": os.getcwd(),
                "env": {k: v for k, v in os.environ.items() if k.startswith("GABAI_")},
                "default_level": _NAMES.get(_S["default"]), "component_levels": {k: _NAMES.get(v) for k, v in _S["levels"].items()}}
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        _prune(log_dir, int(keep_runs if keep_runs is not None else env("GABAI_LOG_KEEP_RUNS", 10)), keep=run_dir)
        get("log").info("log.start", run_dir=run_dir, level=_NAMES.get(_S["default"]), components=meta["component_levels"])
        return run_dir


def _prune(log_dir, keep_runs, keep):
    runs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.exists(os.path.join(log_dir, d, "run.json"))]
    runs.sort(key=lambda p: os.path.getmtime(os.path.join(p, "run.json")))
    for old in (runs[:-keep_runs] if keep_runs > 0 else []):
        if os.path.abspath(old) != os.path.abspath(keep):
            shutil.rmtree(old, ignore_errors=True)


def run_dir():
    return _S["run_dir"]


def dropped_count():
    return _S["dropped"]


def _pause_writer_for_test(pause):
    (_resume.clear if pause else _resume.set)()


def shutdown():
    """Write everything still queued, then close. Called automatically at exit."""
    with _lock:
        lst = _S.get("listener")
        if lst is None:
            return
        _resume.set()
        q, deadline = _S["queue"], time.monotonic() + 5.0
        while q.full() and time.monotonic() < deadline:   # stop() needs one free slot for its end marker
            time.sleep(0.005)
        lst.stop()
        lost = _S["dropped"] - _S["reported"]
        try:
            _S["fh"].close()
        except Exception:
            pass
        if lost > 0:
            with open(_S["path"], "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "mono": round(time.monotonic(), 4), "run": _S["run"],
                                    "comp": "log", "lvl": "WARNING", "ev": "log.dropped", "count": lost,
                                    "note": "records lost because the writer fell behind"}) + "\n")
        _S.update(configured=False, listener=None, handler=None)


atexit.register(shutdown)
