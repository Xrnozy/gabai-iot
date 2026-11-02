"""CPU, memory and GPU readings for the diagnostics. A reading that cannot be taken is None, never 0.

The GPU is read with `nvidia-smi`, which takes tens of milliseconds to start: that must never happen inside the frame loop, so a
background thread refreshes the numbers every `period` seconds and snapshot() only returns the latest ones (None until the first
reading, and None again if nvidia-smi is missing or fails).
"""
import subprocess
import threading
import time

try:
    import psutil
except ImportError:                       # the Pi may not have it: everything below then reports None
    psutil = None

_gpu = {"v": (None, None, None), "t": None, "started": False, "error": None}
_lock = threading.Lock()
_proc = None


def _poll(period):
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            t, u, m = [float(x) for x in out[0].split(",")]
            v, err = (t, u, m), None
        except Exception as e:
            v, err = (None, None, None), f"{type(e).__name__}: {e}"
        with _lock:
            _gpu["v"], _gpu["t"], _gpu["error"] = v, time.monotonic(), err
        time.sleep(period)


def start_gpu_poll(period=2.0):
    with _lock:
        if _gpu["started"]:
            return
        _gpu["started"] = True
    threading.Thread(target=_poll, args=(period,), daemon=True, name="gpu-poll").start()


def gpu_read():
    """(temp C, utilisation %, memory MB), as of the last background reading. Never blocks."""
    start_gpu_poll()
    with _lock:
        return _gpu["v"]


def gpu_error():
    with _lock:
        return _gpu["error"]


def snapshot():
    global _proc
    s = {"cpu_pct": None, "ram_pct": None, "proc_rss_mb": None, "gpu_temp_c": None, "gpu_util_pct": None, "gpu_mem_mb": None}
    if psutil is not None:
        if _proc is None:
            _proc = psutil.Process()
        s["cpu_pct"] = float(psutil.cpu_percent(interval=None))
        s["ram_pct"] = float(psutil.virtual_memory().percent)
        s["proc_rss_mb"] = round(_proc.memory_info().rss / 1e6, 1)
    s["gpu_temp_c"], s["gpu_util_pct"], s["gpu_mem_mb"] = gpu_read()
    return s
