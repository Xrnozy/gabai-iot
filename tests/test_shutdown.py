"""Ctrl-C must reach a run that is waiting for a source that went quiet (the Pi stopped sending). On Windows a queue.get() without a timeout cannot be interrupted, so the run could
not be stopped. Each case runs in a child process with a time limit, because a regression here is a hang."""
import subprocess
import sys

SCRIPT = """
import _thread, threading, time
from gabai.pipeline.perf import Prefetch

def quiet_source():
    time.sleep(30)                       # no picture for a long time
    yield 1

pf = Prefetch(quiet_source())
threading.Timer(0.5, _thread.interrupt_main).start()      # the Ctrl-C
t0 = time.time()
try:
    next(pf)
    print("NOT-INTERRUPTED")
except KeyboardInterrupt:
    print("interrupted after %.1f" % (time.time() - t0))
"""


def test_ctrl_c_reaches_a_run_waiting_for_a_silent_source():
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=20)
    assert "interrupted after" in r.stdout, (r.stdout, r.stderr[-300:])
    assert float(r.stdout.split()[-1]) < 2.0


def test_the_stage_worker_close_gives_up_instead_of_hanging():
    code = """
import threading, time
from gabai.pipeline.perf import StageWorker
w = StageWorker(lambda item: time.sleep(60), maxsize=1)
w.submit(1)
time.sleep(0.3)
t0 = time.time()
w.close(timeout=1.0)
print("closed after %.1f" % (time.time() - t0))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert "closed after" in r.stdout and float(r.stdout.split()[-1]) < 3.0, (r.stdout, r.stderr[-300:])


def test_the_ctrl_c_handler_ends_a_process_that_does_not_stop_by_itself():
    if sys.platform != "win32":
        return
    code = """
import time
from gabai.pipeline import shutdown
assert shutdown.install(grace_s=1.0)
shutdown._keep[0](0)                      # what Windows calls on Ctrl-C
time.sleep(30)                            # the program is stuck
print("NOT-ENDED")
"""
    t0 = __import__("time").time()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert r.returncode == 130 and "NOT-ENDED" not in r.stdout, (r.returncode, r.stdout, r.stderr[-300:])
    assert __import__("time").time() - t0 < 10
