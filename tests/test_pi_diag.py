import argparse
import json
import os
import tempfile
import threading
import time
from collections import deque

from gabai.diag import checks as C
from gabai.diag import log as dlog
from gabai.diag.pi_status import active_flags, decode_throttled
from gabai.pc.receiver import LiveReceiver
from gabai.pi import sender as S


def events(run):
    dlog.shutdown()
    return [json.loads(l) for l in open(os.path.join(run, "gabai.jsonl"), encoding="utf-8") if l.strip()]


def args(port, **kw):
    a = argparse.Namespace(host="127.0.0.1", port=port, token="gabai", width=640, height=480, fps=15, sndbuf=65536, health_every=1.0,
                           stats_every=1.0, imu_obj=None, mag_obj=None, quality=80)
    a.__dict__.update(kw)
    return a


def test_throttle_flags_are_decoded_into_names():
    d = decode_throttled("0x50005")
    assert d["under_voltage_now"] and d["throttled_now"] and d["under_voltage_since_boot"] and d["throttled_since_boot"]
    assert not d["freq_capped_now"]
    assert active_flags("0x0") == [] and active_flags(None) is None and active_flags("zz") is None


def test_pi_power_check_tells_a_live_fault_from_a_past_one():
    assert C.check_pi_power("0x0").status == "pass"
    assert C.check_pi_power("0x50000").status == "suspect"                 # happened since boot, clear now
    r = C.check_pi_power("0x50005")
    assert r.status == "fail" and "under-voltage" in r.detail and "pi-power-guide" in r.hint
    assert C.check_pi_power(None).status == "unknown"


def test_a_session_logs_connection_throughput_and_send_times_with_the_pc_seeing_the_same_frames():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="pi", level="INFO")
    rx = LiveReceiver(port=0, token="gabai")
    rx.start()
    a = args(rx.port)
    th = threading.Thread(target=lambda: _quiet(S.run_session, a, S.TestPattern(a)), daemon=True)
    th.start()
    end = time.time() + 6
    while time.time() < end and rx.stats()["frames"] < 30:
        time.sleep(0.05)
    sent_frames = rx.stats()["frames"]
    rx.stop()
    th.join(timeout=8)
    ev = events(run)
    names = [e["ev"] for e in ev]
    assert sent_frames >= 30 and "pi.connecting" in names and "pi.connected" in names
    st = [e for e in ev if e["ev"] == "pi.stats"]
    assert st and 8 <= st[-1]["sent_fps"] <= 18 and st[-1]["send_ms_max"] is not None and st[-1]["mbit_s"] > 0
    assert "pi.session_end" in names                                     # the end of the session is recorded with a reason
    assert [e for e in ev if e["ev"] == "pi.session_end"][0]["frames_sent"] >= 30


def _quiet(fn, *a):
    try:
        fn(*a)
    except Exception:
        pass


def test_connection_refused_is_logged_with_the_fix_not_just_an_exception():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="pi2", level="INFO")
    a = args(1)                                                          # nothing listens on port 1
    slept = []
    S.run_forever(a, S.TestPattern(a), sleep=slept.append, max_attempts=3)
    ev = events(run)
    down = [e for e in ev if e["ev"] == "pi.link_down"]
    assert len(down) == 3 and slept == [1, 2]                            # backed off 1 s then 2 s
    assert any(("refused" in e["reason"]) or ("timed out" in e["reason"]) or ("network error" in e["reason"]) for e in down)
    assert down[0]["attempt"] == 1 and down[2]["attempt"] == 3


def test_a_receiver_that_goes_away_mid_stream_is_reported_with_how_much_was_sent():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="pi3", level="INFO")
    rx = LiveReceiver(port=0, token="gabai")
    rx.start()
    a = args(rx.port)
    th = threading.Thread(target=lambda: _quiet(S.run_session, a, S.TestPattern(a)), daemon=True)
    th.start()
    end = time.time() + 5
    while time.time() < end and rx.stats()["frames"] < 10:
        time.sleep(0.05)
    rx.stop()
    th.join(timeout=10)
    ev = events(run)
    e = [x for x in ev if x["ev"] == "pi.session_end"]
    assert e and e[0]["frames_sent"] >= 10 and e[0]["reason"]


class FakeCap:
    def __init__(self, script):
        self.script, self.i = list(script), 0

    def grab(self):
        r = self.script[self.i % len(self.script)]
        self.i += 1
        if self.i > 400:
            time.sleep(0.01)
        return r

    def retrieve(self):
        import numpy as np
        return True, np.frombuffer(b"\xff\xd8" + b"x" * 100, np.uint8).reshape(1, -1)


def test_a_camera_that_stops_delivering_is_counted_and_never_kills_its_thread():
    from gabai.pi.cams import SyncedCameras
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="cam", level="INFO")
    cams = object.__new__(SyncedCameras)
    cams.q, cams.min_gap = 80, 1 / 15
    cams.bufs = {"L": deque(maxlen=8), "R": deque(maxlen=8)}
    cams.counts = {"L": deque(maxlen=60), "R": deque(maxlen=60)}
    cams.dts = deque(maxlen=60)
    cams._last_tl, cams._last_out, cams._stop = None, 0.0, False
    cams.cnt = {k: {"frames": 0, "grab_fail": 0, "bad_buf": 0, "last_bytes": None} for k in "LR"}
    cams.caps = {"L": FakeCap([True]), "R": FakeCap([False, False, False, True])}      # the right camera fails 3 of 4 grabs
    cams.threads = {k: threading.Thread(target=cams._reader, args=(k,), daemon=True) for k in "LR"}
    for t in cams.threads.values():
        t.start()
    time.sleep(0.4)
    cams._stop = True
    time.sleep(0.1)
    st = cams.stats()
    assert st["frames_left"] > 20 and st["grab_fail_right"] > 20 and st["grab_fail_left"] == 0
    assert st["jpeg_bytes_left"] == 102
    ev = events(run)
    assert any(e["ev"] == "cam.grab_failed" and e["cam"] == "R" for e in ev)
    assert any(e["ev"] == "cam.recovered" for e in ev)


def test_the_pi_log_needs_only_the_standard_library_and_numpy():
    import subprocess
    import sys
    code = ("import sys\nfrom gabai.diag import log, trace, pi_status\n"
            "bad=[m for m in ('pandas','scipy','torch','matplotlib','psutil') if m in sys.modules]\n"
            "assert not bad, bad\nprint('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.stdout.strip() == "ok", r.stderr[-300:]
