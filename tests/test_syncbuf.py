import json
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np

from gabai.net import protocol as P
from gabai.pc.receiver import KltPoints, LiveReceiver
from gabai.pc.syncbuf import SyncBuffer
from gabai.pi.imu import Bmi160


def row(t, v=0.0):
    return (t, v, 0.0, 0.0, 0.0, 0.0, 1.0)


def kp(seq, t):
    return KltPoints(t, seq, np.arange(3), np.zeros((3, 2), np.float32), np.zeros(3, np.uint8), np.zeros(3, np.uint8), None, 1.0)


def buf(**kw):
    clock = SimpleNamespace(now=0.0)
    b = SyncBuffer(clock=lambda: clock.now, **kw)
    b.enable()
    return b, clock


def test_a_group_with_everything_present_is_released_at_once_with_its_whole_imu_interval():
    b, c = buf()
    b.add_imu([row(0.01 * i) for i in range(1, 21)])               # 10 ms .. 200 ms
    b.add_frame(0, 0, 0.100, b"L", b"R", 1.0, now=0.0)
    b.add_klt(kp(0, 0.100))
    g = b.pop()
    assert g is not None and g.klt.frame_seq == 0 and len(g.imu) == 10
    assert g.info["imu_complete"] and not g.info["klt_missing"] and g.info["imu_margin_ms"] > 0
    b.add_frame(1, 1, 0.200, b"L", b"R", 1.0, now=0.0)
    b.add_klt(kp(1, 0.200))
    g2 = b.pop()
    assert len(g.imu) + len(g2.imu) == 20 and g2.imu[0][0] > g.imu[-1][0]                # every sample once, in order, none in two groups


def test_a_pair_waits_for_its_imu_then_goes_out_marked_incomplete():
    b, c = buf(imu_wait_s=0.15)
    b.add_imu([row(0.01)])
    b.add_frame(0, 0, 0.100, b"L", b"R", None, now=0.0)
    assert b.pop() is None                                                               # the IMU has not reached 0.1 s
    c.now = 0.1
    assert b.pop() is None
    b.add_imu([row(0.11)])
    g = b.pop()
    assert g is not None and g.info["imu_complete"]                                      # it arrived in time: no need to wait the full 150 ms
    b.add_frame(1, 1, 0.300, b"L", b"R", None, now=0.1)
    c.now = 0.3
    g = b.pop()
    assert g is not None and not g.info["imu_complete"] and b.stats()["incomplete"] == 1


def test_klt_points_belong_to_the_pair_with_their_frame_number_and_never_to_another():
    b, c = buf(klt_wait_s=0.15)
    b.add_imu([row(1.0)])
    b.add_klt(kp(5, 0.50))
    b.add_frame(5, 5, 0.50, b"L", b"R", None, now=0.0)
    b.add_frame(6, 6, 0.60, b"L", b"R", None, now=0.0)
    assert b.pop().klt.frame_seq == 5
    assert b.pop() is None                                                               # pair 6 has no points yet: waits, does not borrow another pair's
    c.now = 0.2
    g = b.pop()
    assert g.seq == 6 and g.klt is None and g.info["klt_missing"] and b.stats()["klt_missing"] == 1
    b.add_klt(kp(7, 0.99))                                                               # same counter, wrong time: not this pair's points
    b.add_frame(7, 7, 0.70, b"L", b"R", None, now=0.2)
    c.now = 0.4
    g = b.pop()
    assert g.klt is None and b.stats()["klt_mismatch"] == 1


def test_a_repeated_pair_is_dropped_and_pairs_the_pi_skipped_are_counted():
    b, _ = buf()
    for seq in (0, 1, 1, 4):
        b.add_frame(seq, seq, 0.1 * seq, b"L", b"R", None)
    s = b.stats()
    assert s["frames"] == 3 and s["dup_frames"] == 1 and s["pi_skipped"] == 2


def test_imu_samples_out_of_order_or_repeated_by_time_are_integrated_once():
    b, _ = buf()
    b.add_imu([row(0.1), row(0.2), row(0.15), row(0.2), row(0.3)])
    assert [r[0] for r in b.imu] == [0.1, 0.2, 0.3] and b.stats()["imu_dup"] == 2


def test_falling_behind_skips_old_pictures_but_keeps_every_imu_sample():
    b, c = buf(max_lag_s=1.0, keep_lag_s=0.3)
    b.add_imu([row(0.01 * i) for i in range(1, 301)])                                    # 3 s of IMU
    for i in range(30):                                                                  # 30 pairs arrive over 3 s: the consumer was busy
        b.add_frame(i, i, 0.1 * i, b"L", b"R", None, now=0.1 * i)
    c.now = 3.0
    g = b.pop()
    s = b.stats()
    assert s["recoveries"] == 1 and s["recovery_skipped"] > 15 and g.info["skipped_before"] == s["recovery_skipped"]
    assert g.seq >= 20 and g.imu[0][0] <= 0.011 and abs(g.imu[-1][0] - (g.t)) < 0.0101  # its interval starts at the very first sample: nothing of the skipped time is lost


def test_max_lag_zero_never_skips_and_the_queue_is_still_bounded():
    b, c = buf(max_lag_s=0.0, max_groups=5)
    for i in range(8):
        b.add_frame(i, i, 0.1 * i, b"L", b"R", None, now=0.0)
    c.now = 100.0
    assert b.stats()["groups"] == 5 and b.stats()["overflow"] == 3 and b.stats()["recoveries"] == 0
    assert b.pop().seq == 3


def test_a_hole_in_the_imu_stream_is_reported_with_the_group_that_spans_it():
    b, _ = buf()
    b.add_imu([row(0.01 * i) for i in range(1, 11)])
    b.add_imu([row(0.5 + 0.01 * i) for i in range(10)], missing=40)                      # samples 0.11 .. 0.5 never arrived
    b.add_frame(0, 0, 0.55, b"L", b"R", None, now=0.0)
    g = b.pop()
    assert b.stats()["imu_missing"] == 40 and g.info["imu_gap_s"] > 0.3


def test_a_buffer_nobody_enabled_keeps_nothing():
    b = SyncBuffer()
    b.add_frame(0, 0, 0.1, b"L", b"R", None)
    b.add_imu([row(0.1)])
    assert b.stats()["frames"] == 0 and b.stats()["groups"] == 0 and b.pop() is None


# -- the link: sample numbers, repeats after a reconnect, the ACK ------------------------------------------------------------------------------------
def _hello(token="gabai"):
    return P.pack_json(P.HELLO, 0.0, {"token": token, "name": "t", "width": 640, "height": 360, "fps": 15})


def _imu(seq, ts):
    return P.pack_json(P.IMU, ts[-1], {"seq": seq, "samples": [[t, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for t in ts]})


def _until(fn, s=3.0):
    end = time.time() + s
    while time.time() < end and not fn():
        time.sleep(0.01)
    return fn()


def test_the_receiver_tells_a_reconnecting_pi_where_its_imu_record_ends_and_drops_the_repeats():
    rx = LiveReceiver(port=0, token="gabai")
    sb = rx.enable_sync()
    rx.start()
    try:
        c1 = socket.create_connection(("127.0.0.1", rx.port))
        c1.sendall(_hello() + _imu(0, [0.01, 0.02, 0.03, 0.04, 0.05]))
        assert _until(lambda: sb.stats()["imu_rows"] == 5)
        c1.close()
        assert _until(lambda: not rx.stats()["connected"])
        c2 = socket.create_connection(("127.0.0.1", rx.port))
        c2.settimeout(3)
        c2.sendall(_hello())
        t, _, payload = P.read_message(c2)
        assert t == P.ACK and json.loads(payload)["imu_next"] == 5
        c2.sendall(_imu(3, [0.04, 0.05, 0.06, 0.07, 0.08]))                              # the Pi resends from 3: two repeats, three new
        assert _until(lambda: sb.stats()["imu_rows"] == 8)
        c2.sendall(_imu(20, [0.30, 0.31]))                                               # samples 8..19 were lost for good
        assert _until(lambda: sb.stats()["imu_rows"] == 10)
        s, st = sb.stats(), rx.stats()
        assert s["imu_missing"] == 12 and st["imu_dup"] == 2 and st["imu_next"] == 22
        c2.close()
    finally:
        rx.stop()


def test_a_frame_with_counters_reaches_the_buffer_with_its_klt_points_matched_by_number():
    rx = LiveReceiver(port=0, token="gabai")
    sb = rx.enable_sync()
    rx.start()
    try:
        c = socket.create_connection(("127.0.0.1", rx.port))
        import cv2
        jp = cv2.imencode(".jpg", np.full((36, 64, 3), 90, np.uint8))[1].tobytes()
        c.sendall(_hello() + _imu(0, [0.9, 1.0, 1.1]) + P.pack(P.FRAME, 1.0, P.pack_frame(jp, jp, 2.5, (41, 42)))
                  + P.pack(P.KLT, 1.0, P.pack_klt(41, np.arange(4), np.ones((4, 2), np.float32), np.zeros(4), np.zeros(4, np.uint8))))
        assert _until(lambda: sb.stats()["frames"] == 1 and sb.stats()["klt_waiting"] == 1)
        g = sb.pop()
        assert g.seq == 41 and g.seq_r == 42 and g.klt.frame_seq == 41 and len(g.klt.ids) == 4 and abs(g.dt_ms - 2.5) < 1e-6 and len(g.imu) == 2
        c.close()
    finally:
        rx.stop()


# -- the Pi's IMU: nothing skipped, a resend on request ----------------------------------------------------------------------------------------------
def _imu_with(n):
    imu = object.__new__(Bmi160)
    from collections import deque
    imu.full, imu.n, imu._next, imu._lock = deque(maxlen=6000), 0, 0, threading.Lock()
    for i in range(n):
        imu.full.append((i * 0.005, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0))
        imu.n += 1
    return imu


def test_the_pi_hands_out_every_imu_sample_oldest_first_not_just_the_newest_120():
    imu = _imu_with(1000)                                                                # a 5 s stall
    got, seq0 = [], None
    while True:
        s, rows = imu.drain_seq(400)
        if not rows:
            break
        seq0 = s if seq0 is None else seq0
        got += rows
    assert seq0 == 0 and len(got) == 1000 and got[0][0] == 0.0 and got[-1][0] == round(999 * 0.005, 5)


def test_the_pi_sends_again_from_the_number_the_pc_asks_for_and_a_sample_gone_from_the_ring_shows_as_a_jump():
    imu = _imu_with(100)
    imu.drain_seq(400)
    imu.rewind(60)
    s, rows = imu.drain_seq(400)
    assert s == 60 and len(rows) == 40
    small = _imu_with(100)
    small.full = type(small.full)(small.full, maxlen=50)                                 # only the last 50 are still held
    s, rows = small.drain_seq(400)
    assert s == 50 and len(rows) == 50                                                   # the first 50 are gone: the PC sees seq jump 0 -> 50 and counts them missing


# -- the whole path: Pi sender -> link -> SyncBuffer ----------------------------------------------------------------------------------------------
class FakeImu:
    """200 Hz samples numbered like Bmi160.drain_seq, generated from the clock."""
    errors, full = 0, []

    def __init__(self):
        self.t0, self.n, self.next = time.monotonic(), 0, 0

    def _have(self):
        return int((time.monotonic() - self.t0) * 200)

    def drain_seq(self, limit=400):
        rows = [[round(self.t0 + (k + 1) / 200, 5), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for k in range(self.next, min(self._have(), self.next + limit))]
        s, self.next = self.next, self.next + len(rows)
        return s, rows

    def rewind(self, seq):
        self.next = min(self.next, seq)

    def snapshot(self):
        return {"gyro_dps": [0, 0, 0], "peak_dps": 0.0, "accel_g": [0, 0, 1], "i2c_errors": 0}


def test_sender_to_buffer_gives_every_group_its_own_klt_points_and_a_gapless_imu_interval():
    import argparse
    import tempfile
    from gabai.diag import log as dlog
    from gabai.pi import sender as S
    from gabai.pi.klt import KltTracker
    dlog.configure(log_dir=tempfile.mkdtemp(), run_name="pi", level="WARNING")
    rx = LiveReceiver(port=0, token="gabai")
    sb = rx.enable_sync(max_lag_s=0.0)
    rx.start()
    a = argparse.Namespace(host="127.0.0.1", port=rx.port, token="gabai", width=640, height=480, fps=15, sndbuf=65536, health_every=1.0, stats_every=1.0,
                           imu_obj=FakeImu(), mag_obj=None, quality=80, klt_obj=KltTracker(640, 480))
    th = threading.Thread(target=lambda: _run_quiet(S.run_session, a, S.TestPattern(a)), daemon=True)
    th.start()
    groups, end = [], time.time() + 12
    while time.time() < end and len(groups) < 25:
        g = sb.pop()
        groups.append(g) if g is not None else time.sleep(0.005)
    rx.stop()
    th.join(timeout=8)
    from gabai.diag import log as dlog2
    dlog2.shutdown()
    assert len(groups) >= 25
    assert all(g.klt is not None and g.klt.frame_seq == g.seq for g in groups)                      # each pair has the points computed from it
    assert [g.seq for g in groups] == list(range(groups[0].seq, groups[0].seq + len(groups)))        # none skipped, none repeated, in order
    t = [r[0] for g in groups[1:] for r in g.imu]
    assert len(t) > 200 and max(np.diff(t)) < 0.0051 and min(np.diff(t)) > 0.0049                   # the IMU record is continuous across all the groups
    assert all(g.info["imu_complete"] and g.info["imu_gap_s"] == 0 for g in groups)
    assert sb.stats()["imu_missing"] == 0 and sb.stats()["klt_missing"] == 0


def _run_quiet(fn, *a):
    try:
        fn(*a)
    except Exception:
        pass
