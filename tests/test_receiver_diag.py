import json
import os
import socket
import tempfile
import time

import cv2
import numpy as np

from gabai.diag import log as dlog
from gabai.net import protocol as P
from gabai.pc.receiver import LiveReceiver


def jpeg(color=(10, 20, 30)):
    return cv2.imencode(".jpg", np.full((48, 64, 3), color, np.uint8))[1].tobytes()


def start(token="gabai"):
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="rx", level="INFO")
    rx = LiveReceiver(port=0, token=token)
    rx.start()
    return run, rx


def client(rx, token="gabai", **hello):
    s = socket.create_connection(("127.0.0.1", rx.port), timeout=3)
    s.sendall(P.pack(P.HELLO, time.monotonic(), json.dumps({"token": token, "name": "test-pi", **hello}).encode()))
    return s


def frame(s, t=1.0, left=None, right=None):
    s.sendall(P.pack(P.FRAME, t, P.pack_frame(left or jpeg(), right or jpeg())))


def wait(cond, secs=3.0):
    end = time.time() + secs
    while time.time() < end and not cond():
        time.sleep(0.02)
    return cond()


def events(run):
    dlog.shutdown()
    return [json.loads(l) for l in open(os.path.join(run, "gabai.jsonl"), encoding="utf-8") if l.strip()]


def test_a_wrong_token_is_rejected_logged_with_the_peer_and_the_token_is_never_written():
    run, rx = start(token="secret-xyz")
    try:
        s = client(rx, token="guess")
        assert wait(lambda: rx.stats()["rejected"] == 1)
    finally:
        rx.stop()
    ev = events(run)
    r = [e for e in ev if e["ev"] == "link.rejected"][0]
    assert r["peer"] == "127.0.0.1" and "token" in r["reason"]
    text = open(os.path.join(run, "gabai.jsonl"), encoding="utf-8").read()
    assert "secret-xyz" not in text and "guess" not in text


def test_hello_is_logged_with_what_the_pi_says_it_is():
    run, rx = start()
    try:
        s = client(rx, width=640, height=480, fps=15)
        assert wait(lambda: rx.stats()["connected"])
    finally:
        rx.stop()
    h = [e for e in events(run) if e["ev"] == "link.hello"][0]
    assert h["name"] == "test-pi" and h["width"] == 640 and h["fps"] == 15


def test_garbage_on_the_wire_ends_the_connection_with_a_stated_reason():
    run, rx = start()
    try:
        s = client(rx)
        assert wait(lambda: rx.stats()["connected"])
        s.sendall(b"XXXX" + b"\x00" * 40)
        assert wait(lambda: not rx.stats()["connected"])
        err = rx.stats()["last_error"]
    finally:
        rx.stop()
    d = [e for e in events(run) if e["ev"] == "link.disconnected"][0]
    assert "bad magic" in d["reason"] and "bad magic" in err


def test_an_undecodable_frame_is_counted_and_logged_with_its_size_not_ignored():
    run, rx = start()
    try:
        s = client(rx)
        frame(s, left=b"not a jpeg at all", right=jpeg())
        assert wait(lambda: rx.stats()["decode_errors"] == 1)
        frame(s)                                                    # the stream continues after a bad frame
        assert wait(lambda: rx.stats()["frames"] == 1)
    finally:
        rx.stop()
    e = [x for x in events(run) if x["ev"] == "link.decode_error"][0]
    assert e["left_bytes"] == 17 and e["left_is_jpeg"] is False and e["right_is_jpeg"] is True


def test_frames_that_arrive_faster_than_they_are_taken_are_counted_as_overwritten():
    run, rx = start()
    try:
        s = client(rx)
        for i in range(10):
            frame(s, t=1.0 + i / 15)
        assert wait(lambda: rx.stats()["frames"] == 10)
        assert rx.stats()["overwritten"] == 9                       # nobody looked at the first nine
        before = rx.stats()["overwritten"]
        for i in range(5):
            frame(s, t=2.0 + i / 15)
            assert wait(lambda: rx.stats()["frames"] == 11 + i)
            rx.latest()                                             # this time each one is taken before the next arrives
        assert rx.stats()["overwritten"] == before + 1              # only the first of these replaced an untaken frame
    finally:
        rx.stop()
    dlog.shutdown()


def test_a_long_gap_between_frames_is_measured_and_warned():
    run, rx = start()
    try:
        s = client(rx)
        frame(s, 1.0)
        assert wait(lambda: rx.stats()["frames"] == 1)
        time.sleep(0.45)
        frame(s, 1.5)
        assert wait(lambda: rx.stats()["frames"] == 2)
        gap = rx.stats()["inter_arrival_max_ms"]
    finally:
        rx.stop()
    assert gap >= 400
    g = [e for e in events(run) if e["ev"] == "link.gap"][0]
    assert g["gap_ms"] >= 400


def test_a_gap_line_carries_what_the_pi_last_reported_about_itself():
    run, rx = start()
    try:
        s = client(rx)
        s.sendall(P.pack_json(P.HEALTH, 1.0, {"throttled": "0x50005", "fps": 1.7, "cam_fps_left": 29.1, "cam_fps_right": 11.0, "rtt_s": 0.25, "load1": 3.2}))
        frame(s, 1.0)
        assert wait(lambda: rx.stats()["frames"] == 1)
        time.sleep(0.45)
        frame(s, 1.5)
        assert wait(lambda: rx.stats()["frames"] == 2)
    finally:
        rx.stop()
    g = [e for e in events(run) if e["ev"] == "link.gap"][0]
    assert "under_voltage_now" in g["pi_power_now"] and g["pi_send_fps"] == 1.7 and g["pi_cam_fps"] == [29.1, 11.0] and g["pi_rtt_ms"] == 250 and g["pi_health_age_s"] >= 0.4


def test_a_clean_disconnect_says_the_peer_closed_and_reports_what_was_received():
    run, rx = start()
    try:
        s = client(rx)
        frame(s)
        assert wait(lambda: rx.stats()["frames"] == 1)
        s.close()
        assert wait(lambda: not rx.stats()["connected"])
    finally:
        rx.stop()
    d = [e for e in events(run) if e["ev"] == "link.disconnected"][0]
    assert "peer closed" in d["reason"] and d["frames"] == 1 and d["duration_s"] >= 0


def test_an_unknown_message_type_is_reported():
    run, rx = start()
    try:
        s = client(rx)
        s.sendall(P.pack(99, 1.0, b"{}"))
        assert wait(lambda: rx.stats()["msg_counts"].get("99") == 1)
    finally:
        rx.stop()
    assert any(e["ev"] == "link.unknown_message" and e["msg_type"] == 99 for e in events(run))


def test_imu_samples_lost_to_a_full_queue_are_counted():
    run, rx = start()
    try:
        s = client(rx)
        rx._imu_q.extend([(0, 0, 0, 0, 0, 0, 1)] * 5990)
        samples = [[1.0 + i * 0.005, 0, 0, 0, 0, 0, 1.0] for i in range(40)]
        s.sendall(P.pack_json(P.IMU, 1.0, {"gyro_dps": [0, 0, 0], "peak_dps": 0.0, "samples": samples}))
        assert wait(lambda: rx.stats()["msg_counts"].get("6") == 1)
        assert rx.stats()["imu_overflow"] >= 30
    finally:
        rx.stop()
    dlog.shutdown()
