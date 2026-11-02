import io
import json
import socket
import time

import cv2
import numpy as np

from gabai.net import protocol as P
from gabai.pc.receiver import LiveReceiver


class Pipe:
    """Feeds bytes to read_message in awkward small pieces, like a real TCP stream."""

    def __init__(self, data, step=3):
        self.buf, self.step = io.BytesIO(data), step

    def recv(self, n):
        return self.buf.read(min(n, self.step))


def jpeg(color):
    img = np.full((48, 64, 3), color, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_roundtrip_with_fragmented_reads():
    left, right = b"L" * 1000, b"R" * 2000
    data = P.pack(P.FRAME, 12.5, P.pack_frame(left, right)) + P.pack(P.HEALTH, 13.0, b'{"cpu":1}')
    pipe = Pipe(data)
    t, ts, payload = P.read_message(pipe)
    assert (t, ts) == (P.FRAME, 12.5)
    assert P.unpack_frame(payload) == (left, right)
    t, ts, payload = P.read_message(pipe)
    assert t == P.HEALTH and json.loads(payload) == {"cpu": 1}


def test_bad_magic_and_oversize_are_rejected():
    bad = b"XXXX" + P.pack(P.HELLO, 0.0, b"{}")[4:]
    try:
        P.read_message(Pipe(bad))
        assert False, "bad magic accepted"
    except P.ProtocolError:
        pass
    huge = P.HEADER.pack(P.MAGIC, P.FRAME, 0.0, P.MAX_PAYLOAD + 1)
    try:
        P.read_message(Pipe(huge))
        assert False, "oversize accepted"
    except P.ProtocolError:
        pass


def test_truncated_stream_raises_connection_error():
    data = P.pack(P.FRAME, 1.0, P.pack_frame(b"a" * 50, b"b" * 50))[:-10]
    try:
        P.read_message(Pipe(data))
        assert False
    except ConnectionError:
        pass


def _client(port, token="gabai"):
    s = socket.create_connection(("127.0.0.1", port), timeout=3)
    s.sendall(P.pack(P.HELLO, time.monotonic(), json.dumps({"token": token, "name": "test-pi"}).encode()))
    return s


def test_receiver_delivers_latest_pair_and_stats():
    rx = LiveReceiver(port=0, token="gabai")
    rx.start()
    try:
        s = _client(rx.port)
        for i in range(5):
            s.sendall(P.pack(P.FRAME, 100.0 + i / 15, P.pack_frame(jpeg((10, 20, 30)), jpeg((200, 100, 50)))))
        s.sendall(P.pack(P.HEALTH, 101.0, json.dumps({"throttled": "0x0", "temp_c": 51.0}).encode()))
        deadline = time.time() + 3
        while time.time() < deadline and rx.stats()["frames"] < 5:
            time.sleep(0.02)
        pair = rx.latest()
        assert rx.stats()["frames"] == 5
        assert pair is not None and pair.left.shape == (48, 64, 3)
        assert abs(int(pair.left[0, 0, 0]) - 10) < 8      # decoded correct image, left is left
        assert abs(int(pair.right[0, 0, 0]) - 200) < 8
        assert pair.t_capture == 100.0 + 4 / 15            # newest frame wins
        time.sleep(0.2)
        assert rx.stats()["health"]["temp_c"] == 51.0
        s.close()
    finally:
        rx.stop()


def test_receiver_rejects_wrong_token():
    rx = LiveReceiver(port=0, token="gabai")
    rx.start()
    try:
        s = _client(rx.port, token="wrong")
        s.sendall(P.pack(P.FRAME, 1.0, P.pack_frame(jpeg((1, 2, 3)), jpeg((1, 2, 3)))))
        time.sleep(0.5)
        assert rx.stats()["frames"] == 0
        assert rx.stats()["rejected"] == 1
        s.close()
    finally:
        rx.stop()


def test_receiver_survives_reconnect_and_answers_ping():
    rx = LiveReceiver(port=0, token="gabai")
    rx.start()
    try:
        s = _client(rx.port)
        s.sendall(P.pack(P.PING, 5.0, json.dumps({"t1": 5.0}).encode()))
        t, _, payload = P.read_message(s)
        assert t == P.PONG and json.loads(payload)["t1"] == 5.0 and "t2" in json.loads(payload)
        s.close()
        time.sleep(0.2)
        s2 = _client(rx.port)
        s2.sendall(P.pack(P.FRAME, 7.0, P.pack_frame(jpeg((9, 9, 9)), jpeg((9, 9, 9)))))
        deadline = time.time() + 3
        while time.time() < deadline and rx.stats()["frames"] < 1:
            time.sleep(0.02)
        assert rx.stats()["frames"] == 1 and rx.stats()["connections"] == 2
        s2.close()
    finally:
        rx.stop()


def test_receiver_rotates_a_flipped_camera():
    img = np.zeros((48, 64, 3), np.uint8)
    img[:10, :10] = (0, 0, 255)  # red marker in the top-left corner
    jb = cv2.imencode(".png", img)[1].tobytes()
    rx = LiveReceiver(port=0, token="gabai", rotate_left=180)
    rx.start()
    try:
        s = _client(rx.port)
        s.sendall(P.pack(P.FRAME, 1.0, P.pack_frame(jb, jb)))
        deadline = time.time() + 3
        while time.time() < deadline and rx.stats()["frames"] < 1:
            time.sleep(0.02)
        pair = rx.latest()
        assert pair.left[-3, -3, 2] > 200 and pair.left[3, 3, 2] < 50   # left rotated: marker bottom-right
        assert pair.right[3, 3, 2] > 200                                  # right untouched
        s.close()
    finally:
        rx.stop()


def test_receiver_keeps_latest_imu_message():
    rx = LiveReceiver(port=0, token="gabai")
    rx.start()
    try:
        s = _client(rx.port)
        s.sendall(P.pack_json(P.IMU, 5.0, {"gyro_dps": [1.0, -2.0, 3.0], "peak_dps": 42.5, "accel_g": [0, 0, 1]}))
        deadline = time.time() + 3
        while time.time() < deadline and not rx.stats().get("imu"):
            time.sleep(0.02)
        assert rx.stats()["imu"]["peak_dps"] == 42.5
        s.close()
    finally:
        rx.stop()


def test_receiver_hands_over_timestamped_imu_samples_once():
    rx = LiveReceiver(port=0, token="gabai")
    rx.start()
    try:
        s = _client(rx.port)
        samples = [[10.00, 1, 2, 3, 0, 0, 1], [10.005, 1.5, 2.5, 3.5, 0, 0, 1]]
        s.sendall(P.pack_json(P.IMU, 10.01, {"samples": samples, "peak_dps": 4.0}))
        deadline = time.time() + 3
        got = []
        while time.time() < deadline and len(got) < 2:
            got += rx.drain_imu()
            time.sleep(0.02)
        assert got == [tuple(x) for x in samples]
        assert rx.drain_imu() == []                      # handed over only once
        s.close()
    finally:
        rx.stop()


def test_receiver_hands_over_compass_readings_once():
    rx = LiveReceiver(port=0, token="gabai")
    rx.start()
    try:
        s = _client(rx.port)
        mag = [[20.0, 100, -200, 300], [20.02, 101, -199, 301]]
        s.sendall(P.pack_json(P.IMU, 20.03, {"mag": mag, "mag_overflows": 2}))
        deadline = time.time() + 3
        got = []
        while time.time() < deadline and len(got) < 2:
            got += rx.drain_mag()
            time.sleep(0.02)
        assert got == [tuple(x) for x in mag]
        assert rx.drain_mag() == []
        assert rx.stats()["mag_overflows"] == 2
        s.close()
    finally:
        rx.stop()
