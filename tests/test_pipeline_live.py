import json
import os
import socket
import tempfile
import time

import cv2
import numpy as np

from gabai.net import protocol as P
from gabai.pipeline import sources
from gabai.pipeline.config import PipelineConfig, validate
from gabai.tools.calibrate_stereo import Rectifier


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _live_cfg():
    c = PipelineConfig()
    c.pipeline.mode, c.source.kind, c.source.port = "stereo_imu", "live", _free_port()
    c.source.folder = tempfile.mkdtemp()
    return c


def test_live_validation_needs_the_stereo_calibration_and_says_what_is_missing():
    c = _live_cfg()
    c.source.calib, c.source.imu_sync, c.source.accel_calib = "nope.npz", "nope.json", "nope2.json"
    errs, warns = validate(c)
    assert any("stereo calibration not found" in e for e in errs)
    assert any("camera-IMU rotation is a guess" in w for w in warns)


def test_live_frames_carry_the_imu_samples_up_to_their_own_stamp():
    c = _live_cfg()
    rect, delay, k = sources.prepare_live(c)
    calib = json.load(open(os.path.join(c.source.folder, "calib.json")))["value0"]
    assert len(calib["T_imu_cam"]) == 2                                              # Basalt gets both cameras' extrinsics
    gen = sources.live_source(c, rect, delay, k)
    w, h = rect.size
    img = cv2.imencode(".jpg", np.full((h, w, 3), 90, np.uint8))[1].tobytes()
    import threading

    def pi():
        s = None
        for _ in range(100):
            try:
                s = socket.create_connection(("127.0.0.1", c.source.port), timeout=1)
                break
            except OSError:
                time.sleep(0.05)
        s.sendall(P.pack(P.HELLO, time.monotonic(), json.dumps({"token": c.source.token, "name": "fake-pi"}).encode()))
        t0 = time.monotonic()
        samples = [[t0 + i * 0.01, 0, 0, 0, 0, 0, 1.0] for i in range(30)]           # 0.3 s of IMU, then a frame stamped after it
        s.sendall(P.pack_json(P.IMU, t0, {"gyro_dps": [0, 0, 0], "peak_dps": 0.0, "samples": samples}))
        time.sleep(0.3)
        s.sendall(P.pack(P.FRAME, t0 + 0.4 + delay, P.pack_frame(img, img)))
        time.sleep(1.0)
        s.close()

    threading.Thread(target=pi, daemon=True).start()
    fr = next(gen)
    gen.close()
    assert fr.sync["kind"] == "live" and fr.cal["metric"] and fr.left.shape[:2] == (int(round(h * k)), int(round(w * k)))
    assert len(fr.imu) >= 25 and all(r[0] <= fr.t for r in fr.imu)
    assert abs(np.linalg.norm(fr.imu[0][2]) - 9.80665) < 1e-6                         # accelerometer in m/s^2 (1 g)


def test_the_baseline_check_names_the_square_size_that_would_match_the_rig():
    from gabai.tools.calibrate_stereo import check_baseline
    assert "agrees" in check_baseline({"baseline": 0.0421}, 25.0, 42.0)
    msg = check_baseline({"baseline": 0.0506}, 25.0, 42.0)
    assert "DOES NOT AGREE" in msg and "20.75 mm" in msg


def test_big_pictures_are_shrunk_after_rectifying_and_the_camera_parameters_follow():
    c = _live_cfg()
    c.source.live_width = 320                                                          # the calibration is 640 wide: shrink by half
    rect, delay, k = sources.prepare_live(c)
    assert k == 320 / rect.size[0]
    calib = json.load(open(os.path.join(c.source.folder, "calib.json")))["value0"]
    k0 = calib["intrinsics"][0]["intrinsics"]
    assert abs(k0["fx"] - rect.f * k) < 1e-6 and abs(k0["cy"] - rect.cy * k) < 1e-6
    assert calib["resolution"][0] == [320, int(round(rect.size[1] * k))]


def test_a_swapped_stereo_calibration_is_refused_for_the_live_run():
    c = _live_cfg()
    z = dict(np.load(c.source.calib))
    path = os.path.join(tempfile.mkdtemp(), "swapped.npz")
    z["T"] = np.array([[0.05], [0.0], [0.0]])                                          # right camera on the LEFT of the left one
    np.savez(path, **z)
    c.source.calib = path
    errs, _ = validate(c)
    assert any("SWAPPED" in e for e in errs)


def test_a_picture_turn_that_differs_from_the_calibration_is_applied_and_warned_about():
    c = _live_cfg()
    assert sources.live_turns(c.source, (180, 0)) == ((180, 0), "")                    # default: what the calibration was made with
    c.source.rotate_left, c.source.rotate_right = 0, 180
    rot, warn = sources.live_turns(c.source, (180, 0))
    assert rot == (0, 180) and "WARNING: pictures are turned" in warn and "--rotate-left 0 --rotate-right 180" in warn
    c.source.rotate_left = 45
    assert any("rotate_left" in e for e in validate(c)[0])


def test_pictures_of_another_size_than_the_calibration_stop_the_run_with_a_clear_message():
    import threading
    import pytest
    c = _live_cfg()
    rect, delay, k = sources.prepare_live(c)
    gen = sources.live_source(c, rect, delay, k)
    w, h = rect.size
    img = cv2.imencode(".jpg", np.full((h + 120, w, 3), 90, np.uint8))[1].tobytes()           # taller than the calibration
    def pi():
        for _ in range(100):
            try:
                s = socket.create_connection(("127.0.0.1", c.source.port), timeout=1)
                break
            except OSError:
                time.sleep(0.05)
        s.sendall(P.pack(P.HELLO, time.monotonic(), json.dumps({"token": c.source.token, "name": "fake-pi"}).encode()))
        s.sendall(P.pack(P.FRAME, time.monotonic(), P.pack_frame(img, img)))
        time.sleep(1.0)
    threading.Thread(target=pi, daemon=True).start()
    with pytest.raises(RuntimeError, match=f"--width {w} --height {h}"):
        next(gen)
