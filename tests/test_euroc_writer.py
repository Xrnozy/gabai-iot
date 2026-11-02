import csv
import json
import os
import tempfile

import numpy as np

from gabai.slam.euroc import G, EurocWriter, basalt_calib


def Rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_calibration_has_two_pinhole_cameras_and_the_right_one_is_plus_baseline_along_x():
    c = basalt_calib(f=881.0, cx=319.5, cy=239.5, size=(640, 480), B=0.0507, R_ic=np.eye(3))["value0"]
    assert [i["camera_type"] for i in c["intrinsics"]] == ["pinhole", "pinhole"]
    assert c["intrinsics"][0]["intrinsics"]["fx"] == 881.0 and c["resolution"] == [[640, 480], [640, 480]]
    assert abs(c["T_imu_cam"][1]["px"] - 0.0507) < 1e-9 and c["T_imu_cam"][1]["py"] == 0.0
    assert c["cam_time_offset_ns"] == 0
    json.dumps(basalt_calib(881.0, 319.5, 239.5, (640, 480), 0.0507, np.eye(3)))        # serialisable


def test_a_rotated_camera_rotates_the_baseline_into_imu_axes():
    R = Rz(90)                                                                              # camera x axis is IMU y
    c = basalt_calib(881.0, 319.5, 239.5, (640, 480), 0.0507, R)["value0"]
    assert abs(c["T_imu_cam"][1]["py"] - 0.0507) < 1e-9 and abs(c["T_imu_cam"][1]["px"]) < 1e-9
    q = c["T_imu_cam"][0]
    assert abs(q["qz"] - np.sin(np.radians(45))) < 1e-9 and abs(q["qw"] - np.cos(np.radians(45))) < 1e-9


def test_noise_defaults_come_from_the_datasheet_not_from_zero():
    c = basalt_calib(881.0, 319.5, 239.5, (640, 480), 0.0507, np.eye(3), rate_hz=100.0)["value0"]
    assert 1e-4 < c["gyro_noise_std"][0] < 1e-2 and 1e-3 < c["accel_noise_std"][0] < 0.2


def test_writer_makes_the_layout_basalt_reads_with_units_and_time_shift():
    with tempfile.TemporaryDirectory() as d:
        w = EurocWriter(d, cam_delay_s=0.045)
        img = np.full((48, 64), 90, np.uint8)
        assert w.add_frame(100.000, img, img)
        assert w.add_frame(100.066, img, img)
        assert not w.add_frame(100.066, img, img)                                          # stamp did not advance: dropped, not duplicated
        w.add_imu([[99.990, 0.0, 0.0, 180.0, 0.0, 0.0, 1.0], [100.005, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   [100.005, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])                              # same stamp as the previous: skipped
        w.close()
        cam = list(csv.reader(l for l in open(os.path.join(d, "mav0/cam0/data.csv")) if not l.startswith("#")))
        imu = list(csv.reader(l for l in open(os.path.join(d, "mav0/imu0/data.csv")) if not l.startswith("#")))
        assert [int(r[0]) for r in cam] == [99955000000, 100021000000]                        # 45 ms earlier
        assert os.path.exists(os.path.join(d, "mav0/cam1/data", cam[0][1]))
        assert len(imu) == 2 and int(imu[0][0]) == 99990000000
        assert abs(float(imu[0][3]) - np.pi) < 1e-6 and abs(float(imu[0][6]) - G) < 1e-6    # 180 deg/s -> pi rad/s, 1 g -> 9.80665
        assert abs(float(imu[1][1]) - np.radians(10.0)) < 1e-6 and abs(float(imu[1][5]) - G) < 1e-6
        assert w.n_frames == 2 and w.n_imu == 2


def test_missing_sync_file_is_reported_as_not_calibrated_not_silently_identity():
    from gabai.slam.euroc import load_imu_extrinsics
    cfg = load_imu_extrinsics("no_such_file.json")
    assert cfg["calibrated"] is False and np.allclose(cfg["R_ic"], np.eye(3)) and cfg["delay_s"] == 0.045
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "imu_sync.json")
        json.dump({"R_ci": Rz(90).tolist(), "delay_s": 0.07}, open(p, "w"))
        cfg = load_imu_extrinsics(p)
    assert cfg["calibrated"] and abs(cfg["delay_s"] - 0.07) < 1e-12 and np.allclose(cfg["R_ic"], Rz(90).T)


def test_accel_parameters_pass_into_the_basalt_calibration_and_default_to_zero():
    from gabai.slam.euroc import load_accel_calib
    p = [0.1, 0.2, 0.3, 0.01, 0.0, 0.0, 0.02, 0.0, 0.03]
    assert basalt_calib(881.0, 319.5, 239.5, (640, 480), 0.0507, np.eye(3), accel_params=p)["value0"]["calib_accel_bias"] == p
    assert basalt_calib(881.0, 319.5, 239.5, (640, 480), 0.0507, np.eye(3))["value0"]["calib_accel_bias"] == [0.0] * 9
    assert load_accel_calib("no_such_file.json") is None
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "accel_calib.json")
        json.dump({"basalt_params": p}, open(f, "w"))
        assert load_accel_calib(f) == p
