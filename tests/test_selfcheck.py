import json
import os
import shutil
import tempfile

import numpy as np

from gabai.tools import selfcheck as S


def mini_root(**have):
    """A throw-away project folder with only the pieces asked for."""
    r = tempfile.mkdtemp()
    os.makedirs(os.path.join(r, "data"))
    os.makedirs(os.path.join(r, "gabai", "audio", "alerts"))
    if have.get("stereo", False):
        np.savez(os.path.join(r, "data", "stereo_calib.npz"), size=np.array([640, 480]), focal=881.0, baseline=0.0507, cx=319.5, cy=239.5)
        json.dump({"baseline": 0.0507, "focal": 881.0, "cx": 319.5, "cy": 239.5, "size": [640, 480], "rms_stereo": 0.6, "swapped": have.get("swapped", False),
                   "verdict": "OK", "rotate_left": 180, "rotate_right": 0}, open(os.path.join(r, "data", "stereo_calib.json"), "w"))
    if have.get("imu", False):
        a = np.radians(8.0)                                              # a fitted rotation, not a perfect axis swap
        R = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        json.dump({"delay_s": 0.03, "R_ci": R, "quality": 0.8}, open(os.path.join(r, "data", "imu_sync.json"), "w"))
    if have.get("accel", False):
        json.dump({"basalt_params": [0] * 9, "bias": [0, 0, 0], "scale": [0, 0, 0], "rms_after": 0.02}, open(os.path.join(r, "data", "accel_calib.json"), "w"))
    for i in range(have.get("clips", 0)):
        open(os.path.join(r, "gabai", "audio", "alerts", f"c{i}.wav"), "wb").write(b"RIFF")
    return r


def by_name(results):
    return {r.name: r for r in results}


def test_every_failure_has_a_fix_and_the_summary_counts_by_status():
    res = S.check_calibration(mini_root())
    assert res and all(r.status in ("PASS", "WARN", "FAIL", "SKIP") for r in res)
    bad = [r for r in res if r.status in ("WARN", "FAIL")]
    assert bad and all(r.fix for r in bad)
    s = S.summarize(res)
    assert s["FAIL"] + s["WARN"] >= 1 and sum(s.values()) == len(res)


def test_a_missing_stereo_calibration_is_a_fail_with_the_command_that_makes_one():
    r = by_name(S.check_calibration(mini_root()))["stereo calibration"]
    assert r.status == "FAIL" and "calibrate_stereo" in r.fix


def test_a_swapped_camera_calibration_is_a_fail():
    r = by_name(S.check_calibration(mini_root(stereo=True, swapped=True, imu=True, accel=True)))["stereo calibration"]
    assert r.status == "FAIL" and "swapped" in r.detail.lower()


def test_missing_imu_and_accelerometer_calibration_are_warnings_that_explain_the_consequence():
    r = by_name(S.check_calibration(mini_root(stereo=True)))
    assert r["IMU-camera rotation"].status == "WARN" and "diverge" in r["IMU-camera rotation"].detail.lower() + r["IMU-camera rotation"].fix.lower()
    assert r["accelerometer calibration"].status == "WARN" and "calibrate_accel" in r["accelerometer calibration"].fix


def test_a_complete_calibration_set_passes():
    r = S.check_calibration(mini_root(stereo=True, imu=True, accel=True))
    assert [x.status for x in r if x.name in ("stereo calibration", "IMU-camera rotation", "accelerometer calibration")] == ["PASS", "PASS", "PASS"]


def test_alert_clips_must_all_be_there():
    assert S.check_alert_audio(mini_root(clips=3)).status == "FAIL"
    assert S.check_alert_audio(mini_root(clips=23)).status == "PASS"


def test_a_busy_port_is_reported_as_the_other_program_holding_it():
    import socket
    s = socket.socket()
    s.bind(("0.0.0.0", 0))
    s.listen(1)
    port = s.getsockname()[1]
    r = S.check_port(port, "test", exclusive=False)
    s.close()
    assert r.status in ("FAIL", "WARN") and "in use" in r.detail
    assert S.check_port(port, "test", exclusive=False).status == "PASS"


def test_disk_and_log_folder_checks():
    d = tempfile.mkdtemp()
    assert S.check_log_dir(d).status == "PASS"
    assert S.check_log_dir(os.path.join(d, "no", "such", "\0bad")).status == "FAIL"


def test_the_pure_software_smoke_test_checks_the_decision_chain_against_known_geometry():
    r = S.check_software_chain()
    assert r.status == "PASS", r.detail
    assert "ahead" in r.detail


def test_exit_code_is_nonzero_only_for_a_fail():
    ok = [S.Result("a", "PASS", "", ""), S.Result("b", "WARN", "x", "y")]
    assert S.exit_code(ok) == 0 and S.exit_code(ok + [S.Result("c", "FAIL", "x", "y")]) == 1


def test_an_exact_axis_swap_rotation_is_called_out_as_unfitted():
    r = tempfile.mkdtemp()
    os.makedirs(os.path.join(r, "data"))
    json.dump({"delay_s": 0.03, "R_ci": [[0, -1, 0], [-1, 0, 0], [0, 0, -1]], "quality": 0.2}, open(os.path.join(r, "data", "imu_sync.json"), "w"))
    x = [c for c in S.check_calibration(r) if c.name == "IMU-camera rotation"][0]
    assert x.status == "WARN" and "check_extrinsics" in x.fix
