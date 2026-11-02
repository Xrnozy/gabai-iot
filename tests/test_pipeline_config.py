import json
import os
import tempfile

import numpy as np

from gabai.pipeline.config import MODES, PipelineConfig, validate


def make_recording(imu=True, calibrated=True, extr=True, rate_hz=160, seconds=3.0):
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "mav0", "cam0", "data"))
    os.makedirs(os.path.join(d, "mav0", "imu0"))
    n_cam = int(seconds * 13)
    with open(os.path.join(d, "mav0", "cam0", "data.csv"), "w") as f:
        f.write("#t,f\n")
        for i in range(n_cam):
            f.write(f"{int((1 + i / 13) * 1e9)},{i}.png\n")
    if imu:
        t = (1 + np.arange(int(seconds * rate_hz)) / rate_hz) * 1e9
        np.savetxt(os.path.join(d, "mav0", "imu0", "data.csv"), np.c_[t, np.zeros((len(t), 3)), np.tile([0, 0, 9.81], (len(t), 1))], delimiter=",", header="#t", comments="")
    calib = {"value0": {"T_imu_cam": [{"px": 0, "py": 0, "pz": 0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}, {"px": 0.05, "py": 0, "pz": 0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}] if extr else [],
                         "intrinsics": [{"intrinsics": {"fx": 800, "fy": 800, "cx": 320, "cy": 240}}], "resolution": [[640, 480], [640, 480]]}}
    json.dump(calib, open(os.path.join(d, "calib.json"), "w"))
    json.dump({"imu_extrinsics_calibrated": calibrated, "cam_delay_ms": 30.0}, open(os.path.join(d, "recording_info.json"), "w"))
    return d


def cfg(**kw):
    c = PipelineConfig()
    for k, v in kw.items():
        sec, key = k.split("__")
        setattr(getattr(c, sec), key, v)
    return c


def test_defaults_are_stereo_only_on_video_and_modes_are_the_two_named_ones():
    c = PipelineConfig()
    assert c.pipeline.mode == "stereo_only" and set(MODES) == {"stereo_only", "stereo_imu"} and c.source.kind == "video"


def test_loading_a_toml_file_and_overriding_from_the_command_line():
    p = os.path.join(tempfile.mkdtemp(), "c.toml")
    open(p, "w").write('[pipeline]\nmode = "stereo_imu"\n[source]\nkind = "recording"\nfolder = "x"\nskip = 5\n[map]\nvoxel = 0.1\n')
    c = PipelineConfig.load(p, overrides={"pipeline.mode": "stereo_only", "source.skip": 3})
    assert c.pipeline.mode == "stereo_only" and c.source.skip == 3 and c.map.voxel == 0.1 and c.source.folder == "x"


def test_unknown_keys_are_rejected_not_ignored():
    p = os.path.join(tempfile.mkdtemp(), "c.toml")
    open(p, "w").write('[pipeline]\nmod = "stereo_imu"\n')
    try:
        PipelineConfig.load(p)
        assert False
    except ValueError as e:
        assert "mod" in str(e) and "mode" in str(e)


def test_the_shipped_config_file_loads_and_is_valid_for_the_default_videos_if_present():
    c = PipelineConfig.load(os.path.join("config", "pipeline.toml"))
    assert c.pipeline.mode in MODES
    errs, warns = validate(c, check_files=False)
    assert errs == []


def test_stereo_imu_on_plain_video_is_an_error_that_says_why():
    errs, _ = validate(cfg(pipeline__mode="stereo_imu"), check_files=False)
    assert errs and "IMU" in errs[0] and "video" in errs[0]


def test_stereo_only_never_looks_at_the_imu_so_a_recording_without_one_is_fine():
    d = make_recording(imu=False)
    errs, warns = validate(cfg(pipeline__mode="stereo_only", source__kind="recording", source__folder=d))
    assert errs == []


def test_stereo_imu_needs_the_imu_file_and_says_which_one_is_missing():
    d = make_recording(imu=False)
    errs, _ = validate(cfg(pipeline__mode="stereo_imu", source__kind="recording", source__folder=d))
    assert any("imu0" in e for e in errs)


def test_stereo_imu_needs_camera_imu_extrinsics_in_the_calibration():
    d = make_recording(extr=False)
    errs, _ = validate(cfg(pipeline__mode="stereo_imu", source__kind="recording", source__folder=d))
    assert any("T_imu_cam" in e for e in errs)


def test_uncalibrated_imu_extrinsics_is_a_warning_not_a_silent_pass():
    d = make_recording(calibrated=False)
    errs, warns = validate(cfg(pipeline__mode="stereo_imu", source__kind="recording", source__folder=d))
    assert not any("extrinsics" in e for e in errs) and any("not calibrated" in w for w in warns)


def test_a_slow_or_gappy_imu_is_reported():
    d = make_recording(rate_hz=40)
    _, warns = validate(cfg(pipeline__mode="stereo_imu", source__kind="recording", source__folder=d))
    assert any("IMU rate" in w for w in warns)


def test_imu_that_does_not_overlap_the_camera_time_is_an_error():
    d = make_recording()
    t = (100 + np.arange(400) / 160) * 1e9                              # IMU stamped 100 s later than the camera
    np.savetxt(os.path.join(d, "mav0", "imu0", "data.csv"), np.c_[t, np.zeros((400, 3)), np.tile([0, 0, 9.81], (400, 1))], delimiter=",", header="#t", comments="")
    errs, _ = validate(cfg(pipeline__mode="stereo_imu", source__kind="recording", source__folder=d))
    assert any("overlap" in e for e in errs)


def test_video_mode_warns_that_scale_is_assumed():
    _, warns = validate(cfg(), check_files=False)
    assert any("assumed" in w and "metric" in w for w in warns)
