"""Failure behaviour of the decision chain: bad or missing input must say so and never turn into a confident 'path clear'."""
import json
import tempfile

import numpy as np

from gabai.diag import log as dlog
from gabai.tools import replay_pipeline as RP

CAL = {"f": 881.0, "B": 0.0507, "cx": 319.5, "cy": 239.5}


def chain(cam_h=1.2):
    dlog.configure(log_dir=tempfile.mkdtemp(), run_name="t", level="DEBUG")
    return RP.Chain(CAL, cam_h, imu_cal=True)


def test_a_model_that_returns_no_depth_is_cant_see_ahead_not_clear():
    c = chain()
    res, _ = c.frame(1000, 1.0, np.full((480, 640), np.nan, np.float32), np.zeros((480, 640), bool), 20.0)
    dlog.shutdown()
    assert res.alert[0] == "cant_see_ahead"


def test_all_zero_or_negative_disparity_is_treated_as_invalid():
    c = chain()
    res, _ = c.frame(1000, 1.0, np.zeros((480, 640), np.float32), np.ones((480, 640), bool), 20.0)
    dlog.shutdown()
    assert res.alert[0] == "cant_see_ahead"


def test_recovery_after_blind_frames_returns_to_normal():
    c = chain()
    blind = np.full((480, 640), np.nan, np.float32)
    seen = [RP.sim_frames(RP.SCENARIOS["clear_floor"]).__next__()]
    out = []
    for k in range(3):
        out.append(c.frame(1000 + k, k / 14, blind, np.zeros_like(blind, bool), 20.0)[0].alert)
    f = seen[0]
    for k in range(3, 6):
        out.append(c.frame(1000 + k, k / 14, f["disp"], np.isfinite(f["disp"]), 20.0, R_wc=f["R_wc"])[0].alert)
    dlog.shutdown()
    assert [a[0] for a in out[:3]] == ["cant_see_ahead"] * 3 and out[3:] == [None, None, None]


def test_a_frame_with_no_imu_orientation_still_decides_and_the_trace_says_the_floor_model_was_level():
    c = chain()
    f = RP.sim_frames(RP.SCENARIOS["clear_floor"]).__next__()
    res, _ = c.frame(1000, 1.0, f["disp"], np.isfinite(f["disp"]), 20.0)      # no R_wc, no imu samples
    dlog.shutdown()
    assert res.evidence["floor_model"] == "level camera"                     # recorded, so a wrong result can be traced to it


def test_unknown_camera_height_disables_floor_model_and_drop_check_and_says_so():
    c = chain(cam_h=None)
    f = RP.sim_frames(RP.SCENARIOS["clear_floor"]).__next__()
    res, _ = c.frame(1000, 1.0, f["disp"], np.isfinite(f["disp"]), 20.0, R_wc=f["R_wc"])
    dlog.shutdown()
    assert res.evidence["floor_model"].startswith("none")


def test_a_missing_picture_in_a_recording_is_logged_and_skipped():
    import cv2
    from pathlib import Path
    tmp_path = Path(tempfile.mkdtemp())
    rec = tmp_path / "rec"
    for cam in ("cam0", "cam1"):
        (rec / "mav0" / cam / "data").mkdir(parents=True)
    (rec / "mav0" / "imu0").mkdir(parents=True)
    cv2.imwrite(str(rec / "mav0" / "cam0" / "data" / "1000000000.png"), np.zeros((48, 64), np.uint8))   # right picture is missing
    (rec / "mav0" / "cam0" / "data.csv").write_text("#t,f\n1000000000,1000000000.png\n")
    (rec / "mav0" / "imu0" / "data.csv").write_text("#t\n")
    (rec / "calib.json").write_text(json.dumps({"value0": {"intrinsics": [{"intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0}}], "resolution": [[64, 48]],
                                                          "T_imu_cam": [{"px": 0, "py": 0, "pz": 0}, {"px": 0.05, "py": 0, "pz": 0}]}}))
    dlog.configure(log_dir=str(tmp_path / "l"), run_name="t", level="DEBUG")
    assert list(RP.recording_frames(str(rec))) == []
    dlog.shutdown()
    import glob
    ev = [json.loads(l) for l in open(glob.glob(str(tmp_path / "l" / "*" / "gabai.jsonl"))[0])]
    assert any(e["ev"] == "replay.frame_missing" and e["lvl"] == "ERROR" for e in ev)
