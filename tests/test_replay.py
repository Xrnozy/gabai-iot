import json
import os
import tempfile

import pytest

from gabai.diag import log as dlog
from gabai.tools import replay_pipeline as RP

REC = os.path.join("data", "slam", "rec_20261026_155811")


def run(name, **kw):
    out = tempfile.mkdtemp()
    s = RP.run_scenario(name, out_dir=out, log_dir=out, **kw)
    dlog.shutdown()
    return s, out


def events(out):
    import glob
    f = glob.glob(os.path.join(out, "*", "gabai.jsonl"))[0]
    return [json.loads(l) for l in open(f, encoding="utf-8")]


def test_a_clear_floor_gives_no_alert():
    s, _ = run("clear_floor")
    assert s["verdict"] == "PASS" and s["alerts_after_warmup"] == 0 and s["frames"] >= 20


def test_a_box_two_metres_ahead_is_found_and_its_distance_is_within_half_a_metre_of_the_truth():
    s, _ = run("box_2m")
    assert s["verdict"] == "PASS"
    assert s["truth_horizontal_m"] == 2.0 and abs(s["spoken_m_median"] - 2.0) <= 0.5 and s["alert_fraction"] >= 0.9


def test_approaching_a_wall_the_spoken_distance_shrinks_with_the_truth():
    s, _ = run("approach")
    assert s["verdict"] == "PASS"
    rows = [r for r in s["timeline"] if r["spoken_m"] is not None]
    assert len(rows) > 10 and rows[0]["spoken_m"] > rows[-1]["spoken_m"]
    assert all(abs(r["spoken_m"] - r["truth_m"]) <= 0.6 for r in rows), [(r["spoken_m"], r["truth_m"]) for r in rows if abs(r["spoken_m"] - r["truth_m"]) > 0.6]


def test_a_wrong_camera_height_is_a_false_alarm_and_the_log_shows_the_believed_height():
    s, out = run("clear_floor_wrong_height")
    assert s["verdict"] == "FALSE_ALARM" and s["believed_cam_h"] == 0.6 and s["true_cam_h"] == 1.2
    traces = [e for e in events(out) if e["ev"] == "frame.trace"]
    assert traces and traces[0]["stages"]["zones"]["cam_h_m"] == 0.6                    # an investigator can see what height the system believed
    assert any(e["ev"] == "alert.spoken" for e in events(out))


def test_every_replayed_frame_leaves_a_trace_with_a_frame_id_and_the_decision():
    s, out = run("box_2m")
    tr = [e for e in events(out) if e["ev"] == "frame.trace" and e.get("sampled")]
    assert len(tr) >= 20 and all(isinstance(t["fid"], int) for t in tr)
    assert any(d["name"] == "alert" for d in tr[5]["decisions"])
    assert os.path.exists(os.path.join(out, "replay_summary.json"))


def test_unknown_scenario_and_missing_folder_are_clear_errors():
    with pytest.raises(SystemExit) as e:
        RP.run_scenario("no_such", out_dir=tempfile.mkdtemp(), log_dir=tempfile.mkdtemp())
    assert "box_2m" in str(e.value)                                    # it lists the ones that exist
    with pytest.raises(SystemExit) as e:
        RP.recording_frames("definitely/not/here")
    assert "not found" in str(e.value).lower()


@pytest.mark.skipif(not os.path.exists(os.path.join(REC, "calib.json")), reason="the recording is not on this machine")
def test_a_real_recording_replays_through_the_real_depth_model_without_any_hardware():
    out = tempfile.mkdtemp()
    s = RP.run_recording(REC, model="s", cam_h=1.2, max_frames=12, skip=4, out_dir=out, log_dir=out)
    dlog.shutdown()
    assert s["frames"] == 12 and s["model"] == "las2-s" and s["errors"] == 0
    ev = events(out)
    tr = [e for e in ev if e["ev"] == "frame.trace"]
    assert tr and tr[0]["stages"]["depth"]["model_ms"] > 0 and "zones" in tr[0]["stages"]
    assert all(isinstance(e["fid"], int) for e in tr)
    assert os.path.exists(os.path.join(out, "replay_summary.json"))
