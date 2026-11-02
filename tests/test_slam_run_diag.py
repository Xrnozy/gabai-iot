import json
import os
import tempfile

import numpy as np

from gabai.diag import log as dlog
from gabai.slam.run_diag import SlamRunDiag
from gabai.slam.still import StillGate

G = 9.80665


def events(run):
    dlog.shutdown()
    return [json.loads(l) for l in open(os.path.join(run, "gabai.jsonl"), encoding="utf-8") if l.strip()]


def img(seed=0, blur=False):
    import cv2
    a = np.random.default_rng(seed).integers(0, 255, (480, 640), dtype=np.uint8)
    return cv2.GaussianBlur(a, (0, 0), 15) if blur else a


def pose(t, x, y=0.0, timing=None):
    return {"t_ns": int(t * 1e9), "p": (x, y, 0.0), "q": (0.0, 0.0, 0.0, 1.0), "v": (0.0, 0.0, 0.0), "timing": timing}


def run_diag(**kw):
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="s", level="INFO")
    out = tempfile.mkdtemp()
    return run, out, SlamRunDiag(out, f_px=881.0, cam_delay_s=0.022, period_s=1.0, **kw)


def test_a_jump_is_logged_with_its_frame_id_and_the_context_that_may_explain_it():
    run, out, d = run_diag()
    d.note_image(img(blur=True), fid=17000)
    for i in range(10):
        d.note_pose(pose(100 + i * 0.07, i * 0.04), push_wall=None, now=float(i) * 0.07)
    d.note_pose(pose(100.7, 4.0), push_wall=None, now=0.7)               # 4 m in 70 ms
    ev = events(run)
    j = [e for e in ev if e["ev"] == "slam.jump"][0]
    assert j["step_m"] > 3.5 and j["fid"] == int(round(100.7 * 1000 + 22)) and "image" in j and j["image"]["status"] in ("suspect", "fail")


def test_no_pose_for_a_second_while_frames_keep_arriving_is_a_warning_with_counts():
    run, out, d = run_diag()
    d.note_pose(pose(100.0, 0.0), push_wall=None, now=0.0)
    for i in range(20):
        d.note_frame_pushed(fid=1000 + i)
    d.tick(2.5, gate=None, n_frames=20, n_imu=400)
    e = [x for x in events(run) if x["ev"] == "slam.no_pose"][0]
    assert e["frames_pushed_since_last_pose"] == 20 and e["seconds_without_pose"] >= 2.0


def test_a_gate_that_never_opens_says_why():
    run, out, d = run_diag()
    gate = StillGate(1.0)
    for i in range(400):
        gate.feed(int(i * 5e6), np.array([0.0, 0.0, G]), np.radians([20.0, 0, 0]))     # turning at 20 deg/s
    d.tick(30.0, gate=gate, n_frames=0, n_imu=400, started_wall=0.0)
    e = [x for x in events(run) if x["ev"] == "slam.gate_waiting"][0]
    assert e["progress"] == 0.0 and abs(e["last_gyro_dps"] - 20.0) < 0.5 and "hold" in e["hint"].lower()


def test_periodic_stats_include_pose_delay_and_where_basalt_spent_its_time():
    run, out, d = run_diag()
    timing = {"queue_wait_ms": 1.0, "frontend_ms": 5.0, "backend_ms": 8.0, "basalt_total_ms": 14.0, "to_consumer_ms": 0.2}
    d.tick(10.0, gate=None, n_frames=0, n_imu=0)                         # the loop calls tick() from its first pass: this starts the window
    for i in range(20):
        d.note_frame_pushed(fid=i)
        d.note_pose(pose(100 + i * 0.07, i * 0.04, timing=timing), push_wall=10.0 + i * 0.07 - 0.02, now=10.0 + i * 0.07)
    d.tick(12.0, gate=None, n_frames=20, n_imu=400)
    s = [e for e in events(run) if e["ev"] == "slam.stats"][0]
    assert s["pose_delay_ms_p50"] is not None and s["basalt_ms"]["backend_ms"] == 8.0 and s["poses_in_window"] == 20


def test_finish_writes_a_summary_with_findings_and_logs_them_by_level():
    run, out, d = run_diag()
    for i in range(60):
        x = i * 0.04 + (3.0 if i >= 30 else 0.0)                         # one jump halfway
        d.note_pose(pose(100 + i * 0.07, x), push_wall=None, now=i * 0.07)
    imu = [(100 + i * 0.005, 0.0, 0.0, 0.0, 0.0, 0.0, G) for i in range(900)]
    for s in imu:
        d.note_imu(s[0], s[1:4], s[4:7])
    summary = d.finish()
    assert summary["trajectory"]["n_jumps"] == 1
    assert any(f["code"] == "pose_jump" for f in summary["findings"])
    saved = json.load(open(os.path.join(out, "slam_summary.json")))
    assert saved["findings"][0]["level"] == "confirmed" and "no ground truth" in saved["trajectory"]["caveat"].lower()
    ev = events(run)
    assert any(e["ev"] == "slam.finding" and e["lvl"] == "WARNING" and e["code"] == "pose_jump" for e in ev)


def test_a_clean_run_reports_no_findings_and_says_so_without_claiming_accuracy():
    run, out, d = run_diag()
    for i in range(80):
        d.note_pose(pose(100 + i * 0.07, i * 0.04), push_wall=None, now=i * 0.07)
    s = d.finish()
    assert s["findings"] == []
    assert "plausible" in s["verdict"] and "correct" not in s["verdict"].replace("not whether it is correct", "")
