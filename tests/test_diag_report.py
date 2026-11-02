import os
import tempfile

import numpy as np

from gabai.diag import log as dlog
from gabai.diag.session import PerceptionDiag
from gabai.live_alerts import AlertController
from gabai.speaker import AlertSpeaker
from gabai.tools.diag_report import analyze_events, format_report, load_events
from gabai.zones import analyze, floor_disparity

F, B, CX, CY = 881.0, 0.0507, 319.5, 239.5


def scene(z=1.2, cam_h=0.5):
    d = floor_disparity((480, 640), F, B, cam_h, CX, CY)
    d[250:420, 260:380] = F * B / z
    return d


def make_run(n=90, link=None, img=None, disp_fn=None, cam_h=0.5, imu_cal=True, trusted=True, speech=150.0):
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="t", level="INFO")
    diag = PerceptionDiag("test", F, B, CX, CY, sample_every=1, system_every=0, image_every=1)
    ctl = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=True)
    link = link or {"age_s": 0.05, "pair_dt_ms": 3.0, "turn_dps": 2.0, "latency_s": 0.04}
    for i in range(n):
        d = disp_fn(i) if disp_fn else scene()
        tr = diag.begin(10000 + i * 70, 1.0, 1.03, link, left=img)
        diag.after_depth(tr, d, np.isfinite(d) if trusted else None, 40.0, "las2-m", lr_check=trusted)
        res = analyze(np.where(d > 0.5, d, np.nan), F, B, cam_h, CY, cx=CX, evidence=True)
        said = ctl.on_frame(100.0 + i * 0.07, res.alert, "steady", link["latency_s"])
        diag.after_decision(tr, res, ctl, said, loop_ms=55.0, speech_ms=speech if said else None, cam_h=cam_h, imu_calibrated=imu_cal, have_up_vector=imu_cal)
    return run


def report(run):
    dlog.shutdown()
    return analyze_events(load_events(run))


def codes(rep):
    return {(f["level"], f["code"]) for f in rep["findings"]}


def good_image():
    return np.random.default_rng(0).integers(40, 220, (480, 640), dtype=np.uint8)


def test_a_healthy_run_has_no_findings_and_the_report_does_not_claim_accuracy():
    rep = report(make_run(img=good_image()))
    assert rep["findings"] == [], rep["findings"]
    text = format_report(rep)
    assert "no problems" in text.lower() and "not whether" in text.lower()          # says what it does not know
    assert rep["pipeline"]["traces_sampled"] >= 80


def test_stale_frames_are_a_confirmed_finding_with_the_measured_age():
    rep = report(make_run(link={"age_s": 0.9, "pair_dt_ms": 3.0, "turn_dps": 2.0, "latency_s": 0.95}, img=good_image()))
    f = [x for x in rep["findings"] if x["code"] == "stale_frames"][0]
    assert f["level"] == "confirmed" and f["evidence"]["fraction_failed"] > 0.9 and f["possible_causes"]


def test_a_dark_scene_with_no_depth_is_traced_to_the_image_not_the_model():
    dark = np.full((480, 640), 20, np.uint8)
    rep = report(make_run(img=dark, disp_fn=lambda i: np.full((480, 640), np.nan, np.float32)))
    f = {x["code"]: x for x in rep["findings"]}
    assert "frequent_cant_see" in f and f["frequent_cant_see"]["level"] == "confirmed"
    assert f["frequent_cant_see"]["evidence"]["dark_or_flat_images_fraction"] > 0.9
    assert any("light" in c.lower() for c in f["frequent_cant_see"]["possible_causes"])


def test_pixels_piled_up_at_the_model_limit_are_flagged_as_possible_not_confirmed():
    def d(i):
        x = scene()
        x[:200] = 191.0
        return x
    rep = report(make_run(img=good_image(), disp_fn=d))
    assert ("possible", "objects_at_model_limit") in codes(rep)


def test_a_missing_floor_model_is_a_confirmed_explanation_for_wrong_alerts():
    rep = report(make_run(img=good_image(), cam_h=None, imu_cal=False))
    assert ("confirmed", "floor_model_unreliable") in codes(rep)


def test_late_speech_names_the_slowest_stage():
    rep = report(make_run(img=good_image(), speech=700.0))
    f = [x for x in rep["findings"] if x["code"] == "latency_over_target"][0]
    assert f["level"] == "confirmed" and f["evidence"]["speech_ms_p50"] >= 700 and "stages_ms_p50" in f["evidence"]


def test_link_events_are_counted_with_their_reasons():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="l", level="INFO")
    lg = dlog.get("link")
    for _ in range(3):
        lg.warning("link.disconnected", peer="1.2.3.4", reason="no data for 10 s (receive timeout)", frames=50, duration_s=30)
    lg.warning("link.decode_error", left_bytes=17)
    lg.warning("link.stats_dummy_not_counted")
    dlog.get("pi").warning("pi.power", flags=["under_voltage_now", "throttled_now"], throttled="0x50005")
    dlog.get("log").warning("log.dropped", count=40)
    rep = report(run)
    c = {x["code"]: x for x in rep["findings"]}
    assert c["link_dropouts"]["level"] == "confirmed" and c["link_dropouts"]["evidence"]["reasons"]["no data for 10 s (receive timeout)"] == 3
    assert c["pi_power"]["level"] == "confirmed" and "pi-power-guide" in " ".join(c["pi_power"]["next_steps"])
    assert c["log_incomplete"]["level"] == "confirmed"


def test_a_crash_is_reported_first_with_its_error_line():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="c", level="INFO")
    dlog.get("viewer").error("viewer.crash", error="ValueError: boom", exc="Traceback ...\nValueError: boom")
    rep = report(run)
    assert rep["findings"][0]["code"] == "crash" and "boom" in rep["findings"][0]["message"]


def test_operator_marks_are_listed_with_the_evidence_file():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="m", level="INFO")
    dlog.get("viewer").warning("viewer.mark", kind="false_alarm", fid=1234, evidence="marks/false_alarm_1234.jsonl", alert=["ahead", 1.0])
    rep = report(run)
    assert rep["operator_marks"][0]["kind"] == "false_alarm" and rep["operator_marks"][0]["fid"] == 1234
    assert "false_alarm_1234" in format_report(rep)


def test_too_little_data_is_said_plainly():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="e", level="INFO")
    dlog.get("viewer").info("viewer.start")
    rep = report(run)
    assert "not enough" in format_report(rep).lower()


def test_failure_rates_come_from_the_regular_sample_only():
    # 1 frame in 10 is sampled; the failing frames are all 'interesting' extras. The rate must not be inflated by them.
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="s", level="INFO")
    diag = PerceptionDiag("test", F, B, CX, CY, sample_every=10, system_every=0, image_every=0)
    ctl = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=True)
    for i in range(200):
        link = {"age_s": 0.9 if i % 10 == 3 else 0.05, "pair_dt_ms": 3.0, "turn_dps": 2.0, "latency_s": 0.05}
        d = scene()
        tr = diag.begin(i * 70, 1.0, 1.03, link, left=None)
        diag.after_depth(tr, d, None, 40.0, "las2-m")
        res = analyze(np.where(d > 0.5, d, np.nan), F, B, 0.5, CY, cx=CX, evidence=True)
        diag.after_decision(tr, res, ctl, [], loop_ms=55.0, speech_ms=None, cam_h=0.5, imu_calibrated=True, have_up_vector=True)
    rep = report(run)
    assert rep["pipeline"]["traces_sampled"] == 20
    assert not [f for f in rep["findings"] if f["code"] == "stale_frames"]        # none of the sampled frames was stale
    assert rep["pipeline"]["failed_in_sample"].get("frame_age", 0) == 0
