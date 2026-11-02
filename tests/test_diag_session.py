import json
import os
import tempfile

import numpy as np

from gabai.diag import log as dlog
from gabai.diag.session import PerceptionDiag
from gabai.live_alerts import AlertController
from gabai.speaker import AlertSpeaker
from gabai.zones import analyze, floor_disparity

F, B, CX, CY = 881.0, 0.0507, 319.5, 239.5


def rows(run):
    return [json.loads(l) for l in open(os.path.join(run, "gabai.jsonl"), encoding="utf-8") if l.strip()]


def scene(z=1.2, cam_h=0.5):
    d = floor_disparity((480, 640), F, B, cam_h, CX, CY)
    d[250:420, 260:380] = F * B / z
    return d


def one_frame(diag, fid, disp, controller, link=None, trusted=None, cam_h=0.5, img=None):
    link = link or {"age_s": 0.05, "pair_dt_ms": 4.0, "turn_dps": 3.0, "latency_s": 0.04}
    tr = diag.begin(fid, t_capture=fid / 1000.0, t_recv=fid / 1000.0 + 0.03, link=link, left=img)
    diag.after_depth(tr, disp, trusted, model_ms=40.0, model="las2-m", lr_check=trusted is not None)
    res = analyze(np.where(disp > 0.5, disp, np.nan), F, B, cam_h, CY, cx=CX, evidence=True)
    said = controller.on_frame(100.0 + fid / 1000.0, res.alert, "steady", link["latency_s"])
    diag.after_decision(tr, res, controller, said, loop_ms=55.0, speech_ms=150.0, cam_h=cam_h, imu_calibrated=True, have_up_vector=True)
    return tr, res, said


def test_one_frame_leaves_a_trace_that_explains_the_alert_end_to_end():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="d", level="INFO")
    diag = PerceptionDiag(source="test", f=F, B=B, cx=CX, cy=CY, sample_every=1, system_every=0)
    ctl = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=True)
    tr, res, said = one_frame(diag, 17785712, scene(1.2), ctl)
    dlog.shutdown()
    d = tr.to_dict()
    assert d["fid"] == 17785712
    assert d["stages"]["depth"]["model"] == "las2-m" and d["stages"]["depth"]["disparity"]["kind"] == "model_prediction"
    assert d["stages"]["zones"]["bands"]["ahead"]["p5_depth_m"] and d["stages"]["zones"]["decision_path"]
    dec = [x for x in d["decisions"] if x["name"] == "alert"][0]
    assert dec["accepted"] is True and dec["outcome"] == "spoken" and said
    evs = [r["ev"] for r in rows(run)]
    assert "frame.trace" in evs and "alert.spoken" in evs
    a = [r for r in rows(run) if r["ev"] == "alert.spoken"][0]
    assert a["fid"] == 17785712 and "ahead" in a["phrase"] and a["nearest_px"] and a["depth_kind"] == "estimated"


def test_a_suppressed_alert_is_logged_with_its_reason_not_hidden():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="d2", level="INFO")
    diag = PerceptionDiag(source="test", f=F, B=B, cx=CX, cy=CY, sample_every=1, system_every=0)
    ctl = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=True)
    one_frame(diag, 1000, scene(1.2), ctl)
    tr, _, said = one_frame(diag, 1070, scene(1.2), ctl)               # same alert 70 ms later: the repeat rule holds it back
    dlog.shutdown()
    assert said == []
    dec = [x for x in tr.to_dict()["decisions"] if x["name"] == "alert"][0]
    assert dec["accepted"] is False and "already said" in " ".join(dec["reasons"])
    assert any(r["ev"] == "alert.suppressed" and "already said" in r["reason"] for r in rows(run))


def test_failed_checks_are_named_in_the_trace_and_logged_once_per_period():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="d3", level="INFO")
    diag = PerceptionDiag(source="test", f=F, B=B, cx=CX, cy=CY, sample_every=1000, system_every=0)
    ctl = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=True)
    stale = {"age_s": 0.9, "pair_dt_ms": 30.0, "turn_dps": 90.0, "latency_s": 0.95}
    for i in range(30):
        tr, _, _ = one_frame(diag, 2000 + i * 70, scene(1.2), ctl, link=stale)
    dlog.shutdown()
    failed = tr.to_dict()["failed_checks"]
    assert "frame_age" in failed and "pair_sync" in failed
    n = len([r for r in rows(run) if r["ev"] == "check.fail" and r["check"] == "frame_age"])
    assert 1 <= n <= 2                                                  # 30 bad frames, one line (plus at most one repeat)


def test_missing_inputs_are_reported_as_unknown_not_as_good():
    diag = PerceptionDiag(source="test", f=F, B=B, cx=CX, cy=CY, sample_every=1000, system_every=0)
    ctl = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=True)
    tr, _, _ = one_frame(diag, 1, scene(), ctl, link={"age_s": None, "pair_dt_ms": None, "turn_dps": None, "latency_s": None})
    st = {c["name"]: c["status"] for c in tr.to_dict()["checks"]}
    assert st["frame_age"] == "unknown" and st["pair_sync"] == "unknown"


def test_unreliable_depth_is_flagged_before_it_can_become_a_confident_decision():
    diag = PerceptionDiag(source="test", f=F, B=B, cx=CX, cy=CY, sample_every=1000, system_every=0)
    ctl = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=True)
    d = scene(1.2)
    d[:200] = 191.0                                                      # a big patch at the model limit
    tr, _, _ = one_frame(diag, 5, d, ctl)
    cap = [c for c in tr.to_dict()["checks"] if c["name"] == "disparity_cap"][0]
    assert cap["status"] in ("suspect", "fail") and "limit" in cap["detail"]
    assert tr.to_dict()["stages"]["depth"]["disparity"]["kind"] == "model_prediction"        # never promoted to validated by itself


def test_lr_consistency_is_the_only_thing_that_may_validate_a_depth_pixel_and_it_is_named():
    diag = PerceptionDiag(source="test", f=F, B=B, cx=CX, cy=CY, sample_every=1000, system_every=0)
    ctl = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=True)
    d = scene(1.2)
    trusted = np.isfinite(d)
    tr, _, _ = one_frame(diag, 6, d, ctl, trusted=trusted)
    dd = tr.to_dict()["stages"]["depth"]
    assert dd["trusted_fraction"]["kind"] == "validated" and dd["trusted_fraction"]["by"] == "left-right consistency"
    tr2, _, _ = one_frame(diag, 7, d, ctl, trusted=None)
    assert "trusted_fraction" not in tr2.to_dict()["stages"]["depth"]


def test_diagnostics_can_be_switched_off_and_then_do_nothing():
    diag = PerceptionDiag(source="test", f=F, B=B, cx=CX, cy=CY, enabled=False)
    ctl = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=True)
    tr, res, said = one_frame(diag, 8, scene(), ctl)
    assert tr.to_dict()["checks"] == [] and said                        # the alert path itself is untouched
