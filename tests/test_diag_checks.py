import numpy as np

from gabai.diag import checks as C
from gabai.diag.trace import ESTIMATED, MEASURED, MODEL, STALE, UNAVAILABLE, VALIDATED, FrameTrace, frame_id, q, validated


def test_frame_id_is_the_pi_capture_time_in_milliseconds_so_both_machines_agree():
    assert frame_id(17785.7123) == 17785712
    assert frame_id(None) is None


def test_a_model_output_is_never_called_validated_without_naming_the_independent_check():
    d = q(1.2, MODEL, unit="m", source="las2-m")
    assert d["kind"] == MODEL
    try:
        validated(d, by="")
        raise AssertionError("must refuse")
    except ValueError:
        pass
    v = validated(d, by="left-right consistency")
    assert v["kind"] == VALIDATED and v["by"] == "left-right consistency" and d["kind"] == MODEL


def test_unavailable_data_has_no_value_not_zero():
    u = q(None, MEASURED, source="imu")
    assert u["kind"] == UNAVAILABLE and u["v"] is None


def test_stale_data_is_labelled_with_its_age():
    s = C.freshness(q(5.0, MEASURED), age_s=0.8, limit_s=0.3)
    assert s["kind"] == STALE and s["age_s"] == 0.8
    assert C.freshness(q(5.0, MEASURED), age_s=0.1, limit_s=0.3)["kind"] == MEASURED


def test_trace_records_stage_timing_checks_and_the_decision():
    t = FrameTrace(fid=123, t_capture=1.0)
    with t.stage("depth") as s:
        s["model"] = "las2-m"
    t.add_check(C.check_frame_age(0.5))
    t.decision("alert", outcome="spoken", reasons=["nearest band ahead 1.2 m"], accepted=True)
    d = t.to_dict()
    assert d["fid"] == 123 and d["stages"]["depth"]["model"] == "las2-m" and d["stages"]["depth"]["dur_ms"] >= 0
    assert d["checks"][0]["name"] == "frame_age" and d["checks"][0]["status"] == "fail"
    assert d["decisions"][0]["accepted"] is True and "failed_checks" in d


def test_frame_age_check():
    assert C.check_frame_age(0.1).status == "pass"
    assert C.check_frame_age(0.5).status == "fail"
    assert C.check_frame_age(None).status == "unknown"


def test_pair_sync_check_uses_the_turn_rate_to_say_what_the_offset_costs_in_pixels():
    ok = C.check_pair_sync(dt_ms=5.0, omega_dps=10.0, f_px=881.0)       # 881 * 0.1745 rad/s * 0.005 s = 0.77 px... above 0.5
    assert ok.status in ("suspect", "pass")
    bad = C.check_pair_sync(dt_ms=25.0, omega_dps=100.0, f_px=881.0)
    assert bad.status == "fail" and bad.value > 30
    assert C.check_pair_sync(dt_ms=5.0, omega_dps=0.0, f_px=881.0).status == "pass"
    assert C.check_pair_sync(dt_ms=None, omega_dps=0.0, f_px=881.0).status == "unknown"


def test_image_check_finds_black_dark_flat_and_blurred_frames():
    rng = np.random.default_rng(0)
    good = rng.integers(40, 220, (480, 640), dtype=np.uint8)
    assert C.check_image(good).status == "pass"
    assert C.check_image(np.zeros((480, 640), np.uint8)).status == "fail"
    assert C.check_image(np.full((480, 640), 25, np.uint8)).status in ("suspect", "fail")
    import cv2
    blurred = cv2.GaussianBlur(good, (0, 0), 12)
    r = C.check_image(blurred)
    assert r.status in ("suspect", "fail") and "blur" in r.detail.lower() + " " + str(r.value)
    assert C.check_image(None).status == "fail"


def test_disparity_cap_check_flags_pixels_piled_up_at_the_model_limit():
    d = np.full((100, 100), 40.0, np.float32)
    assert C.check_disparity_cap(d, cap=192).status == "pass"
    d[:20] = 190.0
    r = C.check_disparity_cap(d, cap=192)
    assert r.status in ("suspect", "fail") and r.value > 0.15


def test_depth_coverage_and_trust_checks():
    assert C.check_depth_coverage(0.8).status == "pass"
    assert C.check_depth_coverage(0.2).status == "fail"
    assert C.check_trusted_fraction(0.05).status in ("suspect", "fail")
    assert C.check_depth_coverage(None).status == "unknown"


def test_imu_checks_catch_a_low_rate_gaps_and_a_miscalibrated_accelerometer():
    assert C.check_imu_rate(rate_hz=195, max_gap_ms=12).status == "pass"
    assert C.check_imu_rate(rate_hz=60, max_gap_ms=12).status == "fail"
    assert C.check_imu_rate(rate_hz=190, max_gap_ms=400).status == "fail"
    assert C.check_gravity(9.81).status == "pass"
    r = C.check_gravity(9.42)
    assert r.status in ("suspect", "fail") and "accel" in r.hint.lower()


def test_pose_checks_reject_impossible_walking_speeds_and_jumps():
    assert C.check_pose_step(step_m=0.05, dt_s=0.07).status == "pass"
    r = C.check_pose_step(step_m=1.5, dt_s=0.07)
    assert r.status == "fail" and "jump" in r.detail
    assert C.check_pose_step(step_m=None, dt_s=0.07).status == "unknown"


def test_latency_queue_and_resource_checks():
    assert C.check_latency(180).status == "pass"
    assert C.check_latency(900).status == "fail"
    assert C.check_queue(10, 300).status == "pass" and C.check_queue(295, 300).status == "fail"
    assert C.check_resources(cpu_pct=30, ram_pct=50, gpu_temp_c=60).status == "pass"
    assert C.check_resources(cpu_pct=99, ram_pct=50, gpu_temp_c=60).status in ("suspect", "fail")
    assert C.check_resources(cpu_pct=None, ram_pct=None, gpu_temp_c=None).status == "unknown"


def test_every_check_serialises_with_a_hint_for_failures():
    chk = C.check_frame_age(0.9)
    d = chk.as_dict()
    assert d["status"] == "fail" and d["hint"] and d["component"] and "limit" in d


def test_failed_checks_are_listed_by_name_in_the_trace_summary():
    t = FrameTrace(fid=1)
    t.add_check(C.check_frame_age(0.9))
    t.add_check(C.check_latency(100))
    assert t.to_dict()["failed_checks"] == ["frame_age"]


def test_grid_agreement_flags_a_zone_obstacle_the_grid_cannot_see_and_a_grid_obstacle_the_zones_missed():
    seen = {"nearest_occupied_m": 1.3, "unknown_fraction": 0.1, "free_fraction": 0.6}
    none = {"nearest_occupied_m": None, "unknown_fraction": 0.1, "free_fraction": 0.9}
    assert C.check_grid_agreement(("ahead", 1.5), seen, 10).status == "pass"
    r = C.check_grid_agreement(("ahead", 1.5), none, 10)
    assert r.status == "fail" and "false alarm" in r.hint
    m = C.check_grid_agreement(None, seen, 10)
    assert m.status == "fail" and "MISSED" in m.hint
    assert C.check_grid_agreement(("ahead", 1.5), {"nearest_occupied_m": 2.4, "unknown_fraction": 0.1, "free_fraction": 0.5}, 10).status == "suspect"
    assert C.check_grid_agreement(("ahead", 1.5), none, 1).status == "unknown"                 # a young grid is not judged
    assert C.check_grid_agreement(("ahead", 1.5), {"nearest_occupied_m": None, "unknown_fraction": 0.9, "free_fraction": 0.0}, 10).status == "suspect"


def test_range_vs_depth_check_reports_how_optimistic_the_spoken_distance_is():
    assert C.check_range_vs_depth(1.69, 1.50).status == "pass"
    r = C.check_range_vs_depth(2.4, 1.9)
    assert r.status == "suspect" and r.value["optimistic_by_m"] == 0.5 and "overstates" in r.hint
    assert C.check_range_vs_depth(1.0, None).status == "unknown"
