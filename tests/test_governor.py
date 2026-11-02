import json
import os
import tempfile

from gabai.diag import log as dlog
from gabai.governor import Governor


def run(gov, seconds, gpu=None, cpu=None, lanes=None, t0=0.0, step=0.5, moving=True):
    out = None
    t = t0
    while t < t0 + seconds:
        out = gov.update(t, gpu_c=gpu, cpu_c=cpu, lane_ms=lanes or {"depth": 50.0, "grid": 5.0}, user_moving=moving)
        t += step
    return out


def test_cool_machine_runs_everything_at_full_rate():
    g = Governor()
    s = run(g, 6, gpu=60, cpu=50)
    assert s["tier"] == "normal" and s["depth_every"] == 1 and s["detect_every"] == 2 and s["boxer_hz"] == 1.0 and not s["say_reduced_accuracy"]


def test_a_warm_gpu_halves_the_detector_and_boxer_but_never_the_depth_below_7_5_hz():
    g = Governor()
    s = run(g, 6, gpu=75, cpu=50)
    assert s["tier"] == "hot" and s["detect_every"] == 4 and s["boxer_hz"] == 0.3
    assert s["depth_every"] == 2 and s["depth_hz_floor"] == 7.5            # every 2nd frame of 15 FPS is 7.5 Hz: the floor, not below it
    assert s["basalt_every"] == 1                                          # tracking is never slowed


def test_an_overheating_gpu_switches_to_the_cheap_fallback_and_says_reduced_accuracy():
    g = Governor()
    s = run(g, 6, gpu=83, cpu=50)
    assert s["tier"] == "critical" and s["use_fallback_depth"] and s["boxer_hz"] == 0 and s["say_reduced_accuracy"]
    assert s["basalt_every"] == 1 and s["depth_every"] == 1                # the fallback still runs on every frame


def test_a_slow_lane_counts_as_hot_even_when_the_temperatures_are_fine():
    g = Governor()
    s = run(g, 6, gpu=55, cpu=50, lanes={"depth": 130.0, "grid": 5.0})
    assert s["tier"] == "hot" and "lane depth" in s["reason"]


def test_the_governor_decides_only_every_two_seconds():
    g = Governor()
    s0 = g.update(0.0, gpu_c=60, cpu_c=50, lane_ms={}, user_moving=True)
    s1 = g.update(0.5, gpu_c=95, cpu_c=50, lane_ms={}, user_moving=True)    # far too hot, but it is not time yet
    assert s0["tier"] == "normal" and s1["tier"] == "normal"
    s2 = g.update(2.1, gpu_c=95, cpu_c=50, lane_ms={}, user_moving=True)
    assert s2["tier"] == "critical"


def test_it_does_not_flap_around_a_threshold():
    g = Governor()
    assert run(g, 4, gpu=72)["tier"] == "hot"
    assert run(g, 4, gpu=69, t0=4.0)["tier"] == "hot"                       # just under 70 is not cool enough to leave
    assert run(g, 14, gpu=64, t0=8.0)["tier"] == "normal"                   # 6 degrees under, calm for 10 s: back to normal


def test_no_temperature_reading_is_not_treated_as_cool():
    run_dir = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="g", level="INFO")
    g = Governor()
    s = run(g, 14, gpu=None, cpu=None)
    dlog.shutdown()
    assert s["temps_unknown"] is True and s["tier"] == "normal"            # it cannot know, and says so; it does not pretend to be cool
    ev = [json.loads(l) for l in open(os.path.join(run_dir, "gabai.jsonl"), encoding="utf-8")]
    assert any(e["ev"] == "governor.no_temperature" for e in ev)


def test_an_unreadable_sensor_after_a_hot_reading_keeps_the_hot_tier():
    g = Governor()
    assert run(g, 4, gpu=78)["tier"] == "hot"
    s = run(g, 14, gpu=None, t0=4.0)
    assert s["tier"] == "hot"                                              # losing the sensor must not release the limits


def test_background_work_waits_for_a_still_user_and_a_cool_gpu():
    g = Governor()
    assert run(g, 4, gpu=60, moving=True)["background_ok"] is False         # shape reconstruction never runs while walking
    assert run(g, 4, gpu=60, moving=False, t0=4.0)["background_ok"] is True
    assert run(g, 4, gpu=66, moving=False, t0=8.0)["background_ok"] is False


def test_tier_changes_are_logged_with_the_numbers_that_caused_them():
    run_dir = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="g2", level="INFO")
    g = Governor()
    run(g, 4, gpu=60)
    run(g, 4, gpu=76, t0=4.0)
    dlog.shutdown()
    ev = [json.loads(l) for l in open(os.path.join(run_dir, "gabai.jsonl"), encoding="utf-8")]
    t = [e for e in ev if e["ev"] == "governor.tier"]
    assert t and t[-1]["tier"] == "hot" and t[-1]["gpu_c"] == 76 and t[-1]["previous"] == "normal" and "GPU" in t[-1]["reason"]
