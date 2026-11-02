import csv
import os
import tempfile

import numpy as np

from gabai.report.metrics import alert_episodes, load_marks, load_run, parse_throttled, pct, summarize


def write_run(rows, path):
    fields = ["t", "fps_in", "latency_ms", "model_ms", "loop_ms", "left", "ahead", "right", "valid_pct", "alert",
              "speech_ms", "pi_temp", "throttled", "mem_mb", "gpu_temp"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def rows(n=100, **over):
    out = []
    for i in range(n):
        r = dict(t=1000 + i / 15, fps_in=14.8, latency_ms=100 + i, model_ms=40, loop_ms=60, left=2.0, ahead=3.0, right=2.5,
                 valid_pct=80, alert="", speech_ms="", pi_temp=50, throttled="0x0", mem_mb=300, gpu_temp=55)
        r.update({k: (v(i) if callable(v) else v) for k, v in over.items()})
        out.append(r)
    return out


def test_percentiles_ignore_blanks_and_nan():
    assert pct([1, 2, 3, 4, 5, float("nan")], 50) == 3
    assert np.isnan(pct([], 50))


def test_throttle_bits_are_decoded_into_now_and_since_boot():
    assert parse_throttled("0x50005") == {"under_voltage_now": True, "freq_capped_now": False, "throttled_now": True,
                                          "temp_limit_now": False, "under_voltage_ever": True, "throttled_ever": True}
    assert parse_throttled("0x0")["under_voltage_ever"] is False
    assert parse_throttled("") is None and parse_throttled(None) is None


def test_alert_episodes_count_each_alert_once_not_once_per_frame():
    alerts = ["", "ahead", "ahead", "ahead", "", "left", "left", "", "", "ahead", "cant_see_ahead", "cant_see_ahead"]
    eps = alert_episodes(alerts)
    assert [e["key"] for e in eps] == ["ahead", "left", "ahead", "cant_see_ahead"]
    assert eps[0]["frames"] == 3 and eps[0]["start"] == 1


def test_summary_reports_latency_percentiles_and_alert_counts():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "run.csv")
        write_run(rows(alert=lambda i: "ahead" if 20 <= i < 30 else ("cant_see_ahead" if 60 <= i < 62 else ""),
                       speech_ms=lambda i: 180 if i == 20 else ("" if i != 60 else 250)), p)
        s = summarize(load_run(p))
    assert s["frames"] == 100
    assert abs(s["latency_ms"]["p50"] - 149.5) < 1 and s["latency_ms"]["p95"] > 190
    assert s["alerts"]["ahead"] == 1 and s["alerts"]["cant_see_ahead"] == 1
    assert s["speech_ms"]["n"] == 2 and s["speech_ms"]["p50"] == 215
    assert abs(s["cant_see_frame_pct"] - 2.0) < 1e-6


def test_missing_columns_are_reported_as_not_measured_not_zero():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "run.csv")
        with open(p, "w", newline="") as f:
            f.write("t,fps_in,loop_ms,alert\n1,15,60,\n2,15,70,\n")
        s = summarize(load_run(p))
    assert s["latency_ms"] is None and s["speech_ms"] is None and s["gpu_temp_max"] is None


def test_supervisor_marks_give_false_and_missed_alarm_counts():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "run.csv.marks.csv")
        with open(p, "w", newline="") as f:
            f.write("t,kind\n1000.5,false_alarm\n1001.0,missed\n1002.0,false_alarm\n")
        m = load_marks(p)
        assert m == {"false_alarm": 2, "missed": 1}
        assert load_marks(os.path.join(d, "none.csv")) is None


def test_throttle_flags_in_a_run_are_counted():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "run.csv")
        write_run(rows(throttled=lambda i: "0x50005" if i >= 50 else "0x0"), p)
        s = summarize(load_run(p))
    assert abs(s["under_voltage_now_pct"] - 50.0) < 1e-6 and s["under_voltage_ever"]


def test_speech_latency_adds_link_latency_and_time_spent_on_the_pc():
    from gabai.report.metrics import speech_latency_ms
    assert speech_latency_ms(now=10.250, t_recv=10.100, lat_s=0.080) == 230
    assert speech_latency_ms(now=10.250, t_recv=10.100, lat_s=None) is None      # clock offset unknown: not guessed


def test_stalls_and_tracking_delayed_announcements_are_counted():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "run.csv")
        rs = rows(100, spoken=lambda i: "tracking delayed" if i in (50, 51) else "")
        for i, r in enumerate(rs):
            if i >= 50:
                r["t"] += 1.2                                  # a 1.2 s hole in the stream before frame 50
        write_run_with_spoken(rs, p)
        s = summarize(load_run(p))
    assert s["stalls_over_500ms"] == 1 and abs(s["max_gap_s"] - (1.2 + 1 / 15)) < 1e-6
    assert s["tracking_delayed_spoken"] == 2


def write_run_with_spoken(rs, path):
    fields = list(rs[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rs)


def test_frames_with_a_zone_at_the_model_limit_are_flagged():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "run.csv")
        write_run(rows(100, ahead=lambda i: 0.24 if i < 40 else 3.0), p)
        s = summarize(load_run(p))
    assert abs(s["near_limit_frame_pct"] - 40.0) < 1e-6
