import json
import os
import tempfile

import numpy as np

from gabai.diag import log as dlog
from gabai.diag.watch import LinkWatcher

G = 9.80665


def events(run):
    dlog.shutdown()
    return [json.loads(l) for l in open(os.path.join(run, "gabai.jsonl"), encoding="utf-8") if l.strip()]


def stats(**kw):
    s = dict(connected=True, frames=0, bytes=0, fps=14.5, frame_age_s=0.05, rejected=0, connections=1, decode_errors=0, overwritten=0,
             imu_overflow=0, mag_overflow=0, inter_arrival_max_ms=80.0, imu_queue=10, mag_queue=0, msg_counts={}, health={}, imu={},
             last_error=None, peer="192.168.1.63")
    s.update(kw)
    return s


def imu_block(t0, seconds, hz=200, gyro_dps=0.0, accel_g=1.0):
    n = int(seconds * hz)
    return [(t0 + i / hz, gyro_dps, 0.0, 0.0, 0.0, 0.0, accel_g) for i in range(n)]


def test_connect_and_disconnect_transitions_are_logged_with_the_last_error():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="w", level="INFO")
    w = LinkWatcher(period_s=100)
    w.update(0.0, stats(connected=False), {})
    w.update(1.0, stats(connected=True, peer="192.168.1.63"), {})
    w.update(2.0, stats(connected=False, last_error="peer closed or reset the connection"), {})
    ev = events(run)
    names = [e["ev"] for e in ev]
    assert "link.up" in names and "link.lost" in names
    lost = [e for e in ev if e["ev"] == "link.lost"][0]
    assert lost["last_error"].startswith("peer closed") and lost["lvl"] == "WARNING"


def test_periodic_stats_report_rates_per_window_not_just_totals():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="w2", level="INFO")
    w = LinkWatcher(period_s=5)
    w.update(0.0, stats(frames=0, bytes=0, overwritten=0), {})
    w.update(5.0, stats(frames=70, bytes=5_000_000, overwritten=20, decode_errors=1), {"temp_c": 51.0, "throttled": "0x0", "rtt_s": 0.06})
    r = [e for e in events(run) if e["ev"] == "link.stats"][0]
    assert abs(r["recv_fps"] - 14.0) < 0.1 and abs(r["mbit_s"] - 8.0) < 0.1
    assert r["overwritten_in_window"] == 20 and r["decode_errors_in_window"] == 1 and r["rtt_ms"] == 60.0


def test_a_live_under_voltage_flag_is_a_warning_with_the_fix():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="w3", level="INFO")
    w = LinkWatcher(period_s=5)
    w.update(0.0, stats(), {"throttled": "0x0"})
    w.update(5.0, stats(), {"throttled": "0x50005", "temp_c": 49.0})
    ev = events(run)
    p = [e for e in ev if e["ev"] == "check.fail" and e["check"] == "pi_power"]
    assert p and "pi-power-guide" in p[0]["hint"]


def test_imu_rate_gaps_and_still_gravity_are_measured_from_the_samples_received():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="w4", level="INFO")
    w = LinkWatcher(period_s=5)
    w.update(0.0, stats(), {})
    w.note_imu(imu_block(0.0, 2.0, accel_g=0.96))                    # a still rig whose accelerometer reads 0.96 g
    w.note_imu(imu_block(2.4, 2.6, accel_g=0.96))                    # a 0.4 s hole in the samples
    w.update(5.0, stats(), {})
    ev = events(run)
    r = [e for e in ev if e["ev"] == "link.stats"][0]
    assert 150 <= r["imu_hz"] <= 210 and r["imu_max_gap_ms"] >= 390
    assert any(e["ev"] == "check.fail" and e["check"] == "imu_rate" for e in ev)         # the gap fails the rate check
    g = [e for e in ev if e["ev"] in ("check.fail", "check.suspect") and e.get("check") == "gravity"]
    assert g and "calibrate_accel" in g[0]["hint"]


def test_a_stale_stream_with_the_link_still_up_is_called_out():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="w5", level="INFO")
    w = LinkWatcher(period_s=5)
    w.update(0.0, stats(), {})
    w.update(5.0, stats(frame_age_s=2.5, fps=0.0), {})
    assert any(e["ev"] == "check.fail" and e["check"] == "frame_age" for e in events(run))
