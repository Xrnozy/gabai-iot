"""Numbers for the Phase M report, computed from one run's CSV log (gabai.pc.telemetry.TelemetryLog).

A column that is missing or empty is reported as None ("not measured"), never as zero.
"""
import csv
import os

import numpy as np
import pandas as pd

NEAR_LIMIT_M = 0.26          # LAS2 cannot match closer than f*B/192 = 0.23 m on this rig: readings at the floor of the range are suspect

THROTTLE_BITS = {"under_voltage_now": 0x1, "freq_capped_now": 0x2, "throttled_now": 0x4, "temp_limit_now": 0x8,
                 "under_voltage_ever": 0x10000, "throttled_ever": 0x40000}


def pct(values, q):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def parse_throttled(v):
    """`vcgencmd get_throttled` value (hex string or int) -> flags, or None when it was not reported."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return None
    x = int(str(v), 16) if isinstance(v, str) else int(v)
    return {k: bool(x & bit) for k, bit in THROTTLE_BITS.items()}


def alert_episodes(alerts):
    """One entry per run of the same alert key: [{key, start (frame index), frames}]. Empty/NaN means no alert."""
    out, cur = [], None
    for i, a in enumerate(alerts):
        key = "" if (a is None or (isinstance(a, float) and np.isnan(a))) else str(a)
        if key and cur is not None and cur["key"] == key:
            cur["frames"] += 1
        elif key:
            cur = {"key": key, "start": i, "frames": 1}
            out.append(cur)
        else:
            cur = None
    return out


def load_run(path):
    return pd.read_csv(path, dtype={"throttled": str, "alert": str})


def load_marks(path):
    """Supervisor key presses during the run -> {kind: count}, or None when no marks file exists."""
    if not os.path.exists(path):
        return None
    counts = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    return counts


def _stat(df, col):
    if col not in df.columns:
        return None
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    return {"n": int(x.size), "p50": pct(x, 50), "p95": pct(x, 95), "max": float(x.max()), "min": float(x.min())}


def summarize(df, marks=None):
    s = {"frames": int(len(df))}
    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(float) if "t" in df.columns else np.array([])
    s["duration_s"] = float(np.nanmax(t) - np.nanmin(t)) if t.size > 1 else None
    for name, col in (("fps_in", "fps_in"), ("latency_ms", "latency_ms"), ("model_ms", "model_ms"), ("loop_ms", "loop_ms"),
                      ("speech_ms", "speech_ms"), ("valid_pct", "valid_pct"), ("rtt_ms", "rtt_ms"),
                      ("pair_dt_ms", "pair_dt_ms"), ("pi_load", "pi_load")):
        s[name] = _stat(df, col)
    temp, gpu, mem = _stat(df, "pi_temp"), _stat(df, "gpu_temp"), _stat(df, "mem_mb")
    s["pi_temp_max"] = temp["max"] if temp else None
    s["gpu_temp_max"] = gpu["max"] if gpu else None
    s["mem_min_mb"] = mem["min"] if mem else None
    alerts = df["alert"].tolist() if "alert" in df.columns else []
    eps = alert_episodes(alerts)
    s["alerts"] = {}
    for e in eps:
        s["alerts"][e["key"]] = s["alerts"].get(e["key"], 0) + 1
    n = max(len(alerts), 1)
    s["cant_see_frame_pct"] = 100.0 * sum(e["frames"] for e in eps if e["key"] == "cant_see_ahead") / n
    s["drop_frame_pct"] = 100.0 * sum(e["frames"] for e in eps if e["key"] == "drop_ahead") / n
    flags = [parse_throttled(v) for v in df["throttled"]] if "throttled" in df.columns else []
    seen = [f for f in flags if f is not None]
    s["under_voltage_now_pct"] = 100.0 * sum(f["under_voltage_now"] for f in seen) / len(seen) if seen else None
    s["throttled_now_pct"] = 100.0 * sum(f["throttled_now"] for f in seen) / len(seen) if seen else None
    s["under_voltage_ever"] = any(f["under_voltage_ever"] for f in seen) if seen else None
    if t.size > 1:
        gaps = np.diff(t)
        s["max_gap_s"] = float(np.nanmax(gaps))
        s["stalls_over_500ms"] = int((gaps > 0.5).sum())
    else:
        s["max_gap_s"], s["stalls_over_500ms"] = None, None
    spoken = df["spoken"].fillna("").astype(str) if "spoken" in df.columns else None
    s["tracking_delayed_spoken"] = int(spoken.str.contains("tracking delayed").sum()) if spoken is not None else None
    zone = [pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in ("left", "ahead", "right") if c in df.columns]
    if zone:
        z = np.vstack(zone)
        with np.errstate(all="ignore"):
            nearest = np.nanmin(np.where(np.isfinite(z), z, np.inf), axis=0)
        known = np.isfinite(nearest)
        s["near_limit_frame_pct"] = 100.0 * float((nearest[known] < NEAR_LIMIT_M).mean()) if known.any() else None
    else:
        s["near_limit_frame_pct"] = None
    s["marks"] = marks
    return s


def speech_latency_ms(now, t_recv, lat_s):
    """Capture to 'clip handed to the sound card', in ms: the link latency at receipt (Pi capture to PC receipt, from the
    measured clock offset) plus the time the frame then spent on the PC. None when the clock offset is not known yet."""
    if lat_s is None or t_recv is None:
        return None
    return round(((now - t_recv) + lat_s) * 1000)
