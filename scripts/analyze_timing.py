"""Frame timing and left/right synchronization of recordings made with gabai.tools.record_pairs.

    py scripts/analyze_timing.py                       every folder in data/recordings
    py scripts/analyze_timing.py data/recordings/rapid_20261029_235536

Per recording, from pairs.csv: the rate and the gaps between consecutive pairs (a stutter is a gap well above the sender's period), how long the pictures took
from the Pi to the PC compared with the fastest pair (variation of t_recv - t_capture: network and queueing delay, the absolute value is not known because the two
clocks differ), and the left/right capture offset the Pi measured. If the recording holds the Pi's power flags (recordings made after the sender reported them), the
share of pairs taken while the Pi was under-voltage or throttled, and the gaps seen then, are listed too.
"""
import csv
import glob
import json
import os
import sys

import numpy as np


def analyze(folder):
    rows = list(csv.DictReader(open(os.path.join(folder, "pairs.csv"))))
    meta = json.load(open(os.path.join(folder, "meta.json")))
    t = np.array([float(r["t_capture"]) for r in rows])
    rx = np.array([float(r["t_recv"]) for r in rows])
    dt = np.array([float(r["dt_ms"]) if r["dt_ms"] != "" else np.nan for r in rows])
    gap = np.diff(t) * 1000
    delay = (rx - t) * 1000
    delay -= delay.min()
    out = {"recording": os.path.basename(folder), "pairs": len(t), "seconds": round(float(t[-1] - t[0]), 1), "pairs_per_s": round(float((len(t) - 1) / (t[-1] - t[0])), 2),
           "gap_ms": {"median": round(float(np.median(gap)), 1), "p95": round(float(np.percentile(gap, 95)), 1), "max": round(float(gap.max()), 1)},
           "gaps_over": {f"{x} ms": int((gap > x).sum()) for x in (100, 200, 500)},
           "time_lost_to_gaps_over_150ms_pct": round(float(gap[gap > 150].sum() / gap.sum() * 100), 1),
           "arrival_delay_above_fastest_ms": {"median": round(float(np.median(delay)), 1), "p95": round(float(np.percentile(delay, 95)), 1), "max": round(float(delay.max()), 1)},
           "receiver_missed_pairs": meta.get("pairs_missed_by_the_receiver")}
    if np.isfinite(dt).any():
        out["left_right_offset_ms"] = {"median": round(float(np.nanmedian(dt)), 2), "std": round(float(np.nanstd(dt)), 2), "min": round(float(np.nanmin(dt)), 2), "max": round(float(np.nanmax(dt)), 2)}
    if "throttled" in rows[0]:
        th = np.array([int(float(r["throttled"])) if r["throttled"] != "" else 0 for r in rows])
        now = (th & 0x5) != 0
        out["pi_power"] = {"pairs_with_under_voltage_or_throttle_now_pct": round(float(now.mean() * 100), 1),
                           "gaps_over_150ms_during_power_trouble": int(((gap > 150) & now[1:]).sum()), "gaps_over_150ms_otherwise": int(((gap > 150) & ~now[1:]).sum())}
    else:
        out["pi_power"] = "not in this recording (made before the recorder logged the Pi's power flags)"
    h = meta.get("pi_health_at_end") or {}
    out["pi_health_at_end"] = {k: h.get(k) for k in ("throttled", "temp_c", "load1", "rtt_s") if k in h}
    return out


def main():
    folders = sys.argv[1:] or sorted(glob.glob(os.path.join("data", "recordings", "*")))
    for f in folders:
        if os.path.exists(os.path.join(f, "pairs.csv")):
            print(json.dumps(analyze(f), indent=1))


if __name__ == "__main__":
    main()
