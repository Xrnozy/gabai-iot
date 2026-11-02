"""Read a diagnostic log and say what the evidence shows: which checks failed, how often, where the time went, what the link and the
Pi reported, what SLAM found. Every finding is marked

    confirmed   a number measured in these logs broke a stated limit (the claim is a fact about the run), or
    possible    a symptom that points at a cause the log cannot prove

and lists the evidence, the possible causes, and the next thing to look at. The report never says the depth or the position is
correct: the logs contain no ground truth. Per-frame failure rates use only the regular sample of frames (every Nth frame), never
the extra "interesting" frames that are written on top of it, because those would exaggerate how often things fail.

    py -m gabai.tools.diag_report                       newest run in logs/
    py -m gabai.tools.diag_report logs/viewer_2026...   a specific run folder (or a .jsonl file)
    py -m gabai.tools.diag_report <run> --json out.json --md out.md
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

CRASH_EVENTS = ("viewer.crash", "slam.crash", "pi.fatal")
MIN_SAMPLED = 10
CAVEAT = ("This report shows whether the logged checks passed, not whether the distances or positions are correct: the logs contain "
          "no ground truth. To test accuracy use gabai.tools.target_test (tape-measured targets).")


def load_events(path):
    """All events of a run (rotated files first, oldest to newest). Lines that are not valid JSON are counted, not hidden."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "gabai.jsonl.*")), key=lambda p: -int(p.rsplit(".", 1)[1])) + [os.path.join(path, "gabai.jsonl")]
    else:
        files = [path]
    ev, bad = [], 0
    for f in files:
        if not os.path.exists(f):
            continue
        for line in open(f, encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            try:
                ev.append(json.loads(line))
            except ValueError:
                bad += 1
    if bad:
        ev.append({"ev": "report.bad_lines", "comp": "report", "lvl": "WARNING", "n": bad})
    return ev


def _pct(a, q):
    a = [x for x in a if x is not None]
    return round(float(np.percentile(a, q)), 2) if a else None


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def analyze_events(events):
    rep = {"findings": [], "caveat": CAVEAT, "operator_marks": []}
    by_ev = {}
    for e in events:
        by_ev.setdefault(e.get("ev"), []).append(e)
    levels = Counter(e.get("lvl") for e in events)
    start = (by_ev.get("log.start") or [{}])[0]
    mono = [e["mono"] for e in events if "mono" in e]
    rep["run"] = {"run": (events[0].get("run") if events else None), "events": len(events), "by_level": dict(levels), "duration_s": round(max(mono) - min(mono), 1) if mono else None,
                  "components": dict(Counter(e.get("comp") for e in events)), "log_dir": start.get("run_dir")}
    add = lambda level, code, comp, msg, ev, causes=(), steps=(): rep["findings"].append(
        {"level": level, "code": code, "component": comp, "message": msg, "evidence": ev, "possible_causes": list(causes), "next_steps": list(steps)})

    # -- crashes first ----------------------------------------------------------------------------------------------------------
    for name in CRASH_EVENTS:
        for e in by_ev.get(name, [])[:3]:
            add("confirmed", "crash", e.get("comp", "?"), f"{name}: {e.get('error') or str(e.get('exc', ''))[-160:]}", {"event": name, "traceback_tail": str(e.get("exc", ""))[-600:]},
                steps=["read the traceback above; the fix is in the file and line it names"])

    # -- per-frame traces: the regular sample only ------------------------------------------------------------------------------
    traces = by_ev.get("frame.trace", [])
    sampled = [t for t in traces if t.get("sampled")]
    n = len(sampled)
    pl = {"traces_total": len(traces), "traces_sampled": n}
    if n:
        cnt = {}
        for t in sampled:
            for c in t.get("checks", []):
                cnt.setdefault(c["name"], Counter())[c["status"]] += 1
        pl["checks"] = {k: dict(v) for k, v in cnt.items()}
        pl["failed_in_sample"] = {k: v.get("fail", 0) for k, v in cnt.items() if v.get("fail")}
        pl["timing_ms"] = {"model_p50": _pct([_g(t, "stages", "depth", "model_ms") for t in sampled], 50), "model_p95": _pct([_g(t, "stages", "depth", "model_ms") for t in sampled], 95),
                           "loop_p50": _pct([_g(t, "stages", "timing", "loop_ms") for t in sampled], 50), "loop_p95": _pct([_g(t, "stages", "timing", "loop_ms") for t in sampled], 95),
                           "link_latency_p50": _pct([None if _g(t, "stages", "input", "latency_s", "v") is None else _g(t, "stages", "input", "latency_s", "v") * 1000 for t in sampled], 50)}
    rep["pipeline"] = pl

    def frac(name, *statuses):
        c = cnt.get(name) if n else None
        if not c:
            return None, 0
        known = sum(v for k, v in c.items() if k != "unknown")
        bad = sum(c.get(s, 0) for s in statuses)
        return (bad / known if known else None), known

    def examples(check, *statuses):
        out = []
        for t in sampled:
            for c in t.get("checks", []):
                if c["name"] == check and c["status"] in statuses and len(out) < 4:
                    out.append({"fid": t.get("fid"), "value": c.get("value"), "detail": c.get("detail")})
        return out

    if n >= MIN_SAMPLED:
        f, k = frac("frame_age", "fail")
        if f is not None and f >= 0.05:
            ages = [_g(t, "stages", "input", "age_s", "v") for t in sampled]
            add("confirmed", "stale_frames", "link", f"{f * 100:.0f}% of sampled frames were older than 300 ms when processed",
                {"fraction_failed": round(f, 3), "frames_checked": k, "age_s_p50": _pct(ages, 50), "age_s_max": _pct(ages, 100), "examples": examples("frame_age", "fail"),
                 "link_gaps_logged": len(by_ev.get("link.gap", [])), "pi_send_slow_logged": len(by_ev.get("pi.send_slow", []))},
                ["Wi-Fi stalls or weak signal (see link.gap and pi.send_slow counts)", "the PC loop is slower than the camera rate (see timing_ms)", "Pi under-voltage or overload (see pi_power)"],
                ["compare link.stats rtt_ms and recv_fps over time", "run with --log-level DEBUG on both machines and match fids"])
        f, k = frac("depth_coverage", "fail")
        if f is not None and f >= 0.30:
            imgs = [c for t in sampled for c in t.get("checks", []) if c["name"] == "image"]
            dark = [c for c in imgs if c["status"] != "pass" and any(w in c.get("detail", "") for w in ("dark", "flat", "black", "white", "low contrast"))]
            blur = [c for c in imgs if "blur" in c.get("detail", "")]
            ev = {"fraction_failed": round(f, 3), "frames_checked": k, "images_checked": len(imgs),
                  "dark_or_flat_images_fraction": round(len(dark) / len(imgs), 3) if imgs else None, "blurred_or_textureless_images_fraction": round(len(blur) / len(imgs), 3) if imgs else None,
                  "trusted_fraction_failed": frac("trusted_fraction", "suspect", "fail")[0]}
            causes = []
            if imgs and len(dark) / len(imgs) > 0.5:
                causes.append(f"the picture itself is dark or flat in {len(dark) / len(imgs) * 100:.0f}% of checked frames: more light or more camera gain (the most likely cause here)")
            if ev["trusted_fraction_failed"] and ev["trusted_fraction_failed"] > 0.3:
                causes.append("the left-right check rejects most pixels: blank or glossy surfaces, or left/right images swapped or mis-rectified")
            causes += ["surfaces with no texture (blank walls, glass)", "an object closer than the model's minimum range"]
            add("confirmed", "frequent_cant_see", "depth", f"in {f * 100:.0f}% of sampled frames too little of the floor view had trusted depth, so 'can't see ahead' applies", ev, causes,
                ["look at the saved depth pictures (--diag-save depth)", "py -m gabai.tools.noise_view to see what the model rejects"])
        f, k = frac("disparity_cap", "suspect", "fail")
        if f is not None and f >= 0.10:
            add("possible", "objects_at_model_limit", "depth", f"{f * 100:.0f}% of sampled frames had many pixels at the model's maximum disparity (an object nearer than about 0.23 m, or a wrong match)",
                {"fraction": round(f, 3), "examples": examples("disparity_cap", "suspect", "fail")}, ["something very close to the lens", "wrong matches on repeated texture"],
                ["try --near with --lr-check (reaches 0.12 m)", "check what was in front of the cameras"])
        f, k = frac("trusted_fraction", "fail")
        if f is not None and f >= 0.30 and not any(x["code"] == "frequent_cant_see" for x in rep["findings"]):
            add("possible", "left_right_rejects_most", "depth", f"the left-right check rejected most pixels in {f * 100:.0f}% of sampled frames", {"fraction": round(f, 3), "examples": examples("trusted_fraction", "fail", "suspect")},
                ["left/right cameras swapped or the rectification is wrong", "glare, blank or dark surfaces"], ["py -m gabai.tools.calibrate_stereo (check the SWAPPED warning)"])
        f, k = frac("pair_sync", "fail")
        if f is not None and f >= 0.10:
            add("confirmed", "pair_sync_error", "cameras", f"in {f * 100:.0f}% of sampled frames the left/right capture offset caused more than 1 px of false disparity while turning",
                {"fraction_failed": round(f, 3), "examples": examples("pair_sync", "fail")}, ["fast turning with unsynchronised cameras", "capture rate too low (--low-power uses 20 FPS)"], ["slow down while turning", "see docs/sync-guide.md"])
        else:
            f2, _ = frac("pair_sync", "suspect", "fail")
            if f2 is not None and f2 >= 0.20:
                add("possible", "pair_sync_error", "cameras", f"the left/right offset costs 0.5-1 px of false disparity in {f2 * 100:.0f}% of sampled frames", {"fraction": round(f2, 3)}, ["turning while moving"], [])
        f, k = frac("floor_model", "suspect", "fail")
        if f is not None and f >= 0.5:
            why = Counter(c.get("detail") for t in sampled for c in t.get("checks", []) if c["name"] == "floor_model" and c["status"] != "pass")
            add("confirmed", "floor_model_unreliable", "zones", f"the floor model was not reliable in {f * 100:.0f}% of sampled frames", {"fraction": round(f, 3), "reasons": dict(why)},
                ["no --cam-h (no floor model, obstacle speech is off)", "IMU axes not calibrated: a tilted camera makes the floor and every distance wrong"],
                ["start the viewer with --cam-h <metres>", "run sync_imu_camera and check_extrinsics"])
        f, k = frac("resources", "suspect", "fail")
        if f is not None and f >= 0.2:
            add("possible", "machine_saturated", "system", f"CPU, memory or GPU temperature was high in {f * 100:.0f}% of the readings", {"fraction": round(f, 3), "examples": examples("resources", "suspect", "fail")},
                ["other programs", "GPU heat"], ["see governor.tier events"])
    # latency uses every trace that carries an alert time (alerts are what matter), not just the sample
    sp = [(t, _g(t, "stages", "timing", "speech_ms")) for t in traces]
    sp = [(t, v) for t, v in sp if v is not None]
    if sp:
        v = [x for _, x in sp]
        pl["speech_ms_p50"], pl["speech_ms_p95"], pl["speech_n"] = _pct(v, 50), _pct(v, 95), len(v)
        if _pct(v, 50) > 400 or sum(x > 400 for x in v) / len(v) >= 0.2:
            st = {"model_ms": _pct([_g(t, "stages", "depth", "model_ms") for t, _ in sp], 50), "loop_ms": _pct([_g(t, "stages", "timing", "loop_ms") for t, _ in sp], 50),
                  "link_latency_ms": _pct([None if _g(t, "stages", "input", "latency_s", "v") is None else _g(t, "stages", "input", "latency_s", "v") * 1000 for t, _ in sp], 50)}
            big = max((k for k in st if st[k] is not None), key=lambda k: st[k], default=None)
            add("confirmed", "latency_over_target", "pipeline", f"capture-to-speech median {_pct(v, 50):.0f} ms against the 400 ms target (p95 {_pct(v, 95):.0f} ms, {len(v)} alerts); largest stage: {big}",
                {"speech_ms_p50": _pct(v, 50), "speech_ms_p95": _pct(v, 95), "n_alerts": len(v), "stages_ms_p50": st}, [f"the {big} stage" if big else "unknown"],
                ["see link.stats rtt_ms for the network part", "the target is the spec's, not a measured pass mark"])
    gd = [(t, c) for t in traces for c in t.get("checks", []) if c["name"] == "grid_agreement" and c["status"] == "fail"]
    if gd:
        add("confirmed", "grid_disagrees", "grid", f"the rolling grid and the zone check disagreed in {len(gd)} logged frame(s)",
            {"n": len(gd), "examples": [{"fid": t.get("fid"), "detail": c["detail"], "hint": c["hint"]} for t, c in gd[:4]]},
            ["floor model (camera height or tilt) wrong in one of them", "the grid height limits"], ["open the saved grid picture (--diag-save grid) for those fids"])

    # -- link, Pi and logging events -------------------------------------------------------------------------------------------
    dis = by_ev.get("link.disconnected", [])
    if dis:
        add("confirmed", "link_dropouts", "link", f"{len(dis)} connection(s) from the Pi ended", {"reasons": dict(Counter(e.get("reason") for e in dis)), "frames_before": [e.get("frames") for e in dis[:5]],
                                                                                                    "durations_s": [e.get("duration_s") for e in dis[:5]]},
            ["Wi-Fi drops", "Pi rebooted or lost power", "the Pi sender was stopped"], ["read pi.session_end and pi.link_down in the Pi log for the same time"])
    gaps = by_ev.get("link.gap", [])
    if len(gaps) >= 3:
        add("confirmed", "link_gaps", "link", f"{len(gaps)} gaps longer than 300 ms between frames", {"max_gap_ms": max(e.get("gap_ms", 0) for e in gaps), "n": len(gaps)},
            ["Wi-Fi stalls", "Pi under-voltage", "Pi overloaded"], ["compare with pi.send_slow and pi.power"])
    for code, evn, msg in (("decode_errors", "link.decode_error", "frames arrived that could not be decoded"), ("connection_rejected", "link.rejected", "connections were rejected")):
        es = by_ev.get(evn, [])
        if es:
            add("confirmed", code, "link", f"{len(es)} {msg}", {"first": {k: v for k, v in es[0].items() if k not in ("ts", "mono", "run")}},
                ["corrupt frames from the Pi (power or USB)"] if code == "decode_errors" else ["wrong --token on the Pi sender"], [])
    pw = [e for e in by_ev.get("pi.power", []) if any(str(f).endswith("_now") for f in (e.get("flags") or []))]
    pw += [e for e in by_ev.get("check.fail", []) if e.get("check") == "pi_power"]
    stats_flag = [e for e in by_ev.get("link.stats", []) if "under_voltage_now" in (e.get("pi_flags") or [])]
    if pw or stats_flag:
        add("confirmed", "pi_power", "pi", "the Pi reported under-voltage or throttling while running", {"events": len(pw), "stat_windows_with_flag": len(stats_flag), "first": (pw or [{}])[0].get("throttled") or (pw[0].get("value") if pw else None)},
            ["power bank or cable cannot hold 5 V under load"], ["docs/pi-power-guide.md"])
    for code, evn, msg, causes in (
            ("pi_send_slow", "pi.send_slow", "the Pi's sends were slow", ["network congestion", "PC busy"]),
            ("imu_rate_low", "pi.imu_rate_low", "the IMU rate on the Pi was below 100 Hz", ["Pi overloaded or I2C errors"]),
            ("camera_failures", "cam.grab_failed", "a camera stopped delivering frames", ["USB power or cable", "overload"]),
            ("camera_no_pair", "cam.no_pair", "no left/right pair could be formed for more than 2 s", ["one camera is dead (see newest_frame_age_s)"])):
        es = by_ev.get(evn, [])
        if es:
            add("confirmed", code, "pi", f"{len(es)} {msg}", {"first": {k: v for k, v in es[0].items() if k not in ("ts", "mono", "run", "comp", "lvl", "ev")}}, causes, [])
    drop = by_ev.get("log.dropped", [])
    if drop:
        add("confirmed", "log_incomplete", "log", f"{sum(e.get('count', 0) for e in drop)} log records were lost because the writer fell behind: the evidence has gaps",
            {"lost": sum(e.get("count", 0) for e in drop)}, ["too much DEBUG logging for the disk"], ["lower GABAI_LOG_LEVEL or raise GABAI_LOG_QUEUE"])
    for e in by_ev.get("slam.finding", []):
        add(e.get("level", "possible"), "slam_" + str(e.get("code")), "slam", e.get("message"), e.get("evidence") or {}, e.get("possible_causes") or [], [])
    for e in by_ev.get("slam.gate_waiting", [])[:1]:
        add("confirmed", "slam_never_started", "slam", f"Basalt was still waiting for the rig to be still after {e.get('seconds')} s", {"last_gyro_dps": e.get("last_gyro_dps"), "last_accel_err_ms2": e.get("last_accel_err_ms2")},
            ["the rig was moving", "gyro bias wrong from the Pi start-up"], [e.get("hint", "")])
    for e in by_ev.get("slam.accel_calib_missing", [])[:1] + by_ev.get("slam.imu_sync_missing", [])[:1]:
        add("confirmed", "calibration_missing", "slam", f"{e.get('ev')}: {e.get('hint')}", {"path": e.get("path")}, ["without it Basalt diverges"], [e.get("hint", "")])
    rep["operator_marks"] = [{"kind": e.get("kind"), "fid": e.get("fid"), "alert": e.get("alert"), "evidence_file": e.get("evidence")} for e in by_ev.get("viewer.mark", [])]
    rep["checks_logged"] = dict(Counter(e.get("check") for e in by_ev.get("check.fail", [])))
    order = {"confirmed": 0, "possible": 1}
    rep["findings"].sort(key=lambda f: (f["code"] != "crash", order.get(f["level"], 2)))
    rep["enough_frame_data"] = n >= MIN_SAMPLED
    return rep


def format_report(rep):
    L = ["DIAGNOSTIC REPORT", "=" * 60]
    r = rep["run"]
    L.append(f"run {r['run']}  events {r['events']}  duration {r['duration_s']} s  levels {r['by_level']}")
    p = rep.get("pipeline", {})
    if p.get("traces_sampled"):
        t = p.get("timing_ms", {})
        L.append(f"frames: {p['traces_sampled']} sampled traces ({p['traces_total']} written)  depth model p50/p95 {t.get('model_p50')}/{t.get('model_p95')} ms  "
                 f"loop p50/p95 {t.get('loop_p50')}/{t.get('loop_p95')} ms  link latency p50 {t.get('link_latency_p50')} ms")
        if p.get("speech_n"):
            L.append(f"speech: capture-to-speech p50/p95 {p['speech_ms_p50']}/{p['speech_ms_p95']} ms over {p['speech_n']} alerts (to the sound card, not the sound)")
        if p.get("failed_in_sample"):
            L.append("checks that failed in the sample: " + ", ".join(f"{k} x{v}" for k, v in p["failed_in_sample"].items()))
    if not rep["enough_frame_data"]:
        L.append("NOT ENOUGH per-frame data to judge frame-level checks (need at least 10 sampled frames).")
    L.append("")
    if not rep["findings"]:
        L.append("No problems were found in the data that was logged." if rep["enough_frame_data"] else "No problems found, but there is not enough data to say much.")
    for i, f in enumerate(rep["findings"], 1):
        L.append(f"{i}. [{f['level'].upper()}] {f['code']} ({f['component']}): {f['message']}")
        L.append(f"     evidence: {json.dumps(f['evidence'], default=str)[:600]}")
        for c in f["possible_causes"]:
            L.append(f"     possible cause: {c}")
        for s in f["next_steps"]:
            if s:
                L.append(f"     next: {s}")
    if rep["operator_marks"]:
        L.append("")
        L.append("Operator marks (judgement made while running):")
        for m in rep["operator_marks"]:
            L.append(f"  {m['kind']} at frame {m['fid']} (alert then: {m['alert']}); evidence: {m['evidence_file']}")
    L += ["", rep["caveat"]]
    return "\n".join(L)


def newest_run(log_dir="logs"):
    runs = [d for d in glob.glob(os.path.join(log_dir, "*")) if os.path.exists(os.path.join(d, "gabai.jsonl"))]
    return max(runs, key=os.path.getmtime) if runs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="a run folder or .jsonl file (default: the newest run in logs/)")
    ap.add_argument("--json", help="also write the structured result here")
    ap.add_argument("--md", help="also write the text report here")
    a = ap.parse_args()
    path = a.path or newest_run()
    if not path:
        raise SystemExit("no log found: run a viewer or recorder first, or give the path to a run folder")
    rep = analyze_events(load_events(path))
    text = format_report(rep)
    print(f"log: {path}\n")
    print(text)
    if a.json:
        json.dump(rep, open(a.json, "w", encoding="utf-8"), indent=2, default=str)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
