"""Graphs for the Phase M report. Every function takes the run DataFrame and an output path, draws only what was
measured, and returns the path (or None when the needed column is missing or empty)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gabai.report.metrics import alert_episodes, parse_throttled, pct

SPEC_LATENCY_MS = 400.0          # the spec's target, drawn as a target line, never as a result


def _col(df, name):
    if name not in df.columns:
        return None
    x = pd.to_numeric(df[name], errors="coerce").to_numpy(float)
    return x if np.isfinite(x).any() else None


def _time(df):
    t = _col(df, "t")
    return (t - np.nanmin(t)) if t is not None else np.arange(len(df), dtype=float)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def latency(df, path):
    lat = _col(df, "latency_ms")
    if lat is None:
        return None
    t = _time(df)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [2, 1]})
    a.plot(t, lat, lw=1)
    a.axhline(SPEC_LATENCY_MS, color="tab:red", ls="--", lw=1, label=f"spec target {SPEC_LATENCY_MS:.0f} ms (a target, not a result)")
    a.set(xlabel="run time (s)", ylabel="capture to PC receipt (ms)", title="Link latency over the run")
    a.legend(fontsize=8)
    p50, p95 = pct(lat, 50), pct(lat, 95)
    b.hist(lat[np.isfinite(lat)], bins=30, color="tab:blue")
    for v, n in ((p50, "p50"), (p95, "p95")):
        b.axvline(v, color="k", ls=":")
        b.text(v, b.get_ylim()[1] * 0.9, f" {n} {v:.0f}", fontsize=8)
    b.set(xlabel="ms", title="Distribution")
    return _save(fig, path)


def speech(df, path):
    s = _col(df, "speech_ms")
    if s is None or np.isfinite(s).sum() < 1:
        return None
    s = s[np.isfinite(s)]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(s, bins=min(25, max(5, len(s))), color="tab:green")
    ax.axvline(SPEC_LATENCY_MS, color="tab:red", ls="--", lw=1, label="spec target 400 ms (a target, not a result)")
    ax.axvline(pct(s, 50), color="k", ls=":", label=f"p50 {pct(s, 50):.0f} ms")
    ax.set(xlabel="capture to clip handed to the sound card (ms)", ylabel="alerts", title=f"Capture to speech, {len(s)} alerts")
    ax.legend(fontsize=8)
    return _save(fig, path)


def pipeline(df, path):
    fps, model, loop = _col(df, "fps_in"), _col(df, "model_ms"), _col(df, "loop_ms")
    if fps is None and model is None and loop is None:
        return None
    t = _time(df)
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True)
    if fps is not None:
        a.plot(t, fps, color="tab:green", lw=1)
    a.set(ylabel="frames in (FPS)", title="Throughput")
    if model is not None:
        b.plot(t, model, lw=1, label="depth model")
    if loop is not None:
        b.plot(t, loop, lw=1, label="whole loop")
    b.set(xlabel="run time (s)", ylabel="ms per frame")
    b.legend(fontsize=8)
    return _save(fig, path)


def health(df, path):
    temp, gpu, mem = _col(df, "pi_temp"), _col(df, "gpu_temp"), _col(df, "mem_mb")
    if temp is None and gpu is None and mem is None:
        return None
    t = _time(df)
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True)
    if temp is not None:
        a.plot(t, temp, label="Pi temperature", color="tab:orange")
    if gpu is not None:
        a.plot(t, gpu, label="PC GPU temperature", color="tab:purple")
    a.set(ylabel="deg C", title="Temperatures and Pi supply")
    a.legend(fontsize=8)
    if "throttled" in df.columns:
        flags = [parse_throttled(v) for v in df["throttled"]]
        under = np.array([bool(f and f["under_voltage_now"]) for f in flags])
        if under.any():
            for tt in t[under]:
                a.axvspan(tt, tt + 0.1, color="tab:red", alpha=0.25, lw=0)
            a.text(0.01, 0.92, "red = Pi reports under-voltage right now", transform=a.transAxes, color="tab:red", fontsize=8)
    if mem is not None:
        b.plot(t, mem, color="tab:blue")
    b.set(xlabel="run time (s)", ylabel="Pi free memory (MB)")
    return _save(fig, path)


def zones(df, path):
    cols = [c for c in ("left", "ahead", "right") if _col(df, c) is not None]
    if not cols:
        return None
    t = _time(df)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for c, colr in zip(("left", "ahead", "right"), ("tab:blue", "tab:red", "tab:green")):
        if c in cols:
            ax.plot(t, _col(df, c), lw=1, color=colr, label=f"nearest {c}")
    if "alert" in df.columns:
        for e in alert_episodes(df["alert"].tolist()):
            t0, t1 = t[e["start"]], t[min(e["start"] + e["frames"], len(t) - 1)]
            ax.axvspan(t0, t1, color="gold" if e["key"] in ("left", "ahead", "right") else "tab:gray", alpha=0.3, lw=0)
    ax.set(xlabel="run time (s)", ylabel="nearest valid distance (m)",
           title="Nearest distance per zone (yellow = obstacle alert, grey = drop / can't-see alert)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def accuracy(targets, ranges, path):
    """targets: rows from target_test.load_targets; ranges: (true_m, est_m, p10, p90) tuples from range_test. Error vs true distance."""
    pts = []
    for r in targets or []:
        if r["est_m"] is not None:
            pts.append((r["label"], r["true_m"], (r["est_m"] - r["true_m"]) / r["true_m"] * 100))
    for t, e, _, _ in ranges or []:
        pts.append(("checkerboard", t, (e - t) / t * 100))
    nodepth = [r for r in targets or [] if r["est_m"] is None]
    if not pts and not nodepth:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = sorted({p[0] for p in pts} | {r["label"] for r in nodepth})
    for lab in labels:
        xs = [p[1] for p in pts if p[0] == lab]
        ys = [p[2] for p in pts if p[0] == lab]
        if xs:
            ax.scatter(xs, ys, label=lab, s=28)
    for r in nodepth:
        ax.scatter([r["true_m"]], [0], marker="x", color="k", s=50)
    if nodepth:
        ax.scatter([], [], marker="x", color="k", label="no depth at all (plotted on 0 line)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(xlabel="true distance (m, tape measure)", ylabel="depth error (%)", title="Depth error vs real distance")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, path)
