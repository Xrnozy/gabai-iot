"""Replay a recording made with gabai.tools.record_pairs through the depth pipeline, once per setting, and compare depth noise, missing regions and point cloud stability.

    py scripts/evaluate_recording.py data/recordings/rapid_20261030_101500
    py scripts/evaluate_recording.py data/recordings/rapid_... --model h --variants before,layout,layout+offset

Variants (every one replays the SAME recorded pairs, with the recorded capture times and left/right offsets):
    before                  pictures as received, the calibration used as it is, no offset correction (what the pipeline did before)
    layout                  + the camera-layout check: if the pictures only line up after being turned 180 deg and swapped, they are (source.auto_layout)
    layout+offset           + the left/right offset correction
    layout+offset+scale1.5  + the model run on pictures enlarged 1.5x (slower)
Each frame is put in a motion class from how fast the picture moves (median optical flow between consecutive left pictures, px/s; 100 px/s is about 9 deg/s at
this focal length): still (< 30), normal (30-200), rapid (> 200). Reported per class: valid depth (share of pixels kept), the cause of the rest, depth noise (how much a
pixel's disparity changes from one frame to the next after following the picture motion; px) and, over the whole recording, the 3D map: voxels, voxels seen 3+ times, median
distance between the map and each new frame's depth (cm), path length. There is no ground-truth depth for a recording: noise and stability are measured, accuracy is not.
Writes <recording>/evaluation/{results.json, depth_compare.png, cloud_compare.png}.
"""
import argparse
import csv
import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from gabai import depthq as Q                              # noqa: E402
from gabai.pipeline import run as R                         # noqa: E402
from gabai.pipeline.config import PipelineConfig             # noqa: E402
from gabai.relocalize import load_map                        # noqa: E402

VARIANTS = {"before": {"source.auto_layout": False, "source.skew_comp": False}, "layout": {"source.auto_layout": True, "source.skew_comp": False},
            "layout+offset": {"source.auto_layout": True, "source.skew_comp": True}, "layout+offset+scale1.5": {"source.auto_layout": True, "source.skew_comp": True, "stereo.scale": 1.5}}
CLASSES = (("still", 0, 30), ("normal", 30, 200), ("rapid", 200, 1e9))


def picture_speed(folder):
    """px/s of picture motion at every pair (0 for the first), from the left video and the recorded capture times."""
    t = [float(r["t_capture"]) for r in csv.DictReader(open(os.path.join(folder, "pairs.csv")))]
    cap = cv2.VideoCapture(os.path.join(folder, "left.avi"))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    prev, out = None, []
    for i in range(len(t)):
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (320, 180))
        out.append(0.0 if prev is None else float(np.median(np.hypot(*np.moveaxis(dis.calc(prev, g, None), -1, 0))) * 2.0 / max(t[i] - t[i - 1], 1e-3)))
        prev = g
    return np.array(out)


class Probe:
    """Wraps the depth function: per frame the valid share, the causes, the frame-to-frame disparity change, and the disparity of chosen frames."""

    def __init__(self, base, keep):
        self.base, self.keep, self.rows, self.shots, self.prev, self.n = base, set(keep), [], {}, None, 0
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.reasons = None

    def __call__(self, left, right):
        d, ok, ms = self.base(left, right)
        self.reasons = self.base.reasons
        why = Q.reason_fractions(self.reasons)
        change = float("nan")
        g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        if self.prev is not None:
            pg, pd, pok = self.prev
            f = self.flow.calc(g, pg, None)
            h, w = g.shape
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            d_prev = cv2.remap(np.nan_to_num(pd), gx + f[..., 0], gy + f[..., 1], cv2.INTER_NEAREST)
            ok_prev = cv2.remap(pok.astype(np.uint8), gx + f[..., 0], gy + f[..., 1], cv2.INTER_NEAREST).astype(bool)
            m = ok_prev & ok
            if m.sum() > 500:
                change = float(np.median(np.abs(d - d_prev)[m]))
        self.prev = (g, d.copy(), ok.copy())
        self.rows.append({"valid": float(ok.mean()), "change": change, **{k: v for k, v in why.items()}, "ms": ms})
        if self.n in self.keep:
            self.shots[self.n] = (left.copy(), d.copy(), ok.copy())
        self.n += 1
        return d, ok, ms


def run_variant(folder, name, overrides, model, keep, out, max_frames):
    base_over = {"pipeline.mode": "stereo_only", "source.kind": "replay", "source.folder": folder, "source.max_frames": max_frames, "stereo.model": model,
                 "hands.enabled": False, "detector.enabled": False, "viz.viewer": False, "map.save": os.path.join(out, f"map_{name.replace('+', '_')}.npz"),
                 "perf.drop_when_busy": False, "perf.prefetch": False}
    cfg = PipelineConfig.load("config/pipeline.toml", {**base_over, **overrides})
    probe = Probe(R.default_depth(model, cfg.stereo.speckle_px, cfg.stereo.speckle_diff, cfg.stereo.mono, cfg.stereo.mono_mode, cfg.stereo.scale), keep)
    summary = R.run(cfg, depth_fn=probe, log_dir=tempfile.mkdtemp(), quiet=True)
    return probe, summary


def by_class(probe, speed):
    out = {}
    n = min(len(probe.rows), len(speed))
    for cname, lo, hi in CLASSES:
        idx = [i for i in range(n) if lo <= speed[i] < hi]
        if not idx:
            continue
        rows = [probe.rows[i] for i in idx]
        ch = [r["change"] for r in rows if np.isfinite(r["change"])]
        out[cname] = {"frames": len(idx), "valid_pct": round(float(np.mean([r["valid"] for r in rows])) * 100, 1),
                      "disagree_pct": round(float(np.mean([r["left/right disagree"] for r in rows])) * 100, 1), "no_texture_pct": round(float(np.mean([r["no texture"] for r in rows])) * 100, 1),
                      "outside_view_pct": round(float(np.mean([r["outside the right view"] for r in rows])) * 100, 1),
                      "depth_noise_px_frame_to_frame": None if not ch else round(float(np.median(ch)), 3)}
    return out


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    return cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="h", choices=["s", "m", "h"])
    ap.add_argument("--variants", default="before,layout,layout+offset")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = the whole recording")
    a = ap.parse_args()
    out = os.path.join(a.folder, "evaluation")
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(a.folder, "meta.json")))
    speed = picture_speed(a.folder)
    n_total = len(speed) if not a.max_frames else min(a.max_frames, len(speed))
    speed = speed[:n_total]
    classes = {c: int(((speed >= lo) & (speed < hi)).sum()) for c, lo, hi in CLASSES}
    print(f"{a.folder}: {n_total} pairs, {meta['pairs_per_second']} pairs/s, offset between the cameras {meta['left_right_offset_ms']} ms; frames by motion {classes}", flush=True)
    if meta["left_right_offset_ms"] is None:
        print("WARNING: this recording has no left/right offsets (old sender): the offset correction cannot do anything", flush=True)
    keep = []
    if speed.max() > 0:
        keep = [int(np.argmax(speed)), int(np.argsort(np.abs(speed - np.median(speed)))[0])]
    results, probes, maps = {"recording": a.folder, "pairs": n_total, "frames_by_motion": classes, "model": f"LAS2-{a.model.upper()}",
                             "note": "no ground-truth depth: noise and stability are measured, accuracy is not"}, {}, []
    results["variants"] = {}
    names = [v for v in a.variants.split(",")]
    for name in names:
        probe, s = run_variant(a.folder, name, VARIANTS[name], a.model, keep, out, a.max_frames)
        probes[name] = probe
        maps.append(load_map(os.path.join(out, f"map_{name.replace('+', '_')}.npz"))[0])
        results["variants"][name] = {"by_motion": by_class(probe, speed),
                                     "map": {"voxels": s["map_voxels"], "voxels_seen_3_or_more_times": s["map_solid_voxels"], "obs_per_voxel_median": s["map_obs_per_voxel_median"],
                                             "map_vs_frame_depth_median_cm": None if not s["map_alignment_median_cm"] else s["map_alignment_median_cm"]["median"],
                                             "map_vs_frame_depth_last_cm": None if not s["map_alignment_median_cm"] else s["map_alignment_median_cm"]["last"],
                                             "path_m": s["path_m"], "poses_ok": s["poses_ok"], "poses_held": s["poses_held"], "frames_not_added_inconsistent": s["frames_skipped_inconsistent"],
                                             "map_resets": s["map_resets"]},
                                     "speed": {"depth_ms_median": s["depth_ms_median"], "frame_ms_p95": s["frame_ms_p95"], "fps_steady": s["fps_steady"]}}
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
    if keep:
        rows = []
        for idx in keep:
            shots = [probes[n].shots.get(idx) for n in names]
            if any(x is None for x in shots):
                continue
            top = max(float(np.percentile(x[1][x[2]], 98)) if x[2].any() else 1.0 for x in shots)
            cols = [label(shots[0][0].copy(), f"left picture, pair {idx}, motion {speed[idx]:.0f} px/s")]
            for n, (l, d, ok) in zip(names, shots):
                v = cv2.applyColorMap(np.clip(np.nan_to_num(d) / top * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                v[~ok] = 0
                cols.append(label(v, f"{n}: {ok.mean() * 100:.0f}% valid"))
            rows.append(np.hstack(cols))
        if rows:
            cv2.imwrite(os.path.join(out, "depth_compare.png"), np.vstack(rows))
    pts = [m.points(min_obs=2) for m in maps]
    if all(len(p) for p in pts):
        allp = np.concatenate(pts)
        lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
        from bench_depth_quality import project
        spans = {"top": ((lo[0], lo[2]), (hi[0], hi[2])), "side": ((lo[2], -hi[1]), (hi[2], -lo[1]))}
        cols = [np.vstack([label(project(p, v, span=spans[v]), f"{n}: {len(p)} voxels ({v} view)") for v in ("top", "side")]) for p, n in zip(pts, names)]
        cv2.imwrite(os.path.join(out, "cloud_compare.png"), np.hstack(cols))
    print(json.dumps(results["variants"], indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
