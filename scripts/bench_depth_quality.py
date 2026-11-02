"""Before / after comparison of the left/right offset correction (gabai/skewcomp.py) on one recorded stereo sequence.

    py scripts/bench_depth_quality.py                                        data/left.mp4 + data/right.mp4, 60 frames, LAS2-H
    py scripts/bench_depth_quality.py --start 30 --frames 80 --skew-frames 1 --model h

The two USB cameras of the real rig are captured up to ~17 ms apart. A recorded video has no such offset, so it is simulated: the right picture of
frame i is replaced by the one of frame i + skew-frames (1 frame = 33 ms at 30 fps, the worst case, so the effect is easy to see). Three variants:
  clean       the true pair (left i, right i). Its LAS2 result is the reference: there is no ground-truth depth for this video, so every number
              below says how far a variant moved away from the clean result, not how far it is from the true distances.
  original    the offset pair, used as it is (what the pipeline did before)
  corrected   the offset pair after SkewCompensator
Depth level: valid-point coverage, error against clean, share of pixels more than 2 px off, frame-to-frame change.
Map level: the whole pipeline (visual odometry + voxel map) on each variant: map voxels, solid voxels, map-vs-depth alignment.
Writes data/bench/depth_quality/{results.json, depth_compare.png, cloud_compare.png}.
"""
import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from gabai.pipeline import run as R                      # noqa: E402
from gabai.pipeline.config import PipelineConfig          # noqa: E402
from gabai.pipeline.sources import open_source            # noqa: E402
from gabai.relocalize import load_map                     # noqa: E402
from gabai.skewcomp import SkewCompensator                # noqa: E402

TURBO = cv2.COLORMAP_TURBO


def load_frames(a):
    cfg = PipelineConfig.load("config/pipeline.toml", {"source.left": a.left, "source.right": a.right, "source.start": a.start, "source.skip": 1,
                                                       "source.max_frames": a.frames + a.skew_frames, "stereo.model": a.model})
    return list(open_source(cfg)), cfg


def variants(frames, a):
    """(clean, original, corrected) lists of (left, right) pictures and the compensation time per frame in ms."""
    sc, dt_ms = SkewCompensator(), a.skew_frames * 1000.0 / a.fps
    clean, orig, corr, ms = [], [], [], []
    for i in range(len(frames) - a.skew_frames):
        l, r_true, r_late = frames[i].left, frames[i].right, frames[i + a.skew_frames].right
        t0 = time.perf_counter()
        r_fix, _ = sc.apply(l, r_late, frames[i].t, dt_ms)
        ms.append((time.perf_counter() - t0) * 1000)
        clean.append((l, r_true)), orig.append((l, r_late)), corr.append((l, r_fix))
    return clean, orig, corr, ms


def depth_all(fn, pairs):
    out = []
    for l, r in pairs:
        d, ok, _ = fn(l, r)
        out.append((d, ok))
    return out


def level_metrics(res, ref, frames):
    cov, err_all, flick = [], [], []
    for (d, ok), (d0, ok0) in zip(res, ref):
        cov.append(float(ok.mean()))
        m = ok & ok0
        err_all.append(np.abs(d - d0)[m])
    e = np.concatenate(err_all)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for i in range(len(res) - 1):                                      # change of one pixel's disparity from frame i to i + 1, following the picture motion
        g0, g1 = (cv2.cvtColor(frames[j].left, cv2.COLOR_BGR2GRAY) for j in (i, i + 1))
        f = dis.calc(g1, g0, None)
        h, w = g0.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        d_prev = cv2.remap(np.nan_to_num(res[i][0]), gx + f[..., 0], gy + f[..., 1], cv2.INTER_NEAREST)
        ok_prev = cv2.remap(res[i][1].astype(np.uint8), gx + f[..., 0], gy + f[..., 1], cv2.INTER_NEAREST).astype(bool)
        m = ok_prev & res[i + 1][1]
        flick.append(np.abs(res[i + 1][0] - d_prev)[m])
    fl = np.concatenate(flick)
    return {"valid_point_coverage_pct": round(float(np.mean(cov)) * 100, 2), "median_abs_error_vs_clean_px": round(float(np.median(e)), 3),
            "pixels_over_2px_off_pct": round(float((e > 2).mean() * 100), 2), "frame_to_frame_disparity_change_px": round(float(np.median(fl)), 3)}


def map_run(fn, cfg, source_frames, name, out_dir, a):
    c = PipelineConfig.load("config/pipeline.toml", {"source.max_frames": 0, "stereo.model": a.model, "map.save": os.path.join(out_dir, f"map_{name}.npz"),
                                                    "hands.enabled": False, "detector.enabled": False, "viz.viewer": False, "pipeline.mode": "stereo_only"})
    real = R.open_source
    R.open_source = lambda _cfg: iter(source_frames)
    try:
        s = R.run(c, depth_fn=fn, log_dir=tempfile.mkdtemp(), quiet=True)
    finally:
        R.open_source = real
    keep = ("frames", "poses_ok", "map_frames", "map_voxels", "map_solid_voxels", "map_obs_per_voxel_median", "map_alignment_median_cm", "path_m", "depth_ms_median", "fps_steady")
    return {k: s[k] for k in keep}


def with_pairs(frames, pairs):
    return [dataclasses.replace(fr, left=l, right=r) for fr, (l, r) in zip(frames, pairs)]


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    return cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def depth_img(d, ok, top):
    v = cv2.applyColorMap(np.clip(np.nan_to_num(d) / top * 255, 0, 255).astype(np.uint8), TURBO)
    v[~ok] = 0
    return v


def depth_figure(frames, ref, res3, names, idxs, path):
    rows = []
    for i in idxs:
        valid = ref[i][1]
        top = float(np.percentile(ref[i][0][valid], 98))
        cols = [label(frames[i].left.copy(), f"left picture, frame {i}")]
        for (d, ok), n in zip([ref[i]] + [r[i] for r in res3], names):
            cols.append(label(depth_img(d, ok, top), f"{n}: {ok.mean() * 100:.0f}% valid"))
        rows.append(np.hstack(cols))
    cv2.imwrite(path, np.vstack(rows))


def project(points, view, size=(520, 400), span=None):
    """Orthographic view of a point set: 'top' looks down the y axis, 'side' along x. Colour = distance along the camera axis."""
    img = np.zeros((size[1], size[0], 3), np.uint8)
    if len(points) == 0:
        return img
    p = np.asarray(points, np.float32)
    a, b = (p[:, 0], p[:, 2]) if view == "top" else (p[:, 2], -p[:, 1])
    lo, hi = span if span is not None else ((a.min(), b.min()), (a.max(), b.max()))
    ux = ((a - lo[0]) / max(hi[0] - lo[0], 1e-6) * (size[0] - 1)).astype(int)
    uy = ((b - lo[1]) / max(hi[1] - lo[1], 1e-6) * (size[1] - 1)).astype(int)
    ok = (ux >= 0) & (ux < size[0]) & (uy >= 0) & (uy < size[1])
    z = p[:, 2] if view == "top" else p[:, 0]
    zc = np.clip((z - np.percentile(z, 2)) / max(np.percentile(z, 98) - np.percentile(z, 2), 1e-6), 0, 1)
    col = cv2.applyColorMap((zc * 255).astype(np.uint8)[:, None], TURBO)[:, 0]
    img[uy[ok], ux[ok]] = col[ok]
    return cv2.dilate(img, np.ones((2, 2), np.uint8))


def cloud_figure(maps, names, path):
    pts = [m.points(min_obs=2) for m in maps]
    allp = np.concatenate([p for p in pts if len(p)])
    lo = np.percentile(allp, 1, axis=0)
    hi = np.percentile(allp, 99, axis=0)
    spans = {"top": ((lo[0], lo[2]), (hi[0], hi[2])), "side": ((lo[2], -hi[1]), (hi[2], -lo[1]))}
    cols = []
    for p, n in zip(pts, names):
        col = [label(project(p, v, span=spans[v]), f"{n}: {len(p)} voxels ({v} view)") for v in ("top", "side")]
        cols.append(np.vstack(col))
    cv2.imwrite(path, np.hstack(cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="data/left.mp4"), ap.add_argument("--right", default="data/right.mp4")
    ap.add_argument("--start", type=float, default=30.0, help="seconds into the video (30 s has fast hand-held motion)")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--skew-frames", type=int, default=1, help="simulated left/right offset in video frames")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--model", default="h", choices=["s", "m", "h"])
    ap.add_argument("--no-map", action="store_true", help="depth-level numbers only")
    ap.add_argument("--out", default=os.path.join("data", "bench", "depth_quality"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames, cfg = load_frames(a)
    n = len(frames) - a.skew_frames
    clean, orig, corr, comp_ms = variants(frames, a)
    import torch
    fn = R.default_depth(a.model, cfg.stereo.speckle_px, cfg.stereo.speckle_diff)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    ref, res_o, res_c = (depth_all(fn, p) for p in (clean, orig, corr))
    wall = time.perf_counter() - t0
    names = ["clean (reference)", "original", "corrected"]
    out = {"sequence": {"left": a.left, "start_s": a.start, "frames": n, "simulated_offset_ms": round(a.skew_frames * 1000 / a.fps, 1), "model": f"LAS2-{a.model.upper()}"},
           "note": "errors are measured against the clean pair's own LAS2 result: the video has no ground-truth depth",
           "depth": {"original": level_metrics(res_o, ref, frames), "corrected": level_metrics(res_c, ref, frames)},
           "cost": {"correction_ms_per_frame_median": round(float(np.median(comp_ms)), 2), "depth_model_ms_per_pair": round(wall / (3 * n) * 1000, 1),
                    "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None}}
    shifts = [float(np.abs(a_.astype(np.float32) - b_.astype(np.float32)).mean()) for (_, a_), (_, b_) in zip(orig, corr)]
    out["cost"]["picture_change_mean_grey_levels"] = round(float(np.mean(shifts)), 2)
    err = [np.abs(res_o[i][0] - ref[i][0])[res_o[i][1] & ref[i][1]].mean() for i in range(n)]
    worst = int(np.argmax(err))
    depth_figure(frames, ref, (res_o, res_c), names, [n // 2, worst], os.path.join(a.out, "depth_compare.png"))
    if not a.no_map:
        maps = []
        out["map"] = {}
        for name, pairs in (("clean", clean), ("original", orig), ("corrected", corr)):
            out["map"][name] = map_run(fn, cfg, with_pairs(frames[:n], pairs), name, a.out, a)
            maps.append(load_map(os.path.join(a.out, f"map_{name}.npz"))[0])
        cloud_figure(maps, names, os.path.join(a.out, "cloud_compare.png"))
    json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
