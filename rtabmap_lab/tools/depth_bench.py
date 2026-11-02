"""Benchmarks every depth source ON ITS OWN, on the same rectified pairs, before any fusion: speed, GPU memory, coverage, how flat its flat things are, and how much the sources agree.

  py -m rtabmap_lab.tools.depth_bench data/recordings/vio_20261101_014854 [--frames 24] [--step 25]

Sources: LAS2 stereo (the main system's, with its left-right check), Fast-FoundationStereo (NVIDIA C-Fast weights, FP16), OpenCV SGBM with RTAB-Map's settings (a stand-in for RTAB-Map's own matcher, which runs
inside the bridge), Depth Anything V2 metric on the left and on the right picture (raw, and after a robust scale fit to LAS2). One model is loaded at a time: loaded, run, measured, freed.
Planarity is measured on ONE depth picture at a time (no poses): RANSAC planes at 2 cm, share of points on a plane and their RMS thickness (rtabmap_lab/flatness.py). Agreement is the share of pixels where
two sources, both valid, differ by less than 5% in depth.
"""
import argparse
import gc
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from rtabmap_lab import flatness
from rtabmap_lab.gabai_depth import points_from_depth


def frames(folder, n, step, config="config/pipeline.toml"):
    from gabai.pipeline.config import PipelineConfig
    from gabai.pipeline.sources import prepare_replay, replay_source
    cfg = PipelineConfig.load(config, {"source.kind": "replay", "source.folder": folder, "source.max_frames": 0, "source.skew_comp": False, "source.flip180": False})
    rect, delay, k = prepare_replay(cfg)
    out = []
    for i, fr in enumerate(replay_source(cfg, rect, delay, k)):
        if i % step == 0:
            out.append(fr)
        if len(out) >= n:
            break
    return out


def run_source(name, make, fn, pairs, cal):
    """Loads one model, runs it on every pair, returns per-pair depth (m, NaN = none), extra info, ms per pair, model MB, peak MB."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    model = make()
    loaded = (torch.cuda.memory_allocated() - base) / 2 ** 20
    fn(model, pairs[0])                                                    # warm-up: the first call pays for GPU set-up
    torch.cuda.reset_peak_memory_stats()
    out, ms = [], []
    for p in pairs:
        t0 = time.perf_counter()
        out.append(fn(model, p))
        ms.append((time.perf_counter() - t0) * 1000)
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out, float(np.median(ms)), loaded, peak


def to_depth(disp, cal, lo=0.3, hi=4.0):
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(disp > 0.5, cal["f"] * cal["B"] / disp, np.nan)
    return np.where((z >= lo) & (z <= hi), z, np.nan).astype(np.float32)


def planarity(z, cal):
    d = np.where(np.isfinite(z), z * 1000.0, 0).astype(np.uint16)
    p, _ = points_from_depth(d, np.zeros(z.shape + (3,), np.uint8), cal["f"], cal["f"], cal["cx"], cal["cy"], 2)
    if len(p) < 3000:
        return None
    r = flatness.report(p, n_planes=2, min_points=800)
    return (r["plane_share"], r["mean_thickness_mm"]) if r["planes"] else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--ffs-lr", action="store_true", help="also give FFS its left-right check (a second inference per pair)")
    ap.add_argument("--config", default="config/pipeline.toml", help="read-only: which calibration the pairs are rectified with")
    a = ap.parse_args(argv)
    fr = frames(a.folder, a.frames, a.step, a.config)
    cal = dict(fr[0].cal)
    pairs = [(f.left[:, :, ::-1].copy(), f.right[:, :, ::-1].copy()) for f in fr]          # RGB
    print(f"{len(pairs)} rectified pairs {pairs[0][0].shape[1]}x{pairs[0][0].shape[0]}, f={cal['f']:.1f} px, baseline {cal['B'] * 1000:.1f} mm\n", flush=True)
    res = {}

    from gabai.spike.las2_runner import Las2Depth
    out, ms, mb, peak = run_source("LAS2", lambda: Las2Depth("s"), lambda m, p: m.disparity_lr_batched(*p), pairs, cal)
    res["LAS2 stereo (+LR check)"] = dict(z=[to_depth(np.where(t, d, np.nan), cal) for d, t, _ in out], ms=ms, mb=mb, peak=peak, trusted=np.mean([t.mean() for _, t, _ in out]))
    las_disp = [d for d, _, _ in out]

    from gabai.spike.ffs_runner import FfsDepth
    out, ms, mb, peak = run_source("FFS", lambda: FfsDepth(), lambda m, p: m.disparity(*p)[0], pairs, cal)
    res["Fast-FoundationStereo (FP16)"] = dict(z=[to_depth(d, cal) for d in out], ms=ms, mb=mb, peak=peak)
    if a.ffs_lr:
        out2, ms2, mb2, peak2 = run_source("FFS+LR", lambda: FfsDepth(), lambda m, p: m.disparity_lr(*p)[:2], pairs, cal)
        res["Fast-FoundationStereo + LR check"] = dict(z=[to_depth(np.where(t, d, np.nan), cal) for d, t in out2], ms=ms2, mb=mb2, peak=peak2, trusted=np.mean([t.mean() for _, t in out2]))

    def sgbm(_, p):
        g = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in p]
        s = cv2.StereoSGBM_create(0, 96, 7, 8 * 49, 32 * 49, 1, 31, 10, 100, 4, cv2.STEREO_SGBM_MODE_HH)
        return s.compute(g[0], g[1]).astype(np.float32) / 16.0
    out, ms, _, _ = run_source("SGBM", lambda: None, sgbm, pairs, cal)
    res["OpenCV SGBM (RTAB-Map settings)"] = dict(z=[to_depth(np.where(d > 0.5, d, np.nan), cal) for d in out], ms=ms, mb=0, peak=0)

    from gabai.spike.mono_depth import MonoDepth
    out, ms, mb, peak = run_source("DAv2", lambda: MonoDepth("dav2"), lambda m, p: (m.depth(p[0]), m.depth(p[1])), pairs, cal)
    from gabai.fuse_depth import fit_affine
    zl, zr, zl_al = [], [], []
    for (ml, mr), d in zip(out, las_disp):
        zl.append(np.where((ml > 0.3) & (ml < 4.0), ml, np.nan).astype(np.float32))
        zr.append(np.where((mr > 0.3) & (mr < 4.0), mr, np.nan).astype(np.float32))
        dm = cal["f"] * cal["B"] / np.clip(ml, 0.2, 30.0)                  # mono as disparity on the true scale, then fitted to LAS2 (scale and offset)
        ok = (d > 3) & np.isfinite(dm)
        aff = fit_affine(dm, d, ok) if ok.sum() > 300 else (1.0, 0.0)
        zl_al.append(to_depth(aff[0] * dm + aff[1], cal))
    res["DAv2 left, raw metric"] = dict(z=zl, ms=ms / 2, mb=mb, peak=peak)
    res["DAv2 right, raw metric"] = dict(z=zr, ms=ms / 2, mb=mb, peak=peak)
    res["DAv2 left, fitted to LAS2"] = dict(z=zl_al, ms=ms / 2, mb=mb, peak=peak)

    ref_names = ["LAS2 stereo (+LR check)", "Fast-FoundationStereo (FP16)"]
    print(f"{'source':36s} {'ms':>6s} {'model MB':>8s} {'peak MB':>8s} {'valid':>6s} {'planar':>7s} {'thick mm':>9s} {'agree LAS2':>10s} {'agree FFS':>10s}")
    for name, r in res.items():
        valid = np.mean([np.isfinite(z).mean() for z in r["z"]])
        pl = [x for x in (planarity(z, cal) for z in r["z"][::3]) if x]
        share, thick = (np.mean([x[0] for x in pl]), np.mean([x[1] for x in pl])) if pl else (np.nan, np.nan)
        agree = []
        for ref in ref_names:
            vals = []
            for z, zr_ in zip(r["z"], res[ref]["z"]):
                both = np.isfinite(z) & np.isfinite(zr_)
                if both.sum() > 500:
                    vals.append(np.mean(np.abs(z[both] - zr_[both]) / zr_[both] < 0.05))
            agree.append(np.mean(vals) if vals else np.nan)
        print(f"{name:36s} {r['ms']:6.0f} {r['mb']:8.0f} {r['peak']:8.0f} {valid * 100:5.0f}% {share * 100:6.0f}% {thick:9.1f} {agree[0] * 100:9.0f}% {agree[1] * 100:9.0f}%", flush=True)
    print("\nplanar / thick: one depth picture alone, RANSAC planes at 2 cm. agree: share of pixels, both valid, within 5% in depth. LAS2 'valid' is after its left-right check.")


if __name__ == "__main__":
    main()
