"""M1 acceptance: TensorRT FFS + confidence against the PyTorch FFS on the same rectified pairs.

  py -m mapper.bench_depth data/recordings/walk_20261102_230115 --config config/pipeline_calib45.toml [--frames 24 --step 30]

Prints latency, agreement with the PyTorch disparity, and whether confidence means anything: the disparity scatter around fitted planes (px, +-3 px band, same for every bucket)
for high-confidence against low-confidence pixels. A calibrated confidence has a clearly smaller scatter in the high bucket.
"""
import argparse

import numpy as np

from mapper.depth_trt import StereoTrt, paths
from rtabmap_lab.tools.depth_bench import frames


def plane_residuals(disp, conf, n_planes=3, band=3.0, iters=300, seed=0):
    """Sequential RANSAC planes in (u, v, disparity). Returns (residual px, confidence) of every pixel within `band` of its plane."""
    v, u = np.nonzero(np.isfinite(disp) & (disp > 3))
    rng = np.random.default_rng(seed)
    if len(v) > 40000:
        k = rng.choice(len(v), 40000, replace=False)
        v, u = v[k], u[k]
    X = np.stack([u, v, np.ones(len(u))], 1).astype(np.float64)
    d, c = disp[v, u].astype(np.float64), conf[v, u]
    res, cs = [], []
    for _ in range(n_planes):
        if len(d) < 1500:
            break
        best = None
        for _ in range(iters):
            i = rng.choice(len(d), 3, replace=False)
            try:
                w = np.linalg.solve(X[i], d[i])
            except np.linalg.LinAlgError:
                continue
            inl = np.abs(X @ w - d) < 1.0
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < 1500:
            break
        w = np.linalg.lstsq(X[best], d[best], rcond=None)[0]
        r = X @ w - d
        near = np.abs(r) < band
        res.append(r[near]); cs.append(c[near])
        X, d, c = X[~near], d[~near], c[~near]
    return (np.concatenate(res), np.concatenate(cs)) if res else (np.zeros(0), np.zeros(0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--config", default="config/pipeline.toml")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--depth", choices=["retinify", "las2", "ffs"], default="retinify")
    ap.add_argument("--retinify-mode", choices=["fast", "balanced", "accurate"], default="balanced")
    ap.add_argument("--las2-size", choices=["s", "m", "h"], default="s")
    ap.add_argument("--iters", type=int, default=8, help="which engine (see depth_trt build --iters)")
    ap.add_argument("--max-disp", type=int, default=192)
    ap.add_argument("--torch", action="store_true", help="also run the PyTorch FFS and compare")
    a = ap.parse_args(argv)
    fr = frames(a.folder, a.frames, a.step, a.config)
    cal = fr[0].cal
    if a.depth == "retinify":
        from mapper.depth_retinify import RetinifyStereo
        m = RetinifyStereo(cal["f"], cal["B"], a.retinify_mode)
    elif a.depth == "las2":
        from mapper.depth_las2 import Las2Stereo
        m = Las2Stereo(cal["f"], cal["B"], a.las2_size)
    else:
        m = StereoTrt(cal["f"], cal["B"], paths(a.iters, a.max_disp)[1])
    rgb = lambda x: np.ascontiguousarray(x[:, :, ::-1])
    pairs = [(rgb(f.left), rgb(f.right)) for f in fr]
    m(*pairs[0])
    out = [m(*p) for p in pairs]
    print(f"{len(pairs)} pairs, f*B = {cal['f'] * cal['B']:.1f} px*m")
    print(f"{a.depth} batch 2 + confidence: {np.median([o.ms for o in out]):.1f} ms median, {max(o.ms for o in out):.1f} ms worst")
    print(f"depth kept: {np.mean([(o.depth > 0).mean() for o in out]) * 100:.0f}% of pixels; conf>0.8: {np.mean([(o.conf > 0.8).mean() for o in out]) * 100:.0f}%")
    if a.torch:
        from gabai.spike.ffs_runner import FfsDepth
        t = FfsDepth()
        ref = [t.disparity(*p)[0] for p in pairs]
        diff = np.concatenate([np.abs(o.disp - r)[r > 3] for o, r in zip(out, ref)])
        print(f"vs PyTorch FFS disparity: median |diff| {np.median(diff):.3f} px, 90th {np.percentile(diff, 90):.3f} px")
    rs, cs = zip(*[plane_residuals(np.where(o.depth > 0, o.disp, np.nan), o.conf) for o in out])
    rs, cs = np.concatenate(rs), np.concatenate(cs)
    for name, sel in (("conf > 0.8", cs > 0.8), ("0.3 - 0.8", (cs > 0.3) & (cs <= 0.8)), ("conf <= 0.3", cs <= 0.3)):
        if sel.sum() > 100:
            print(f"plane scatter, {name:11s}: {np.sqrt(np.mean(rs[sel] ** 2)):.3f} px  ({sel.sum()} px)")


if __name__ == "__main__":
    main()
