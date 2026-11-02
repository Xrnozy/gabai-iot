"""Does asking the RIGHT monocular map to agree make the fill better? Hold-out test like evaluate_mono_fill.py: hide patches of trusted stereo pixels, fill them
from the left monocular map, and score the fill against the hidden stereo values, once for every filled pixel and once only for those where the right monocular map
(own scale fitted on the visible stereo pixels, put into the left view) agrees with the left one within tol px.

py scripts/evaluate_mono_pair.py data/recordings/normal_20261029_235458 [--model h] [--frames 24]
"""
import argparse

import cv2
import numpy as np

from evaluate_mono_fill import hide
from gabai import depthcal
from gabai.fuse_depth import DepthFuser, combine_mono_pair, fit_affine, warp_right_to_left
from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.run import MONO_FB
from gabai.pipeline.sources import open_source


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="h")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--every", type=int, default=8)
    a = ap.parse_args()
    from gabai.spike.las2_runner import Las2Depth
    from gabai.spike.mono_depth import MonoDepth
    las, mono = Las2Depth(a.model), MonoDepth("dav2")
    corr = depthcal.load(depthcal.DEFAULT_PATH, a.model)
    cfg = PipelineConfig.load("config/pipeline.toml", {"pipeline.mode": "stereo_only", "source.kind": "replay", "source.folder": a.folder, "source.max_frames": 0})
    rng, fuser = np.random.default_rng(0), DepthFuser()
    tols = (1.0, 2.0, 3.0, 4.0)
    err_all, err_agree, kept = [], {t: [] for t in tols}, {t: [] for t in tols}
    n = 0
    for i, fr in enumerate(open_source(cfg)):
        if i % a.every:
            continue
        l, r = cv2.cvtColor(fr.left, cv2.COLOR_BGR2RGB), cv2.cvtColor(fr.right, cv2.COLOR_BGR2RGB)
        d, trusted, _ = las.disparity_lr_batched(l, r)
        d = corr(d) if corr is not None else d
        dl, dr = mono.depth_pair(l, r)
        dm, dmr = MONO_FB / np.clip(dl, 0.2, 30.0), MONO_FB / np.clip(dr, 0.2, 30.0)
        good = trusted & (d > 1)
        holes = hide(good, rng)
        if not holes.any():
            continue
        fit_mask = good & ~holes
        aligned, fill, info = fuser.fill(np.where(fit_mask, d, 0), fit_mask, dm)
        if not info["ok"]:
            continue
        rw = warp_right_to_left(dmr, np.where(fit_mask, d, aligned))
        okr = fit_mask & np.isfinite(rw)
        if okr.sum() < 300:
            continue
        a_r, b_r = fit_affine(rw, d, okr)
        rl = a_r * rw + b_r
        f = fr.cal["f"] * fr.cal["B"]
        tz = f / d[holes]
        err_all.extend(np.abs((f / np.maximum(aligned[holes], 0.5) - tz) / tz).tolist())
        for t in tols:
            comb, agree = combine_mono_pair(aligned, rl, t)
            m = holes & agree
            kept[t].append(float(m.sum() / holes.sum()))
            if m.any():
                err_agree[t].extend(np.abs((f / np.maximum(comb[m], 0.5) - f / d[m]) / (f / d[m])).tolist())
        n += 1
        if n >= a.frames:
            break
    print(f"{n} frames; error of the fill vs the hidden stereo distance (median / p90):")
    print(f"  left monocular only, every filled pixel:   {np.median(err_all) * 100:.1f}% / {np.percentile(err_all, 90) * 100:.1f}%")
    for t in tols:
        print(f"  left+right agree within {t:g} px:           {np.median(err_agree[t]) * 100:.1f}% / {np.percentile(err_agree[t], 90) * 100:.1f}%   keeps {np.mean(kept[t]) * 100:.0f}% of the pixels")


if __name__ == "__main__":
    main()
