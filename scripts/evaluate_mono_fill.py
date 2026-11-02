"""How well does the monocular fill reproduce depth stereo really measured? Hold-out test on a recording, no ground truth needed beyond stereo itself.

py scripts/evaluate_mono_fill.py data/recordings/normal_20261029_235458 [--model h] [--frames 24]

Per frame: run LAS2 (with the left-right check) and the monocular model. Hide a few square patches of TRUSTED stereo pixels (they become holes: not used for the
scale fit, not in the trusted mask), let DepthFuser.fill supply them, and compare what it supplied with the stereo values that were hidden. The error is of the
distance, (filled - stereo) / stereo, over the hidden pixels. 'scale only' is the same with one global scale and no offset or low-frequency correction, to show what
the alignment buys. It says how closely the fill follows a stereo measurement. It does not say stereo is right: both stay bounded by the stereo calibration (see
gabai.tools.fit_depth_correction for that error).
"""
import argparse

import numpy as np

from gabai.fuse_depth import DepthFuser
from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.run import MONO_FB
from gabai.pipeline.sources import open_source
from gabai import depthcal


def hide(trusted, rng, n=6, size=40):
    """Mask of n square patches fully inside the trusted area (at least 90% trusted), the pseudo-holes."""
    h, w = trusted.shape
    out = np.zeros_like(trusted)
    for _ in range(60):
        y, x = int(rng.integers(0, h - size)), int(rng.integers(0, w - size))
        if trusted[y:y + size, x:x + size].mean() > 0.9:
            out[y:y + size, x:x + size] = True
        if out.sum() >= n * size * size:
            break
    return out & trusted


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
    rng = np.random.default_rng(0)
    fuser = DepthFuser()
    err_fill, err_scale, covered, resid, refused, whys, err_refused = [], [], [], [], 0, [], []
    n = 0
    for i, fr in enumerate(open_source(cfg)):
        if i % a.every:
            continue
        import cv2
        l, r = cv2.cvtColor(fr.left, cv2.COLOR_BGR2RGB), cv2.cvtColor(fr.right, cv2.COLOR_BGR2RGB)
        d, trusted, _ = las.disparity_lr_batched(l, r)
        d = corr(d) if corr is not None else d
        dm = MONO_FB / np.clip(mono.depth(l), 0.2, 30.0)
        good = trusted & (d > 1)
        holes = hide(good, rng)
        if not holes.any():
            continue
        fit_mask = good & ~holes
        aligned, fill, info = fuser.fill(np.where(fit_mask, d, 0), fit_mask, dm)
        n += 1
        f = fr.cal["f"] * fr.cal["B"]
        if not info["ok"]:
            refused += 1
            whys.append(info['why'][:60])
            if "resid_px" in info:                                  # how bad the frames the check turns away really are
                tz = f / d[holes]
                err_refused.extend(np.abs((f / np.maximum(aligned[holes], 0.5) - tz) / tz).tolist())
            continue
        resid.append(info["resid_px"])
        f = fr.cal["f"] * fr.cal["B"]
        hz, tz = f / np.maximum(aligned[holes], 0.5), f / d[holes]
        err_fill.extend(((hz - tz) / tz).tolist())
        covered.append(float((fill & holes).sum() / holes.sum()))
        a_s = float(np.median(d[fit_mask] / dm[fit_mask]))
        err_scale.extend(((f / np.maximum(a_s * dm[holes], 0.5) - tz) / tz).tolist())
        if n >= a.frames:
            break
    for w in sorted(set(x.split(" (")[0].split(" is ")[0][:40] + (" resid" if "px from" in x else " corr" if "uncorr" in x else "") for x in whys)):
        pass
    print("refused by: residual", sum("px from" in x for x in whys), "| uncorrelated", sum("uncorr" in x for x in whys), "| too few pixels", sum("only" in x for x in whys))
    if err_refused:
        print(f"frames turned away by the fit check would have had error: median {np.median(err_refused)*100:.1f}%  p90 {np.percentile(err_refused, 90)*100:.1f}%")
    e, s = np.abs(err_fill), np.abs(err_scale)
    print(f"{n} frames tried, {refused} refused by the fit check, {len(resid)} scored; fit residual on trusted pixels median {np.median(resid):.2f} px")
    print(f"hidden stereo pixels covered by the fill: {np.mean(covered)*100:.0f}%")
    print(f"fill error vs the hidden stereo distance: median {np.median(e)*100:.1f}%  p90 {np.percentile(e, 90)*100:.1f}%  bias {np.median(err_fill)*100:+.1f}%")
    print(f"scale-only, no alignment detail:           median {np.median(s)*100:.1f}%  p90 {np.percentile(s, 90)*100:.1f}%  bias {np.median(err_scale)*100:+.1f}%")


if __name__ == "__main__":
    main()
