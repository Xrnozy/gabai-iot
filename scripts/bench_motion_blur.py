"""How much do motion blur and sensor noise each hurt LAS2 depth? Sharp daylight video pairs are degraded in a controlled way and compared with LAS2's
own result on the clean pair (the video has no ground-truth depth, so numbers are 'distance from the clean result').

    py scripts/bench_motion_blur.py [--model h] [--frames 12]

blur = straight smear of L px along a 30 deg diagonal (L = turn rate in px/s x exposure time); noise = Gaussian grey-level noise sigma (what a higher
camera gain adds). A camera that shortens its exposure to cut blur has to raise its gain, so the last block compares the two at equal brightness
(a 4x shorter exposure needs about 4x the gain, which roughly doubles the noise of a shot-noise-limited sensor).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from gabai.spike.las2_runner import Las2Depth, lr_consistency      # noqa: E402


def smear(img, length, angle_deg=30.0):
    if length < 1.5:
        return img
    k = np.zeros((int(length) | 1,) * 2, np.float32)
    c = k.shape[0] // 2
    dx, dy = np.cos(np.radians(angle_deg)) * length / 2, np.sin(np.radians(angle_deg)) * length / 2
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    return cv2.filter2D(img, -1, k / k.sum(), borderType=cv2.BORDER_REPLICATE)


def noisy(img, sigma, rng):
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="h"), ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--seq", default=os.path.join("data", "bench", "seq_clean.npz"), help="npz with L, R pictures; made from data/left.mp4 when missing")
    a = ap.parse_args()
    if not os.path.exists(a.seq):
        from gabai.pipeline.config import PipelineConfig
        from gabai.pipeline.sources import open_source
        fr = list(open_source(PipelineConfig.load("config/pipeline.toml", {"source.start": 60.0, "source.skip": 6, "source.max_frames": a.frames, "stereo.model": a.model})))
        np.savez_compressed(a.seq, L=np.array([f.left for f in fr]), R=np.array([f.right for f in fr]))
    z = np.load(a.seq)
    L, R = z["L"][:a.frames], z["R"][:a.frames]
    las = Las2Depth(a.model)
    rng = np.random.default_rng(0)

    def disp(l, r):
        d, ok, _ = las.disparity_lr_batched(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        return d, ok
    ref = [disp(l, r) for l, r in zip(L, R)]

    def score(blur, sigma):
        err, cov = [], []
        for (l, r), (d0, ok0) in zip(zip(L, R), ref):
            d, ok = disp(noisy(smear(l, blur), sigma, rng), noisy(smear(r, blur), sigma, rng))
            err.append(np.abs(d - d0)[ok & ok0]), cov.append(ok.mean())
        e = np.concatenate(err)
        return {"valid_pct": round(float(np.mean(cov)) * 100, 1), "median_err_px": round(float(np.median(e)), 2), "over_2px_pct": round(float((e > 2).mean() * 100), 1)}
    out = {"blur_only (noise sigma 2)": {f"{b}px": score(b, 2) for b in (0, 4, 8, 16, 32)},
           "noise_only (no blur)": {f"sigma {s}": score(0, s) for s in (2, 8, 16, 24)},
           "same brightness, shorter exposure + more gain": {"33 ms: blur 32 px, noise 2": score(32, 2), "8 ms: blur 8 px, noise 4": score(8, 4),
                                                              "4 ms: blur 4 px, noise 6": score(4, 6), "2 ms: blur 2 px, noise 9": score(2, 9)}}
    json.dump(out, open(os.path.join("data", "bench", "motion_blur.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
