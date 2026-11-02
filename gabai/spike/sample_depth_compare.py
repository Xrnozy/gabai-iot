"""Compare LAS2-S and OpenCV SGBM at webcam-like resolution on LiteAnyStereo's sample ZED pairs.

There is no ground truth here. The reference is LAS2-S run at full resolution and downscaled, so the
numbers show agreement with a stronger run, not true metric accuracy. Real accuracy needs tape-measured scenes.
"""
import argparse
import glob
import os

import cv2
import numpy as np

from gabai.spike.las2_runner import Las2Depth, ROOT

F_FULL, BASELINE = 1068.63, 0.120134  # ZED2 HD2K left fx and baseline from LiteAnyStereo's demo.py


def sgbm_disparity(l, r, max_disp=128):
    g1, g2 = cv2.cvtColor(l, cv2.COLOR_RGB2GRAY), cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
    s = cv2.StereoSGBM_create(0, max_disp, 5, 8 * 25, 32 * 25, uniquenessRatio=10, speckleWindowSize=100,
                              speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = s.compute(g1, g2).astype(np.float32) / 16
    d[d <= 0] = np.nan
    return d


def stats(d, ref, f, name):
    ok = np.isfinite(d) & np.isfinite(ref) & (ref > 1)
    err = np.abs(d - ref)[ok]
    z = f * BASELINE / np.maximum(ref[ok], 1e-3)
    near = z < 3
    zd = f * BASELINE / np.maximum(d[ok], 1e-3)
    zrel = np.abs(zd - z) / z
    print(f"  {name:6s} valid={np.isfinite(d).mean()*100:5.1f}%  median|disp err|={np.median(err):5.2f}px  "
          f"p90={np.percentile(err, 90):5.2f}px  median depth err (<3 m)={np.median(zrel[near])*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out", default="docs/sample_compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    las = Las2Depth("s")
    files = sorted(glob.glob(os.path.join(ROOT, "assets", "Explorer_*.png")))
    for p in files:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        W = img.shape[1] // 2
        L, R = img[:, :W], img[:, W:2 * W]
        scale = a.width / W
        size = (a.width, int(round(L.shape[0] * scale / 32)) * 32 // 1)
        size = (a.width, int(L.shape[0] * scale))
        ref_full, _ = las.disparity(L, R)
        ref = cv2.resize(ref_full, size, interpolation=cv2.INTER_AREA) * scale
        l, r = cv2.resize(L, size, interpolation=cv2.INTER_AREA), cv2.resize(R, size, interpolation=cv2.INTER_AREA)
        las.disparity(l, r)  # warm up
        d_las, dt = las.disparity(l, r)
        d_sg = sgbm_disparity(l, r)
        f = F_FULL * scale
        print(os.path.basename(p), f"-> {size[0]}x{size[1]}  LAS2-S model time {dt*1000:.1f} ms")
        stats(d_las, ref, f, "LAS2-S")
        stats(d_sg, ref, f, "SGBM")
        vis = lambda d: cv2.applyColorMap(np.nan_to_num(np.clip(d * 255 / 96, 0, 255)).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(a.out, os.path.basename(p)[:-4] + "_cmp.png"),
                    np.hstack([cv2.cvtColor(l, cv2.COLOR_RGB2BGR), vis(d_las), vis(d_sg)]))


if __name__ == "__main__":
    main()
