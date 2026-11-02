"""PC-side. Rectifies stereo pairs, runs OpenCV SGBM, saves rectified PNGs for LiteAnyStereo, and
reports depth at a chosen pixel for distance-error checks.

  python -m gabai.spike.depth_compare PAIRS_DIR --calib calib.npz --out depth_out
Then run LiteAnyStereo's demo.py on depth_out/rect_L_*.png / rect_R_*.png and compare.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_dir")
    ap.add_argument("--calib", default="calib.npz")
    ap.add_argument("--out", default="depth_out")
    ap.add_argument("--probe", default=None, help="x,y pixel to print SGBM depth at (rectified image)")
    a = ap.parse_args()

    c = np.load(a.calib)
    size = tuple(int(v) for v in c["size"])
    m1 = cv2.initUndistortRectifyMap(c["K1"], c["D1"], c["R1"], c["P1"], size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(c["K2"], c["D2"], c["R2"], c["P2"], size, cv2.CV_32FC1)
    f, B = float(c["focal"]), float(c["baseline"])
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
                                 P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    os.makedirs(a.out, exist_ok=True)
    for lf in sorted(glob.glob(os.path.join(a.pairs_dir, "L_*.png"))):
        rf = lf.replace("L_", "R_")
        l = cv2.remap(cv2.imread(lf), *m1, cv2.INTER_LINEAR)
        r = cv2.remap(cv2.imread(rf), *m2, cv2.INTER_LINEAR)
        k = os.path.basename(lf)[2:-4]
        cv2.imwrite(os.path.join(a.out, f"rect_L_{k}.png"), l)
        cv2.imwrite(os.path.join(a.out, f"rect_R_{k}.png"), r)
        d = sgbm.compute(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)) / 16.0
        valid = d > 0
        vis = cv2.applyColorMap(cv2.convertScaleAbs(np.where(valid, d, 0), alpha=255 / 128), cv2.COLORMAP_TURBO)
        vis[~valid] = 0
        cv2.imwrite(os.path.join(a.out, f"sgbm_{k}.png"), vis)
        msg = f"{k}: valid={valid.mean()*100:.0f}%"
        if a.probe:
            x, y = map(int, a.probe.split(","))
            patch = d[y - 5:y + 6, x - 5:x + 6]
            patch = patch[patch > 0]
            if patch.size:
                msg += f" depth@probe={f*B/np.median(patch):.2f} m"
            else:
                msg += " depth@probe=invalid"
        print(msg)


if __name__ == "__main__":
    main()
