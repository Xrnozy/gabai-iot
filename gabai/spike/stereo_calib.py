"""PC-side quick stereo calibration from checkerboard pairs saved by `pi_capture.py pairs`.

Writes calib.npz with intrinsics, extrinsics, baseline (m) and focal length (px).
Print the board with a known square size and keep it flat; show it at many angles/distances.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_dir")
    ap.add_argument("--cols", type=int, required=True, help="inner corners per row")
    ap.add_argument("--rows", type=int, required=True, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, required=True)
    ap.add_argument("--out", default="calib.npz")
    a = ap.parse_args()

    pattern = (a.cols, a.rows)
    objp = np.zeros((a.cols * a.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square_mm / 1000
    objpts, lpts, rpts = [], [], []
    size = None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    for lf in sorted(glob.glob(os.path.join(a.pairs_dir, "L_*.png"))):
        rf = lf.replace("L_", "R_")
        if not os.path.exists(rf):
            continue
        gl = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
        gr = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        size = gl.shape[::-1]
        okl, cl = cv2.findChessboardCorners(gl, pattern)
        okr, cr = cv2.findChessboardCorners(gr, pattern)
        if okl and okr:
            lpts.append(cv2.cornerSubPix(gl, cl, (11, 11), (-1, -1), crit))
            rpts.append(cv2.cornerSubPix(gr, cr, (11, 11), (-1, -1), crit))
            objpts.append(objp)
    if len(objpts) < 12:
        raise SystemExit(f"only {len(objpts)} usable pairs; need at least 12 (aim for 20+)")
    _, K1, D1, _, _ = cv2.calibrateCamera(objpts, lpts, size, None, None)
    _, K2, D2, _, _ = cv2.calibrateCamera(objpts, rpts, size, None, None)
    rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        objpts, lpts, rpts, K1, D1, K2, D2, size, flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)
    baseline = float(np.linalg.norm(T))
    f = float(P1[0, 0])
    np.savez(a.out, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             size=np.array(size), baseline=baseline, focal=f)
    print(f"pairs used={len(objpts)} stereo RMS reprojection error={rms:.3f}px (want < 0.5)")
    print(f"baseline={baseline*100:.1f} cm  focal={f:.1f}px  size={size}")
    print(f"depth error rule of thumb: dZ ~ Z^2 / (f*B) * dDisp -> at 3 m, 0.5px disparity error = "
          f"{9/(f*baseline)*0.5*100:.1f} cm")


if __name__ == "__main__":
    main()
