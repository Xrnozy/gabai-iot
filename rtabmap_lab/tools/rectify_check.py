"""Does a calibration rectify THIS recording? Row error of left/right matches, per calibration file, with and without the rolling-shutter correction, and a picture with horizontal lines.

  py -m rtabmap_lab.tools.rectify_check data/recordings/<folder> --calib data/stereo_calib_charuco_45.npz data/stereo_calib_charuco_42.npz [--out rect.png]

Row error = median |y_right - y_left| of SIFT matches of the rectified pair (0 when the rows line up). 'still' pairs: the rig turns slower than 10 deg/s, so neither motion nor rolling shutter
blames the calibration. A calibration made before the cameras moved in their mounts shows up here as several px on still pairs.
The picture: rectified left | right of a few still pairs, a green line every 20 px; the same feature must sit on the same line in both halves.
"""
import argparse

import cv2
import numpy as np

from rtabmap_lab.calib_lab import params as P
from rtabmap_lab.calib_lab.recording import Recording
from rtabmap_lab.rolling_shutter import gyro_rate, row_stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--calib", nargs="+", required=True)
    ap.add_argument("--pairs", type=int, default=60, help="pairs sampled over the recording")
    ap.add_argument("--out", help="write the line picture (first calibration) here")
    a = ap.parse_args(argv)
    for k, path in enumerate(a.calib):
        rect = P.CalibParams.from_npz(path).rect()
        rec = Recording(a.folder).decide(rect)
        idx = np.linspace(0, rec.n - 1, a.pairs).astype(int)
        res, still, tiles = {"none": [], "corrected": []}, [], []
        for i in idx:
            w = gyro_rate(rec.imu, rec.t[i], rec.imu_delay) if rec.imu is not None else None
            for name, sh in (("none", None), ("corrected", "default")):
                if name == "corrected" and rec.shutter is None:
                    continue
                l, r = rec.prepare(i, rect, shutter=sh)
                s = row_stats(l, r)
                res[name].append(np.median(np.abs(s[1])) if s else np.nan)
                if name == "none" and k == 0 and a.out and s and len(s[0]) > 300 and w is not None and np.degrees(np.linalg.norm(w)) < 15 and len(tiles) < 3:
                    pair = np.hstack([l, r])
                    for y in range(10, pair.shape[0], 20):
                        cv2.line(pair, (0, y), (pair.shape[1], y), (0, 255, 0), 1)
                    tiles.append(pair)
            still.append(w is not None and np.degrees(np.linalg.norm(w)) < 10)
        still = np.array(still)
        line = f"{path}: {rec.layout_note}; row error median over {len(idx)} pairs: {np.nanmedian(res['none']):.2f} px (still pairs {np.nanmedian(np.array(res['none'])[still]):.2f} px, n={still.sum()})"
        if res["corrected"]:
            line += f"; with the rolling-shutter correction {np.nanmedian(res['corrected']):.2f} px"
        print(line, flush=True)
        if tiles:
            cv2.imwrite(a.out, np.vstack(tiles))
            print("written", a.out)


if __name__ == "__main__":
    main()
