"""Stereo distance against a tape measure: the only check of the metre (the calibration boards can only check themselves).

Record one short clip per distance, the rig STILL and square-on to a flat wall (or a door, a large box face), the tape from the front of the lenses to the wall:
  py -m gabai.tools.record_pairs ...      (one folder per distance, e.g. data/recordings/wall_100, wall_150, wall_200, wall_300)
then:
  py -m rtabmap_lab.tools.known_distance --calib data/stereo_calib_charuco_45.npz data/recordings/wall_100=1.00 data/recordings/wall_150=1.50 data/recordings/wall_200=2.00 data/recordings/wall_300=3.00

Per clip: the largest plane in the middle of the rectified pair (OpenCV SGBM disparity, plane fitted in disparity space, no smoothing) gives the distance along the optical axis at the
picture centre; the median over the clip's pairs is compared with the tape. Printed: estimate, error in mm and %, the disparity, and the error 0.5 px of disparity would cause at that distance.
If the error grows in proportion to the distance (same % everywhere) the baseline or focal length is off: the last line gives the baseline that fits the tape (least squares in disparity).
A constant offset in disparity (error growing with the square of the distance) points at the principal point / yaw instead.
"""
import argparse

import cv2
import numpy as np

from rtabmap_lab.calib_lab import params as P
from rtabmap_lab.calib_lab.recording import Recording
from rtabmap_lab.planar_recon import frame_planes


def clip_distance(rec, rect, every=3):
    sg = cv2.StereoSGBM_create(0, 96, 5, 8 * 3 * 25, 32 * 3 * 25, 1, 63, 10, 100, 2, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    fB, out = rect.f * rec.k * rect.B, []
    for i in range(0, rec.n, every):
        l, r = rec.prepare(i, rect)
        h, w = l.shape[:2]
        d = sg.compute(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16
        d[:, : int(w * 0.15)] = 0                                             # the left strip has no right-picture partner
        z = np.where(d > 1, fB / np.maximum(d, 1e-3) * 1000, 0).astype(np.uint16)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        best = None
        for n, dd, piece, cells, (a, b, c) in frame_planes(z, rect.f * rec.k, rect.cx * rec.k, rect.cy * rec.k, fB, max_planes=3):
            ys, xs = np.nonzero(piece)
            near_centre = np.mean((np.abs(xs * 4 - cx) < w / 4) & (np.abs(ys * 4 - cy) < h / 4))
            if best is None or near_centre * cells > best[0]:
                best = (near_centre * cells, a * (rect.cx * rec.k) + b * (rect.cy * rec.k) + c)  # the plane's disparity on the optical axis
        if best and best[1] > 0.5:
            out.append(best[1])
    return np.array(out), fB


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="folder=metres")
    ap.add_argument("--calib", required=True)
    a = ap.parse_args(argv)
    rect = P.CalibParams.from_npz(a.calib).rect()
    D, T = [], []
    print(f"calibration {a.calib}: f {rect.f:.1f} px, baseline {rect.B * 1000:.2f} mm")
    print(f"{'clip':40s} {'tape m':>7s} {'stereo m':>9s} {'error mm':>9s} {'error %':>8s} {'disp px':>8s} {'0.5px = mm':>10s}")
    for item in a.clips:
        folder, tape = item.rsplit("=", 1)
        tape = float(tape)
        rec = Recording(folder).decide(rect)
        disp, fB = clip_distance(rec, rect)
        if not len(disp):
            print(f"{folder:40s} no plane found")
            continue
        dm = float(np.median(disp))
        z = fB / dm
        print(f"{folder:40s} {tape:7.3f} {z:9.3f} {(z - tape) * 1000:+9.0f} {(z - tape) / tape * 100:+8.1f} {dm:8.2f} {tape ** 2 * 0.5 / fB * 1000:10.0f}")
        D.append(dm)
        T.append(tape)
    if len(D) >= 2:
        D, T = np.array(D), np.array(T)
        fB_fit = float(np.sum(D / T) / np.sum(1 / T ** 2))                      # disparity = fB / tape, least squares
        A = np.stack([1 / T, np.ones_like(T)], 1)
        k, off = np.linalg.lstsq(A, D, rcond=None)[0]
        print(f"baseline that fits the tape: {fB_fit / (rect.f * rec.k) * 1000:.2f} mm (calibration {rect.B * 1000:.2f} mm); "
              f"with a free disparity offset: baseline {k / (rect.f * rec.k) * 1000:.2f} mm, offset {off:+.2f} px")


if __name__ == "__main__":
    main()
