"""Are the left and right pictures row-aligned? Stereo depth assumes a point has the same row in both pictures. If it does not, depth is wrong
(or missing) and Basalt is fed wrong geometry. This measures the vertical offset of matched features between the two pictures.

    py -m gabai.tools.stereo_alignment_check data/slam/live_20261026_181526        a recorded folder (EuRoC layout)
    py -m gabai.tools.stereo_alignment_check data/slam/<folder> --every 10

What it reports, and how to read it:
  median_dy_px      vertical offset (right row minus left row) of matched features. About 0 (under 1 px) when aligned.
  static            if the recording has an IMU: the offset on frames where the camera was nearly still (gyro under 10 deg/s). An offset that is there while
                    the camera is still is a rectification / calibration / mounting problem (static_misalignment).
  motion            the offset on moving frames. If it is clearly larger there than when still, the two cameras expose at different times (a timing offset),
                    because a turning camera shifts the scene between the two exposures.
Matches are ORB features; only matches that agree with the median offset and have a positive disparity are used. The verdict is unknown when too few
frames have enough texture.
"""
import argparse
import json
import os

import cv2
import numpy as np

STILL_DPS, MOVING_DPS = 10.0, 40.0
ALIGNED_PX = 1.5
MIN_FRAMES = 4


def pair_offset(left, right, n_feat=2000):
    """Median vertical and horizontal offset of matched features, or None when there are too few reliable matches."""
    orb = cv2.ORB_create(n_feat)
    k1, d1 = orb.detectAndCompute(left, None)
    k2, d2 = orb.detectAndCompute(right, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return None
    dy = np.array([k2[a.trainIdx].pt[1] - k1[a.queryIdx].pt[1] for a in good])
    dx = np.array([k1[a.queryIdx].pt[0] - k2[a.trainIdx].pt[0] for a in good])
    keep = (np.abs(dy - np.median(dy)) < 4) & (dx > 0)
    if keep.sum() < 10:
        return None
    return {"dy_px": float(np.median(dy[keep])), "dx_px": float(np.median(dx[keep])), "n": int(keep.sum())}


def diagnose(folder, every=10):
    cam = [l.strip().split(",") for l in open(os.path.join(folder, "mav0", "cam0", "data.csv")) if not l.startswith("#")]
    imu_p = os.path.join(folder, "mav0", "imu0", "data.csv")
    imu = np.loadtxt(imu_p, delimiter=",", comments="#") if os.path.exists(imu_p) else None
    rows = []
    for ts, name in cam[::every]:
        l = cv2.imread(os.path.join(folder, "mav0", "cam0", "data", name), 0)
        r = cv2.imread(os.path.join(folder, "mav0", "cam1", "data", name), 0)
        if l is None or r is None:
            continue
        o = pair_offset(l, r)
        if o is None:
            continue
        w = None
        if imu is not None:
            sel = (imu[:, 0] > int(ts) - 50e6) & (imu[:, 0] < int(ts) + 50e6)
            if sel.any():
                w = float(np.degrees(np.median(np.linalg.norm(imu[sel, 1:4], axis=1))))
        rows.append({"t_ns": int(ts), "gyro_dps": w, **o})
    out = {"folder": folder, "frames_measured": len(rows), "frames_tried": len(cam[::every])}
    if len(rows) < MIN_FRAMES:
        out.update(verdict="unknown", advice="too few frames with enough texture to measure the offset (need %d): try a more textured scene or a smaller --every" % MIN_FRAMES)
        return out
    dy = np.array([r["dy_px"] for r in rows])
    out.update(median_dy_px=float(np.median(dy)), p10_p90_dy_px=[float(np.percentile(dy, 10)), float(np.percentile(dy, 90))], median_dx_px=float(np.median([r["dx_px"] for r in rows])))
    still = [abs(r["dy_px"]) for r in rows if r["gyro_dps"] is not None and r["gyro_dps"] < STILL_DPS]
    moving = [abs(r["dy_px"]) for r in rows if r["gyro_dps"] is not None and r["gyro_dps"] >= MOVING_DPS]
    out["static"] = None if not still else {"median_abs_dy_px": float(np.median(still)), "frames": len(still)}
    out["motion"] = None if not moving else {"median_abs_dy_px": float(np.median(moving)), "frames": len(moving)}
    if abs(np.median(dy)) <= ALIGNED_PX and (not still or np.median(still) <= ALIGNED_PX):
        out.update(verdict="aligned", advice="rows agree within %.1f px" % ALIGNED_PX)
    elif still and np.median(still) > ALIGNED_PX * 2:
        extra = ""
        if moving and np.median(moving) > 2 * np.median(still):
            extra = " It is also clearly larger while moving, so a left/right timing offset may add to it."
        out.update(verdict="static_misalignment", advice=("the offset is there while the camera is still, so the pictures are not row-aligned: the rectification does not match the cameras "
                                                         "(camera order or rotation wrong, cameras moved since calibration, or a bad calibration). Recalibrate with gabai.tools.calibrate_stereo "
                                                         "and re-check." + extra))
    elif moving and still and np.median(moving) > 2 * max(np.median(still), 0.5):
        out.update(verdict="timing_offset", advice="the offset grows with rotation: the two cameras do not capture at the same moment. Raise the capture frame rate or use a synchronised pair.")
    else:
        out.update(verdict="misaligned", advice="the offset is larger than %.1f px but there is no still/moving split to say why (no IMU in this recording). Recalibrate and re-check." % ALIGNED_PX)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    r = diagnose(a.folder, a.every)
    print(json.dumps(r, indent=2))
    print(f"\nverdict: {r['verdict']}: {r['advice']}")


if __name__ == "__main__":
    main()
