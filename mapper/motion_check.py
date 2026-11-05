"""Independent motion check from the Pi's KLT tracks: does the odometry's rotation agree with what the points did and with what the gyro measured?

KLT is not ground truth and sets no pose. Per pair of consecutive frames, the tracks (forward-backward checked, as on the Pi) give a camera rotation through the essential matrix; its angle is compared
with the gyro's and with the odometry's. The verdict is a flag the mapper uses to keep a doubtful pair from becoming a map node, nothing more.

  py -m mapper.motion_check data/recordings/walk_20261102_230115 --calib data/stereo_calib_charuco_45.npz [--lost lost.json]

Without a Pi in the loop the recording's left pictures are tracked here with the same method as gabai/pi/klt.py (LK pyramid + forward-backward check).
"""
import argparse
import json

import cv2
import numpy as np


def _kabsch(b0, b1):
    U, _, Vt = np.linalg.svd(b1.T @ b0)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def rotation_from_tracks(p0, p1, K, D, R1=None, min_tracks=25, thresh_px=3.0, iters=200, seed=0):
    """p0, p1: (N, 2) pixels of the same points in two raw left pictures. Returns (camera rotation, inlier ratio, inliers); rotation is None when there are too few tracks.
    Rotation only: the bearing vectors of consecutive pictures differ by the camera rotation plus a parallax of about translation / depth (about 1 deg at 4 cm and 3 m), small against the
    3 deg gate. An essential matrix was tried first and discarded: at this little parallax recoverPose's depth cut-off throws most points away (16% of pairs usable).
    R1 (the left rectification rotation) re-expresses the rotation in the frame the odometry and the gyro use."""
    if len(p0) < min_tracks:
        return None, 0.0, len(p0)
    b0, b1 = (np.c_[cv2.undistortPoints(np.asarray(p, np.float64).reshape(-1, 1, 2), K, D).reshape(-1, 2), np.ones(len(p))] for p in (p0, p1))
    b0 /= np.linalg.norm(b0, axis=1, keepdims=True)
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    rng, best, tol = np.random.default_rng(seed), None, thresh_px / K[0, 0]
    for _ in range(iters):
        i = rng.choice(len(b0), 3, replace=False)
        inl = np.linalg.norm(b1 - b0 @ _kabsch(b0[i], b1[i]).T, axis=1) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() < max(min_tracks // 2, 8):
        return None, float(best.sum()) / len(b0), int(best.sum())
    R = _kabsch(b0[best], b1[best])                      # b1 = R b0: the camera's rotation expressed as the points moving the other way, so this is the inverse camera rotation
    R = R.T
    if R1 is not None:
        R = R1 @ R @ R1.T
    return R, float(best.sum()) / len(b0), int(best.sum())


def angle_between(Ra, Rb):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(Ra.T @ Rb)[0])))


def verdict(R_klt, ratio, R_gyro, R_odom, tol_deg=3.0, min_ratio=0.5):
    """Flags one pair. R_* are 3x3 camera rotations between the two pictures in one common frame (any may be None). 'ok' is False only when the odometry disagrees with
    a measurement that exists AND the other independent measurement does not contradict that one."""
    out = {"klt_deg": None, "d_klt_gyro": None, "d_odom_gyro": None, "d_odom_klt": None, "ok": True, "reason": ""}
    if R_klt is not None:
        out["klt_deg"] = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R_klt)[0])))
    if R_klt is not None and ratio >= min_ratio:
        if R_gyro is not None:
            out["d_klt_gyro"] = angle_between(R_klt, R_gyro)
        if R_odom is not None:
            out["d_odom_klt"] = angle_between(R_klt, R_odom)
    if R_gyro is not None and R_odom is not None:
        out["d_odom_gyro"] = angle_between(R_gyro, R_odom)
    d = [out[k] for k in ("d_odom_klt", "d_odom_gyro") if out[k] is not None]
    if d and max(d) > tol_deg:
        other = out["d_klt_gyro"]
        if other is None or other <= tol_deg:
            out["ok"], out["reason"] = False, f"odometry rotation off by {max(d):.1f} deg from " + ("gyro and KLT" if len(d) == 2 else "the other sensor")
    return out


def track(prev_gray, gray, pts, fb_px=1.0):
    """LK from prev to cur with a forward-backward check, as on the Pi. Returns (p0, p1) of the points that survive."""
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(15, 15), maxLevel=2)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, winSize=(15, 15), maxLevel=2)
    ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(p0b - pts, axis=2)[:, 0] < fb_px)
    return pts[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--calib", default="data/stereo_calib_charuco_45.npz")
    ap.add_argument("--clip", type=int, nargs=2, default=(0, 300))
    ap.add_argument("--lost", help="JSON list of pair indices the odometry lost (odom_eval --save-lost)")
    a = ap.parse_args(argv)
    from rtabmap_lab.calib_lab import params as P
    from rtabmap_lab.calib_lab.recording import Recording
    base = P.CalibParams.from_npz(a.calib)
    rect = base.rect()
    rec = Recording(a.folder).decide(rect)
    K, D = base.arrays()[0], base.arrays()[1]
    rows, prev = [], None
    for i in range(a.clip[0], min(a.clip[1], rec.n)):
        l, _, _ = rec.calibrated_pair(i)
        g = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY) if l.ndim == 3 else l
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, 200, 0.01, 10)
            if pts is None or len(pts) < 25:
                rows.append((i, None)); prev = g; continue
            p0, p1 = track(prev, g, pts)
            Rk, ratio, _ = rotation_from_tracks(p0, p1, K, D, rect.R1)
            gv = rec.gyro_rotvec(i - 1, i)
            rows.append((i, verdict(Rk, ratio, None if gv is None else cv2.Rodrigues(gv)[0], None)))
        prev = g
    got = [(i, v) for i, v in rows if v and v["d_klt_gyro"] is not None]
    d = np.array([v["d_klt_gyro"] for _, v in got])
    print(f"{len(rows)} pairs; KLT rotation usable on {len(got)} ({len(got) / max(len(rows), 1) * 100:.0f}%)")
    if len(d):
        print(f"KLT vs gyro rotation disagreement: median {np.median(d):.2f} deg, 90th {np.percentile(d, 90):.2f} deg")
    if a.lost:
        lost = set(json.load(open(a.lost)))
        usable = {i for i, _ in got}
        miss = [i for i, _ in rows if i not in usable]
        print(f"pairs with no usable KLT rotation: {len(miss)}, of which odometry lost: {len([i for i in miss if i in lost])}; odometry lost in total: {len(lost)}")


if __name__ == "__main__":
    main()
