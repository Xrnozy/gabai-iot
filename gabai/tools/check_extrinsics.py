"""Check the camera-to-IMU rotation and delay from a RECORDED sequence (data/slam/rec_*), without Basalt.

    py -m gabai.tools.check_extrinsics data/slam/rec_20261026_155811

The camera's rotation between frames is measured from the images (essential matrix on tracked corners), the gyro is
integrated over the same intervals, and the two are compared. Rotation size is independent of the axis map, so it gives
the delay; the axes then come from a trimmed least-squares fit. Compare the result with data/imu_sync.json: a large
angle between them means Basalt will diverge, because it trusts that rotation completely.
"""
import json
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

STRIDE = 2


def gyro_rotvec(ti, w, dt, t0, t1):
    m = (ti >= t0) & (ti < t1)
    return (w[m] * dt[m, None]).sum(0) if m.sum() > 2 else None


def fit_axes(vis, gyro, trim=0.7, rounds=8):
    """Rotation Rf with gyro ~ Rf @ vis (Kabsch), refit on the best `trim` share of pairs. Returns Rf, inlier mask, relative errors."""
    inl = np.ones(len(vis), bool)
    for _ in range(rounds):
        U, _, Vt = np.linalg.svd(vis[inl].T @ gyro[inl])
        Rf = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
        err = np.linalg.norm(gyro - vis @ Rf.T, axis=1) / np.maximum(np.linalg.norm(gyro, axis=1), 1e-9)
        inl = err < max(np.percentile(err, 100 * trim), 0.25)
    return Rf, inl, err


def measure_pairs(imgs, K):
    pairs = []
    for i in range(len(imgs) - STRIDE):
        j = i + STRIDE
        p0 = cv2.goodFeaturesToTrack(imgs[i], 400, 0.01, 8)
        if p0 is None or len(p0) < 40:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(imgs[i], imgs[j], p0, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 40:
            continue
        a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
        E, m = cv2.findEssentialMat(a, b, K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3) or m.sum() < 30:
            continue
        _, Rr, _, _ = cv2.recoverPose(E, a, b, K, mask=m.copy())
        pairs.append((i, j, -R.from_matrix(Rr).as_rotvec()))      # body rotation of the camera from frame i to frame j
    return pairs


def main():
    d = sys.argv[1]
    sync = sys.argv[2] if len(sys.argv) > 2 else "data/imu_sync.json"
    c = json.load(open(f"{d}/calib.json"))["value0"]["intrinsics"][0]["intrinsics"]
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])
    imu = np.loadtxt(f"{d}/mav0/imu0/data.csv", delimiter=",", comments="#")
    ti, w = imu[:, 0] * 1e-9, imu[:, 1:4]
    dt = np.gradient(ti)
    cam = [l.strip().split(",") for l in open(f"{d}/mav0/cam0/data.csv") if not l.startswith("#")]
    tc = np.array([int(a) * 1e-9 for a, _ in cam])
    imgs = [cv2.imread(f"{d}/mav0/cam0/data/{fn}", 0) for _, fn in cam]
    pairs = measure_pairs(imgs, K)
    vis = np.array([p[2] for p in pairs])
    mag = np.linalg.norm(vis, axis=1)
    keep = mag < np.radians(10)
    print(f"{len(pairs)} image pairs; image rotation per pair: median {np.degrees(np.median(mag)):.2f} deg, "
          f"{(~keep).sum()} discarded over 10 deg")

    def gyro_all(delta):
        return [gyro_rotvec(ti, w, dt, tc[i] + delta, tc[j] + delta) for i, j, _ in pairs]

    best = None
    print("\nshift of the camera stamps (ms)   correlation of rotation size, image vs gyro")
    for delta in np.arange(-0.12, 0.121, 0.01):
        g = gyro_all(delta)
        ok = np.array([x is not None for x in g]) & keep
        gm = np.array([np.linalg.norm(x) if x is not None else 0.0 for x in g])
        cc = np.corrcoef(mag[ok], gm[ok])[0, 1]
        print(f"   {delta * 1000:6.0f}                           {cc:6.3f}")
        if best is None or cc > best[0]:
            best = (cc, delta)
    cc, delta = best
    print(f"\nbest shift {delta * 1000:.0f} ms, size correlation {cc:.3f}  (1.0 is perfect; below about 0.7 the data cannot fix the delay)")
    # delay again, now with the axes fitted at each candidate (the vector error is more sensitive than rotation size)
    cand = []
    for dl in np.arange(-0.06, 0.061, 0.01):
        g = gyro_all(dl)
        idx = [k for k, x in enumerate(g) if x is not None and keep[k] and mag[k] > np.radians(1.0)]
        Rf, inl, err = fit_axes(vis[idx], np.array([g[k] for k in idx]))
        cand.append((float(np.median(err[inl])), dl, Rf, int(inl.sum()), len(idx)))
    e, delta, Rf, n_in, n_all = min(cand, key=lambda c: c[0])
    flat = [c for c in cand if c[0] < e + 0.01]
    print(f"axes fitted on {n_in} of {n_all} pairs; median relative error {e:.2f} (0 = perfect, 1 = useless)")
    print(f"delay: best camera-stamp shift {delta * 1000:.0f} ms, but every shift from {min(c[1] for c in flat) * 1000:.0f} to "
          f"{max(c[1] for c in flat) * 1000:.0f} ms fits about as well: the delay is only known to that range")
    print("R_ic from images + gyro (camera axes to IMU axes):\n", np.round(Rf, 3))
    cur = np.array(json.load(open(sync))["R_ci"]).T
    print(f"R_ic in {sync}:\n", np.round(cur, 3))
    print(f"angle between them: {np.degrees(R.from_matrix(Rf @ cur.T).magnitude()):.1f} deg  (under about 3 deg is fine for Basalt)")
    if "--write" in sys.argv:
        import os
        import shutil
        import time
        old = json.load(open(sync))
        os.makedirs("data/backup", exist_ok=True)
        bak = f"data/backup/imu_sync_{time.strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy(sync, bak)
        # the recording already had old['delay_s'] removed from its camera stamps; a further +shift means the true delay is smaller
        new = {"delay_s": float(old["delay_s"] - delta), "R_ci": Rf.T.tolist(), "quality": float(1 - e),
               "date": time.strftime("%Y-%m-%d %H:%M:%S"), "source": f"check_extrinsics on {d}", "delay_range_ms": [
                   float((old["delay_s"] - max(c[1] for c in flat)) * 1000), float((old["delay_s"] - min(c[1] for c in flat)) * 1000)]}
        json.dump(new, open(sync, "w"), indent=2)
        print(f"WROTE {sync} (delay {new['delay_s'] * 1000:.0f} ms, range {new['delay_range_ms'][0]:.0f} to {new['delay_range_ms'][1]:.0f} ms); old file saved as {bak}")


if __name__ == "__main__":
    main()
