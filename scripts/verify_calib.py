"""Independent check of the ChArUco stereo calibration: free translation (baseline, vertical offset, recess = Tz), held-out
epipolar error, and what forcing Tz=0 (no recess) costs. Usage: py scripts/verify_calib.py [pairs_dir] [square_mm]"""
import glob, json, os, sys
import cv2, numpy as np
from gabai.tools.charuco_calibrate import Board

folder = sys.argv[1] if len(sys.argv) > 1 else "data/charuco_pairs"
sq = float(sys.argv[2]) if len(sys.argv) > 2 else 22.29
rot = json.load(open(os.path.join(folder, "rotation.json")))
turn = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
board = Board(square_mm=sq, marker_mm=sq * 18 / 25)
O, L, R = [], [], []
for lf in sorted(glob.glob(os.path.join(folder, "*_L.png"))):
    im = [cv2.imread(lf, 0), cv2.imread(lf.replace("_L", "_R"), 0)]
    pass  # saved pairs are already turned; rotation.json is metadata
    d = [board.detect(i) for i in im]
    if None in d: continue
    ids = np.intersect1d(d[0][0], d[1][0])
    if len(ids) < 12: continue
    pick = lambda dd: dd[1][np.searchsorted(dd[0], ids)] if (np.diff(dd[0]) > 0).all() else dd[1][[list(dd[0]).index(i) for i in ids]]
    O.append(board.obj[ids]); L.append(pick(d[0]).reshape(-1, 1, 2)); R.append(pick(d[1]).reshape(-1, 1, 2))
size = im[0].shape[::-1]
print(f"{len(O)} usable pairs, image {size}")

def run(flags_extra, tag, idx):
    o = [O[i] for i in idx]; l = [L[i] for i in idx]; r = [R[i] for i in idx]
    fl = cv2.CALIB_ZERO_TANGENT_DIST
    _, K1, D1, _, _ = cv2.calibrateCamera(o, l, size, None, None, flags=fl)
    _, K2, D2, _, _ = cv2.calibrateCamera(o, r, size, None, None, flags=fl)
    rms, K1, D1, K2, D2, Rm, T, _, _ = cv2.stereoCalibrate(o, l, r, K1, D1, K2, D2, size, flags=cv2.CALIB_USE_INTRINSIC_GUESS | fl | flags_extra,
                                                         criteria=(3, 200, 1e-7))
    return K1, D1, K2, D2, Rm, T, rms

def epi(cal, idx):
    K1, D1, K2, D2, Rm, T, _ = cal
    R1, R2, P1, P2, *_ = cv2.stereoRectify(K1, D1, K2, D2, size, Rm, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    ey, ed = [], []
    for i in idx:
        a = cv2.undistortPoints(L[i], K1, D1, R=R1, P=P1).reshape(-1, 2); b = cv2.undistortPoints(R[i], K2, D2, R=R2, P=P2).reshape(-1, 2)
        ey += list(np.abs(a[:, 1] - b[:, 1])); ed += list(a[:, 0] - b[:, 0])
    return np.mean(ey), np.percentile(ey, 95), P1[0, 0]

n = len(O); idx = list(range(n)); held = idx[::5]; train = [i for i in idx if i not in held]
for tag, extra in (("free T (Tz measured)", 0),):
    c = run(extra, tag, idx)
    T = c[5].ravel() * 1000
    print(f"{tag}: rms {c[6]:.3f}px  T=[{T[0]:.2f} {T[1]:.2f} {T[2]:.2f}] mm  |T|={np.linalg.norm(T):.2f} mm  f={c[0][0,0]:.1f}/{c[2][0,0]:.1f}  rot(deg)={np.degrees(np.linalg.norm(cv2.Rodrigues(c[4])[0])):.3f}")
    print("   epipolar err all views mean/95pct: %.3f / %.3f px" % epi(c, idx)[:2])
    ct = run(0, "train", train); print("   held-out views (calibrated without them): %.3f / %.3f px" % epi(ct, held)[:2])
# no-recess hypothesis: force Tz = 0 by rectifying with T_z zeroed and see the epipolar penalty
K1, D1, K2, D2, Rm, T, _ = c
T0 = T.copy(); T0[2] = 0
print("   Tz forced to 0 -> epipolar err mean/95pct: %.3f / %.3f px" % epi((K1, D1, K2, D2, Rm, T0, 0), idx)[:2])
ang = float(np.degrees(np.arctan2(float(c[5][2, 0]), abs(float(c[5][0, 0])))))
print(f"   recess: right lens is {-float(T[2]):+.2f} mm along the optical axis vs left; that tilts the baseline by {ang:.2f} deg")
