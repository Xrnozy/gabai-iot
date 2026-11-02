import sys, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
z = np.load(sys.argv[1]); calls, gray, K = z["calls"], z["gray"], z["K"]
zp = np.load(sys.argv[2]); idx, Rs, ts = zp["idx"], zp["R"], zp["t"]
pos = {int(i): k for k, i in enumerate(idx)}
orb = cv2.ORB_create(1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
feat = {int(c): orb.detectAndCompute(g, None) for c, g in zip(calls, gray)}
def rel(a, b):
    ka, da = feat[a]; kb, db = feat[b]
    if da is None or db is None or len(da) < 30 or len(db) < 30: return None
    m = [p[0] for p in bf.knnMatch(da, db, 2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(m) < 30: return None
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    E, inl = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.5)
    if E is None or inl is None or inl.sum() < 25: return None
    _, Rr, tr, _ = cv2.recoverPose(E, pa, pb, K, mask=inl)
    return Rr, int(inl.sum())
# frames far apart in time that still overlap: rotation by essential matrix vs VO
cs = [int(c) for c in calls if int(c) in pos]
import csv
sys.path.insert(0, ".")
from gabai.slam.euroc import load_imu_extrinsics
folder = "data/recordings/vio_20261101_014854"
tc = np.array([float(r["t_capture"]) for r in csv.DictReader(open(folder + "/pairs.csv"))])
imu = np.loadtxt(folder + "/imu.csv", delimiter=",", skiprows=1)
Ric = load_imu_extrinsics("data/imu_sync_corrected.json")["R_ic"]
g = imu[:, 1:4]; tt = imu[:, 0]
w = 164; kq = min(range(0, len(g) - w, 40), key=lambda k: np.linalg.norm(g[k:k+w].mean(0)) + np.linalg.norm(g[k:k+w].std(0)))
bias_q = g[kq:kq+w].mean(0)
# cumulative gyro rotation (camera axes) at every sample, for several biases
def cum(bias):
    dt = np.diff(tt, prepend=tt[0]); wc = (np.radians(g - bias) @ Ric) * dt[:, None]      # R_ic.T @ w == w @ R_ic
    out = [R.identity()]
    r = R.identity()
    for v in wc[1:]:
        r = r * R.from_rotvec(v); out.append(r)
    return out
C0, Cq = cum(np.zeros(3)), cum(bias_q)
def gyro_rel(cumr, a, b):
    ia, ib = np.searchsorted(tt, tc[a]), np.searchsorted(tt, tc[b])
    return cumr[ia].inv() * cumr[ib]
rows = []
for i, a in enumerate(cs):
    for b in cs[i + 20:]:                                  # at least ~60 frames (4 s) apart
        r = rel(a, b)
        if r is None: continue
        Rr, n = r                                          # X_b = Rr X_a
        vo = R.from_matrix(Rs[pos[a]]).inv() * R.from_matrix(Rs[pos[b]])   # R_a^T R_b = camera b in camera a
        est = R.from_matrix(Rr.T)                          # camera b in camera a from the images
        err = np.degrees((vo.inv() * est).magnitude())
        e0 = np.degrees((gyro_rel(C0, a, b).inv() * est).magnitude()); eq = np.degrees((gyro_rel(Cq, a, b).inv() * est).magnitude())
        rows.append((a, b, n, err, np.degrees(est.magnitude()), e0, eq))
rows = np.array(rows)
print("pairs with enough matches:", len(rows))
if len(rows):
    for lo, hi in ((60, 150), (150, 300), (300, 900)):
        s = rows[(rows[:, 1] - rows[:, 0] >= lo) & (rows[:, 1] - rows[:, 0] < hi)]
        if len(s): print(f"frames {lo}-{hi} apart: {len(s)} pairs, VO vs image rotation error median {np.median(s[:,3]):.1f} deg  p90 {np.percentile(s[:,3],90):.1f}  (rotation between them median {np.median(s[:,4]):.0f} deg) | gyro, no bias removed: median {np.median(s[:,5]):.1f}  gyro, quiet-window bias removed: median {np.median(s[:,6]):.1f}")
