"""Over a stall, does predicting the camera's translation from the IMU beat holding still? Measured against the pictures' own poses at full rate.

py scripts/stall_prediction_check.py poses.npz     (poses.npz: a scripts/trackeval.py --stutter none --save run; the recording's imu.csv is read from the folder below)

For frame pairs i -> j that are dt seconds apart (0.3 ... 3 s, what a stall of the Pi looks like) three predictions of the displacement t_j - t_i are compared with what the pictures say:
  hold        no motion
  const vel   the speed seen just before frame i, kept for dt
  IMU         that speed plus the accelerometer, gravity removed with the orientation turned by the gyro from frame i on (what a loosely coupled filter could do)
  IMU (bias removed)  the same with the accelerometer bias taken from the quietest second (best case: the bias is known)
The pictures' poses are not ground truth (about 1-2 cm over a short gap), so differences below that are noise.
"""
import csv
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, ".")
from gabai.slam.euroc import load_imu_extrinsics

FOLDER = "data/recordings/vio_20261101_014854"
G = 9.81
z = np.load(sys.argv[1])
idx, ok, Rs, ts = z["idx"], z["ok"], z["R"], z["t"]
tc = np.array([float(r["t_capture"]) for r in csv.DictReader(open(FOLDER + "/pairs.csv"))])
imu = np.loadtxt(FOLDER + "/imu.csv", delimiter=",", skiprows=1)
Ric = load_imu_extrinsics("data/imu_sync_corrected.json")["R_ic"]
tt = imu[:, 0]
w_c = np.radians(imu[:, 1:4]) @ Ric                                  # camera axes
f_c = imu[:, 4:7] * G @ Ric                                          # specific force, camera axes (reads +up at rest)
sel = np.where(ok)[0]
T = tc[idx[sel]]
P, Rm = ts[sel], Rs[sel]
# accelerometer bias from the quietest second: at rest f_w should be (0, 0, +G)
win = int(1.0 / np.median(np.diff(tt)))
quiet = min(range(0, len(tt) - win, win // 4), key=lambda k: np.linalg.norm(w_c[k:k + win].mean(0)) + np.linalg.norm(w_c[k:k + win].std(0)))
qa = np.searchsorted(T, tt[quiet])
Rq = Rm[min(qa, len(Rm) - 1)]
bias_c = f_c[quiet:quiet + win].mean(0) - Rq.T @ np.array([0, 0, G])


def predict(i, j, bias):
    """Displacement of the camera from frame i to j with orientation turned by the gyro and the accelerometer integrated twice, start speed from the pictures."""
    k = max(i - 3, 0)
    v0 = (P[i] - P[k]) / max(T[i] - T[k], 1e-3)
    a, b = np.searchsorted(tt, T[i]), np.searchsorted(tt, T[j])
    r, v, d = R.from_matrix(Rm[i]), v0.copy(), np.zeros(3)
    for n in range(a, min(b, len(tt) - 1)):
        dt = tt[n + 1] - tt[n]
        r = r * R.from_rotvec(w_c[n] * dt)
        acc = r.apply(f_c[n] - bias) - np.array([0, 0, G])
        d += v * dt + 0.5 * acc * dt * dt
        v = v + acc * dt
    return v0, d


print(f"accelerometer bias from the quietest second (camera axes, m/s2): {np.round(bias_c, 2)}")
print(f"{'dt s':>5} {'pairs':>6} | median error cm:  {'hold':>6} {'const vel':>10} {'IMU':>6} {'IMU (bias removed)':>20} | p90:  {'hold':>6} {'const vel':>10} {'IMU':>6} {'IMU (bias removed)':>20}   median true motion cm")
for dt_s in (0.3, 0.7, 1.4, 2.8):
    rows = []
    for i in range(3, len(T) - 1, 7):
        j = int(np.searchsorted(T, T[i] + dt_s))
        if j >= len(T) or abs(T[j] - T[i] - dt_s) > 0.06:
            continue
        true = P[j] - P[i]
        v0, d_imu = predict(i, j, np.zeros(3))
        _, d_bias = predict(i, j, bias_c)
        rows.append([np.linalg.norm(true), np.linalg.norm(true), np.linalg.norm(true - v0 * (T[j] - T[i])), np.linalg.norm(true - d_imu), np.linalg.norm(true - d_bias)])
    if rows:
        r = np.array(rows) * 100
        print(f"{dt_s:5.1f} {len(r):6d} | {'':18}{np.median(r[:, 1]):6.1f} {np.median(r[:, 2]):10.1f} {np.median(r[:, 3]):6.1f} {np.median(r[:, 4]):20.1f} | {'':6}{np.percentile(r[:, 1], 90):6.1f} {np.percentile(r[:, 2], 90):10.1f} {np.percentile(r[:, 3], 90):6.1f} {np.percentile(r[:, 4], 90):20.1f}   {np.median(r[:, 0]):.1f}")
