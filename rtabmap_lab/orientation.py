"""Which way up are the pictures RTAB-Map is given? (an upside-down picture gives an upside-down map: see dashboard.py)

RTAB-Map has no gravity and is not given any. This module uses the accelerometer as an OUTSIDE JUDGE only, once, at the start: it measures where 'up' is in the camera's own axes
while the rig is roughly still, and compares it with the 'up' RTAB-Map assumes (the camera level and upright: minus the optical y axis). From the accelerometer nothing reaches RTAB-Map except
the decision to turn the pictures 180 deg or not. The gyro's rotation between pictures (gyro_rotvec) does: the bridge uses it to keep the heading when the odometry is lost.

It relies on data/imu_sync_corrected.json (the camera-IMU rotation for the pictures as the pipeline produces them). If the verdict contradicts what you see in the camera panel,
trust your eyes and pass --orientation as-is or --orientation flip.
"""
import json
import os

import numpy as np

ASSUMED_UP = np.array([0.0, -1.0, 0.0])
D = np.diag([-1.0, -1.0, 1.0])                         # the camera turned 180 deg about its optical axis


def load_R_ic(path="data/imu_sync_corrected.json"):
    if not os.path.exists(path):
        return None
    return np.array(json.load(open(path))["R_ci"], float).T


def gyro_rotvec(t, gyro_rad, R_ic, flip, t_end):
    """Camera rotation over the gyro samples (times t, rad/s in the IMU's axes, each held until the next sample, the last until t_end), as a rotation vector in the
    optical frame of the pictures RTAB-Map gets (flip: the pictures were turned 180 deg). Checked on walk_20261102_230115: 0.9 deg median from the visual odometry."""
    import cv2
    if R_ic is None or not len(t):
        return None
    F = D if flip else np.eye(3)
    dt = np.diff(np.append(np.asarray(t, float), t_end))
    R = np.eye(3)
    for w, d in zip(np.asarray(gyro_rad, float), dt):
        R = R @ cv2.Rodrigues(F @ R_ic.T @ w * max(d, 0.0))[0]
    return cv2.Rodrigues(R)[0].ravel()


def judge(accel, R_ic):
    """accel: (n, 3) accelerometer rows in the IMU's axes (any unit; reads +up when still). Returns dict(angle_as_is, angle_flipped, decision, reason)."""
    acc = np.asarray(accel, float)
    if R_ic is None or len(acc) < 20:
        return {"decision": "as-is", "reason": "no camera-IMU rotation file or too few IMU samples to judge: pictures left as they are"}
    mean = acc.mean(axis=0)
    n = float(np.linalg.norm(mean))
    spread = float(np.linalg.norm(acc - mean, axis=1).mean() / max(n, 1e-9))
    up = mean / n
    up_c = R_ic.T @ up
    a0 = float(np.degrees(np.arccos(np.clip(up_c @ ASSUMED_UP, -1, 1))))
    a1 = 180.0 - a0
    if spread > 0.35:
        return {"angle_as_is": a0, "angle_flipped": a1, "decision": "as-is", "reason": f"the rig was moving too much to judge (acceleration spread {spread:.2f}): pictures left as they are"}
    if a0 < 70:
        return {"angle_as_is": a0, "angle_flipped": a1, "decision": "as-is", "reason": f"pictures are upright ({a0:.0f} deg from level-up)"}
    if a1 < 70:
        return {"angle_as_is": a0, "angle_flipped": a1, "decision": "flip", "reason": f"pictures are upside down ({a0:.0f} deg from up): turning them 180 deg"}
    return {"angle_as_is": a0, "angle_flipped": a1, "decision": "as-is", "reason": f"the camera starts tilted {min(a0, a1):.0f} deg from level, so upright or upside down cannot be told: pictures left as they are"}


def from_recording(folder, seconds=3.0, imu_sync="data/imu_sync_corrected.json"):
    p = os.path.join(folder, "imu.csv")
    if not os.path.exists(p):
        return {"decision": "as-is", "reason": "this recording has no imu.csv: pictures left as they are"}
    imu = np.loadtxt(p, delimiter=",", skiprows=1)
    imu = imu[imu[:, 0] <= imu[0, 0] + seconds]
    return judge(imu[:, 4:7], load_R_ic(imu_sync))
