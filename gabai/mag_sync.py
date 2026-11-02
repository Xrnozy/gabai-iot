"""Line the compass up with the gyro: how late its time stamps are, and how its axes map onto the IMU's axes.

Idea: when a rigid body rotates with angular velocity w (in its own axes) inside a fixed magnetic field, the field
vector it measures changes as  dm/dt = -w x m. The gyro gives w, the compass gives m. So the right axis map and the
right time shift are the ones that make  dm/dt + w x m  vanish. No knowledge of where north is is needed, and no
camera. The compass must be calibrated first (hard/soft iron removed, see gabai.mag.calibrate).
"""
import itertools

import numpy as np


def rotations_24():
    """All 24 proper rotations that map each axis onto an axis (signed permutation matrices with determinant +1)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i, p] = s
            if abs(np.linalg.det(M) - 1) < 1e-9:
                out.append(M)
    return out


def _cross(a, b):
    return np.cross(a, b)


def _smooth(x, n):
    k = np.ones(n) / n
    return np.stack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])], axis=1)


def _prep(t_mag, mag, t_imu, gyro_rad, delay, win=3):
    """Smoothed field, its derivative, and the gyro rate at the same (delay-corrected) times."""
    m = _smooth(mag, win)
    dt = np.gradient(t_mag)
    dm = np.gradient(m, axis=0) / dt[:, None]
    dm = _smooth(dm, win)
    keep = slice(win + 1, -(win + 1))
    t = t_mag[keep]
    w = np.stack([np.interp(t - delay, t_imu, gyro_rad[:, i]) for i in range(3)], axis=1)
    return m[keep], dm[keep], w


def estimate_mag_alignment(t_mag, mag, t_imu, gyro_rad, delay):
    """Best axis map for a given delay. Returns (R_im, score): imu_vector = R_im @ mag_vector, and a score in
    (-inf, 1] = fraction of the field's rate of change explained by the rotation (1 = perfect)."""
    m, dm, w = _prep(np.asarray(t_mag), np.asarray(mag, float), np.asarray(t_imu), np.asarray(gyro_rad, float), delay)
    base = float(np.sum(dm ** 2))
    best = (None, -np.inf)
    for R in rotations_24():
        mi, dmi = m @ R.T, dm @ R.T
        resid = dmi + _cross(w, mi)
        score = 1.0 - float(np.sum(resid ** 2)) / base
        if score > best[1]:
            best = (R, score)
    return best


def estimate_mag_delay(t_mag, mag, t_imu, gyro_rad, lags=None):
    """Delay (s) of the compass's time stamps, the axis map, and the score at that delay.
    Positive delay = the compass stamp is later than the field it reports."""
    lags = np.arange(-0.10, 0.2501, 0.005) if lags is None else lags
    best = (0.0, None, -np.inf)
    for lag in lags:
        R, s = estimate_mag_alignment(t_mag, mag, t_imu, gyro_rad, lag)
        if s > best[2]:
            best = (float(lag), R, s)
    return best
