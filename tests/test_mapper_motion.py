import cv2
import numpy as np

from mapper.motion_check import rotation_from_tracks, verdict

K = np.array([[600.0, 0, 320], [0, 600.0, 180], [0, 0, 1]])
D = np.zeros(5)


def _project(X, R, t):
    Xc = (X - t) @ R                                    # camera rotated by R and moved to t: world point in the new camera frame
    uv = (K @ Xc.T).T
    return uv[:, :2] / uv[:, 2:3]


def _scene(deg, outliers=0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.c_[rng.uniform(-2, 2, 200), rng.uniform(-1.5, 1.5, 200), rng.uniform(2, 6, 200)]
    R = cv2.Rodrigues(np.radians([0, deg, 0.0]))[0]
    p0, p1 = _project(X, np.eye(3), np.zeros(3)), _project(X, R, np.array([0.03, 0.0, 0.01]))
    p1 = p1 + rng.normal(0, 0.2, p1.shape)
    p1[:outliers] = rng.uniform(0, 600, (outliers, 2))
    return p0, p1, R


def test_recovers_rotation_with_outliers():
    p0, p1, R = _scene(4.0, outliers=40)
    Rk, ratio, _ = rotation_from_tracks(p0, p1, K, D)
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(Rk.T @ R)[0])) < 1.0 and 0.6 < ratio < 0.95


def test_too_few_tracks_gives_none():
    p0, p1, _ = _scene(4.0)
    assert rotation_from_tracks(p0[:10], p1[:10], K, D)[0] is None


def test_verdict_flags_odometry_when_gyro_and_klt_agree():
    R = cv2.Rodrigues(np.radians([0, 5.0, 0]))[0]
    Rbad = cv2.Rodrigues(np.radians([0, 12.0, 0]))[0]
    assert verdict(R, 0.9, R, R)["ok"]
    v = verdict(R, 0.9, R, Rbad)
    assert not v["ok"] and "odometry" in v["reason"]


def test_verdict_does_not_blame_odometry_when_the_sensors_disagree():
    R = cv2.Rodrigues(np.radians([0, 5.0, 0]))[0]
    Rg = cv2.Rodrigues(np.radians([0, 15.0, 0]))[0]
    assert verdict(R, 0.9, Rg, R)["ok"]                 # klt == odom, gyro is the odd one: odometry stays
