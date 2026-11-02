import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from gabai.net import protocol as P
from gabai.pi.klt import KltResult, MotionEstimator

F, B, CX, CY = 400.0, 0.06, 320.0, 180.0


@pytest.fixture
def calib(tmp_path):
    """A calibration whose rectification does nothing, so raw pixels are rectified pixels and the geometry can be written down."""
    K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1.0]])
    P1, P2 = np.hstack([K, np.zeros((3, 1))]), np.hstack([K, [[-F * B], [0], [0]]])
    p = tmp_path / "c.npz"
    np.savez(p, K1=K, D1=np.zeros((1, 5)), R1=np.eye(3), P1=P1, K2=K, D2=np.zeros((1, 5)), R2=np.eye(3), P2=P2, size=np.array([640, 360]), baseline=B, focal=F, cx=CX, cy=CY,
             rotate_left=0, rotate_right=0)
    return str(p)


def scene(n=120, seed=0):
    rng = np.random.default_rng(seed)
    u, v, z = rng.uniform(60, 580, n), rng.uniform(50, 310, n), rng.uniform(1.0, 4.0, n)
    return np.stack([(u - CX) * z / F, (v - CY) * z / F, z], 1)


def result(ids, X_cam, row_shift=0.0, status=None, noise=0.0, seed=1):
    """What the tracker reports for scene points seen from a camera: left pixels, right pixels (shifted left by the disparity)."""
    rng = np.random.default_rng(seed)
    ul = np.stack([F * X_cam[:, 0] / X_cam[:, 2] + CX, F * X_cam[:, 1] / X_cam[:, 2] + CY], 1) + rng.normal(0, noise, (len(X_cam), 2))
    ur = ul - np.stack([F * B / X_cam[:, 2], -np.full(len(X_cam), row_shift)], 1)
    st = np.full(len(ids), P.KLT_TRACKED, np.uint8) if status is None else status
    return KltResult(ids, ul.astype(np.float32), st, ur.astype(np.float32), 1.0, (0, 0, 0, 0), np.zeros(len(ids), np.uint8))


def test_the_pi_recovers_the_motion_between_two_pairs_from_the_stereo_tracks(calib):
    X = scene()
    ids = np.arange(len(X))
    est = MotionEstimator(calib, (640, 360))
    assert est.update(result(ids, X), 1.00, 10) is None                              # nothing to compare with yet
    Rt, tt = R.from_rotvec([0.01, 0.03, -0.02]).as_matrix(), np.array([0.04, -0.01, 0.12])     # the current camera, seen from the previous one
    X_cur = (X - tt) @ Rt                                                               # the same points in the current camera's frame
    m = est.update(result(ids, X_cur, noise=0.2), 1.07, 11)
    assert m is not None and m["prev_seq"] == 10 and abs(m["dt"] - 0.07) < 1e-9 and m["n_inl"] > 80
    assert np.linalg.norm(m["t"] - tt) < 0.01 and np.degrees(np.linalg.norm(R.from_rotvec(m["rvec"]).inv().as_matrix() @ Rt - np.eye(3))) < 0.5 and m["rms"] < 1.0


def test_lost_points_and_a_camera_order_that_does_not_match_the_calibration_give_no_motion(calib):
    X = scene()
    ids = np.arange(len(X))
    est = MotionEstimator(calib, (640, 360))
    est.update(result(ids, X, row_shift=6.0), 1.0, 1)                                 # the right rows are 6 px off: not a rectified pair
    assert est.prev is None and est.update(result(ids, X, row_shift=6.0), 1.07, 2) is None
    est = MotionEstimator(calib, (640, 360))
    est.update(result(ids, X), 1.0, 1)
    lost = np.full(len(ids), P.KLT_LOST, np.uint8)
    assert est.update(result(ids, X, status=lost), 1.07, 2) is None                   # every track lost: nothing to estimate from
