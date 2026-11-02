import numpy as np
import cv2

from gabai.tools.calibrate_stereo import calibrate_from_corners, board_points, pose_is_new

K = np.array([[420.0, 0, 320], [0, 420.0, 240], [0, 0, 1]])
D = np.array([-0.25, 0.08, 0, 0, 0.0])
SIZE = (640, 480)
T_TRUE = np.array([-0.065, 0.001, 0.0005])  # right camera is 6.5 cm to the right


def synth(n=24, noise=0.3, seed=0, cols=9, rows=6, square=0.025):
    rng = np.random.default_rng(seed)
    obj = board_points(cols, rows, square)
    L, R, O = [], [], []
    while len(O) < n:
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1), rng.uniform(0.5, 1.3)])
        pl, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        Rm, _ = cv2.Rodrigues(rvec)
        # same board seen from the right camera: X1 = X0 + T (cameras are parallel)
        pr, _ = cv2.projectPoints(obj, rvec, tvec + T_TRUE, K, D)
        pl, pr = pl.reshape(-1, 2), pr.reshape(-1, 2)
        ok = lambda p: (p[:, 0].min() > 5 and p[:, 0].max() < 635 and p[:, 1].min() > 5 and p[:, 1].max() < 475)
        if ok(pl) and ok(pr):
            L.append((pl + rng.normal(0, noise, pl.shape)).astype(np.float32).reshape(-1, 1, 2))
            R.append((pr + rng.normal(0, noise, pr.shape)).astype(np.float32).reshape(-1, 1, 2))
            O.append(obj)
    return O, L, R


def test_recovers_baseline_and_focal_from_synthetic_board():
    O, L, R = synth()
    c = calibrate_from_corners(O, L, R, SIZE)
    assert abs(c["baseline"] - 0.065) / 0.065 < 0.02, c["baseline"]
    assert abs(c["K1"][0, 0] - 420) / 420 < 0.03, c["K1"][0, 0]
    assert c["rms_stereo"] < 0.6
    assert c["epipolar_err_px"] < 0.5


def test_bad_view_is_pruned_not_trusted():
    O, L, R = synth(n=20)
    L[3] = L[3] + np.random.default_rng(5).normal(0, 12, L[3].shape).astype(np.float32)  # smeared detection
    c = calibrate_from_corners(O, L, R, SIZE)
    assert 3 not in c["used"]
    assert abs(c["baseline"] - 0.065) / 0.065 < 0.03


def test_too_few_views_refused():
    O, L, R = synth(n=6)
    try:
        calibrate_from_corners(O, L, R, SIZE)
        assert False
    except ValueError:
        pass


def test_pose_is_new_requires_movement():
    a = np.stack([np.arange(54) * 5.0 % 300 + 100, np.arange(54) * 3.0 % 200 + 100], 1).astype(np.float32)
    assert pose_is_new(a, [], 40)
    assert not pose_is_new(a + 5, [a], 40)       # barely moved
    assert pose_is_new(a + 90, [a], 40)          # moved clearly


def test_swapped_cameras_are_flagged():
    O, L, R = synth()
    assert calibrate_from_corners(O, L, R, SIZE)["swapped"] is False
    assert calibrate_from_corners(O, R, L, SIZE)["swapped"] is True   # left/right exchanged
