import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.kfvo import _t_only

K = np.array([[600.0, 0, 320], [0, 600.0, 180], [0, 0, 1]])


def scene(n=120, bad=0.6, seed=0):
    rng = np.random.default_rng(seed)
    X = np.c_[rng.uniform(-1.5, 1.5, n), rng.uniform(-1, 1, n), rng.uniform(1.5, 4, n)]
    Rm = R.from_euler("xyz", [4, -7, 3], degrees=True).as_matrix()
    t = np.array([0.15, -0.05, 0.3])
    Xc = X @ Rm.T + t
    uv = np.c_[600 * Xc[:, 0] / Xc[:, 2] + 320, 600 * Xc[:, 1] / Xc[:, 2] + 180] + rng.normal(0, 0.5, (n, 2))
    wrong = rng.random(n) < bad
    uv[wrong] += rng.uniform(-60, 60, (wrong.sum(), 2))                    # tracks that followed something else
    return X, uv, Rm, t, wrong


def test_translation_with_a_known_rotation_survives_sixty_percent_wrong_tracks():
    X, uv, Rm, t, wrong = scene()
    est, inl = _t_only(X, uv, K, Rm)
    assert np.linalg.norm(est - t) < 0.03 and len(inl) >= 0.8 * (~wrong).sum() and not wrong[inl].mean() > 0.1


def test_a_wrong_rotation_is_not_hidden_by_the_translation_search():
    X, uv, Rm, t, _ = scene(bad=0.0)
    Rwrong = R.from_euler("xyz", [0, 25, 0], degrees=True).as_matrix() @ Rm                # 25 deg off: nothing fits
    res = _t_only(X, uv, K, Rwrong)
    assert res is None or len(res[1]) < 0.5 * len(X)
