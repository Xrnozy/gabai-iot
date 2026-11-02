import numpy as np
from scipy.spatial.transform import Rotation as Rot

from gabai import posegraph as pg


def scene(n=30, drift_deg=0.6, seed=0):
    """A path in a circle that ends where it began; odometry with a yaw drift of drift_deg per step; three fixed landmarks seen from the first and the last keyframes."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi * 0.97, n)
    t_true = np.c_[2 * np.cos(ang) - 2, 2 * np.sin(ang), np.zeros(n)]
    R_true = Rot.from_euler("z", ang + np.pi / 2).as_matrix()
    X = np.array([[1.0, 0.2, 0.5], [-0.5, 1.5, 0.2], [0.3, -1.0, 0.4]])
    R_od, t_od = [R_true[0]], [t_true[0]]
    edges = []
    for k in range(n - 1):
        R_ij, t_ij = R_true[k].T @ R_true[k + 1], R_true[k].T @ (t_true[k + 1] - t_true[k])
        R_ij = R_ij @ Rot.from_euler("z", np.radians(drift_deg)).as_matrix()                 # the odometry turns a little too much every step
        edges.append((k, k + 1, R_ij, t_ij, pg.ODO_SIGMA_ROT, 0.02))
        R_od.append(R_od[-1] @ R_ij)
        t_od.append(t_od[-1] + R_od[-2] @ t_ij)
    obs = []
    for k in [0, 1, 2, n - 3, n - 2, n - 1]:
        for l in range(3):
            obs.append((k, l, R_true[k].T @ (X[l] - t_true[k]) + rng.normal(0, 0.01, 3), 0.03))
    return np.array(R_od), np.array(t_od), edges, obs, X, R_true, t_true


def test_landmark_revisit_pulls_the_drifted_end_back():
    R, t, edges, obs, X, R_true, t_true = scene()
    X0 = np.array([R_true[0] @ o[2] + t_true[0] for o in obs[:3]])                            # landmarks start from the first look
    R2, t2, info = pg.optimize(R, t, edges, lm_obs=obs, lm_x=X0)
    before, after = np.linalg.norm(t[-1] - t_true[-1]), np.linalg.norm(t2[-1] - t_true[-1])
    assert before > 0.5 and after < 0.15 * before
    assert info["lm_cm_after"] < info["lm_cm_before"] and info["landmarks"].shape == (3, 3)
    assert np.abs(info["landmarks"] - X).max() < 0.1                                           # the landmarks themselves are where they were put


def test_one_wrong_observation_is_down_weighted():
    R, t, edges, obs, X, R_true, t_true = scene()
    bad = list(obs)
    k, l, p, s = bad[-1]
    bad[-1] = (k, l, p + np.array([1.0, 0.0, 0.0]), s)                                          # a box on the wrong object, 1 m off
    X0 = np.array([R_true[0] @ o[2] + t_true[0] for o in obs[:3]])
    _, t2, info = pg.optimize(R, t, edges, lm_obs=bad, lm_x=X0)
    assert info["lm_outliers"] >= 1 and np.linalg.norm(t2[-1] - t_true[-1]) < 0.3


def test_no_landmarks_is_the_old_behaviour():
    R, t, edges, obs, X, R_true, t_true = scene()
    R2, t2, info = pg.optimize(R, t, edges)
    assert info["landmarks"] is None and np.allclose(t2, t, atol=1e-3)                          # a pure odometry chain has nothing to correct
