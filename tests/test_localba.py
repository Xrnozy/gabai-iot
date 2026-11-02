import time

import numpy as np
from scipy.spatial.transform import Rotation

from gabai import localba

F, B, CX, CY = 600.0, 0.065, 320.0, 240.0
K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1.0]])


def scene(n_nodes=5, n_lm=200, seed=0):
    """Landmarks on a wall 2.5 m away; keyframes sliding 0.25 m sideways each, looking at it."""
    r = np.random.default_rng(seed)
    X = np.c_[r.uniform(-1.2, 1.2, n_lm), r.uniform(-0.8, 0.8, n_lm), r.uniform(2.3, 2.7, n_lm)]
    R = np.stack([np.eye(3)] * n_nodes)
    t = np.array([[0.25 * k, 0.0, 0.0] for k in range(n_nodes)])
    return X, R, t


def observe(X, R, t, seed=1, px=0.5, dnoise=0.0, dbias=None):
    r = np.random.default_rng(seed)
    on, ol, ouv, od = [], [], [], []
    for k in range(len(R)):
        Xc = (X - t[k]) @ R[k]
        uv = np.stack([F * Xc[:, 0] / Xc[:, 2] + CX, F * Xc[:, 1] / Xc[:, 2] + CY], 1)
        inside = (uv[:, 0] > 0) & (uv[:, 0] < 640) & (uv[:, 1] > 0) & (uv[:, 1] < 480)
        for i in np.nonzero(inside)[0]:
            d = F * B / Xc[i, 2] + r.normal(0, dnoise) + (0 if dbias is None else dbias[i])
            on.append(k), ol.append(i), ouv.append(uv[i] + r.normal(0, px, 2)), od.append(d)
    return np.array(on), np.array(ol), np.array(ouv), np.array(od)


def run(X0, R, t, obs, n_fixed=2, **kw):
    return localba.optimize(K, B, R, t, n_fixed, *obs, X0, **kw)


def test_perturbed_poses_and_landmarks_come_back_to_the_truth_with_two_poses_held():
    X, R, t = scene()
    obs = observe(X, R, t)
    rng = np.random.default_rng(5)
    t_bad = t + np.r_[np.zeros((2, 3)), rng.normal(0, 0.03, (3, 3))]
    X_bad = X + rng.normal(0, 0.05, X.shape)
    R2, t2, X2, info = run(X_bad, R, t_bad, obs)
    assert np.linalg.norm(t2[2:] - t[2:], axis=1).max() < 0.01, np.linalg.norm(t2[2:] - t[2:], axis=1)
    assert np.linalg.norm(X2 - X, axis=1).mean() < 0.4 * np.linalg.norm(X_bad - X, axis=1).mean()
    assert info["cost_after"] < 0.2 * info["cost_before"]


def test_the_held_poses_do_not_move():
    X, R, t = scene()
    obs = observe(X, R, t)
    R2, t2, X2, _ = run(X + 0.02, R, t + 0.01, obs)
    assert np.allclose(t2[:2], t[:2] + 0.01) and np.allclose(R2[:2], R[:2])


def test_parallax_beats_a_persistent_depth_error_that_repeated_stereo_cannot_remove():
    """Every keyframe reports the same wrong disparity for a landmark (a persistent stereo error). The landmark depth the window finds must be closer to the truth than the stereo depth alone."""
    X, R, t = scene(n_nodes=6, n_lm=250)
    bias = np.random.default_rng(7).normal(0, 0.7, len(X))                    # px of disparity, the same in every keyframe
    obs = observe(X, R, t, dbias=bias)
    z_stereo = F * B / (F * B / X[:, 2] + bias)
    X_init = X.copy()
    X_init[:, 2] = z_stereo                                                    # start from the stereo depth (lateral position as seen)
    X_init[:, 0] = X[:, 0] * z_stereo / X[:, 2]
    X_init[:, 1] = X[:, 1] * z_stereo / X[:, 2]
    R2, t2, X2, info = run(X_init, R, t, obs, n_fixed=2)
    err_before = np.abs(X_init[:, 2] - X[:, 2]).mean()
    err_after = np.abs(X2[:, 2] - X[:, 2]).mean()
    assert err_after < 0.6 * err_before, (err_before, err_after)


def test_a_wrong_track_does_not_drag_the_pose_much():
    X, R, t = scene()
    on, ol, ouv, od = observe(X, R, t)
    bad = np.arange(0, len(on), 25)                                           # 4 percent of the observations are tracked to the wrong place
    ouv[bad] += 40.0
    R2, t2, X2, _ = run(X, R, t + np.r_[np.zeros((2, 3)), np.full((3, 3), 0.02)], (on, ol, ouv, od))
    assert np.linalg.norm(t2[2:] - t[2:], axis=1).max() < 0.02


def test_no_observations_or_nothing_free_returns_the_input():
    X, R, t = scene()
    R2, t2, X2, info = localba.optimize(K, B, R, t, len(R), np.array([], int), np.array([], int), np.zeros((0, 2)), np.zeros(0), X)
    assert np.array_equal(t2, t) and info["nfev"] == 0


def test_a_window_of_six_keyframes_and_300_landmarks_is_fast_enough_for_a_worker_thread():
    X, R, t = scene(n_nodes=6, n_lm=300)
    obs = observe(X, R, t)
    t0 = time.perf_counter()
    run(X + 0.03, R, t + np.r_[np.zeros((2, 3)), np.full((4, 3), 0.02)], obs)
    ms = (time.perf_counter() - t0) * 1000
    print(f"window BA: {ms:.0f} ms for {len(obs[0])} observations")
    assert ms < 3000


def test_the_tracker_with_window_ba_on_runs_applies_adjustments_and_does_not_do_worse_than_without():
    from gabai.kfvo import KeyframeVO
    from gabai.tools import vo_bench as vb
    from gabai.tools.slam_bench import run_laps
    res = {}
    for name, kw in (("off", {}), ("on", dict(window_ba=True, ba_every=3, ba_window=6, ba_max_landmarks=250))):
        def mk(kw=kw):
            v = KeyframeVO(vb.F, vb.F, vb.W / 2, vb.H / 2, **kw)
            v.set_baseline(vb.B)
            return v
        res[name] = np.mean([run_laps(mk, 150, 2, 0.4, s)["online_rms"] for s in range(1)])
    assert res["on"] <= res["off"] * 1.1, res
