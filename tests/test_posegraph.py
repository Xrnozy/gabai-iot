import time

import numpy as np
from scipy.spatial.transform import Rotation

from gabai import posegraph as pg


def loop_truth(n=60, radius=1.0):
    """Camera poses around a circle, heading along the circle."""
    R, t = [], []
    for k in range(n):
        a = 2 * np.pi * k / n
        t.append([radius * np.cos(a), radius * np.sin(a), 0.0])
        R.append(Rotation.from_euler("z", a + np.pi / 2).as_matrix())
    return np.array(R), np.array(t)


def drifting_odometry(R, t, rot_bias_deg=0.4, scale=1.01):
    """Integrate the true relative motions with a bias: a yaw error and a scale error per step, as drift does."""
    Ro, to = [R[0]], [t[0]]
    bias = Rotation.from_euler("z", np.radians(rot_bias_deg)).as_matrix()
    for k in range(1, len(R)):
        rel_R, rel_t = R[k - 1].T @ R[k], R[k - 1].T @ (t[k] - t[k - 1])
        Ro.append(Ro[-1] @ rel_R @ bias)
        to.append(to[-1] + Ro[-2] @ (rel_t * scale))
    return np.array(Ro), np.array(to)


def edges_from(R, t, loops=()):
    e = [pg.odometry_edge(k, k + 1, R[k], t[k], R[k + 1], t[k + 1]) for k in range(len(R) - 1)]
    return e + list(loops)


def ate(t_est, t_true):
    return float(np.sqrt(np.mean(np.sum((t_est - t_true) ** 2, axis=1))))


def test_without_a_loop_edge_the_optimiser_changes_nothing_the_odometry_chain_is_already_consistent():
    R, t = loop_truth()
    Ro, to = drifting_odometry(R, t)
    R2, t2, info = pg.optimize(Ro, to, edges_from(Ro, to))
    assert np.allclose(t2, to, atol=1e-6) and info["max_shift_m"] < 1e-5


def test_a_loop_edge_closes_the_path_and_shares_the_error_along_it():
    R, t = loop_truth()
    Ro, to = drifting_odometry(R, t)
    gap = np.linalg.norm(to[-1] - t[-1])
    assert gap > 0.1                                                          # the drifted path really does not close
    # a loop edge from the last node back to node 0: the truth says node 59 is one step before node 0 on the circle
    R_l, t_l = R[0].T @ R[-1], R[0].T @ (t[-1] - t[0])
    R2, t2, info = pg.optimize(Ro, to, edges_from(Ro, to, [pg.loop_edge(0, len(R) - 1, R_l, t_l)]))
    assert ate(t2, t) < 0.4 * ate(to, t), (ate(t2, t), ate(to, t))
    assert np.linalg.norm(t2[-1] - t[-1]) < 0.3 * gap
    mid = len(R) // 2
    assert np.linalg.norm(t2[mid] - to[mid]) > 0.01                          # the middle of the path moved too, not only the end
    assert info["cost_after"] < info["cost_before"]


def test_node_zero_is_the_anchor_and_does_not_move():
    R, t = loop_truth()
    Ro, to = drifting_odometry(R, t)
    R_l, t_l = R[0].T @ R[-1], R[0].T @ (t[-1] - t[0])
    R2, t2, _ = pg.optimize(Ro, to, edges_from(Ro, to, [pg.loop_edge(0, len(R) - 1, R_l, t_l)]))
    assert np.allclose(t2[0], to[0]) and np.allclose(R2[0], Ro[0])


def test_it_stays_fast_for_a_few_hundred_keyframes():
    R, t = loop_truth(n=300, radius=3.0)
    Ro, to = drifting_odometry(R, t, 0.1, 1.003)
    R_l, t_l = R[0].T @ R[-1], R[0].T @ (t[-1] - t[0])
    edges = edges_from(Ro, to, [pg.loop_edge(0, len(R) - 1, R_l, t_l)])
    t0 = time.perf_counter()
    R2, t2, info = pg.optimize(Ro, to, edges)
    ms = (time.perf_counter() - t0) * 1000
    assert ms < 3000 and ate(t2, t) < 0.2 * ate(to, t), (ms, info)
    print(f"300 keyframes: {ms:.0f} ms, {info['iterations']} evaluations")


def test_an_empty_graph_or_no_free_node_returns_the_input():
    R, t = loop_truth(n=5)
    R2, t2, info = pg.optimize(R, t, [])
    assert np.array_equal(t2, t) and info["edges"] == 0
    R3, t3, _ = pg.optimize(R[:1], t[:1], [pg.loop_edge(0, 0, np.eye(3), np.zeros(3))])
    assert np.array_equal(t3, t[:1])
