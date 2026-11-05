import numpy as np

from mapper.flatten import flatten
from mapper.semantics import CLASSES

SCREEN = CLASSES.index("screen")
PERSON = CLASSES.index("person")


def _sheet(n, x, y0, y1, z0, z1, noise, rng):
    """A vertical sheet at base-x = x (a plane normal to x) covering y, z."""
    P = np.c_[np.full(n, x) + rng.normal(0, noise, n), rng.uniform(y0, y1, n), rng.uniform(z0, z1, n)]
    N = np.tile([1.0, 0, 0], (n, 1))
    return P, N


def test_two_inconsistent_sheets_become_one_and_the_heavy_one_wins():
    rng = np.random.default_rng(0)
    A, NA = _sheet(3000, 2.00, 0, 0.6, 0, 0.35, 0.004, rng)            # good measurements
    B, NB = _sheet(1500, 2.12, 0, 0.6, 0, 0.35, 0.004, rng)            # the same monitor seen through a pose 12 cm off
    P, N = np.concatenate([A, B]), np.concatenate([NA, NB])
    w = np.r_[np.full(len(A), 5.0), np.full(len(B), 1.0)]
    lab = np.full(len(P), SCREEN)
    F, keep, planes = flatten(P, N, w, lab)
    assert keep.sum() == len(A) and not keep[len(A):].any()           # the stacked copy is dropped, nothing was invented
    assert np.abs(F[keep][:, 0] - 2.0).max() < 1e-3 or F[keep][:, 0].std() < 1e-3
    assert abs(F[keep][:, 0].mean() - 2.0) < 0.005
    assert planes[0]["label"] == "screen" and planes[0]["rms_after_mm"] < 1e-3 and planes[0]["dropped"] == len(B)


def test_wall_behind_a_monitor_is_not_taken_for_its_layer():
    rng = np.random.default_rng(1)
    S, NS = _sheet(2500, 1.50, 0.2, 0.8, 0.2, 0.55, 0.004, rng)        # monitor
    Wl, NW = _sheet(8000, 2.00, -1.0, 2.0, 0, 2.0, 0.004, rng)         # wall 50 cm behind
    P, N = np.concatenate([S, Wl]), np.concatenate([NS, NW])
    lab = np.r_[np.full(len(S), SCREEN), np.zeros(len(Wl), int)]
    F, keep, planes = flatten(P, N, np.ones(len(P)), lab)
    assert keep.all()                                                  # nothing dropped: they are different surfaces
    assert {p["label"] for p in planes} == {"screen", "scene"}
    assert abs(F[:len(S), 0].mean() - 1.5) < 0.003 and abs(F[len(S):, 0].mean() - 2.0) < 0.003


def test_non_planar_objects_and_edges_are_left_alone():
    rng = np.random.default_rng(2)
    Wl, NW = _sheet(6000, 2.0, -1, 1, 0, 2, 0.004, rng)
    ang = rng.uniform(0, 2 * np.pi, 800)                               # a person-sized cylinder in front of the wall
    Pc = np.c_[1.4 + 0.12 * np.cos(ang), 0.12 * np.sin(ang), rng.uniform(0, 1.7, 800)]
    Nc = np.c_[np.cos(ang), np.sin(ang), np.zeros(800)]
    P, N = np.concatenate([Wl, Pc]), np.concatenate([NW, Nc])
    lab = np.r_[np.zeros(len(Wl), int), np.full(800, PERSON)]
    F, keep, _ = flatten(P, N, np.ones(len(P)), lab)
    assert keep.all() and np.allclose(F[len(Wl):], Pc.astype(np.float32))     # the person is untouched
