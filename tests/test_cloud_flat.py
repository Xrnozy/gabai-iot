import numpy as np

from mapper.cloud import PointMap
from mapper.cloud_flat import flatten_cloud


def wall(n, rng, noise):
    """n points on the plane x = 2 (a 3 m x 2 m wall), noise m thick."""
    return np.stack([2 + noise * rng.standard_normal(n), rng.uniform(-1.5, 1.5, n), rng.uniform(0, 2, n)], 1).astype(np.float32)


def test_thick_wall_becomes_a_thin_combined_sheet_and_the_clutter_stays():
    rng = np.random.default_rng(0)
    W = np.concatenate([wall(150000, rng, 0.02), wall(30000, rng, 0.01) + [0.1, 0, 0]])            # the wall and a second copy of it 10 cm off (inconsistent poses)
    clutter = (rng.uniform(0, 1, (500, 3)) * [0.3, 0.3, 0.3] + [1.0, 0.2, 0.5]).astype(np.float32)   # a small object well in front of the wall
    P = np.concatenate([W, clutter])
    n = len(P)
    out = flatten_cloud(P, np.full((n, 3), 128, np.uint8), np.full(n, 0.8, np.float32), np.full(n, 3, np.int32), np.zeros(n, np.int32), cell=0.01)
    Pf = out[0]
    assert all(len(a) == len(Pf) for a in out)
    on_wall = Pf[np.abs(Pf[:, 0] - 2.0) < 0.15]
    assert len(Pf) < n / 2                                                  # combined: far fewer points
    assert np.mean(np.abs(on_wall[:, 0] - 2.0) < 0.005) > 0.9              # flat: the slab and the stacked copy became one sheet (a few edge points of the copy may stay)
    assert (np.abs(Pf[:, 0] - 1.15) < 0.2).sum() >= 400                     # the object in front is untouched


def test_too_little_to_fit_a_plane_is_returned_unchanged():
    P = np.random.default_rng(1).uniform(0, 1, (100, 3)).astype(np.float32)
    out = flatten_cloud(P, np.zeros((100, 3), np.uint8), np.ones(100, np.float32), np.ones(100, np.int32), np.zeros(100, np.int32))
    assert out[0] is P


def test_point_map_reports_the_age_of_each_point():
    rng = np.random.default_rng(2)
    pm = PointMap((600, 600, 320, 180), min_views=1, device="cpu")
    for i in range(3):
        d = np.full((360, 640), 2.0, np.float32)
        pm.add_node(i, d, np.ones_like(d), np.full_like(d, 0.01), (rng.random((360, 640, 3)) * 255).astype(np.uint8), conf_min=0.1)
        pm.set_poses({i: np.eye(4)})
    P, C, cf, views = pm.cloud()
    last, n = pm.age
    assert n == 3 and len(last) == len(P) and last.max() == 2
