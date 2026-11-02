import os
import tempfile
import time

import numpy as np

from gabai.cloud import VoxelMap
from gabai.relocalize import apply_correction, load_map, relocalize, save_map


def box_surface(lo, hi, n, rng):
    lo, hi = np.array(lo, float), np.array(hi, float)
    pts = rng.uniform(lo, hi, (n, 3))
    face = rng.integers(0, 6, n)
    ax, side = face // 2, face % 2
    pts[np.arange(n), ax] = np.where(side == 0, lo[ax], hi[ax])
    return pts


def room(rng):
    """An asymmetric room: floor, three walls of different lengths, and furniture of different sizes."""
    floor = np.c_[rng.uniform(0, 8, 6000), rng.uniform(0, 5, 6000), np.zeros(6000)]
    w1 = np.c_[np.zeros(3000), rng.uniform(0, 5, 3000), rng.uniform(0, 2.5, 3000)]
    w2 = np.c_[rng.uniform(0, 8, 4000), np.zeros(4000), rng.uniform(0, 2.5, 4000)]
    w3 = np.c_[np.full(2000, 8.0), rng.uniform(0, 3, 2000), rng.uniform(0, 2.5, 2000)]
    sofa = box_surface((1, 3.5, 0), (3.2, 4.5, 0.8), 2500, rng)
    table = box_surface((4.5, 1.0, 0), (5.5, 2.4, 0.75), 2000, rng)
    shelf = box_surface((7.4, 3.2, 0), (7.9, 4.9, 1.8), 1500, rng)
    return np.vstack([floor, w1, w2, w3, sofa, table, shelf])


def rotz(deg):
    a = np.radians(deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return T


def fused(pts):
    vm = VoxelMap(0.05)
    for _ in range(4):
        vm.integrate(pts)
    return vm


def local_view(map_pts, center, radius, T_world_from_live, rng, noise=0.01):
    """What the rig sees from `center`, written in the (drifted) live frame: the inverse of T applied to the nearby map points."""
    near = map_pts[np.linalg.norm(map_pts - center, axis=1) < radius]
    near = near + rng.normal(0, noise, near.shape)
    inv = np.linalg.inv(T_world_from_live)
    return near @ inv[:3, :3].T + inv[:3, 3]


def pose_error(T, T_true):
    d = np.linalg.inv(T_true) @ T
    return float(np.linalg.norm(d[:3, 3])), float(np.degrees(abs(np.arctan2(d[1, 0], d[0, 0]))))


def test_recovers_a_known_drift_from_several_start_spots_in_under_five_seconds():
    rng = np.random.default_rng(1)
    pts = room(rng)
    vm = fused(pts)
    mp = vm.points(min_obs=1)
    for center, yaw, shift in [((2.0, 2.0, 1.0), 25, (0.4, -0.3, 0)), ((6.0, 1.5, 1.0), -40, (-0.5, 0.2, 0)), ((4.0, 4.0, 1.0), 90, (0.2, 0.6, 0)),
                               ((1.5, 1.0, 1.0), 160, (0.0, 0.0, 0)), ((6.5, 4.0, 1.0), -120, (0.3, 0.3, 0))]:
        T_true = rotz(yaw)
        T_true[:3, 3] = shift
        live = local_view(mp, np.array(center), 4.0, T_true, rng)
        t0 = time.time()
        r = relocalize(mp, live)
        dt = time.time() - t0
        assert r["ok"], (center, r["reason"])
        e_m, e_deg = pose_error(r["T"], T_true)
        assert e_m < 0.08 and e_deg < 2.0, (center, e_m, e_deg)
        assert dt < 5.0, dt


def test_a_view_of_a_different_place_is_rejected_with_a_reason_not_forced():
    rng = np.random.default_rng(2)
    mp = fused(room(rng)).points(min_obs=1)
    other = np.c_[rng.uniform(0, 3, 3000), rng.uniform(0, 3, 3000), rng.uniform(0, 3, 3000)]    # a random blob
    r = relocalize(mp, other)
    assert not r["ok"] and r["reason"] and r["fitness"] < 0.9


def test_a_symmetric_scene_is_reported_as_ambiguous():
    rng = np.random.default_rng(3)
    floor = np.c_[rng.uniform(-3, 3, 6000), rng.uniform(-3, 3, 6000), np.zeros(6000)]
    walls = np.vstack([np.c_[np.full(2000, s * 3.0), rng.uniform(-3, 3, 2000), rng.uniform(0, 2, 2000)] for s in (-1, 1)] +
                      [np.c_[rng.uniform(-3, 3, 2000), np.full(2000, s * 3.0), rng.uniform(0, 2, 2000)] for s in (-1, 1)])
    mp = fused(np.vstack([floor, walls])).points(min_obs=1)
    live = mp[np.linalg.norm(mp[:, :2], axis=1) < 2.5]
    r = relocalize(mp, live)
    assert (not r["ok"]) and r["ambiguous"] and "ambiguous" in r["reason"]


def test_too_few_points_is_a_clear_failure():
    r = relocalize(np.random.rand(500, 3), np.random.rand(10, 3))
    assert not r["ok"] and "points" in r["reason"]


def test_saved_map_round_trips_with_its_metadata():
    rng = np.random.default_rng(4)
    vm = fused(room(rng))
    d = tempfile.mkdtemp()
    p = os.path.join(d, "home.npz")
    save_map(p, vm, meta={"name": "home", "built_from": "test", "frame": "gravity-aligned, z up"})
    vm2, meta = load_map(p)
    assert len(vm2) == len(vm) and np.allclose(vm2.points(1), vm.points(1)) and meta["name"] == "home" and meta["voxel"] == 0.05


def test_correction_moves_a_pose_into_the_map_frame():
    T_corr = rotz(30)
    T_corr[:3, 3] = [1, 2, 0]
    pose = np.eye(4)
    pose[:3, 3] = [1, 0, 0]
    out = apply_correction(T_corr, pose)
    assert np.allclose(out[:3, 3], T_corr[:3, :3] @ [1, 0, 0] + [1, 2, 0])
