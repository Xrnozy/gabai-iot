import os
import tempfile

import numpy as np

from gabai.cloud import VoxelMap, depth_std, depth_weight, map_points
from gabai.mapref import MapRefiner
from gabai.relocalize import load_map, save_map


def room_points(n=6000, seed=0):
    """Floor, back wall and a side wall, sampled densely, in the world frame (z up)."""
    r = np.random.default_rng(seed)
    floor = np.c_[r.uniform(0, 4, n), r.uniform(-2, 2, n), np.zeros(n)]
    wall = np.c_[np.full(n, 4.0), r.uniform(-2, 2, n), r.uniform(0, 2.5, n)]
    side = np.c_[r.uniform(0, 4, n), np.full(n, 2.0), r.uniform(0, 2.5, n)]
    return np.vstack([floor, wall, side])


def cam_pose(x=0.5, y=0.0, z=1.2, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    fwd, right, down = np.array([c, s, 0.0]), np.array([s, -c, 0.0]), np.array([0.0, 0.0, -1.0])
    return np.stack([right, down, fwd], 1), np.array([x, y, z])


def live_points(pts_w, R, t, n=2500, seed=1, noise=0.005):
    """What the camera sees: map points in front of it, in the camera frame, with a little noise."""
    pc = (pts_w - t) @ R
    sel = (pc[:, 2] > 0.5) & (pc[:, 2] < 4.0) & (np.abs(pc[:, 0] / pc[:, 2]) < 0.7) & (np.abs(pc[:, 1] / pc[:, 2]) < 0.5)
    pc = pc[sel]
    r = np.random.default_rng(seed)
    pc = pc[r.choice(len(pc), min(n, len(pc)), replace=False)]
    return pc + r.normal(0, noise, pc.shape)


def built_map():
    vm = VoxelMap(0.05)
    for k in range(4):
        p = room_points(seed=k)
        vm.integrate(p + np.random.default_rng(k).normal(0, 0.004, p.shape))
    return vm


def test_a_pose_that_is_off_by_a_few_centimetres_is_pulled_back_to_the_map():
    vm, pts = built_map(), room_points()
    R, t = cam_pose()
    live = live_points(pts, R, t)
    bad_t = t + np.array([0.03, -0.02, 0.02])
    ref = MapRefiner(every=1)
    R2, t2, info = ref.refine(vm, R, bad_t, live, 0)
    assert info["applied"] and np.linalg.norm(t2 - t) < np.linalg.norm(bad_t - t), info        # part of the way in one step
    for k in range(1, 5):
        R2, t2, info = ref.refine(vm, R2, t2, live, k)
    assert np.linalg.norm(t2 - t) < 0.012 and np.linalg.norm(bad_t - t) > 0.035, (np.linalg.norm(t2 - t), info)


def test_a_pose_that_is_already_right_is_left_alone():
    vm, pts = built_map(), room_points()
    R, t = cam_pose(yaw=0.2)
    R2, t2, info = MapRefiner(every=1).refine(vm, R, t, live_points(pts, R, t), 0)
    assert np.linalg.norm(t2 - t) < 0.01 and np.allclose(R2, R, atol=0.01)


def test_a_correction_larger_than_the_limit_is_refused_and_the_pose_is_unchanged():
    vm, pts = built_map(), room_points()
    R, t = cam_pose()
    bad_t = t + np.array([0.0, 0.0, 0.2])
    R2, t2, info = MapRefiner(every=1, max_shift_m=0.05, inlier_m=0.3).refine(vm, R, bad_t, live_points(pts, R, t), 0)
    assert (not info["applied"]) and np.array_equal(t2, bad_t) and np.array_equal(R2, R)


def test_nothing_is_done_while_the_map_is_too_small_or_the_frame_has_too_few_points():
    ref = MapRefiner(every=1)
    R, t = cam_pose()
    _, t2, info = ref.refine(VoxelMap(0.05), R, t, np.random.rand(3000, 3) + [0, 0, 1], 0)
    assert not info["applied"] and "too small" in info["reason"] and np.array_equal(t2, t)
    _, _, info = ref.refine(built_map(), R, t, np.random.rand(10, 3), 0)
    assert not info["applied"] and "points" in info["reason"]


def test_it_only_runs_every_nth_frame():
    ref = MapRefiner(every=3)
    R, t = cam_pose()
    vm, pts = built_map(), room_points()
    live = live_points(pts, R, t)
    assert "skipped" in ref.refine(vm, R, t, live, 1)[2]["reason"]
    assert "skipped" not in ref.refine(vm, R, t, live, 3)[2]["reason"]


def test_a_flat_wall_alone_does_not_let_the_pose_wander_sideways_beyond_the_limit():
    r = np.random.default_rng(3)
    wall = np.c_[np.full(30000, 3.0), r.uniform(-3, 3, 30000), r.uniform(0, 2.5, 30000)]
    vm = VoxelMap(0.05)
    for k in range(3):
        vm.integrate(wall + r.normal(0, 0.004, wall.shape))
    R, t = cam_pose(x=0.5, z=1.2)
    start = t.copy()
    ref = MapRefiner(every=1, max_shift_m=0.06)
    for k in range(20):
        R, t, _ = ref.refine(vm, R, t, live_points(wall, R, t, seed=10 + k, noise=0.01), k)
    assert np.linalg.norm(t - start) < 0.06, np.linalg.norm(t - start)


# -- the map the refiner works against ----------------------------------------------------------------------------------------------------------------
def test_weights_pull_a_voxels_mean_towards_the_precise_points():
    vm = VoxelMap(0.1)
    vm.integrate(np.array([[0.01, 0.01, 0.01]]), np.array([9.0]))
    vm.integrate(np.array([[0.09, 0.01, 0.01]]), np.array([1.0]))
    p = vm.points(min_obs=1)[0]
    assert abs(p[0] - 0.018) < 1e-9 and vm.n[0] == 2 and abs(vm.w[0] - 10.0) < 1e-9


def test_no_weights_is_the_plain_average_as_before():
    vm = VoxelMap(0.1)
    vm.integrate(np.array([[0.01, 0.0, 0.0], [0.09, 0.0, 0.0]]))
    assert abs(vm.points(min_obs=1)[0][0] - 0.05) < 1e-9 and vm.w[0] == vm.n[0] == 2


def test_depth_std_grows_with_the_square_of_the_distance_and_weight_falls_with_it():
    f, B = 600.0, 0.065
    assert abs(depth_std(2.0, f, B) / depth_std(1.0, f, B) - 4.0) < 1e-9
    assert depth_weight(1.0) == 1.0 and depth_weight(0.5) == 1.0 and abs(depth_weight(2.0) - 1 / 16) < 1e-12


def test_points_with_a_depth_std_over_the_limit_are_dropped_and_counted():
    f, B = 600.0, 0.065
    disp = np.full((60, 80), f * B / 3.5, np.float32)
    ok = np.ones((60, 80), bool)
    p_all, _ = map_points(disp, ok, f, B, 40, 30, max_range=6.0, drop_edges=False, stride=2)
    p_cut, info = map_points(disp, ok, f, B, 40, 30, max_range=6.0, drop_edges=False, stride=2, max_std_m=0.05)
    assert len(p_all) > 0 and len(p_cut) == 0 and info["dropped_uncertain"] == len(p_all)


def test_the_saved_map_keeps_its_weights_and_an_old_file_without_them_still_loads():
    vm = VoxelMap(0.1)
    vm.integrate(np.array([[0.01, 0, 0], [0.5, 0.5, 0.5]]), np.array([2.0, 3.0]))
    d = tempfile.mkdtemp()
    save_map(os.path.join(d, "m.npz"), vm)
    vm2, _ = load_map(os.path.join(d, "m.npz"))
    assert np.allclose(vm2.w, vm.w) and np.allclose(vm2.points(1), vm.points(1))
    np.savez(os.path.join(d, "old.npz"), keys=vm.keys, sum=vm.sum, n=vm.n, meta='{"voxel": 0.1}')
    vm3, _ = load_map(os.path.join(d, "old.npz"))
    assert np.allclose(vm3.w, vm3.n)


def test_on_a_room_with_drifting_odometry_the_refiner_keeps_the_map_closer_to_the_walls():
    from gabai.kfvo import KeyframeVO
    from gabai.tools import vo_bench as vb

    def mk():
        v = KeyframeVO(vb.F, vb.F, vb.W / 2, vb.H / 2)
        v.set_baseline(vb.B)
        return vb.DriftingVO(v)
    a = vb.run_room(mk, False, n=120, seed=0, noise_disp=0.1)
    b = vb.run_room(mk, True, n=120, seed=0, noise_disp=0.1)
    assert b["corrections"] >= 1 and b["final_err_cm"] < a["final_err_cm"] and b["voxels"] <= a["voxels"], (a, b)


def corner_points(rng, n=6000):
    """Points on three perpendicular walls (a corner), 2 m on a side: enough structure to say where the camera is."""
    a, b = rng.uniform(0, 2, (2, n))
    return np.concatenate([np.stack([a, b, np.zeros(n)], 1), np.stack([np.zeros(n), a, b], 1), np.stack([a, np.zeros(n), b], 1)])[::3]


def test_recover_fits_a_pose_that_is_far_off_back_to_the_map_but_refuses_a_wrong_map():
    from gabai.cloud import VoxelMap
    from gabai.mapref import MapRefiner
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(1)
    world = corner_points(rng)
    vm = VoxelMap(0.03)
    for _ in range(3):
        vm.integrate(world + rng.normal(0, 0.003, world.shape), np.ones(len(world)))
    R_true, t_true = Rotation.from_euler("xyz", [200, 20, 30], degrees=True).as_matrix(), np.array([2.2, 1.9, 2.4])
    cam = (world - t_true) @ R_true                                  # the same points as the camera sees them
    R_bad = Rotation.from_rotvec(Rotation.from_matrix(R_true).as_rotvec() + [0.15, -0.1, 0.12]).as_matrix()       # about 14 deg off
    t_bad = t_true + [0.35, -0.25, 0.3]
    ref = MapRefiner()
    R_fix, t_fix, info = ref.recover(vm, R_bad, t_bad, cam[rng.choice(len(cam), 1500, replace=False)], n=0)
    assert info["applied"], info
    assert np.linalg.norm(t_fix - t_true) < 0.02 and np.degrees(np.arccos(np.clip((np.trace(R_fix.T @ R_true) - 1) / 2, -1, 1))) < 1.0
    other = MapRefiner()                                             # a frame of a different place: nothing fits, so nothing is changed
    R_o, t_o, info2 = other.recover(vm, R_bad, t_bad, rng.uniform(-1, 1, (1500, 3)) + [0, 0, 2.0], n=0)
    assert not info2["applied"] and np.allclose(t_o, t_bad)
