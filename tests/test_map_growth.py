"""The map must not grow without bound. Regression tests for the 2026-10-28 runaway: on a close flat surface the map refiner's small corrections fed on
themselves (each correction moved the pose, the moved frame disagreed with the map, which asked for another correction), the pose spun away, every frame
landed in new voxels, and the map passed 500 000 voxels."""
import tempfile

import numpy as np

from gabai.cloud import VoxelMap
from gabai.kfvo import KeyframeVO
from gabai.mapref import MapRefiner
from gabai.pipeline.run import run
from gabai.tools import vo_bench as vb
from gabai.tools.slam_cloud_view import SlamCloud
from tests.test_mapref import cam_pose, live_points
from tests.test_pipeline_run import ListViz, cfg_for, make_recording, F, BASE, H, W


# -- hard cap ------------------------------------------------------------------------------------------------------------------------------------------
def test_the_cap_keeps_the_map_under_the_limit_and_drops_weak_far_voxels_first():
    rng = np.random.default_rng(0)
    vm = VoxelMap(0.05)
    solid = rng.uniform(-1, 1, (3000, 3))
    for _ in range(4):
        vm.integrate(solid)                                           # seen 4 times each: solid
    weak = rng.uniform(-6, 6, (4000, 3))
    vm.integrate(weak)                                                # seen once: noise
    n_solid = int((vm.n >= 3).sum())
    dropped = vm.cap(n_solid + 100, center=np.zeros(3))
    assert len(vm) == n_solid + 100 and dropped > 0 and int((vm.n >= 3).sum()) == n_solid, (len(vm), n_solid)
    vm.cap(n_solid - 50, center=np.zeros(3))                          # still too many: now solid voxels go, farthest first
    assert len(vm) == n_solid - 50 and vm.cap(0, np.zeros(3)) == 0


def test_a_map_far_over_the_cap_is_cut_in_one_step_and_the_weights_stay_in_step_with_the_keys():
    vm = VoxelMap(0.05)
    vm.integrate(np.random.default_rng(1).uniform(-3, 3, (20000, 3)), np.random.default_rng(2).uniform(0.1, 1, 20000))
    vm.cap(1000, np.zeros(3))
    assert len(vm) == len(vm.keys) == len(vm.sum) == len(vm.w) == len(vm.n) == 1000 and np.all(np.diff(vm.keys) > 0)


def test_slam_cloud_never_exceeds_its_cap_even_with_a_pose_that_runs_away():
    sc = SlamCloud(F, BASE, W / 2, H / 2, voxel=0.05, window_m=None, max_voxels=4000, semi_dense=False, stride=4)
    disp = np.full((H, W), F * BASE / 1.5, np.float32)
    gray = np.random.default_rng(0).integers(0, 255, (H, W), dtype=np.uint8)
    ok = np.ones((H, W), bool)
    sizes = []
    for k in range(60):
        t = np.array([0.3 * k, 0.1 * k, 0.0])                         # moves 0.3 m per frame: every frame lands in new voxels
        sc.on_pose(k, np.eye(3), t, gray, disp, ok)
        sizes.append(len(sc.map))
    assert max(sizes) <= 4000 and sc.map.evicted > 0, max(sizes)


# -- the feedback that caused it -----------------------------------------------------------------------------------------------------------------------
def test_a_close_flat_wall_is_refused_by_the_refiner_as_degenerate():
    r = np.random.default_rng(3)
    wall = np.c_[np.full(30000, 3.0), r.uniform(-3, 3, 30000), r.uniform(0, 2.5, 30000)]
    vm = VoxelMap(0.05)
    for k in range(3):
        vm.integrate(wall + r.normal(0, 0.004, wall.shape))
    R, t = cam_pose(x=2.4, z=1.2)
    ref = MapRefiner(every=1)
    for k in range(10):
        R2, t2, info = ref.refine(vm, R, t + 0.01, live_points(wall, R, t, seed=k, noise=0.01), k)
        assert not info["applied"] and np.array_equal(t2, t + 0.01)
    assert ref.stats["why"].get("degenerate", 0) == 10, ref.stats


def test_the_total_of_the_corrections_is_limited_by_the_distance_travelled():
    r = np.random.default_rng(4)
    room = np.vstack([np.c_[r.uniform(0, 4, 9000), r.uniform(-2, 2, 9000), np.zeros(9000)], np.c_[np.full(9000, 4.0), r.uniform(-2, 2, 9000), r.uniform(0, 2.5, 9000)],
                      np.c_[r.uniform(0, 4, 9000), np.full(9000, 2.0), r.uniform(0, 2.5, 9000)]])
    vm = VoxelMap(0.05)
    for k in range(3):
        vm.integrate(room + r.normal(0, 0.004, room.shape))
    R, t = cam_pose()
    ref = MapRefiner(every=1, min_gain=1.01)                          # accept every registration: only the budget can stop them
    applied_m = 0.0
    for k in range(60):
        bad = t + np.array([0.04, 0.0, 0.0])                          # the same 4 cm error offered again and again, camera standing still
        R2, t2, info = ref.refine(vm, R, bad, live_points(room, R, t, seed=k), k)
        applied_m += info.get("shift_cm", 0) / 100 if info["applied"] else 0
    assert applied_m <= 0.03 + 0.15 * ref.path_m + 0.05 and "budget" in ref.stats["why"], (applied_m, ref.stats["why"])


def test_with_the_camera_close_to_a_flat_surface_the_map_stays_small_with_the_refiner_on():
    old = vb.PLANE_Z
    vb.PLANE_Z = 0.7
    try:
        tex = vb.texture()
        sizes = {}
        for refine in (False, True):
            vo = KeyframeVO(vb.F, vb.F, vb.W / 2, vb.H / 2)
            vo.set_baseline(vb.B)
            sc = SlamCloud(vb.F, vb.B, vb.W / 2, vb.H / 2, voxel=0.05, window_m=None, stride=3, max_range=4.0, refiner=MapRefiner() if refine else None)
            rng = np.random.default_rng(0)
            ok = np.ones((vb.H, vb.W), bool)
            for k in range(120):
                t = np.array([0.004 * k, 0.0, 0.0])                   # a slow slide along the wall
                img, disp = vb.render(tex, np.eye(3), t, rng=rng)
                R_w, t_w, info = vo.step(img, disp, ok)
                if info["ok"]:
                    sc.on_pose(0, R_w, t_w, img, disp, ok)
                    if sc.refine_info and sc.refine_info.get("applied"):
                        vo.correct(*sc.pose)
            sizes[refine] = len(sc.map)
    finally:
        vb.PLANE_Z = old
    assert sizes[True] <= 1.3 * sizes[False] + 50, sizes              # the refiner must not blow the map up on a flat close scene


# -- the pipeline's own guards -------------------------------------------------------------------------------------------------------------------------
def depth_that_changes(at):
    n = {"k": 0}

    def fn(left, right):
        n["k"] += 1
        z = 2.0 if n["k"] <= at else 0.9
        return np.full((H, W), F * BASE / z, np.float32), np.ones((H, W), bool), 5.0
    return fn


def test_frames_that_disagree_with_the_map_are_not_added_and_the_map_is_rebuilt_after_a_while():
    d = make_recording(n=70)
    c = cfg_for(d)
    c.map.reset_after_frames = 10
    ListViz.instances.clear()
    s = run(c, depth_fn=depth_that_changes(25), viz_factory=ListViz, log_dir=tempfile.mkdtemp(), quiet=True)
    assert s["frames_skipped_inconsistent"] >= 5 and s["map_resets"] >= 1, s["frames_skipped_inconsistent"]
    assert s["map_voxels"] <= s["map_voxels_peak"]


def test_with_the_guard_off_the_same_scene_adds_every_frame():
    d = make_recording(n=70)
    c = cfg_for(d)
    c.map.max_misalign_cm = 0
    s = run(c, depth_fn=depth_that_changes(25), viz_factory=ListViz, log_dir=tempfile.mkdtemp(), quiet=True)
    assert s["frames_skipped_inconsistent"] == 0 and s["map_resets"] == 0 and s["map_frames"] == s["poses_ok"]


def test_the_voxel_cap_applies_in_the_pipeline_and_is_reported():
    d = make_recording(n=40)
    c = cfg_for(d)
    c.map.max_voxels = 150
    c.map.max_misalign_cm = 0
    s = run(c, depth_fn=depth_that_changes(10), viz_factory=ListViz, log_dir=tempfile.mkdtemp(), quiet=True)
    assert s["map_voxels_peak"] <= 150 and s["map_voxels_evicted"] > 0, (s["map_voxels_peak"], s["map_voxels_evicted"])


def test_a_pose_that_jumps_in_one_frame_is_not_trusted_and_not_mapped():
    from gabai.pipeline.poses import PoseResult
    import gabai.pipeline.run as runmod
    d = make_recording(n=20)
    c = cfg_for(d)
    c.map.max_pose_step_m = 0.2
    real = runmod.make_provider

    def make(cfg, cal):
        prov = real(cfg, cal)
        orig = prov.push
        cnt = {"n": 0}

        def push(t_ns, gl, gr, disp, trusted):
            out = orig(t_ns, gl, gr, disp, trusted)
            cnt["n"] += 1
            if cnt["n"] == 10:
                for p in out:
                    p.t_wc = np.asarray(p.t_wc) + np.array([5.0, 0, 0])      # a wild jump
            return out
        prov.push = push
        return prov
    runmod.make_provider = make
    try:
        s = run(c, depth_fn=lambda l, r: (np.full((H, W), F * BASE / 2.0, np.float32), np.ones((H, W), bool), 5.0), viz_factory=ListViz, log_dir=tempfile.mkdtemp(), quiet=True)
    finally:
        runmod.make_provider = real
    assert s["pose_jumps"] >= 1 and s["poses_held"] >= 1


def test_the_odometry_holds_the_pose_when_one_frame_would_move_the_camera_far():
    import cv2
    img = vb.texture()[:vb.H, :vb.W]
    vo = KeyframeVO(vb.F, vb.F, vb.W / 2, vb.H / 2, max_step_m=0.02)
    vo.set_baseline(vb.B)
    ok = np.ones((vb.H, vb.W), bool)
    disp = np.full((vb.H, vb.W), vb.F * vb.B / 2.0, np.float32)
    vo.step(img, disp, ok)
    vo.step(img, disp, ok)
    vo.step(img, disp, ok)
    t_before = vo.t_wc.copy()
    big = cv2.warpAffine(img, np.float32([[1, 0, -15], [0, 1, 0]]), (vb.W, vb.H), borderMode=cv2.BORDER_REFLECT)          # 15 px = 5 cm at 2 m, over the 2 cm limit set above
    R, t, info = vo.step(big, disp, ok)
    assert not info["ok"] and "jump" in info["reason"] and np.allclose(t, t_before)
