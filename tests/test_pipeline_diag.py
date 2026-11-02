import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.cloud import VoxelMap, map_points, transform
from gabai.pipeline.diagnostics import (check_alignment, check_pose_stability, check_rectification, check_stereo_sync, check_tracking_quality,
                                        map_alignment, stereo_geometry_report)
from gabai.sim import Box, Camera, look_pose, render_disparity

CAM = Camera(881.0, 0.0507, 319.5, 239.5, (640, 480))
WALL = [Box(lo=(3.0, -3, 0), hi=(3.2, 3, 3))]


def build_map(poses, drift=None):
    vm = VoxelMap(0.05)
    for k, (Rm, tm) in enumerate(poses):
        d = render_disparity(CAM, Rm, tm, boxes=WALL, floor=False, noise_px=0.2, seed=k)
        p, _ = map_points(d, np.isfinite(d), CAM.f, CAM.B, CAM.cx, CAM.cy, stride=4, drop_edges=False)
        tm2 = tm if drift is None else tm + drift(k)
        vm.integrate(transform(p, Rm, tm2))
    return vm


def poses(n=20):
    return [look_pose(np.array([0.03 * k, 0.0, 1.0])) for k in range(n)]


def test_a_consistent_map_has_small_residual_against_the_current_depth():
    ps = poses()
    vm = build_map(ps)
    Rm, tm = look_pose(np.array([0.6, 0.0, 1.0]))
    d = render_disparity(CAM, Rm, tm, boxes=WALL, floor=False, noise_px=0.2, seed=99)
    r = map_alignment(vm.points(3), Rm, tm, d, CAM.f, CAM.B, CAM.cx, CAM.cy)
    assert r["n"] > 200 and r["median_abs_m"] < 0.05 and r["within_10cm"] > 0.9
    assert check_alignment(r).status == "pass"


def test_a_pose_that_does_not_match_the_map_is_flagged_with_the_size_of_the_offset():
    vm = build_map(poses())
    Rm, tm = look_pose(np.array([0.6, 0.0, 1.0]))
    d = render_disparity(CAM, Rm, tm, boxes=WALL, floor=False, noise_px=0.2, seed=99)
    r = map_alignment(vm.points(3), Rm, tm + np.array([0.25, 0, 0]), d, CAM.f, CAM.B, CAM.cx, CAM.cy)   # the pose claims 25 cm more forward than it is
    assert r["median_abs_m"] > 0.15 and abs(r["median_signed_m"]) > 0.15
    c = check_alignment(r)
    assert c.status == "fail" and "pose" in c.hint.lower()


def test_a_map_built_with_drifting_poses_disagrees_with_the_depth_the_camera_sees_now():
    vm = build_map(poses(40), drift=lambda k: np.array([0.012 * k, 0, 0]))
    Rm, tm = look_pose(np.array([0.03 * 39, 0.0, 1.0]))
    d = render_disparity(CAM, Rm, tm, boxes=WALL, floor=False, noise_px=0.2, seed=7)
    r = map_alignment(vm.points(3), Rm, tm, d, CAM.f, CAM.B, CAM.cx, CAM.cy)
    assert r["median_abs_m"] > 0.1


def test_too_little_overlap_is_unknown_not_pass():
    Rm, tm = look_pose(np.array([0.0, 0.0, 1.0]))
    d = render_disparity(CAM, Rm, tm, boxes=WALL, floor=False)
    r = map_alignment(np.zeros((0, 3)), Rm, tm, d, CAM.f, CAM.B, CAM.cx, CAM.cy)
    assert r is None and check_alignment(None).status == "unknown"


def test_pose_stability_flags_a_jump_and_a_spin_but_not_a_steady_walk():
    steady = [(i * 0.07, np.array([0.05 * i, 0, 0]), R.identity()) for i in range(10)]
    assert check_pose_stability(steady).status == "pass"
    jump = steady[:5] + [(0.35, np.array([1.5, 0, 0]), R.identity())]
    assert check_pose_stability(jump).status == "fail"
    spin = steady[:5] + [(0.35, steady[4][1], R.from_euler("z", 70, degrees=True))]
    c = check_pose_stability(spin)
    assert c.status in ("fail", "suspect") and "rotation" in c.detail
    assert check_pose_stability(steady[:1]).status == "unknown"


def test_tracking_quality_uses_inliers_and_says_what_it_means():
    assert check_tracking_quality({"inliers": 120, "tracked": 150}).status == "pass"
    assert check_tracking_quality({"inliers": 18, "tracked": 40}).status == "suspect"
    assert check_tracking_quality({"inliers": 3, "tracked": 5}).status == "fail"
    assert check_tracking_quality({}).status == "unknown"


def test_rectification_check_separates_aligned_from_misaligned_rows():
    assert check_rectification(0.4).status == "pass"
    assert check_rectification(3.0).status == "fail" and "recalibrate" in check_rectification(3.0).hint.lower()
    assert check_rectification(float("nan")).status == "unknown"


def test_stereo_sync_check_is_unknown_for_assumed_sync_and_judges_a_measured_offset():
    assert check_stereo_sync({"kind": "assumed", "pair_dt_ms": None}, None).status == "unknown"
    assert check_stereo_sync({"kind": "measured", "pair_dt_ms": 2.0}, 10.0).status == "pass"
    c = check_stereo_sync({"kind": "measured", "pair_dt_ms": 40.0}, 200.0)                # 200 deg/s turn, 40 ms: 8 degrees apart
    assert c.status == "fail"


def test_geometry_report_states_what_is_assumed_and_the_depth_error_per_half_pixel():
    r = stereo_geometry_report(881.0, 0.0507, metric=True, source="calib.json")
    assert r["metric"] and abs(r["depth_error_m_at"][2.0] - 2.0 ** 2 / (881.0 * 0.0507) * 0.5) < 1e-6
    r2 = stereo_geometry_report(384.0, 0.065, metric=False, source="assumed")
    assert not r2["metric"] and "not metric" in r2["note"] and r2["min_range_m"] > 0
