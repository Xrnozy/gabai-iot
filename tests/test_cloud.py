import numpy as np

from gabai.cloud import VoxelMap, backproject, depth_edge_mask, map_points, transform
from gabai.sim import Box, Camera, look_pose, render_disparity

CAM = Camera(881.0, 0.0507, 319.5, 239.5, (640, 480))


def wall(z):
    return np.full((480, 640), CAM.f * CAM.B / z, np.float32)


def test_backprojecting_a_flat_wall_gives_points_at_that_depth_spread_over_the_view():
    p, uv = backproject(wall(2.0), CAM.f, CAM.B, CAM.cx, CAM.cy, stride=8)
    assert abs(np.median(p[:, 2]) - 2.0) < 1e-3 and p[:, 0].min() < -0.5 and p[:, 0].max() > 0.5 and len(p) == len(uv)


def test_nan_and_zero_disparity_give_no_points():
    d = wall(2.0)
    d[:, :320] = np.nan
    d[:100] = 0
    p, _ = backproject(d, CAM.f, CAM.B, CAM.cx, CAM.cy, stride=8)
    assert len(p) > 0 and (p[:, 0] > -0.01).all()


def test_depth_edges_are_found_and_flat_areas_are_not():
    d = wall(1.0)
    d[:, 320:] = CAM.f * CAM.B / 4.0                       # a step: near wall left, far wall right
    e = depth_edge_mask(d, CAM.f, CAM.B)
    assert e[:, 319:321].all() and not e[:, :300].any() and not e[:, 340:].any()


def test_map_points_keeps_only_trusted_pixels_inside_range_and_off_the_excluded_mask():
    d = wall(2.0)
    trusted = np.ones(d.shape, bool)
    trusted[:, :200] = False
    excl = np.zeros(d.shape, bool)
    excl[:, 500:] = True
    p, info = map_points(d, trusted, CAM.f, CAM.B, CAM.cx, CAM.cy, exclude=excl, stride=4)
    assert info["dropped_untrusted"] > 0 and info["dropped_excluded"] > 0 and info["kept"] == len(p)
    u = p[:, 0] / p[:, 2] * CAM.f + CAM.cx
    assert u.min() >= 199 and u.max() < 500
    far, info2 = map_points(wall(6.0), np.ones((480, 640), bool), CAM.f, CAM.B, CAM.cx, CAM.cy, max_range=4.0, stride=4)
    assert len(far) == 0 and info2["dropped_range"] > 0


def test_transform_applies_the_camera_pose():
    R, t = look_pose(np.array([1.0, 2.0, 1.5]), yaw_deg=90)
    w = transform(np.array([[0.0, 0.0, 1.0]]), R, t)           # one metre in front of the camera
    assert np.allclose(w, [[1.0, 3.0, 1.5]], atol=1e-6)


def test_fusing_many_noisy_views_of_a_wall_beats_one_view():
    rng = np.random.default_rng(0)
    vm = VoxelMap(voxel=0.05)
    R, t = look_pose(np.array([0.0, 0.0, 1.0]))
    wallbox = [Box(lo=(3.0, -2, 0), hi=(3.2, 2, 2))]
    first = None
    for i in range(50):
        d = render_disparity(CAM, R, t, boxes=wallbox, noise_px=0.4, seed=i)
        p, _ = map_points(d, np.isfinite(d), CAM.f, CAM.B, CAM.cx, CAM.cy, stride=6, max_range=5.0, drop_edges=False)
        pw = transform(p, R, t)
        pw = pw[(pw[:, 2] > 0.3) & (pw[:, 2] < 1.8) & (abs(pw[:, 1]) < 1.0)]
        if i == 0:
            first = pw[:, 0].std()
        vm.integrate(pw)
    fused = vm.points(min_obs=10, core=0.9)
    sel = fused[(fused[:, 2] > 0.3) & (fused[:, 2] < 1.8) & (abs(fused[:, 1]) < 1.0)]
    assert len(sel) > 100 and abs(np.median(sel[:, 0]) - 3.0) < 0.06
    assert sel[:, 0].std() < 0.5 * first


def test_a_few_observations_are_not_trusted_as_surface():
    vm = VoxelMap(0.05)
    vm.integrate(np.array([[1.0, 1.0, 1.0]]))
    assert len(vm.points(min_obs=3)) == 0 and len(vm.points(min_obs=1)) == 1


def test_rolling_window_keeps_memory_flat_on_a_long_straight_walk():
    vm = VoxelMap(0.05, window_m=6.0)
    sizes = []
    for i in range(200):
        x = i * 0.5                                              # 100 m walk
        pts = np.stack([x + np.linspace(0, 3, 400), np.zeros(400), np.zeros(400) + 0.5 * (np.arange(400) % 3)], axis=1)
        vm.integrate(pts)
        vm.trim(np.array([x, 0, 0]))
        sizes.append(len(vm))
    assert max(sizes[100:]) < 1.3 * max(sizes[30:60]) and vm.evicted > 0


def test_ghost_trails_do_not_appear_when_the_moving_thing_is_masked_out():
    d = wall(2.0)
    person = np.zeros(d.shape, bool)
    person[100:400, 250:390] = True
    d2 = d.copy()
    d2[person] = CAM.f * CAM.B / 1.0
    vm = VoxelMap(0.05)
    for _ in range(5):
        p, _ = map_points(d2, np.ones(d.shape, bool), CAM.f, CAM.B, CAM.cx, CAM.cy, exclude=person, stride=4, drop_edges=False)
        vm.integrate(p)
    pts = vm.points(min_obs=1)
    assert (pts[:, 2] > 1.5).all()                              # nothing from the 1 m person


def test_the_faster_edge_mask_gives_the_same_answer_as_the_depth_formula():
    from gabai.cloud import EDGE_REL, depth_edge_mask
    rng = np.random.default_rng(5)
    d = (rng.random((120, 160)) * 60 + 4).astype(np.float32)
    d[::7, ::5] = np.nan
    d[10:30, 10:50] = 0.2                                                   # below the valid limit
    f, B = 881.0, 0.0507
    valid = np.isfinite(d) & (d > 0.5)
    z = np.where(valid, f * B / np.where(d > 0.5, d, 1), np.nan)
    ref = np.zeros(z.shape, bool)
    with np.errstate(invalid="ignore"):
        dx = np.abs(np.diff(z, axis=1)) > EDGE_REL * np.fmin(z[:, 1:], z[:, :-1])
        dy = np.abs(np.diff(z, axis=0)) > EDGE_REL * np.fmin(z[1:], z[:-1])
    ref[:, 1:] |= dx
    ref[:, :-1] |= dx
    ref[1:] |= dy
    ref[:-1] |= dy
    got = depth_edge_mask(d, f, B)
    assert (got != ref).mean() < 1e-4, int((got != ref).sum())             # only float rounding at exactly-borderline pixels may differ


def test_prune_forgets_what_is_far_and_what_the_camera_passes_through_and_keeps_the_rest():
    vm = VoxelMap(0.05, window_m=6.0)
    vm.integrate(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    gone = vm.prune(np.array([0.0, 0.0, 0.0]), window_m=6.0, carve_m=0.3)
    kept = vm.points(min_obs=1)
    assert gone == 3 and len(kept) == 1 and abs(kept[0, 0] - 3.0) < 0.05           # 20 m away: out of the window; 0 and 0.1 m: where the camera is


def test_a_map_holds_everything_inside_the_window_for_as_long_as_the_camera_stays_near():
    vm = VoxelMap(0.05, window_m=6.0)
    pts = np.stack([np.linspace(1, 5, 500), np.zeros(500), np.zeros(500)], 1)
    vm.integrate(pts)
    for step in range(300):                                                          # a long time standing about in the room: nothing is forgotten
        vm.prune(np.array([0.05 * (step % 7), 0.0, 0.0]), window_m=6.0, carve_m=0.3)
    assert len(vm) >= 75
