"""The map follows the room: a thing that appears is mapped, a thing that moves or goes is carved out, and what is only hidden behind something nearer is kept."""
import numpy as np

from gabai.sim import Box, Camera, look_pose, render_disparity
from gabai.tools.slam_cloud_view import SlamCloud

CAM = Camera(881.0, 0.0507, 319.5, 239.5, (640, 480))
WALL = Box(lo=(4.0, -3, -3), hi=(4.2, 3, 3))
GRAY = np.zeros((480, 640), np.uint8)


def new_cloud(**kw):
    return SlamCloud(CAM.f, CAM.B, CAM.cx, CAM.cy, voxel=0.05, window_m=None, semi_dense=False, stride=4, max_range=6.0, carve_decay=0.5, **kw)


def look(sc, boxes, frames, k0=0, noise=0.15, jitter=0.0, x=0.0):
    rng = np.random.default_rng(k0)
    for k in range(k0, k0 + frames):
        R_wc, t_wc = look_pose(np.array([x, 0.0, 1.0]))
        disp = render_disparity(CAM, R_wc, t_wc, boxes=boxes, floor=False, noise_px=noise, seed=k)
        sc.on_pose(k, R_wc, t_wc + rng.normal(0, jitter, 3), GRAY, disp, np.isfinite(disp))
    return k0 + frames


def count(sc, x0, x1, y0=-3.0, y1=3.0, min_obs=1):
    p = sc.map.points(min_obs=min_obs)
    return int(((p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)).sum())


def test_a_thing_that_appears_is_mapped():
    sc = new_cloud()
    k = look(sc, [WALL], 8)
    assert count(sc, 1.5, 3.0) == 0
    look(sc, [WALL, Box(lo=(2.0, -0.5, 0.6), hi=(2.3, 0.5, 1.5))], 8, k0=k)
    assert count(sc, 1.9, 2.4, min_obs=3) > 100


def test_a_thing_that_is_taken_away_leaves_the_map():
    sc = new_cloud()
    obj = Box(lo=(2.0, -0.5, 0.6), hi=(2.3, 0.5, 1.5))
    k = look(sc, [WALL, obj], 10)
    before = count(sc, 1.9, 2.4)
    assert before > 100
    look(sc, [WALL], 10, k0=k)
    assert count(sc, 1.9, 2.4) < 0.05 * before, (before, count(sc, 1.9, 2.4))
    assert count(sc, 3.9, 4.3, min_obs=3) > 500                                      # the wall behind it is still there
    assert sc.carved > 0


def test_a_thing_that_moves_is_carved_at_the_old_place_and_mapped_at_the_new_one():
    sc = new_cloud()
    k = look(sc, [WALL, Box(lo=(2.0, -1.5, 0.6), hi=(2.3, -0.5, 1.5))], 10)
    old_before = count(sc, 1.9, 2.4, -1.6, -0.4)
    look(sc, [WALL, Box(lo=(2.0, 0.5, 0.6), hi=(2.3, 1.5, 1.5))], 10, k0=k)
    assert old_before > 100
    assert count(sc, 1.9, 2.4, -1.6, -0.4) < 0.05 * old_before
    assert count(sc, 1.9, 2.4, 0.4, 1.6, min_obs=3) > 100


def test_a_still_room_is_not_eroded_by_depth_noise_or_a_little_pose_jitter():
    sc = new_cloud()
    obj = Box(lo=(2.0, -0.5, 0.6), hi=(2.3, 0.5, 1.5))
    k = look(sc, [WALL, obj], 6, noise=0.25, jitter=0.02)
    first = count(sc, 1.9, 4.3)
    look(sc, [WALL, obj], 60, k0=k, noise=0.25, jitter=0.02)                         # a minute of looking at the same room
    assert count(sc, 1.9, 4.3) > 0.9 * first, (first, count(sc, 1.9, 4.3))


def test_what_is_hidden_behind_a_nearer_thing_is_kept():
    sc = new_cloud()
    k = look(sc, [WALL], 10)
    wall_before = count(sc, 3.9, 4.3)
    look(sc, [WALL, Box(lo=(2.0, -3.0, -3.0), hi=(2.3, 3.0, 3.0))], 15, k0=k)          # a screen now hides the whole wall
    assert count(sc, 3.9, 4.3) > 0.95 * wall_before


def test_nothing_is_carved_next_to_a_hole_in_the_depth():
    sc = new_cloud()
    obj = Box(lo=(2.0, -0.5, 0.6), hi=(2.3, 0.5, 1.5))
    k = look(sc, [WALL, obj], 10)
    before = count(sc, 1.9, 2.4)
    for j in range(10):                                                              # the object goes, but stereo cannot measure the middle of that view (a blank patch)
        R_wc, t_wc = look_pose(np.array([0.0, 0.0, 1.0]))
        disp = render_disparity(CAM, R_wc, t_wc, boxes=[WALL], floor=False, noise_px=0.15, seed=100 + j)
        ok = np.isfinite(disp)
        ok[:, 120:520] = False
        sc.on_pose(k + j, R_wc, t_wc, GRAY, disp, ok)
    assert count(sc, 1.9, 2.4) > 0.8 * before                                        # no measurement was made there, so no claim that it is empty


def test_the_filled_layer_is_carved_too():
    sc = new_cloud()
    R_wc, t_wc = look_pose(np.array([0.0, 0.0, 1.0]))
    disp = render_disparity(CAM, R_wc, t_wc, boxes=[WALL, Box(lo=(2.0, -0.5, 0.6), hi=(2.3, 0.5, 1.5))], floor=False)
    ok = np.isfinite(disp)
    hole = np.zeros(ok.shape, bool)
    hole[:, 300:340] = True
    for k in range(6):
        sc.on_pose(k, R_wc, t_wc, GRAY, disp, ok & ~hole, fill=(disp, hole))
    assert len(sc.fill_map) > 20
    only_wall = render_disparity(CAM, R_wc, t_wc, boxes=[WALL], floor=False)
    for k in range(6, 16):
        sc.on_pose(k, R_wc, t_wc, GRAY, only_wall, np.isfinite(only_wall))
    p = sc.fill_map.points(min_obs=1)
    assert len(p) == 0 or (p[:, 0] > 3.5).all()
