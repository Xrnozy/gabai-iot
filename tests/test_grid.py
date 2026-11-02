import numpy as np

from gabai.grid import FREE, OCCUPIED, UNKNOWN, RollingGrid
from gabai.sim import Box, Camera, look_pose, render_disparity

CAM = Camera(f=881.0, B=0.0507, cx=319.5, cy=239.5, size=(640, 480))
H, PITCH = 1.2, 35.0                      # chest height, tilted down so the floor is visible from about 1 m


def view(x=0.0, y=0.0, yaw=0.0, boxes=(), noise=0.0, seed=0):
    R, t = look_pose(np.array([x, y, H]), yaw_deg=yaw, pitch_deg=PITCH)
    return render_disparity(CAM, R, t, boxes=boxes, noise_px=noise, seed=seed), R, t


def feed(grid, disp, R, t, now, mask=None):
    return grid.update(now, disp, CAM.f, CAM.B, CAM.cx, CAM.cy, R, t, floor_z=0.0, person_mask=mask)


def cell_state_at(grid, x, y):
    return grid.state_at(x, y)


def test_a_box_on_the_floor_is_occupied_at_1_2_m_and_2_m_and_the_floor_in_front_is_free():
    for d in (1.2, 2.0):
        g = RollingGrid()
        disp, R, t = view(boxes=[Box(lo=(d, -0.3, 0.0), hi=(d + 0.3, 0.3, 0.4))])
        feed(g, disp, R, t, 0.0)
        feed(g, disp, R, t, 0.07)
        assert cell_state_at(g, d + 0.05, 0.0) == OCCUPIED, d
        assert cell_state_at(g, d - 0.12, 0.0) == FREE, d                 # seen floor between the camera and the box
        assert g.corridor((1.0, 0.0))["nearest_occupied_m"] is not None
        assert abs(g.corridor((1.0, 0.0))["nearest_occupied_m"] - d) < 0.15


def test_what_was_never_seen_is_unknown_not_free():
    g = RollingGrid()
    assert cell_state_at(g, 1.0, 0.0) == UNKNOWN
    c = g.corridor((1.0, 0.0))
    assert c["unknown_fraction"] == 1.0 and c["nearest_occupied_m"] is None and c["free_fraction"] == 0.0
    disp, R, t = view(boxes=[Box(lo=(2.0, -0.3, 0.0), hi=(2.3, 0.3, 0.4))])
    feed(g, disp, R, t, 0.0)
    feed(g, disp, R, t, 0.07)
    assert cell_state_at(g, 2.6, 0.0) == UNKNOWN                         # behind the box nothing was seen
    assert cell_state_at(g, -1.0, 0.0) == UNKNOWN                        # behind the camera nothing was seen


def test_a_wall_seen_again_after_a_loop_lines_up_within_10_cm():
    wall = Box(lo=(2.5, -4, 0.0), hi=(2.7, 4, 3.0))

    def wall_cells(grid):
        st = grid.state_grid()
        ix, iy = np.where(st == OCCUPIED)
        xs = grid.origin[0] + (ix + 0.5) * grid.cell
        return xs[(np.abs(grid.origin[1] + (iy + 0.5) * grid.cell) < 1.0) & (xs > 2.0)]
    g = RollingGrid()
    now = 0.0
    for _ in range(6):                                                   # first look
        disp, R, t = view(boxes=[wall], noise=0.3, seed=int(now * 100))
        feed(g, disp, R, t, now)
        now += 0.07
    first = wall_cells(g)
    for yaw in range(0, 181, 30):                                        # turn round to face away ...
        disp, R, t = view(yaw=yaw, boxes=[wall], noise=0.3)
        feed(g, disp, R, t, now)
        now += 0.07
    for yaw in range(180, -1, -30):                                      # ... and back again
        disp, R, t = view(yaw=yaw, boxes=[wall], noise=0.3)
        feed(g, disp, R, t, now)
        now += 0.07
    for _ in range(6):
        disp, R, t = view(boxes=[wall], noise=0.3, seed=int(now * 100))
        feed(g, disp, R, t, now)
        now += 0.07
    again = wall_cells(g)
    assert len(first) > 5 and len(again) > 5
    assert abs(np.median(first) - 2.5) < 0.10 and abs(np.median(again) - 2.5) < 0.10     # both looks put the wall face at 2.5 m
    assert abs(np.median(first) - np.median(again)) < 0.10


def test_a_person_walking_past_leaves_no_obstacle_behind_after_two_seconds():
    g = RollingGrid()
    now = 0.0
    for _ in range(6):                                                    # empty floor first
        disp, R, t = view()
        feed(g, disp, R, t, now)
        now += 0.07
    saw_person = False
    for i in range(28):                                                   # a person 1.5 m ahead crosses left to right in 2 s
        y = -1.2 + i * 0.0875
        disp, R, t = view(boxes=[Box(lo=(1.5, y - 0.25, 0.0), hi=(1.8, y + 0.25, 1.7))])
        feed(g, disp, R, t, now)
        saw_person |= any(cell_state_at(g, x, y) == OCCUPIED for x in (1.45, 1.55))     # a person shows as their front face, at x = 1.5
        now += 0.07
    assert saw_person                                                     # while they were there, they were an obstacle (the test is empty without this)
    for _ in range(28):                                                   # two seconds of empty floor at 14 fps
        disp, R, t = view()
        feed(g, disp, R, t, now)
        now += 0.07
    st = g.state_grid()
    ix0, iy0 = g.index(1.3, -1.5)
    ix1, iy1 = g.index(1.8, 1.5)
    region = st[ix0:ix1 + 1, iy0:iy1 + 1]
    assert (region == OCCUPIED).sum() == 0


def test_a_person_mask_keeps_people_out_of_the_map():
    g = RollingGrid()
    disp, R, t = view(boxes=[Box(lo=(1.5, -0.25, 0.0), hi=(1.8, 0.25, 1.7))])
    mask = np.zeros(disp.shape, bool)
    mask[:, 200:440] = True                                               # where the detector says the person is
    for k in range(6):
        feed(g, disp, R, t, k * 0.07, mask=mask)
    assert cell_state_at(g, 1.55, 0.0) != OCCUPIED
    g2 = RollingGrid()
    for k in range(6):
        feed(g2, disp, R, t, k * 0.07)
    assert cell_state_at(g2, 1.55, 0.0) == OCCUPIED                       # same scene without the mask is an obstacle


def test_the_window_rolls_with_the_camera_and_forgets_what_is_left_behind():
    g = RollingGrid()
    wall = Box(lo=(2.5, -4, 0.0), hi=(2.7, 4, 3.0))
    for k in range(4):
        disp, R, t = view(boxes=[wall])
        feed(g, disp, R, t, k * 0.07)
    assert cell_state_at(g, 2.55, 0.0) == OCCUPIED
    for k in range(4):                                                    # walk 2 m on: the wall is now 2.55 behind the camera, still inside the window
        disp, R, t = view(x=2.0 + k * 0.01)
        feed(g, disp, R, t, 1.0 + k * 0.07)
    assert cell_state_at(g, 2.55, 0.0) == OCCUPIED                        # kept: the grid is anchored to the world, it rolls with the camera
    for k in range(4):                                                    # then jump 5 m further: more than the whole window
        disp, R, t = view(x=7.0 + k * 0.01)
        feed(g, disp, R, t, 2.0 + k * 0.07)
    assert cell_state_at(g, 2.55, 0.0) == UNKNOWN                         # far outside the window now: dropped, not kept as stale truth
    assert g.origin[0] > 3.5 and (g.state_grid() == OCCUPIED).sum() == 0


def test_old_observations_expire_to_unknown():
    g = RollingGrid(forget_s=10.0)
    disp, R, t = view(boxes=[Box(lo=(2.0, -0.3, 0.0), hi=(2.3, 0.3, 0.4))])
    feed(g, disp, R, t, 0.0)
    feed(g, disp, R, t, 0.07)
    assert cell_state_at(g, 2.05, 0.0) == OCCUPIED
    g.expire(now=30.0)
    assert cell_state_at(g, 2.05, 0.0) == UNKNOWN


def test_points_above_head_height_and_below_the_floor_do_not_make_obstacles():
    g = RollingGrid()
    disp, R, t = view(boxes=[Box(lo=(2.0, -1, 2.6), hi=(2.3, 1, 3.0))])   # a shelf at 2.6 m: overhead, clear of a walking person
    for k in range(6):
        feed(g, disp, R, t, k * 0.07)
    assert cell_state_at(g, 2.1, 0.0) != OCCUPIED


def test_the_diagnostic_picture_marks_the_camera_and_the_three_states():
    g = RollingGrid()
    disp, R, t = view(boxes=[Box(lo=(2.0, -0.3, 0.0), hi=(2.3, 0.3, 0.4))])
    feed(g, disp, R, t, 0.0)
    feed(g, disp, R, t, 0.07)
    img = g.to_image(scale=4, camera_xy=(0.0, 0.0), heading_xy=(1.0, 0.0))
    assert img.shape == (g.n * 4, g.n * 4, 3) and len({tuple(p) for p in img.reshape(-1, 3)[::97]}) >= 3


def test_a_new_obstacle_over_long_observed_clear_floor_is_occupied_within_two_frames():
    g = RollingGrid()
    for k in range(40):                                                   # ten seconds' worth of clear floor
        disp, R, t = view()
        feed(g, disp, R, t, k * 0.07)
    box = Box(lo=(1.8, -0.3, 0.0), hi=(2.1, 0.3, 0.4))
    disp, R, t = view(boxes=[box])
    feed(g, disp, R, t, 3.0)
    assert cell_state_at(g, 1.85, 0.0) != OCCUPIED                         # one frame is not enough: a single noisy frame cannot make an obstacle
    feed(g, disp, R, t, 3.07)
    assert cell_state_at(g, 1.85, 0.0) == OCCUPIED                         # the second confirms it (0.14 s later at 14 fps)


def test_an_obstacle_takes_under_a_second_to_clear_once_the_floor_is_seen_again():
    g = RollingGrid()
    box = Box(lo=(1.8, -0.3, 0.0), hi=(2.1, 0.3, 0.4))
    for k in range(30):
        disp, R, t = view(boxes=[box])
        feed(g, disp, R, t, k * 0.07)
    assert cell_state_at(g, 1.85, 0.0) == OCCUPIED
    cleared_after = None
    for k in range(1, 40):
        disp, R, t = view()
        feed(g, disp, R, t, 2.1 + k * 0.07)
        if cell_state_at(g, 1.85, 0.0) != OCCUPIED:
            cleared_after = k * 0.07
            break
    assert cleared_after is not None and cleared_after < 1.0
