import numpy as np

from gabai.sim import Box, Camera, look_pose, render_disparity

CAM = Camera(f=881.0, B=0.0507, cx=319.5, cy=239.5, size=(640, 480))


def level_pose(x=0.0, y=0.0, h=1.4, yaw_deg=0.0):
    return look_pose(np.array([x, y, h]), yaw_deg=yaw_deg)


def depth_at(disp, u, v):
    return CAM.f * CAM.B / disp[v, u]


def test_a_flat_floor_has_the_disparity_the_floor_model_expects():
    from gabai.zones import floor_disparity
    R, t = level_pose()
    disp = render_disparity(CAM, R, t, boxes=[])
    expect = floor_disparity((480, 640), CAM.f, CAM.B, 1.4, CAM.cx, CAM.cy)
    both = np.isfinite(disp) & np.isfinite(expect)
    assert both.sum() > 50000 and np.nanmax(np.abs(disp[both] - expect[both])) < 0.05


def test_a_wall_straight_ahead_is_at_its_true_distance_across_the_picture():
    R, t = level_pose()
    wall = Box(lo=(2.0, -5, 0.0), hi=(2.2, 5, 3.0))                       # a wall 2.0 m ahead (world +x is the heading at yaw 0)
    disp = render_disparity(CAM, R, t, boxes=[wall])
    zs = depth_at(disp, 320, 100)
    assert abs(zs - 2.0) < 0.01                                          # depth along the optical axis is the wall's x distance
    assert abs(depth_at(disp, 100, 100) - 2.0) < 0.01 and abs(depth_at(disp, 600, 100) - 2.0) < 0.01


def test_a_level_camera_at_1_4_m_cannot_see_the_floor_within_5_m():
    # vertical half field of view is atan(240/881) = 15.2 deg, so the lowest ray meets the floor at 1.4 / tan(15.2 deg) = 5.1 m
    R, t = level_pose(h=1.4)
    disp = render_disparity(CAM, R, t, boxes=[])
    z = CAM.f * CAM.B / disp[np.isfinite(disp)]
    assert z.min() > 5.0


def test_a_box_on_the_floor_is_where_it_is_in_the_world_when_the_camera_is_tilted_down():
    R, t = look_pose(np.array([0.0, 0.0, 1.4]), pitch_deg=25.0)
    box = Box(lo=(1.5, -0.3, 0.0), hi=(1.8, 0.3, 0.4))
    disp = render_disparity(CAM, R, t, boxes=[box])
    h, w = disp.shape
    u = (np.arange(w)[None, :] - CAM.cx) / CAM.f
    v = (np.arange(h)[:, None] - CAM.cy) / CAM.f
    Z = CAM.f * CAM.B / disp
    P = np.stack([u * Z, v * Z, Z], axis=-1) @ R.T + t                   # every pixel's point in the world
    on_box = np.isfinite(Z) & (P[..., 2] > 0.02) & (P[..., 2] < 0.39) & (np.abs(P[..., 1]) < 0.29)
    assert on_box.sum() > 300
    assert abs(P[..., 0][on_box].min() - 1.5) < 0.01                      # the front face is at x = 1.5 m


def test_the_camera_pose_changes_what_is_seen():
    wall = Box(lo=(3.0, -5, 0.0), hi=(3.2, 5, 3.0))
    d0 = depth_at(render_disparity(CAM, *level_pose(x=0.0), boxes=[wall]), 320, 100)
    d1 = depth_at(render_disparity(CAM, *level_pose(x=1.0), boxes=[wall]), 320, 100)
    assert abs(d0 - 3.0) < 0.01 and abs(d1 - 2.0) < 0.01
    turned = render_disparity(CAM, *level_pose(yaw_deg=90.0), boxes=[wall])
    assert not np.isfinite(turned[100, 320]) or depth_at(turned, 320, 100) > 6                   # looking sideways: the wall is not ahead


def test_noise_is_optional_and_repeatable():
    R, t = level_pose()
    a = render_disparity(CAM, R, t, boxes=[], noise_px=0.3, seed=1)
    b = render_disparity(CAM, R, t, boxes=[], noise_px=0.3, seed=1)
    c = render_disparity(CAM, R, t, boxes=[], noise_px=0.0)
    ok = np.isfinite(a)
    assert np.array_equal(a[ok], b[ok]) and 0.2 < np.std((a - c)[ok]) < 0.4
