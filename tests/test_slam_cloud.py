import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.sim import Box, Camera, look_pose, render_disparity
from gabai.tools.slam_cloud_view import ListSink, SlamCloud, cam_pose, gyro_bias_sample, gyro_residual_deg, semi_dense_mask

CAM = Camera(881.0, 0.0507, 319.5, 239.5, (640, 480))


def test_camera_pose_composes_the_imu_pose_with_the_camera_imu_extrinsics():
    R_ic = R.from_euler("z", 90, degrees=True).as_matrix()          # camera axes rotated 90 degrees about z from the IMU's
    t_ic = np.array([0.1, 0.0, 0.0])
    q = R.from_euler("z", 90, degrees=True).as_quat()                # IMU turned 90 degrees in the world
    R_wc, t_wc = cam_pose(np.array([1.0, 2.0, 0.0]), q, R_ic, t_ic)
    assert np.allclose(R_wc, R.from_euler("z", 180, degrees=True).as_matrix(), atol=1e-9)
    assert np.allclose(t_wc, [1.0, 2.1, 0.0], atol=1e-9)             # the 0.1 m lever arm is turned with the IMU


def rows(rate_dps, dt, n, axis=2, bias=0.0):
    t = np.arange(n) * dt
    w = np.zeros((n, 3))
    w[:, axis] = np.radians(rate_dps) + bias
    return t, w


def test_a_gyro_that_agrees_with_the_visual_inertial_rotation_gives_zero_residual():
    t, w = rows(30.0, 0.005, 201)                                    # 30 deg/s for 1 s
    q0 = R.identity().as_quat()
    q1 = R.from_euler("z", 30, degrees=True).as_quat()
    assert gyro_residual_deg(t, w, np.zeros(3), 0.0, 1.0, q0, q1) < 0.3


def test_a_gyro_bias_shows_up_as_residual_and_removing_it_fixes_it():
    t, w = rows(30.0, 0.005, 201, bias=np.radians(5.0))
    q0, q1 = R.identity().as_quat(), R.from_euler("z", 30, degrees=True).as_quat()
    assert gyro_residual_deg(t, w, np.zeros(3), 0.0, 1.0, q0, q1) > 4.0
    assert gyro_residual_deg(t, w, np.array([0, 0, np.radians(5.0)]), 0.0, 1.0, q0, q1) < 0.3


def test_semi_dense_mask_keeps_textured_pixels_and_drops_blank_ones():
    img = np.full((480, 640), 120, np.uint8)
    img[:, 300:310] = 30                                              # a dark stripe: two strong edges
    m = semi_dense_mask(img)
    assert m[:, 298:312].any() and not m[:, :250].any() and not m[:, 400:].any()


def walk(drift_per_frame=0.0, n=40):
    """Camera moves 3 cm per frame toward a wall 4 m away, looking along +x. The pose handed to the map may drift."""
    sink = ListSink()
    sc = SlamCloud(CAM.f, CAM.B, CAM.cx, CAM.cy, sink=sink, voxel=0.05, window_m=20.0, semi_dense=False)
    wall = [Box(lo=(4.0, -3, 0), hi=(4.2, 3, 3))]
    for k in range(n):
        x = 0.03 * k
        R_wc, t_wc = look_pose(np.array([x, 0.0, 1.0]))
        disp = render_disparity(CAM, R_wc, t_wc, boxes=wall, floor=False, noise_px=0.2, seed=k)
        left = np.zeros((480, 640), np.uint8)
        sc.on_pose(k * 1_000_000_00, R_wc, t_wc + np.array([drift_per_frame * k, 0, 0]), left, disp, np.isfinite(disp))
    return sc, sink


def test_with_a_good_pose_the_wall_stays_one_thin_surface_while_the_camera_moves():
    sc, sink = walk(0.0)
    p = sc.map.points(min_obs=5, core=0.9)
    p = p[(abs(p[:, 1]) < 1.0) & (p[:, 2] > 0.3) & (p[:, 2] < 1.8)]
    assert len(p) > 200 and abs(np.median(p[:, 0]) - 4.0) < 0.1 and p[:, 0].std() < 0.05
    assert sink.n_frames == 40 and sink.last["live_points"] is not None


def test_a_drifting_pose_smears_the_wall_which_is_how_bad_tracking_shows_in_the_map():
    good, _ = walk(0.0)
    bad, _ = walk(0.01)                                               # 1 cm of pose error added per frame: 40 cm by the end
    thick = lambda sc: sc.map.points(min_obs=3)[:, 0].std()
    assert thick(bad) > 2 * thick(good)


def test_the_bias_implied_by_the_visual_rotation_is_recovered():
    t, w = rows(30.0, 0.005, 201, bias=np.radians(5.0))
    q0, q1 = R.identity().as_quat(), R.from_euler("z", 30, degrees=True).as_quat()
    b = gyro_bias_sample(t, w, 0.0, 1.0, q0, q1)
    assert abs(np.degrees(b[2]) - 5.0) < 0.3 and abs(b[0]) < 1e-3
    assert gyro_bias_sample(t, w, 5.0, 6.0, q0, q1) is None                       # no gyro data in that window


def wall_cloud(**kw):
    return SlamCloud(CAM.f, CAM.B, CAM.cx, CAM.cy, voxel=0.05, semi_dense=False, stride=4, **kw)


def frame_at(sc, x, k, anchor=None, fill=None):
    R_wc, t_wc = look_pose(np.array([x, 0.0, 1.0]))
    disp = render_disparity(CAM, R_wc, t_wc, boxes=[Box(lo=(x + 3.0, -3, 0), hi=(x + 3.2, 3, 3))], floor=False)
    sc.on_pose(k, R_wc, t_wc, np.zeros((480, 640), np.uint8), disp, np.isfinite(disp), anchor=anchor, fill=fill)


def test_points_stay_while_the_camera_is_near_and_go_once_it_is_several_metres_away():
    sc = wall_cloud(window_m=6.0)
    for k in range(10):
        frame_at(sc, 0.0, k)
    near = len(sc.map)
    for k in range(10, 40):                                                          # the camera stays within 6 m of that first wall: it all stays
        frame_at(sc, 0.5 * (k % 3), k)
    p = sc.map.points(min_obs=1)
    assert near > 100 and (np.abs(p[:, 0] - 3.0) < 0.2).sum() > 50
    for k in range(40, 60):                                                          # then it walks away
        frame_at(sc, 0.5 * (k - 38) * 1.5, k)
    cam_x = 0.5 * (59 - 38) * 1.5
    p = sc.map.points(min_obs=1)
    assert p[:, 0].min() > cam_x - 6.0 - 0.2                                         # nothing left from more than 6 m behind the camera
    assert len(p) > 100                                                              # the wall in front of it is there


def test_a_long_disagreement_drops_only_the_map_near_the_camera():
    sc = wall_cloud(window_m=None)
    for k in range(5):
        frame_at(sc, 0.0, k)
    for k in range(5, 10):
        frame_at(sc, 10.0, k)
    far_before = int((sc.map.points(min_obs=1)[:, 0] > 8).sum())
    sc.forget_near(np.array([10.0, 0.0, 1.0]), 5.0)
    p = sc.map.points(min_obs=1)
    assert far_before > 50 and (p[:, 0] > 8).sum() == 0 and (p[:, 0] < 4).sum() > 50   # the first wall (built while the pose was good) is untouched


def test_the_map_is_laid_again_from_the_corrected_keyframe_poses():
    sc = wall_cloud(window_m=None)
    R0, t0 = np.eye(3), np.zeros(3)
    for k in range(6):
        frame_at(sc, 0.0, k, anchor=(1, R0, t0))                                     # the odometry put keyframe 1 at the origin
    wall_x = sc.map.points(min_obs=1)[:, 0].mean()
    n_frames, dropped, moved = sc.rebuild(lambda i: (R0, t0 + np.array([0.4, 0.0, 0.0])))            # a loop closure moved it 0.4 m
    after = sc.map.points(min_obs=1)[:, 0].mean()
    assert n_frames == 6 and dropped == 0 and abs(moved - 0.4) < 1e-6 and abs((after - wall_x) - 0.4) < 0.03
    sc.rebuild(lambda i: None)                                                       # a keyframe the odometry forgot: its frames go
    assert len(sc.frames) == 0


def test_monocular_fill_adds_the_holes_at_a_lower_weight_and_never_changes_a_stereo_point():
    R_wc, t_wc = look_pose(np.array([0.0, 0.0, 1.0]))
    wall = [Box(lo=(3.0, -3, 0), hi=(3.2, 3, 3))]
    disp = render_disparity(CAM, R_wc, t_wc, boxes=wall, floor=False)
    trusted = np.isfinite(disp)
    trusted[:, 300:340] = False                                                       # a hole stereo could not measure
    stereo_only, with_fill = wall_cloud(window_m=None), wall_cloud(window_m=None, fill_weight=0.3)
    stereo_only.on_pose(0, R_wc, t_wc, np.zeros((480, 640), np.uint8), disp, trusted)
    fill_mask = np.zeros(disp.shape, bool)
    fill_mask[:, 300:340] = True
    info = with_fill.on_pose(0, R_wc, t_wc, np.zeros((480, 640), np.uint8), disp, trusted, fill=(disp, fill_mask))
    assert info["filled_points"] > 20
    a, b = stereo_only.map, with_fill.map
    assert np.array_equal(a.keys, b.keys) and np.array_equal(a.sum, b.sum) and np.array_equal(a.w, b.w)       # the stereo layer is exactly what stereo alone gives
    shown, plain = with_fill.shown_points(min_obs=1), stereo_only.shown_points(min_obs=1)
    assert len(shown) > len(plain)                                                    # the hole is covered...
    extra = shown[len(plain):]
    assert (extra[:, 1] > -3).all() and ((extra[:, 0] > 2.9) & (extra[:, 0] < 3.15)).all()      # ...on the wall, where the hole was
    assert (with_fill.fill_map.w.max() < a.w.max())                                    # at a lower weight
