import math

import numpy as np

from gabai.orient import q_from_axis_angle, q_to_R
from gabai.spike.cam3d import camera_euler_deg, draw_camera_3d


def R_about_world_z(deg):
    return q_to_R(q_from_axis_angle((0, 0, 1), math.radians(deg)))


# A level camera in world coordinates (z up): optical axis along world +y, image-down along world -z, image-right along world +x
LEVEL = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float).T @ np.eye(3)


def level_R():
    # columns are camera x (right), y (down), z (forward) expressed in world coordinates
    return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


def test_level_camera_has_zero_roll_and_pitch():
    roll, pitch, yaw = camera_euler_deg(level_R())
    assert abs(roll) < 1e-6 and abs(pitch) < 1e-6
    assert abs(yaw - 90.0) < 1e-6          # forward is world +y


def test_turning_the_camera_changes_yaw_only():
    R = R_about_world_z(30) @ level_R()
    roll, pitch, yaw = camera_euler_deg(R)
    assert abs(roll) < 1e-6 and abs(pitch) < 1e-6 and abs(yaw - 120.0) < 1e-6


def test_pitching_up_gives_positive_pitch():
    tilt = q_to_R(q_from_axis_angle((1, 0, 0), math.radians(20)))   # rotate about world x: raises the +y direction
    roll, pitch, yaw = camera_euler_deg(tilt @ level_R())
    assert abs(pitch - 20.0) < 1e-6


def test_drawing_changes_with_orientation_and_has_the_right_size():
    a = draw_camera_3d(level_R(), size=200)
    b = draw_camera_3d(R_about_world_z(60) @ level_R(), size=200)
    assert a.shape == (200, 200, 3) and a.dtype == np.uint8
    assert int(np.abs(a.astype(int) - b.astype(int)).sum()) > 5000       # clearly different pictures
    assert int(a.sum()) > 0
