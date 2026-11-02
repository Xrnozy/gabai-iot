"""A small 3D picture of the camera and how it is turned, drawn with plain OpenCV (no extra libraries).

R_wc has the camera's axes as columns, written in world coordinates (z up): column 0 = image right, column 1 =
image down, column 2 = viewing direction. The view is fixed (looking at the world from one side and slightly above),
so when the camera turns you see the box and its lens turn.
"""
import math

import cv2
import numpy as np


def camera_euler_deg(R_wc):
    """(roll, pitch, yaw) of the camera in degrees: pitch up is positive, yaw is the heading in the world x-y plane,
    roll to the right (clockwise seen from behind) is positive."""
    right, down, fwd = R_wc[:, 0], R_wc[:, 1], R_wc[:, 2]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))
    yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
    roll = -math.degrees(math.atan2(right[2], -down[2]))
    return roll, pitch, yaw


def _view_matrix(az_deg=-35.0, el_deg=25.0):
    az, el = math.radians(az_deg), math.radians(el_deg)
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(el - math.pi / 2), -math.sin(el - math.pi / 2)],
                   [0, math.sin(el - math.pi / 2), math.cos(el - math.pi / 2)]])
    return Rx @ Rz


def draw_camera_3d(R_wc, size=220, title=None):
    img = np.full((size, size, 3), 28, np.uint8)
    V = _view_matrix()
    dist, f = 5.0, size * 1.15

    def proj(p):
        c = V @ np.asarray(p, float) + np.array([0.0, 0.0, dist])   # camera looks along +z, 'dist' away
        return int(size / 2 + f * c[0] / c[2]), int(size / 2 - f * c[1] / c[2] * 1.0), c[2]

    def line(a, b, color, w=1):
        pa, pb = proj(a), proj(b)
        cv2.line(img, pa[:2], pb[:2], color, w, cv2.LINE_AA)

    for g in np.linspace(-1.5, 1.5, 7):                              # ground grid at z = -1
        line((g, -1.5, -1), (g, 1.5, -1), (60, 60, 60))
        line((-1.5, g, -1), (1.5, g, -1), (60, 60, 60))
    # world axes (BGR): x red, y green, z blue
    line((0, 0, 0), (1.3, 0, 0), (60, 60, 230), 1)
    line((0, 0, 0), (0, 1.3, 0), (60, 200, 60), 1)
    line((0, 0, 0), (0, 0, 1.3), (230, 120, 60), 1)

    # camera body: box 0.8 wide (x), 0.5 tall (y), 0.5 deep (z) in camera coordinates, lens at the front
    hw, hh, hd = 0.4, 0.25, 0.25
    corners = [(sx * hw, sy * hh, sz * hd) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    P = [R_wc @ np.array(c) for c in corners]
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        line(P[a], P[b], (200, 200, 200), 1)
    # viewing direction cone from the front face
    front = [R_wc @ np.array(c) for c in [(-hw, -hh, hd), (hw, -hh, hd), (hw, hh, hd), (-hw, hh, hd)]]
    tip = R_wc @ np.array((0, 0, 1.4))
    for c in front:
        line(c, tip, (0, 200, 255), 1)
    # camera axes: right (red), down (green), forward (yellow thick)
    line((0, 0, 0), R_wc[:, 0] * 0.9, (60, 60, 230), 2)
    line((0, 0, 0), R_wc[:, 1] * 0.9, (60, 200, 60), 2)
    line((0, 0, 0), R_wc[:, 2] * 1.4, (0, 255, 255), 2)
    roll, pitch, yaw = camera_euler_deg(R_wc)
    cv2.putText(img, title or "camera", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"roll {round(roll) + 0:+d}  pitch {round(pitch) + 0:+d}  yaw {round(yaw) + 0:+d}", (6, size - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (110, 110, 110), 1)
    return img
