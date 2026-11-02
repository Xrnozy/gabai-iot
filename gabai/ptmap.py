"""Points of the raw sent pictures put into the calibration's rectified pictures. numpy + cv2 only, so the Pi can use it too.

The sender's picture is the camera's own (distorted, not turned); the calibration was made on pictures turned by `rotate_left` / `rotate_right` and the rectified result is what
depth and pose are computed on. A point found by the Pi has to go through the same steps to mean the same spot.
"""
import cv2
import numpy as np


def turn_points(xy, deg, w, h):
    """Where points of a w x h picture land when the picture is turned by deg (0/90/180/270, clockwise) like gabai.pc.receiver does."""
    x, y = xy[:, 0], xy[:, 1]
    deg %= 360
    if deg == 90:
        return np.stack([h - 1 - y, x], 1)
    if deg == 180:
        return np.stack([w - 1 - x, h - 1 - y], 1)
    if deg == 270:
        return np.stack([y, w - 1 - x], 1)
    return xy


class RectPoints:
    """Raw left / right picture points -> rectified pixel coordinates of that camera (calibration size). Also the rectified intrinsics and baseline."""

    def __init__(self, calib_path, sent_size):
        z = np.load(calib_path)
        self.cams = {"L": tuple(np.asarray(z[n], np.float64) for n in ("K1", "D1", "R1", "P1")), "R": tuple(np.asarray(z[n], np.float64) for n in ("K2", "D2", "R2", "P2"))}
        self.rotation = (int(z["rotate_left"]), int(z["rotate_right"])) if "rotate_left" in z.files else (0, 0)
        self.size = tuple(int(v) for v in z["size"])
        self.sent = tuple(sent_size)
        P1 = self.cams["L"][3]
        self.f, self.cx, self.cy, self.B = float(P1[0, 0]), float(P1[0, 2]), float(P1[1, 2]), float(z["baseline"])
        self.K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    def __call__(self, xy, side="L", pre180=False, raw=None):
        """(n, 2) float64 rectified pixels, NaN where the input is NaN. pre180: the picture is turned 180 deg again before rectification (a swapped camera layout). raw: which camera the points were found in, when it is not `side`."""
        xy = np.asarray(xy, np.float64)
        out = np.full((len(xy), 2), np.nan)
        ok = np.isfinite(xy).all(axis=1)
        if not ok.any():
            return out
        w, h = self.sent
        rot = self.rotation[0 if (raw or side) == "L" else 1]
        p = turn_points(xy[ok], rot, w, h)
        if pre180:
            rw, rh = (h, w) if rot % 180 else (w, h)
            p = turn_points(p, 180, rw, rh)
        K, D, R, P = self.cams[side]
        out[ok] = cv2.undistortPoints(p.reshape(-1, 1, 2), K, D, R=R, P=P).reshape(-1, 2)
        return out
