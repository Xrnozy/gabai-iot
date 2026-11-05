"""The rolling-shutter correction undoes a known row-by-row shift of the right picture (a turn of the rig while the two cameras read their rows in opposite directions)."""
import cv2
import numpy as np

from rtabmap_lab.rolling_shutter import ShutterCorrection


def test_correction_undoes_a_row_dependent_shift():
    h, w, f = 120, 200, 300.0
    left = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 255, (h, w)).astype(np.float32), (0, 0), 2.0)
    sc = ShutterCorrection(1.0, 50.0)
    wy = 0.5                                                                       # rad/s yaw
    mu = (-f * wy * sc.dt(h)).astype(np.float32)                                   # how far the scene moved in the right picture, per row: about -/+ 3.8 px top/bottom
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    right = cv2.remap(left, u - mu[:, None], v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fixed = sc.correct(right, np.array([0.0, wy, 0.0]), f, w / 2, h / 2)
    inner = (slice(5, -5), slice(15, -15))
    assert np.abs(right - left)[inner].mean() > 5 * np.abs(fixed - left)[inner].mean()
    assert np.array_equal(sc.correct(right, None, f, w / 2, h / 2), right)         # no gyro: unchanged
