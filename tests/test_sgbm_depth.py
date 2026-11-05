"""SgbmDepth on a synthetic textured plane: right picture = left shifted by a known disparity; blank pictures give no depth."""
import numpy as np

from rtabmap_lab.sgbm_depth import SgbmDepth
from rtabmap_lab.tools.stereo_stages import plane_noise

CAL = {"f": 617.0, "B": 0.04684}


def pair(d=20):
    rng = np.random.default_rng(0)
    left = np.clip(cv2_blur(rng.integers(0, 255, (240, 320)).astype(np.float32)), 0, 255).astype(np.uint8)
    right = np.roll(left, -d, axis=1)                                     # a point at x in the left picture is at x - d in the right one
    return np.dstack([left] * 3), np.dstack([right] * 3)


def cv2_blur(x):
    import cv2
    return cv2.GaussianBlur(x, (0, 0), 1.2)


def test_known_disparity_gives_metric_depth():
    left, right = pair(20)
    mm, _ = SgbmDepth(max_std=0).depth_mm(left, right, CAL, 0.3, 4.0)
    z = mm[60:180, 140:300]
    ok = z[z > 0] / 1000.0
    assert len(ok) > 0.5 * z.size
    assert abs(np.median(ok) - CAL["f"] * CAL["B"] / 20) < 0.03, np.median(ok)


def test_blank_pictures_stay_empty_and_noise_gate_works():
    blank = np.full((240, 320, 3), 128, np.uint8)
    assert not SgbmDepth().depth_mm(blank, blank, CAL)[0].any()
    left, right = pair(8)                                               # 3.6 m: noise 0.5 px * 3.6^2 / 28.9 = 22 cm > 8 cm
    assert not SgbmDepth(max_std=0.08).depth_mm(left, right, CAL, 0.3, 6.0)[0][60:180, 140:300].any()


def test_plane_noise_reads_the_noise_it_was_given():
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:240, 0:320]
    d = (0.02 * x + 0.01 * y + 10 + rng.normal(0, 0.3, x.shape)).astype(np.float32)
    share, sigma, _ = plane_noise(d)
    assert share > 0.9 and abs(sigma - 0.3) < 0.05, (share, sigma)
