import cv2
import numpy as np

from gabai.spike.autorect import estimate_row_alignment, vertical_residual


def make_pair(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (360, 640), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2.5)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    left = base
    # right = left shifted horizontally (disparity) plus a small roll and a vertical offset: not row-aligned
    M = cv2.getRotationMatrix2D((320, 180), 2.0, 1.0)
    M[0, 2] -= 25
    M[1, 2] += 9
    right = cv2.warpAffine(base, M, (640, 360), borderMode=cv2.BORDER_REFLECT)
    return left, right


def test_rectification_removes_vertical_misalignment():
    pairs = [make_pair(s) for s in range(4)]
    est = estimate_row_alignment(pairs)
    held = make_pair(10)
    before = vertical_residual(*held)
    after = vertical_residual(held[0], cv2.warpAffine(held[1], est.M, (640, 360)))
    assert before > 4.0, before
    assert after < 1.5, after
