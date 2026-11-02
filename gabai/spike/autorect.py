"""Uncalibrated stereo rectification estimated from the footage itself.

Matches SIFT features across several frame pairs, fits a fundamental matrix with RANSAC, and gets two
homographies that put matching points on the same row (cv2.stereoRectifyUncalibrated). This fixes roll and
vertical offset between the cameras. It does NOT fix lens distortion and does NOT give metric scale.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Rectification:
    H1: np.ndarray
    H2: np.ndarray
    size: tuple
    residual_before: float
    residual_after: float
    n_matches: int


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def match(left, right, max_pts=1500):
    sift = cv2.SIFT_create(max_pts)
    k1, d1 = sift.detectAndCompute(_gray(left), None)
    k2, d2 = sift.detectAndCompute(_gray(right), None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    knn = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return p1, p2


def vertical_residual(left, right):
    """Median |y_left - y_right| of matched features, in pixels (0 = row-aligned)."""
    p1, p2 = match(left, right)
    if len(p1) < 8:
        return float("nan")
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.5, 0.99)
    if F is None:
        return float("nan")
    keep = mask.ravel() > 0
    return float(np.median(np.abs(p1[keep, 1] - p2[keep, 1])))


def estimate_rectification(pairs):
    size = (pairs[0][0].shape[1], pairs[0][0].shape[0])
    P1, P2 = [], []
    for l, r in pairs:
        a, b = match(l, r)
        P1.append(a)
        P2.append(b)
    p1, p2 = np.vstack(P1), np.vstack(P2)
    if len(p1) < 30:
        raise ValueError(f"only {len(p1)} feature matches; footage too blurry or too few frames")
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.5, 0.999)
    keep = mask.ravel() > 0
    p1, p2 = p1[keep], p2[keep]
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(p1, p2, F, size, threshold=3.0)
    if not ok:
        raise ValueError("stereoRectifyUncalibrated failed")
    before = float(np.median(np.abs(p1[:, 1] - p2[:, 1])))
    q1 = cv2.perspectiveTransform(p1[None], H1)[0]
    q2 = cv2.perspectiveTransform(p2[None], H2)[0]
    after = float(np.median(np.abs(q1[:, 1] - q2[:, 1])))
    return Rectification(H1, H2, size, before, after, len(p1))


@dataclass
class RowAlignment:
    M: np.ndarray  # 2x3 affine for the RIGHT image only; x is untouched so disparity is preserved
    residual_before: float
    residual_after: float
    n_matches: int


def estimate_row_alignment(pairs):
    """Fit y_left = a*y_right + b*x_right + c on matched features and return the right-image warp.

    Only rows move (x stays), so horizontal disparity, and therefore depth scale, is not distorted.
    The full homography version above gave a sheared left image and flattened disparities on real footage.
    """
    P1, P2 = [], []
    for l, r in pairs:
        a, b = match(l, r)
        if len(a) >= 8:
            F, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 2.0, 0.99)
            if F is not None:
                k = mask.ravel() > 0
                P1.append(a[k])
                P2.append(b[k])
    if not P1:
        raise ValueError("no usable feature matches")
    p1, p2 = np.vstack(P1), np.vstack(P2)
    if len(p1) < 30:
        raise ValueError(f"only {len(p1)} feature matches; footage too blurry or too few frames")
    A = np.column_stack([p2[:, 1], p2[:, 0], np.ones(len(p2))])
    keep = np.ones(len(p1), bool)
    for _ in range(5):  # iteratively drop outliers
        coef, *_ = np.linalg.lstsq(A[keep], p1[keep, 1], rcond=None)
        res = np.abs(A @ coef - p1[:, 1])
        keep = res < max(3 * np.median(res[keep]), 1.0)
    a_, b_, c_ = coef
    if abs(a_ - 1) > 0.2 or abs(b_) > 0.3:  # more than ~17 degrees of roll: the matches were bad, not the rig
        raise ValueError(f"implausible alignment (scale {a_:.2f}, roll term {b_:.2f}); matches unreliable, "
                         "try a brighter, more textured scene")
    M = np.array([[1, 0, 0], [b_, a_, c_]], np.float64)
    before = float(np.median(np.abs(p1[:, 1] - p2[:, 1])))
    after = float(np.median(np.abs(A @ coef - p1[:, 1])))
    return RowAlignment(M, before, after, int(keep.sum()))
