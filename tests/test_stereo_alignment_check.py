import json
import os
import tempfile

import cv2
import numpy as np

from gabai.tools.stereo_alignment_check import diagnose, pair_offset

W, H = 640, 480


def scene(seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (H, W), dtype=np.uint8), (0, 0), 1.2)
    return cv2.addWeighted(img, 1.5, img, 0, -30)


def shift(img, dx, dy):
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)


def test_an_aligned_pair_has_no_vertical_offset_and_a_positive_disparity():
    l = scene()
    r = shift(l, -20, 0)                                   # right image content is 20 px to the left: disparity +20
    o = pair_offset(l, r)
    assert abs(o["dy_px"]) < 1.0 and 15 < o["dx_px"] < 25 and o["n"] >= 10


def test_a_vertical_shift_is_measured_with_its_sign():
    l = scene(1)
    r = shift(l, -20, -12)                                 # right content 12 px higher
    o = pair_offset(l, r)
    assert abs(o["dy_px"] - (-12)) < 1.5


def test_no_texture_gives_none_not_zero():
    o = pair_offset(np.full((H, W), 100, np.uint8), np.full((H, W), 100, np.uint8))
    assert o is None


def make_rec(dy, n=8, gyro_dps=2.0, scale_with_gyro=False):
    d = tempfile.mkdtemp()
    for cam in ("cam0", "cam1"):
        os.makedirs(os.path.join(d, "mav0", cam, "data"))
    os.makedirs(os.path.join(d, "mav0", "imu0"))
    rows = []
    for k in range(n):
        ts = int((10 + k * 0.1) * 1e9)
        l = scene(k)
        r = shift(l, -20, dy)
        cv2.imwrite(os.path.join(d, "mav0", "cam0", "data", f"{ts}.png"), l)
        cv2.imwrite(os.path.join(d, "mav0", "cam1", "data", f"{ts}.png"), r)
        rows.append(f"{ts},{ts}.png")
    open(os.path.join(d, "mav0", "cam0", "data.csv"), "w").write("#t,f\n" + "\n".join(rows) + "\n")
    t = (9 + np.arange(400) * 0.005) * 1e9
    w = np.zeros((400, 3))
    w[:, 2] = np.radians(gyro_dps)
    np.savetxt(os.path.join(d, "mav0", "imu0", "data.csv"), np.c_[t, w, np.tile([0, 0, 9.8], (400, 1))], delimiter=",", header="#t", comments="")
    return d


def test_a_recording_with_a_constant_vertical_offset_while_still_is_called_a_static_misalignment():
    r = diagnose(make_rec(dy=-14, gyro_dps=2.0), every=1)
    assert r["verdict"] == "static_misalignment" and abs(r["median_dy_px"] + 14) < 2 and "calibrat" in r["advice"].lower()


def test_a_recording_that_is_aligned_passes():
    r = diagnose(make_rec(dy=0), every=1)
    assert r["verdict"] == "aligned" and abs(r["median_dy_px"]) < 1.0


def test_a_recording_without_enough_texture_says_it_cannot_tell():
    d = make_rec(dy=0, n=2)
    for f in os.listdir(os.path.join(d, "mav0", "cam0", "data")):
        cv2.imwrite(os.path.join(d, "mav0", "cam0", "data", f), np.full((H, W), 90, np.uint8))
    r = diagnose(d, every=1)
    assert r["verdict"] == "unknown"
