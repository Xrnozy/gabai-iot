import json
import os
import tempfile

import cv2
import numpy as np

from gabai.tools.capture_calib_pairs import POSES, Capturer, fresh_folder

BOARD = cv2.imread(os.path.join("data", "checkerboard_9x6.png"))


def scene(cx, cy, scale, angle=0.0, disp=18):
    """A 640x360 left/right pair showing the printed board (the right picture sees it `disp` px further left)."""
    h, w = BOARD.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (cx - w / 2, cy - h / 2)
    L = cv2.warpAffine(BOARD, M, (640, 360), borderValue=(255, 255, 255))
    R = cv2.warpAffine(L, np.float32([[1, 0, -disp], [0, 1, 0]]), (640, 360), borderValue=(255, 255, 255))
    return L, R


def feed(cap, pair, dt=1.0, times=3):
    for _ in range(times):                                         # a still board seen in several pairs in a row
        cap.consider(*pair, dt)
    return cap.status


def test_a_steady_new_pair_is_saved_with_matching_names_and_a_log():
    d = tempfile.mkdtemp()
    cap = Capturer(os.path.join(d, "p"))
    pair = scene(330, 180, 0.30)
    assert feed(cap, pair).startswith("READY")
    assert cap.save(*pair) == 1 and cap.n == 1
    assert os.path.exists(os.path.join(d, "p", "pair_01_L.png")) and os.path.exists(os.path.join(d, "p", "pair_01_R.png"))
    assert cv2.imread(os.path.join(d, "p", "pair_01_L.png")).shape[:2] == (360, 640)
    log = json.load(open(os.path.join(d, "p", "capture_log.json")))
    assert log[0]["pair"] == 1 and log[0]["camera_offset_ms"] == 1.0 and log[0]["pose_asked"] == POSES[0]
    assert json.load(open(os.path.join(d, "p", "rotation.json"))) == [180, 0]


def test_the_same_place_again_is_refused_and_a_new_place_is_accepted():
    cap = Capturer(os.path.join(tempfile.mkdtemp(), "p"))
    a = scene(330, 180, 0.30)
    feed(cap, a), cap.save(*a)
    assert "too close" in feed(cap, a)
    assert cap.save(*a) is None and cap.n == 1
    b = scene(250, 150, 0.22, 20)
    assert feed(cap, b).startswith("READY") and cap.save(*b) == 2


def test_nothing_is_saved_for_a_moving_board_a_cut_off_board_or_cameras_out_of_step():
    cap = Capturer(os.path.join(tempfile.mkdtemp(), "p"))
    a, b = scene(330, 180, 0.30), scene(345, 180, 0.30)
    cap.consider(*a, 1.0)
    assert "hold" in (cap.consider(*b, 1.0), cap.status)[1] and cap.save(*b) is None            # it moved between two pairs
    cut = scene(600, 180, 0.40)                                                                  # half of the board is outside the picture
    assert "not found" in feed(cap, cut)
    still = scene(330, 180, 0.30)
    assert "ms apart" in feed(cap, still, dt=25.0) and cap.save(*still) is None
    assert feed(cap, still, dt=-4.0).startswith("READY")                                         # either sign is fine below the limit


def test_pictures_of_the_wrong_size_are_refused_with_the_sender_flags():
    cap = Capturer(os.path.join(tempfile.mkdtemp(), "p"))
    l = np.zeros((480, 640, 3), np.uint8)
    cap.consider(l, l, 0.0)
    assert "--width 640 --height 360" in cap.status and not cap.ok


def test_exactly_the_goal_number_of_pairs_is_saved_and_undo_removes_the_files():
    cap = Capturer(os.path.join(tempfile.mkdtemp(), "p"), pairs=2)
    for sc in (scene(330, 180, 0.30), scene(250, 150, 0.22, 20), scene(400, 200, 0.18, 40)):
        feed(cap, sc)
        cap.save(*sc)
    assert cap.n == 2 and not os.path.exists(os.path.join(cap.out, "pair_03_L.png"))
    assert cap.undo() == 2 and not os.path.exists(os.path.join(cap.out, "pair_02_R.png")) and cap.n == 1
    assert len(json.load(open(os.path.join(cap.out, "capture_log.json")))) == 1


def test_an_older_folder_is_moved_aside_not_overwritten():
    d = os.path.join(tempfile.mkdtemp(), "pairs")
    os.makedirs(d)
    open(os.path.join(d, "pair_01_L.png"), "w").write("x")
    fresh_folder(d)
    assert not os.path.exists(d) and any(n.startswith("pairs_prev_") for n in os.listdir(os.path.dirname(d)))
