"""ChArUco calibration on a synthetic stereo rig: the board is rendered into two cameras with a known focal length and baseline."""
import cv2
import numpy as np

from gabai.tools.charuco_calibrate import Board, Session

F, SIZE, BASE = 700.0, (640, 360), 0.042
K = np.array([[F, 0, 320.0], [0, F, 180.0], [0, 0, 1]])


def render(board, rvec, tvec):
    """Board seen by a camera at the origin looking along +z: the board image is warped onto the picture with the plane's homography."""
    ppm = 4.0                                                             # board image pixels per mm
    sx, sy = board.spec["squares_x"], board.spec["squares_y"]
    sq = board.spec["square_mm"]
    img = board.board.generateImage((int(sx * sq * ppm), int(sy * sq * ppm)), marginSize=0, borderBits=1)
    R = cv2.Rodrigues(rvec)[0]
    S = np.diag([1 / (ppm * 1000.0), 1 / (ppm * 1000.0), 1.0])           # board px -> metres
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec.ravel()]) @ S[:, :]
    out = cv2.warpPerspective(img, H, SIZE, flags=cv2.INTER_AREA, borderValue=128)
    return cv2.cvtColor(cv2.GaussianBlur(out, (3, 3), 0.6), cv2.COLOR_GRAY2BGR)


def test_calibrates_known_rig():
    board = Board(5, 7, 25.0, 18.0, "4x4", min_corners=8)
    rng = np.random.default_rng(3)
    sess = Session(board, pairs=14, min_shift=0.0, max_skew_ms=8.0)
    sess.stable = 99
    for _ in range(60):
        rv = np.radians(rng.uniform(-28, 28, 3)) * np.array([1, 1, 0.3])
        R = cv2.Rodrigues(rv)[0]
        centre = np.array([0.0625, 0.0875, 0]) + 0                       # board centre in board coordinates
        t = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.04, 0.04), rng.uniform(0.5, 0.8)]) - R @ centre
        left = render(board, rv, t)
        right = render(board, rv, t - np.array([BASE, 0, 0]) @ np.eye(3))
        dl, dr, status, kept = sess.offer(left, right, 0.8)
        if dl is not None and dr is not None:
            sess.stable = 99
            sess.views.append(dict(zip(("ids", "l", "r"), __import__("gabai.tools.charuco_calibrate", fromlist=["shared"]).shared(dl, dr))))
        if len(sess.views) >= 14:
            break
    assert len(sess.views) >= 10, "board not detected in enough synthetic pairs"
    c = sess.calibrate()
    assert abs(c["baseline"] - BASE) / BASE < 0.03, c["baseline"]
    assert abs(c["K1"][0, 0] - F) / F < 0.04, c["K1"][0, 0]   # the rectified "focal" is zoomed by alpha=0, K1 is the lens
    assert c["rms_stereo"] < 1.0
    assert not c["swapped"]


def test_skew_gate_and_missing_board():
    board = Board(5, 7, 25.0, 18.0, "4x4", min_corners=8)
    sess = Session(board)
    blank = np.full((360, 640, 3), 128, np.uint8)
    assert sess.offer(blank, blank, 0.0)[3] is False
    img = render(board, np.zeros(3), np.array([-0.0625, -0.0875, 0.6]))
    *_, status, kept = sess.offer(img, img, 80.0)
    assert not kept and "apart" in status


def test_wrong_layout_is_named():
    printed = Board(5, 7, 25.0, 18.0, "4x4", min_corners=8)
    img = render(printed, np.zeros(3), np.array([-0.0625, -0.0875, 0.6]))
    *_, status, kept = Session(Board(8, 5, 25.0, 18.0, "4x4", min_corners=8)).offer(img, img)
    assert not kept and "check --squares-x" in status, status


def test_recessed_left_lens():
    """Right lens 2.5 mm ahead of the left one (left sits deeper in the case): the fit recovers the forward offset."""
    board = Board(5, 7, 25.0, 18.0, "4x4", min_corners=8)
    rng = np.random.default_rng(3)
    sess = Session(board, need_stable=0, min_shift=0.0)
    offset = np.array([BASE, 0, 0.0025])                                  # right camera centre in left-camera coordinates
    while len(sess.views) < 20:                                           # close views and strong tilts: from far views alone the forward offset is off by ~2 mm
        rv = np.radians(rng.uniform(-35, 35, 3)) * np.array([1, 1, 0.3])
        t = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.04, 0.04), rng.uniform(0.35, 0.8)]) - cv2.Rodrigues(rv)[0] @ np.array([0.0625, 0.0875, 0])
        left, right = render(board, rv, t), render(board, rv, t - offset)
        dl, dr, _, kept = sess.offer(left, right)
        if kept:
            sess.keep(left, right, dl, dr)
    c = sess.calibrate()
    centre = (-c["R"].T @ c["T"]).ravel()
    assert abs(centre[0] - BASE) < 0.0015 and abs(centre[2] - 0.0025) < 0.0015, centre


def test_wrong_dictionary_is_named():
    img = render(Board(5, 7, 25.0, 18.0, "6x6", min_corners=8), np.zeros(3), np.array([-0.0625, -0.0875, 0.6]))
    status = Session(Board(5, 7, 25.0, 18.0, "4x4", min_corners=8)).offer(img, img)[2]
    assert "--dict 6x6" in status, status
