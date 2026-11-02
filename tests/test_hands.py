import numpy as np

from gabai.hands import HAND_CONNECTIONS, Hand, HandsUnavailable, HandTracker, hand_mask, lift_hand, open_ratio, finger_state

F, B, CX, CY = 881.0, 0.0507, 319.5, 239.5


def make_hand(open_=True, center=(320, 240), size=80, side="Right"):
    """21 keypoints in pixels: wrist at the bottom, fingers pointing up; closed = fingertips folded back to near the knuckles."""
    cx, cy = center
    k = np.zeros((21, 2), float)
    k[0] = (cx, cy + size)
    mcp = {5: -0.45, 9: -0.15, 13: 0.15, 17: 0.45}
    for base, dx in mcp.items():
        k[base] = (cx + dx * size, cy)
        for j, frac in enumerate((0.0, 0.55, 0.9)):           # pip, dip, tip
            reach = (frac if open_ else 0.15 * (1 - j * 0.3)) * size
            k[base + 1 + j] = (cx + dx * size, cy - reach if open_ else cy + 0.1 * size)
    for j in range(4):
        k[1 + j] = (cx - 0.4 * size - j * 0.1 * size, cy + 0.6 * size - j * 0.25 * size)
    return Hand(side, k, 0.9)


def test_an_open_hand_reads_open_and_a_fist_reads_closed_with_the_ratio_given():
    o, c = open_ratio(make_hand(True)), open_ratio(make_hand(False))
    assert o > 1.5 and c < 1.2 and o > c
    assert finger_state(make_hand(True))["open"] and finger_state(make_hand(False))["closed"]


def test_the_state_between_open_and_closed_is_neither_not_a_guess():
    h = make_hand(True)
    mid = Hand("Right", (h.kp2d + make_hand(False).kp2d) / 2, 0.9)
    s = finger_state(mid)
    assert not (s["open"] and s["closed"])


def test_lifting_gives_metric_xyz_where_depth_is_trusted_and_nan_where_not():
    disp = np.full((480, 640), F * B / 0.8, np.float32)
    disp[:, :300] = np.nan
    h = make_hand(True, center=(400, 240))
    x = lift_hand(h, disp, F, B, CX, CY)
    assert x["xyz"].shape == (21, 3) and np.nanmedian(x["xyz"][:, 2]) > 0.79 and abs(np.nanmedian(x["xyz"][:, 2]) - 0.8) < 0.01
    left = lift_hand(make_hand(True, center=(100, 240)), disp, F, B, CX, CY)
    assert left["valid_fraction"] < 0.2 and np.isnan(left["xyz"][:, 2]).sum() > 15
    assert x["kind"] == "estimated"


def test_one_flying_pixel_in_a_keypoint_window_does_not_move_the_depth():
    disp = np.full((480, 640), F * B / 0.8, np.float32)
    h = make_hand(True)
    px = h.kp2d[8].astype(int)
    disp[px[1], px[0]] = F * B / 3.0
    z = lift_hand(h, disp, F, B, CX, CY)["xyz"][8, 2]
    assert abs(z - 0.8) < 0.02


def test_hand_mask_covers_the_hand_and_is_empty_for_no_hands():
    m = hand_mask([make_hand()], (480, 640), grow_px=4)
    assert m[240, 320] and not m[10, 10] and 2000 < m.sum() < 60000
    assert not hand_mask([], (480, 640)).any()


def test_missing_model_file_raises_with_what_to_do_not_a_download():
    try:
        HandTracker(model_path="definitely/missing.task")
        assert False
    except HandsUnavailable as e:
        assert "hand_landmarker.task" in str(e) and "download" in str(e).lower()


def test_connections_are_the_21_point_hand_skeleton():
    assert len(HAND_CONNECTIONS) == 21 and all(0 <= a < 21 and 0 <= b < 21 for a, b in HAND_CONNECTIONS)
