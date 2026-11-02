import cv2
import numpy as np

from gabai.tools.video_cloud_view import StereoVO

F, B, W, H = 881.0, 0.0507, 640, 480


def texture(seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (H, W), dtype=np.uint8), (0, 0), 1.5)
    return cv2.addWeighted(img, 1.6, np.full_like(img, 0), 0, -40)


def shift(img, dx):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def plane(z):
    return np.full((H, W), F * B / z, np.float32)


def test_sideways_motion_in_front_of_a_plane_is_recovered_in_metres():
    vo = StereoVO(F, F, W / 2, H / 2)
    img = texture()
    z = 2.0
    disp = plane(z)
    poses = [vo.step(img, disp, np.ones((H, W), bool))]
    for k in range(1, 6):
        poses.append(vo.step(shift(img, -10 * k), disp, np.ones((H, W), bool)))     # content moves left 10 px per frame = camera moves right
    R_end, t_end, info = poses[-1]
    moved = np.linalg.norm(t_end - poses[0][1])
    expect = 5 * 10 * z / F                                                       # 0.1135 m
    assert info["ok"] and abs(moved - expect) < 0.25 * expect, (moved, expect)


def test_no_motion_gives_no_movement():
    vo = StereoVO(F, F, W / 2, H / 2)
    img = texture(1)
    for _ in range(4):
        out = vo.step(img, plane(2.0), np.ones((H, W), bool))
    assert np.linalg.norm(out[1] - np.zeros(3)) < 0.01


def test_a_blank_image_is_reported_as_lost_and_the_pose_is_held_not_invented():
    vo = StereoVO(F, F, W / 2, H / 2)
    img = texture(2)
    vo.step(img, plane(2.0), np.ones((H, W), bool))
    R1, t1, i1 = vo.step(shift(img, -8), plane(2.0), np.ones((H, W), bool))
    R2, t2, info = vo.step(np.full((H, W), 128, np.uint8), plane(2.0), np.ones((H, W), bool))
    assert not info["ok"] and "feature" in info["reason"] and np.allclose(t2, t1)


def test_without_trusted_depth_there_is_nothing_to_track_from():
    vo = StereoVO(F, F, W / 2, H / 2)
    img = texture(3)
    d = plane(2.0)
    d[:] = np.nan
    R, t, info = vo.step(img, d, np.zeros((H, W), bool))
    assert not info["ok"]
