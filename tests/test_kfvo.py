import cv2
import numpy as np

from gabai.kfvo import KeyframeVO
from gabai.tools import vo_bench as vb
from gabai.tools.video_cloud_view import StereoVO

F, B, W, H = 881.0, 0.0507, 640, 480
ALL = np.ones((H, W), bool)


def texture(seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, (H, W), dtype=np.uint8), (0, 0), 1.5)


def shift(img, dx):
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REFLECT)


def plane(z):
    return np.full((H, W), F * B / z, np.float32)


def make(**kw):
    vo = KeyframeVO(F, F, W / 2, H / 2, **kw)
    vo.set_baseline(B)
    return vo


def test_sideways_motion_in_front_of_a_plane_is_recovered_in_metres():
    vo = make()
    img = texture()
    first = vo.step(img, plane(2.0), ALL)
    for k in range(1, 6):
        out = vo.step(shift(img, -10 * k), plane(2.0), ALL)
    moved = np.linalg.norm(out[1] - first[1])
    expect = 5 * 10 * 2.0 / F
    assert out[2]["ok"] and abs(moved - expect) < 0.25 * expect, (moved, expect)


def test_no_motion_gives_no_movement_and_no_new_keyframes_pile_up():
    vo = make(kf_max_frames=100)
    img = texture(1)
    for _ in range(10):
        out = vo.step(img, plane(2.0), ALL)
    assert np.linalg.norm(out[1]) < 0.01 and vo.n_keyframes == 1


def test_a_blank_picture_is_reported_lost_and_the_pose_is_held():
    vo = make()
    img = texture(2)
    vo.step(img, plane(2.0), ALL)
    _, t1, _ = vo.step(shift(img, -8), plane(2.0), ALL)
    _, t2, info = vo.step(np.full((H, W), 128, np.uint8), plane(2.0), ALL)
    assert not info["ok"] and info["reason"] and np.allclose(t2, t1)


def test_without_trusted_depth_there_is_nothing_to_track_from():
    vo = make()
    d = plane(2.0)
    d[:] = np.nan
    _, _, info = vo.step(texture(3), d, np.zeros((H, W), bool))
    assert not info["ok"] and "depth" in info["reason"]


def test_a_new_keyframe_is_made_when_the_camera_has_moved_and_the_path_stays_continuous():
    vo = make(kf_dist_m=0.05)
    img = texture(4)
    ts = []
    for k in range(12):
        out = vo.step(shift(img, -12 * k), plane(2.0), ALL)
        ts.append(out[1].copy())
    assert vo.n_keyframes >= 3
    steps = np.linalg.norm(np.diff(np.array(ts), axis=0), axis=1)
    assert steps.max() < 3 * steps.mean() + 0.01, steps                      # no jump at a keyframe change


def test_correct_moves_the_pose_and_later_frames_continue_from_it():
    vo = make()
    img = texture(5)
    vo.step(img, plane(2.0), ALL)
    vo.step(shift(img, -6), plane(2.0), ALL)
    R, t, _ = vo.step(shift(img, -12), plane(2.0), ALL)
    t_fix = t + np.array([0.0, 0.0, 0.10])
    vo.correct(R, t_fix)
    _, t_after, info = vo.step(shift(img, -18), plane(2.0), ALL)
    assert info["ok"] and np.linalg.norm(t_after - t_fix) < 0.1 and abs((t_after - t_fix)[2]) < 0.03, (t_after, t_fix)


def test_a_rotation_prior_that_disagrees_holds_the_pose_and_one_that_agrees_does_not():
    img = texture(6)
    for prior, expect_ok in ((np.eye(3), True), (cv2.Rodrigues(np.array([0.0, 0.5, 0.0]))[0], False)):
        vo = make()
        vo.step(img, plane(2.0), ALL)
        _, _, info = vo.step(shift(img, -5), plane(2.0), ALL, dR_prior=prior)
        assert "prior_disagreement_deg" in info
        assert info["ok"] == expect_ok, info


def test_a_prior_that_agrees_with_the_picture_leaves_the_estimate_where_it_was():
    img = texture(7)
    a, b = make(), make()
    for vo, prior in ((a, None), (b, np.eye(3))):
        vo.step(img, plane(2.0), ALL)
        vo.step(shift(img, -8), plane(2.0), ALL, dR_prior=prior)
    assert np.linalg.norm(a.t_wc - b.t_wc) < 0.005 and np.allclose(a.R_wc, b.R_wc, atol=0.005)


def test_it_has_the_same_interface_as_the_frame_to_frame_odometry():
    img = texture(8)
    for cls in (StereoVO, KeyframeVO):
        vo = cls(F, F, W / 2, H / 2)
        vo.set_baseline(B)
        R, t, info = vo.step(img, plane(2.0), ALL)
        assert R.shape == (3, 3) and t.shape == (3,) and {"ok", "reason", "inliers", "tracked"} <= set(info)
        vo.step(shift(img, -5), plane(2.0), ALL)
        assert vo.last_flow is not None and len(vo.last_flow[0]) == len(vo.last_flow[1])


def test_on_a_rendered_wall_with_noisy_depth_it_drifts_no_more_than_the_original():
    tex = vb.texture()
    res = {}
    for name, cls in (("frame", StereoVO), ("key", KeyframeVO)):
        vo = cls(vb.F, vb.F, vb.W / 2, vb.H / 2)
        vo.set_baseline(vb.B)
        res[name] = vb.run_vo(vo, tex, 80, seed=1)
    assert res["key"]["lost"] == 0 and res["key"]["final_err_cm"] < 3.0, res
    assert res["key"]["final_err_cm"] <= res["frame"]["final_err_cm"] * 1.5 + 0.5, res


def test_an_object_solve_that_does_not_make_the_objects_more_consistent_moves_nothing():
    vo = make()
    vo.step(texture(1), plane(2.0), ALL)
    node = vo.node
    R0, t0 = node["R"].copy(), node["t"].copy()
    info = dict(lm_ids=[0], lm_cm_before=5.0, lm_cm_after=4.9, max_shift_m=0.04, max_rot_deg=1.0, cost_before=1.0, cost_after=0.9, landmarks=np.zeros((1, 3)))
    vo._gout = ([node["id"]], R0[None], t0[None], (R0 + 0.01)[None], (t0 + 0.04)[None], info)
    v = vo.loop_version
    vo._apply_graph()
    assert vo.stats["landmark_unhelpful"] == 1 and vo.loop_version == v and np.allclose(node["t"], t0)
