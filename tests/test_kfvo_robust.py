"""Camera-only robustness of the keyframe odometry: fast rotation, blur, lost pictures, a motion prior (the seam for a gyro later) and loop detection.
The scenes are rendered, so the true camera path is known. They show the method on a flat textured wall with good depth; they do not measure the real world."""
import numpy as np

from gabai.kfvo import KeyframeVO
from gabai.tools import vo_bench as vb

TEX = vb.texture()


def make(**kw):
    vo = KeyframeVO(vb.F, vb.F, vb.W / 2, vb.H / 2, **kw)
    vo.set_baseline(vb.B)
    return vo


OLD = dict(predict=False, refine=False, relocalize=False)


def mean_of(kw, n=80, seeds=2, **run):
    rs = [vb.run_vo(make(**kw), TEX, n, s, **run) for s in range(seeds)]
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}


def test_with_rough_motion_and_blur_the_new_tracker_does_not_lose_more_and_is_not_less_accurate():
    new = mean_of({}, n=100, path=vb.truth_path_fast, blur=True)
    old = mean_of(OLD, n=100, path=vb.truth_path_fast, blur=True)
    assert new["lost"] <= old["lost"] + 0.5 and new["rot_rms_deg"] <= old["rot_rms_deg"] * 1.1 + 0.1 and new["rms_err_cm"] <= old["rms_err_cm"] * 1.2 + 0.3, (new, old)
    assert new["rot_rms_deg"] < 3.0 and new["final_err_cm"] < 12, new


def test_the_pose_comes_back_after_the_picture_is_lost_for_a_few_frames_while_the_camera_keeps_moving():
    kw = dict(path=vb.truth_path_fast, blur=True, dropout=(45, 5))
    with_reloc = mean_of({}, **kw)
    without = mean_of(dict(relocalize=False), **kw)
    assert with_reloc["rot_final_deg"] < 5.0, with_reloc
    assert without["rot_final_deg"] > 2 * with_reloc["rot_final_deg"] + 2, (with_reloc, without)     # without the relocaliser the offset stays for good


def test_the_relocaliser_is_reported_in_the_info_and_in_the_counters():
    vo = make()
    rng = np.random.default_rng(0)
    Rs, ts = vb.truth_path_fast(90, 0)
    ok = np.ones((vb.H, vb.W), bool)
    seen = False
    for i, (R, t) in enumerate(zip(Rs, ts)):
        img, disp = vb.render(TEX, R, t, rng=rng)
        if 55 <= i < 60:
            img = np.full_like(img, 90)
        _, _, info = vo.step(img, disp, ok)
        seen |= bool(info["relocalized"])
    assert seen and vo.stats["relocalized"] >= 1


def test_tracking_starts_from_the_predicted_position_once_there_is_a_motion_to_predict_from():
    vo = make()
    rng = np.random.default_rng(1)
    Rs, ts = vb.truth_path(20)
    inits = []
    for R, t in zip(Rs, ts):
        img, disp = vb.render(TEX, R, t, rng=rng)
        inits.append(vo.step(img, disp, np.ones((vb.H, vb.W), bool))[2]["init"])
    assert inits[1] is None or inits[1] == "previous"                 # nothing to predict from yet
    assert "predicted" in inits[3:] and vo.stats["predicted_ok"] >= 5


def test_a_supplied_rotation_prior_is_used_for_the_search_and_helps_with_fast_turns():
    prior = lambda a, b: a.T @ b                                      # the true rotation between frames: a stand-in for a gyro, only to test the seam
    kw = dict(path=vb.truth_path_fast, blur=True)
    without = mean_of({}, **kw)
    with_prior = mean_of({}, prior_fn=prior, **kw)
    assert with_prior["rot_rms_deg"] <= without["rot_rms_deg"] + 0.2 and with_prior["lost"] <= without["lost"], (with_prior, without)


def test_a_wrong_rotation_prior_cannot_pull_the_estimate_it_can_only_hold_the_frame():
    vo = make()
    rng = np.random.default_rng(2)
    Rs, ts = vb.truth_path(10)
    ok = np.ones((vb.H, vb.W), bool)
    wrong = np.array([[1, 0, 0], [0, np.cos(0.5), -np.sin(0.5)], [0, np.sin(0.5), np.cos(0.5)]])          # 29 degrees about x, nothing like the truth
    held = 0
    for i, (R, t) in enumerate(zip(Rs, ts)):
        img, disp = vb.render(TEX, R, t, rng=rng)
        _, _, info = vo.step(img, disp, ok, dR_prior=None if i < 3 else wrong)
        held += (not info["ok"]) and "prior" in info["reason"]
    assert held >= 1


def test_the_motion_blur_helper_blurs_only_when_there_is_motion():
    img = (np.random.default_rng(0).random((60, 80)) * 255).astype(np.uint8)
    assert np.array_equal(vb.motion_blur(img, 2.0), img) and not np.array_equal(vb.motion_blur(img, 40.0), img)


# -- loop detection ------------------------------------------------------------------------------------------------------------------------------------
def out_and_back(vo, drift=np.array([0.0, 0.06, 0.0]), n=130):
    """Slide along the wall and come back to the start. At the turn, the odometry is moved 6 cm: the drift it would have collected."""
    rng = np.random.default_rng(3)
    ok = np.ones((vb.H, vb.W), bool)
    R0w, t0w = __import__("gabai.sim", fromlist=["look_pose"]).look_pose(np.zeros(3))
    loops, est = [], []
    for i in range(n):
        s = i / (n - 1)
        x = 0.8 * (1 - abs(2 * s - 1))
        img, disp = vb.render(TEX, np.eye(3), np.array([x, 0.0, 0.0]), rng=rng)
        Rw, tw, info = vo.step(img, disp, ok)
        if i == n // 2:
            vo.correct(vo.R_wc, vo.t_wc + drift)
            tw = vo.t_wc
        if info["loop"]:
            loops.append(info["loop"])
        est.append(R0w.T @ (tw - t0w))
    truth_end = np.array([0.0, 0.0, 0.0])
    return loops, float(np.linalg.norm(est[-1] - truth_end))


def test_revisiting_a_place_finds_the_loop_and_measures_the_drift_that_was_collected():
    vo = make(loop_closure=True, loop_apply=False, loop_min_path_m=0.5)
    loops, err_end = out_and_back(vo)
    assert loops and not any(l["applied"] for l in loops)
    assert 3.0 < max(l["drift_cm"] for l in loops) < 10.0, loops        # the injected 6 cm
    assert err_end > 0.04                                               # not applied: the 6 cm is still in the path


def test_with_loop_apply_the_pose_is_pulled_back_after_the_revisit_and_with_it_off_nothing_moves():
    on = make(loop_closure=True, loop_apply=True, loop_min_path_m=0.5)
    _, err_on = out_and_back(on)
    off = make(loop_closure=False, loop_min_path_m=0.5)
    loops_off, err_off = out_and_back(off)
    assert not loops_off and err_off > 0.04
    assert err_on < 0.03 and on.stats["loops_applied"] >= 1, (err_on, on.stats)


def test_two_neighbouring_keyframes_are_not_reported_as_a_loop():
    vo = make(loop_closure=True, loop_apply=True, loop_min_path_m=2.0)
    rng = np.random.default_rng(4)
    ok = np.ones((vb.H, vb.W), bool)
    loops = []
    for i in range(60):
        img, disp = vb.render(TEX, np.eye(3), np.array([0.01 * i, 0, 0]), rng=rng)
        loops.append(vo.step(img, disp, ok)[2]["loop"])
    assert not any(loops)                                              # 0.6 m of travel is under the 2 m a loop needs


def test_a_degenerate_set_of_points_is_a_failed_pose_not_a_crash():
    from gabai.kfvo import _pnp
    obj = np.tile([0.0, 0.0, 2.0], (50, 1))                                  # every point in the same place: OpenCV's solver raises on this
    img = np.random.default_rng(0).uniform(100, 500, (50, 2))
    ok, _, _, inl = _pnp(obj, img, make().K, 100, 2.5)
    assert not ok and inl is None
