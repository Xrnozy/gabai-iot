import os

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from gabai.kltfuse import KltFusion, KltObs, PointMap
from gabai.net import protocol as P

CALIB = os.path.join("data", "stereo_calib_new.npz")
F, B, CX, CY = 400.0, 0.06, 320.0, 180.0
CAL = {"f": F, "B": B, "cx": CX, "cy": CY}
K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1.0]])


# -- PointMap: the same spot, found in the picture the pipeline really builds ------------------------------------------------------------------------
def _dots(pts, w=640, h=360):
    img = np.zeros((h, w), np.float32)
    for x, y in pts:
        cv2.circle(img, (int(round(x)), int(round(y))), 3, 255, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def _pipeline_left(raw_l, raw_r, rot, rect, k, flip, swapped):
    """PairPrep.frame, step by step (gabai/pipeline/sources.py)."""
    turn = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    t = lambda im, d: im if turn[d] is None else cv2.rotate(im, turn[d])
    left, right = t(raw_l, rot[0]), t(raw_r, rot[1])
    if swapped:
        left, right = cv2.rotate(right, cv2.ROTATE_180), cv2.rotate(left, cv2.ROTATE_180)
    l, r = rect.apply(left, right)
    if k < 1:
        l, r = cv2.resize(l, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), cv2.resize(r, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    if flip:
        l, r = cv2.rotate(r, cv2.ROTATE_180), cv2.rotate(l, cv2.ROTATE_180)
    return l


def _pipeline_right(raw_l, raw_r, rot, rect, k, flip, swapped):
    """The pipeline's RIGHT picture: the same steps as _pipeline_left, taking the other slot."""
    turn = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    t = lambda im, d: im if turn[d] is None else cv2.rotate(im, turn[d])
    left, right = t(raw_l, rot[0]), t(raw_r, rot[1])
    if swapped:
        left, right = cv2.rotate(right, cv2.ROTATE_180), cv2.rotate(left, cv2.ROTATE_180)
    l, r = rect.apply(left, right)
    if k < 1:
        l, r = cv2.resize(l, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), cv2.resize(r, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return cv2.rotate(l, cv2.ROTATE_180) if flip else r


def _centroids(img, n):
    m = (img > 40).astype(np.uint8)
    cnt, lab, stats, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, cnt):
        ys, xs = np.nonzero(lab == i)
        wgt = img[ys, xs]
        out.append(((xs * wgt).sum() / wgt.sum(), (ys * wgt).sum() / wgt.sum()))
    return np.array(out)


@pytest.mark.skipif(not os.path.exists(CALIB), reason="needs the stereo calibration file")
@pytest.mark.parametrize("k,flip,swapped", [(1.0, False, False), (0.5, False, False), (1.0, True, False), (1.0, False, True), (0.5, True, True)])
def test_a_point_lands_where_the_pipeline_picture_shows_it(k, flip, swapped):
    from gabai.tools.calibrate_stereo import Rectifier
    rect = Rectifier(CALIB)
    z = np.load(CALIB)
    rot = (int(z["rotate_left"]), int(z["rotate_right"]))
    rng = np.random.default_rng(3)
    pts_l = np.stack([rng.uniform(120, 520, 12), rng.uniform(80, 280, 12)], 1)
    pts_r = np.stack([rng.uniform(120, 520, 12), rng.uniform(80, 280, 12)], 1)
    raw_l, raw_r = _dots(pts_l), _dots(pts_r)
    img = _pipeline_left(raw_l, raw_r, rot, rect, k, flip, swapped)
    mapped = PointMap(CALIB, (640, 360), rot, k, flip, swapped)(pts_l, pts_r)
    found = _centroids(img, 12)
    d = np.linalg.norm(mapped[:, None] - found[None], axis=2).min(axis=1)
    assert len(found) == 12 and d.max() < 0.8, d


@pytest.mark.skipif(not os.path.exists(CALIB), reason="needs the stereo calibration file")
@pytest.mark.parametrize("k,flip,swapped", [(1.0, False, False), (0.5, False, False), (1.0, True, False), (1.0, False, True), (0.5, True, True)])
def test_a_point_lands_where_the_pipeline_RIGHT_picture_shows_it(k, flip, swapped):
    from gabai.tools.calibrate_stereo import Rectifier
    rect = Rectifier(CALIB)
    z = np.load(CALIB)
    rot = (int(z["rotate_left"]), int(z["rotate_right"]))
    rng = np.random.default_rng(5)
    pts_l = np.stack([rng.uniform(120, 520, 12), rng.uniform(80, 280, 12)], 1)
    pts_r = np.stack([rng.uniform(120, 520, 12), rng.uniform(80, 280, 12)], 1)
    img = _pipeline_right(_dots(pts_l), _dots(pts_r), rot, rect, k, flip, swapped)
    mapped = PointMap(CALIB, (640, 360), rot, k, flip, swapped).right(pts_l, pts_r)
    found = _centroids(img, 12)
    d = np.linalg.norm(mapped[:, None] - found[None], axis=2).min(axis=1)
    assert len(found) == 12 and d.max() < 0.8, d


def test_the_pi_motion_is_turned_with_the_picture_and_refused_for_a_swapped_layout():
    if not os.path.exists(CALIB):
        pytest.skip("needs the stereo calibration file")
    m = {"rvec": np.array([0.01, 0.02, 0.03]), "t": np.array([0.1, -0.2, 0.3]), "dt": 0.07, "n_inl": 50, "rms": 0.5, "prev_seq": 1}
    assert PointMap(CALIB, (640, 360), (180, 0)).motion(m) is m
    f = PointMap(CALIB, (640, 360), (180, 0), flip180=True).motion(m)
    assert np.allclose(f["rvec"], [-0.01, -0.02, 0.03]) and np.allclose(f["t"], [-0.1, 0.2, 0.3])
    assert PointMap(CALIB, (640, 360), (180, 0), swapped=True).motion(m) is None and PointMap(CALIB, (640, 360), (180, 0)).motion(None) is None


def test_obs_drops_lost_points_and_points_outside_the_picture():
    if not os.path.exists(CALIB):
        pytest.skip("needs the stereo calibration file")
    from gabai.pc.receiver import KltPoints
    pm = PointMap(CALIB, (640, 360), (180, 0))
    xy = np.array([[320, 180], [320, 180], [2, 2], [638, 358]], np.float32)
    kp = KltPoints(1.0, 9, np.array([1, 2, 3, 4]), xy, np.array([P.KLT_TRACKED, P.KLT_LOST, P.KLT_NEW, P.KLT_TRACKED], np.uint8), np.array([0, 255, 0, 8], np.uint8), None, 1.0)
    o = pm.obs(kp)
    assert 1 in o.ids and 2 not in o.ids and 2 in o.lost_ids and o.frame_seq == 9 and (o.uv >= 0).all() and (o.uv[:, 0] <= 639).all() and (o.uv[:, 1] <= 359).all()


def test_right_picture_points_are_needed_when_the_layout_swaps_and_none_were_sent():
    if not os.path.exists(CALIB):
        pytest.skip("needs the stereo calibration file")
    assert PointMap(CALIB, (640, 360), (180, 0), swapped=True)(np.zeros((3, 2), np.float32), None) is None


# -- KltFusion on a known scene ----------------------------------------------------------------------------------------------------------------------
class FakeVO:
    """Just what KltFusion touches of KeyframeVO."""

    def __init__(self):
        self.R_wc, self.t_wc, self.R_wk, self.t_wk, self.node, self.lost_n = np.eye(3), np.zeros(3), np.eye(3), np.zeros(3), {"id": 1}, 3
        self.nodes = {1: (np.eye(3), np.zeros(3))}

    def node_pose(self, i):
        return self.nodes.get(i)

    def correct(self, Rn, tn):                        # as KeyframeVO.correct
        Rc = Rn @ self.R_wc.T
        self.t_wk = Rc @ self.t_wk + (tn - Rc @ self.t_wc)
        self.R_wk = Rc @ self.R_wk
        self.R_wc, self.t_wc = np.array(Rn), np.array(tn)
        self.nodes[1] = (self.R_wk, self.t_wk)


def _project(X, Rwc, twc):
    Xc = (X - twc) @ Rwc
    return (Xc @ K.T)[:, :2] / Xc[:, 2:3], Xc[:, 2]


def _scene(n=150, seed=0):
    """3D points that the keyframe sees well inside its picture (pixels first, then depth), so they stay in view after a small move."""
    rng = np.random.default_rng(seed)
    u, v, z = rng.uniform(60, 580, n), rng.uniform(50, 310, n), rng.uniform(1.0, 4.0, n)
    return np.stack([(u - CX) * z / F, (v - CY) * z / F, z], 1)


def _disp_at(X, Rwc, twc):
    uv, z = _project(X, Rwc, twc)
    disp = np.full((360, 640), np.nan, np.float32)
    for (u, v), zz in zip(uv, z):
        disp[int(round(v)), int(round(u))] = F * B / zz
    return disp


def _keyframe(fusion, vo, X):
    disp = _disp_at(X, np.eye(3), np.zeros(3))
    uv, _ = _project(X, np.eye(3), np.zeros(3))
    fusion.apply(vo, KltObs(np.arange(len(X)), uv.astype(np.float32)), disp, np.ones_like(disp, bool), np.eye(3), np.zeros(3), {"ok": True, "keyframe": True})
    return disp


R_TRUE, T_TRUE = R.from_rotvec([0.01, 0.03, -0.01]).as_matrix(), np.array([0.05, 0.02, 0.10])


def _frame(fusion, vo, X, disp, tv, Rv=None, ok=True, noise=0.3, keep=None, bad=0.0, seed=1):
    """One frame: the Pi's tracks of the true motion, and the PC tracker's translation tv (its rotation Rv is the IMU's: R_TRUE unless given)."""
    rng = np.random.default_rng(seed)
    uv, _ = _project(X, R_TRUE, T_TRUE)
    uv = uv + rng.normal(0, noise, uv.shape)
    if bad:
        m = rng.random(len(uv)) < bad
        uv[m] = uv[m] + rng.uniform(-60, 60, (m.sum(), 2))
    ids = np.arange(len(X))
    if keep is not None:
        ids, uv = ids[:keep], uv[:keep]
    info = {"ok": ok, "keyframe": False}
    Rv = R_TRUE if Rv is None else Rv
    out = fusion.apply(vo, KltObs(ids, uv.astype(np.float32)), disp, np.ones_like(disp, bool), Rv, tv, info)
    return out, info


def _setup(weight=0.75, **kw):
    vo, X, f = FakeVO(), _scene(), KltFusion(CAL, weight, **kw)
    disp = _keyframe(f, vo, X)
    assert f.kf is not None and len(f.kf["ids"]) > 100
    return vo, X, f, disp


def test_when_the_two_agree_the_translation_leans_toward_klt_and_the_rotation_is_untouched():
    vo, X, f, disp = _setup(0.75, max_step_m=0.05)
    tv = T_TRUE + [0.03, 0, 0]                                                           # the PC tracker is 3 cm off
    (Rf, tf), info = _frame(f, vo, X, disp, tv)
    assert info["klt_used"] == "blend" and 0.5 < info["klt_weight"] <= 0.75
    assert np.linalg.norm(tf - T_TRUE) < 0.5 * np.linalg.norm(tv - T_TRUE)
    assert Rf is R_TRUE or np.array_equal(Rf, R_TRUE)                                    # the IMU's rotation, exactly
    assert np.allclose(vo.t_wc, tf)                                                      # the tracker continues from the fused position


def test_the_rotation_is_never_changed_and_a_rotation_the_tracks_contradict_stops_the_klt_from_being_used():
    vo, X, f, disp = _setup(1.0)
    small = R_TRUE @ R.from_rotvec([0.0, 0.002, 0.0]).as_matrix()                        # 0.1 deg off: the tracks still fit
    (Rf, _), info = _frame(f, vo, X, disp, T_TRUE, Rv=small)
    assert np.array_equal(Rf, small) and info["klt_used"] in ("blend", "limited")
    big = R_TRUE @ R.from_rotvec([0.0, 0.02, 0.0]).as_matrix()                           # 1.1 deg off: about 8 px of misfit, so the KLT translation is not trusted
    vo, X, f, disp = _setup(1.0)
    (Rf, tf), info = _frame(f, vo, X, disp, T_TRUE + [0.01, 0, 0], Rv=big)
    assert np.array_equal(Rf, big) and info["klt_used"] == "vo_only" and np.allclose(tf, T_TRUE + [0.01, 0, 0])


def test_weight_one_is_the_klt_translation_and_weight_zero_the_tracker_translation():
    vo, X, f, disp = _setup(1.0, max_step_m=1.0)
    tv = T_TRUE + [0.03, 0, 0]
    (_, tf), _ = _frame(f, vo, X, disp, tv)
    assert np.linalg.norm(tf - T_TRUE) < 0.01
    vo, X, f, disp = _setup(0.0)
    (_, tf), _ = _frame(f, vo, X, disp, tv)
    assert np.allclose(tf, tv)


def test_a_large_difference_is_closed_over_several_frames_not_in_one_jump():
    vo, X, f, disp = _setup(1.0, max_step_m=0.02)
    tv = T_TRUE + [0.30, 0, 0]                                                           # 30 cm off
    steps, t = [], tv
    for i in range(30):
        (_, t), info = _frame(f, vo, X, disp, t, seed=i)
        steps.append(info["klt_used"])
        if i == 0:
            assert np.linalg.norm(t - tv) <= 0.0201 and info["klt_used"] == "limited"
    assert np.linalg.norm(t - T_TRUE) < 0.02 and steps[0] == "limited"


def test_klt_holds_the_position_of_a_frame_the_pc_tracker_lost_with_the_imu_rotation():
    vo, X, f, disp = _setup()
    (Rf, tf), info = _frame(f, vo, X, disp, np.zeros(3), ok=False)
    assert info["klt_used"] == "rescued" and info["ok"] is True and vo.lost_n == 0
    assert np.array_equal(Rf, R_TRUE) and np.linalg.norm(tf - T_TRUE) < 0.02


def test_with_too_few_tracked_points_the_tracker_translation_stands():
    vo, X, f, disp = _setup()
    tv = T_TRUE + [0.01, 0, 0]
    (_, tf), info = _frame(f, vo, X, disp, tv, keep=6)
    assert info["klt_used"] == "vo_only" and np.allclose(tf, tv)


def test_mismatched_tracks_are_rejected_not_averaged_in():
    vo, X, f, disp = _setup(1.0, max_step_m=1.0)
    (_, tf), info = _frame(f, vo, X, disp, T_TRUE, bad=0.25)
    assert info["klt_used"] in ("blend", "limited") and np.linalg.norm(tf - T_TRUE) < 0.02 and info["klt_inliers"] < 150


def test_new_keyframe_points_are_taken_when_few_of_the_old_ones_are_left():
    vo, X, f, _ = _setup()
    old = f.kf["ids"].copy()
    now = _disp_at(X, R_TRUE, T_TRUE)                                                    # the depth of THIS frame, which is what new keyframe points are taken from
    (_, _), info = _frame(f, vo, X, now, T_TRUE, keep=40)                                # 40 of ~150 points still tracked: below 40%
    assert f.kf["node"] is None and 15 <= len(f.kf["ids"]) <= 40 and set(f.kf["ids"]) <= set(old)
    want = (X[f.kf["ids"]] - T_TRUE) @ R_TRUE                                            # the points, in the new keyframe's own camera frame
    assert np.abs(f.kf["X"] - want).max() < 0.15
