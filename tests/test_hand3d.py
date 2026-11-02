import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.hands import Hand
from gabai.pipeline.hand3d import HandSmoother, HeldObjectModel, extract_object_points, hand_frame, oriented_box, solve_hand_3d

F, B, CX, CY, W, H = 881.0, 0.0507, 319.5, 239.5, 640, 480

LOCAL = np.array([
    [0, 0, 0], [-.03, .02, 0], [-.055, .045, 0], [-.07, .065, 0], [-.08, .085, 0],
    [-.03, .09, 0], [-.03, .125, 0], [-.03, .145, 0], [-.03, .165, 0],
    [0, .095, 0], [0, .135, 0], [0, .16, 0], [0, .185, 0],
    [.027, .09, 0], [.027, .125, 0], [.027, .148, 0], [.027, .17, 0],
    [.052, .08, 0], [.052, .108, 0], [.052, .125, 0], [.052, .143, 0]], float)


def make(R_ch=None, t_ch=(0.0, 0.0, 0.8), bump=0.0):
    """A hand with known 3D joints in the camera frame, its 2D projection and MediaPipe-style relative depth."""
    R_ch = R.from_euler("x", 25, degrees=True).as_matrix() if R_ch is None else R_ch
    loc = LOCAL.copy()
    loc[:, 2] += bump * np.sin(np.arange(21))                          # a little relief
    X = loc @ R_ch.T + np.asarray(t_ch)
    uv = np.stack([X[:, 0] / X[:, 2] * F + CX, X[:, 1] / X[:, 2] * F + CY], 1)
    zp = X[[0, 5, 9, 13, 17], 2].mean()
    z_rel = (X[:, 2] - X[0, 2]) * F / (W * zp)
    return Hand("Right", uv, 0.95, z_rel), X


def disp_for(X, hole_idx=(), bg=2.5, radius=8):
    d = np.full((H, W), F * B / bg, np.float32)
    zp = X[[0, 5, 9, 13, 17], 2].mean()
    uv = np.stack([X[:, 0] / X[:, 2] * F + CX, X[:, 1] / X[:, 2] * F + CY], 1)
    for i, (u, v) in enumerate(uv):
        y0, y1, x0, x1 = int(v) - radius, int(v) + radius, int(u) - radius, int(u) + radius
        d[max(y0, 0):y1, max(x0, 0):x1] = F * B / X[i, 2]
    for i in hole_idx:
        u, v = uv[i]
        d[int(v) - 10:int(v) + 10, int(u) - 10:int(u) + 10] = np.nan
    return d


def rms(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def test_all_joints_with_stereo_depth_are_recovered_within_two_centimetres():
    hand, X = make(bump=0.004)
    r = solve_hand_3d(hand, disp_for(X), F, B, CX, CY)
    assert r["ok"] and rms(r["xyz"], X) < 0.02, rms(r["xyz"], X)
    assert (r["source"] == "stereo").sum() >= 17 and r["scale"]["plausible"]


def test_joints_without_depth_are_filled_from_the_models_relative_depth_and_say_so():
    hand, X = make()
    holes = (4, 8, 12, 16, 20)                                          # the fingertips
    r = solve_hand_3d(hand, disp_for(X, hole_idx=holes), F, B, CX, CY)
    assert r["ok"] and all(str(r["source"][i]).startswith("model") for i in holes)
    assert rms(r["xyz"], X) < 0.03 and np.isfinite(r["xyz"]).all()
    assert r["n_model"] >= 5


def test_a_depth_that_makes_the_hand_inhumanly_large_is_rejected_not_used():
    hand, X = make()
    d = np.full((H, W), F * B / 2.5, np.float32)                        # the pictures say the hand is at 0.8 m; the depth map says 2.5 m
    r = solve_hand_3d(hand, d, F, B, CX, CY)
    assert not r["ok"] and "size" in r["reason"] and r["scale"]["plausible"] is False


def test_no_trusted_depth_anywhere_gives_no_3d_hand_and_the_reason():
    hand, X = make()
    r = solve_hand_3d(hand, np.full((H, W), np.nan, np.float32), F, B, CX, CY)
    assert not r["ok"] and "depth" in r["reason"] and np.isnan(r["xyz"]).all()


def test_a_flying_pixel_that_disagrees_with_the_hand_is_overruled_by_the_model():
    hand, X = make()
    d = disp_for(X)
    u, v = hand.kp2d[8].astype(int)
    d[v - 3:v + 4, u - 3:u + 4] = F * B / 0.4                           # a nearer artefact over the index fingertip
    r = solve_hand_3d(hand, d, F, B, CX, CY)
    assert "disagree" in str(r["source"][8]) and abs(r["xyz"][8, 2] - X[8, 2]) < 0.03


def test_the_hand_frame_follows_the_hand_orientation():
    for yaw, pitch in ((0, 0), (40, 0), (0, 50), (-30, 20)):
        Rh = R.from_euler("zy", [yaw, pitch], degrees=True).as_matrix()
        hand, X = make(Rh)
        Rf, t = hand_frame(X)
        Rf2, t2 = hand_frame(make(Rh, (0.1, 0.05, 0.9))[1])
        rel = R.from_matrix(Rf2 @ Rf.T).magnitude()                      # same hand moved: the frame must not rotate
        assert np.degrees(rel) < 2.0
        assert np.linalg.det(Rf) > 0.99 and np.allclose(Rf.T @ Rf, np.eye(3), atol=1e-6)
    a, _ = hand_frame(make(R.from_euler("z", 0, degrees=True).as_matrix())[1])
    b, _ = hand_frame(make(R.from_euler("z", 45, degrees=True).as_matrix())[1])
    assert abs(np.degrees(R.from_matrix(b @ a.T).magnitude()) - 45) < 6


def test_the_hand_frame_needs_enough_palm_points():
    X = make()[1].copy()
    X[[0, 5, 9], :] = np.nan
    assert hand_frame(X) is None


def test_smoothing_calms_jitter_but_does_not_lag_a_fast_move():
    rng = np.random.default_rng(0)
    sm = HandSmoother()
    base = make()[1]
    raw, out = [], []
    for k in range(40):
        X = base + rng.normal(0, 0.006, base.shape)
        raw.append(X)
        out.append(sm.update("R", X))
    assert np.std(np.array(out)[10:] - base, axis=0).mean() < 0.7 * np.std(np.array(raw)[10:] - base, axis=0).mean()
    sm2 = HandSmoother()
    for k in range(5):
        sm2.update("R", base)
    jumped = sm2.update("R", base + np.array([0.25, 0, 0]))              # the hand really moved 25 cm in one frame
    assert np.linalg.norm(jumped.mean(0) - (base + np.array([0.25, 0, 0])).mean(0)) < 0.02


def trajectory(n=40):
    for k in range(n):
        a = k / (n - 1)
        Rh = R.from_euler("z", 60 * a, degrees=True).as_matrix() @ R.from_euler("x", 25, degrees=True).as_matrix()
        yield Rh, np.array([0.4 * a, 0.1 * a, 0.8 + 0.05 * a])


def test_the_object_model_keeps_the_object_rigidly_in_the_hand_after_it_can_no_longer_be_seen():
    rng = np.random.default_rng(1)
    obj_local = rng.uniform([-0.04, 0.0, 0.02], [0.04, 0.12, 0.10], (3000, 3))          # a block held in the palm
    m = HeldObjectModel("cup0", "cup", voxel=0.01)
    errs = []
    for k, (Rh, th) in enumerate(trajectory()):
        X = LOCAL @ Rh.T + th
        Rf, tf = hand_frame(X)
        truth_world = (obj_local @ Rh.T + th)                                          # the object is rigid in the hand
        if k < 12:                                                                    # visible: points measured with 5 mm noise
            pick = rng.choice(len(truth_world), 700, replace=False)
            m.add_world_points(truth_world[pick] + rng.normal(0, 0.005, (700, 3)), Rf, tf)
        elif k >= 20:
            got = m.points_world(Rf, tf, min_obs=1)
            d = np.linalg.norm(got[:, None, :] - truth_world[None, ::3, :], axis=2).min(axis=1)
            errs.append(float(np.median(d)))
    assert m.n_views == 12 and len(errs) == 20
    assert max(errs) < 0.012, max(errs)                                                # within about a centimetre at the end of a 40 cm, 60 degree move


def test_an_oriented_box_recovers_size_and_turn_of_a_block():
    rng = np.random.default_rng(2)
    half = np.array([0.04, 0.025, 0.06])
    pts = rng.uniform(-half, half, (3000, 3))
    Rb = R.from_euler("z", 30, degrees=True).as_matrix()
    b = oriented_box(pts @ Rb.T + np.array([0.01, 0.2, 0.05]))
    assert b is not None and np.allclose(sorted(b["half_sizes"]), sorted(half), atol=0.008)
    assert np.linalg.norm(b["center"] - [0.01, 0.2, 0.05]) < 0.01
    assert oriented_box(np.zeros((5, 3))) is None


def test_object_points_are_taken_from_the_mask_minus_the_hand_and_within_the_objects_depth():
    disp = np.full((H, W), F * B / 2.0, np.float32)                      # background at 2 m
    disp[200:300, 280:360] = F * B / 0.7                                 # the object at 0.7 m
    mask = np.zeros((H, W), bool)
    mask[180:320, 260:380] = True                                        # the detector's mask also takes in some background
    hand_px = np.zeros((H, W), bool)
    hand_px[200:300, 280:300] = True                                     # the hand covers the left strip of the object
    pts, info = extract_object_points(disp, np.ones((H, W), bool), mask, hand_px, F, B, CX, CY, z_ref=0.7)
    assert len(pts) > 200 and abs(np.median(pts[:, 2]) - 0.7) < 0.01 and pts[:, 2].max() < 0.85
    u = pts[:, 0] / pts[:, 2] * F + CX
    assert u.min() >= 299                                                # nothing from under the hand
    assert info["dropped_background"] > 0 and info["dropped_hand"] > 0


def test_with_assumed_scale_depth_the_size_test_is_skipped_and_the_scale_factor_is_reported():
    hand, X = make()
    d = np.full((H, W), F * B / 2.5, np.float32)                        # same oversize hand as above, but the depth scale is only assumed
    r = solve_hand_3d(hand, d, F, B, CX, CY, metric=False)
    assert r["ok"] and r["scale"]["plausible"] is None and r["scale"]["depth_not_metric"]
    assert 0.2 < r["scale"]["scale_factor_if_hand_is_9p5cm"] < 0.4         # a hand that reads 30 cm would be 9.5 cm with a depth scale 0.32 times as large
