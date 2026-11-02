"""stereo_imu fusion (gabai/pipeline/vio.py) with a stand-in for the visual odometry, so the IMU part is tested alone."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.pipeline import vio
from gabai.pipeline.vio import G, VisualInertialProvider, tilt_to_z
from gabai.sim import look_pose

CAL = dict(f=600.0, cx=320.0, cy=180.0, B=0.04)
T = 0.01


class FakeVO:
    """Returns the pose it is told to; remembers the prior it was given."""
    def __init__(self):
        self.R_wc, self.t_wc = look_pose(np.zeros(3))[0], np.zeros(3)
        self.prior, self.ok, self.corrected = None, True, None

    def correct(self, R_wc, t_wc):
        self.R_wc, self.t_wc, self.corrected = R_wc, t_wc, (R_wc, t_wc)

    def step(self, gray, disp, trusted, frame_s=None, dR_prior=None, **kw):
        self.prior, self.kw = dR_prior, kw
        return self.R_wc.copy(), self.t_wc.copy(), {"ok": self.ok, "inliers": 99 * self.ok, "tracked": 99, "reason": "" if self.ok else "lost"}


def make(R_ic=np.eye(3)):
    p = VisualInertialProvider(CAL, R_ic)
    p.vo = FakeVO()
    return p


def imu(t0, n, gyro=(0, 0, 0), up_cam=(0, -1, 0)):
    """n samples from t0 on, one every 10 ms; accel reads +up (camera axes here, R_ic = identity)."""
    return [(t0 + i * T, np.asarray(gyro, float), np.asarray(up_cam, float) * G) for i in range(n)]


def push(p, k):
    return p.push(int(k * 1e8), None, None, None, None)[0]


def test_no_pose_before_the_imu_has_delivered():
    p = make()
    assert p.push(0, None, None, None, None) == []
    p.feed_imu(imu(0, 60))
    assert p.ready and push(p, 1).ok


def test_world_is_gravity_aligned_when_the_first_camera_is_tilted():
    p = make()
    tilt = R.from_euler("x", 30, degrees=True).as_matrix()              # camera pitched 30 deg: its 'up' is no longer -y
    up_cam = tilt.T @ np.array([0, -1.0, 0])
    p.feed_imu(imu(0, 60, up_cam=up_cam))
    pose = push(p, 1)
    assert np.allclose(pose.R_wc.T @ [0, 0, 1], up_cam, atol=0.02)      # world z, seen from the camera, is the accelerometer's up


def test_gyro_rotation_between_frames_is_the_prior_and_bias_is_removed():
    p = make()
    p.feed_imu(imu(0, 60, gyro=(0, 0.01, 0)))                           # at rest with a constant 0.57 dps offset: that is bias
    push(p, 1)
    assert p.bias_known and abs(p.bias[1] - 0.01) < 1e-4
    p.feed_imu(imu(0.6, 50, gyro=(0, 0.5 + 0.01, 0)))                   # 0.5 rad/s for 0.5 s (plus the bias)
    push(p, 2)
    assert abs(R.from_matrix(p.vo.prior).magnitude() - 0.25) < 0.01


def test_a_rig_at_rest_is_frozen_and_continues_without_a_jump():
    p = make()
    p.feed_imu(imu(0, 60))
    first = push(p, 1)
    for k in range(2, 8):                                               # the pictures wander 1 cm per frame while the IMU says rest
        p.feed_imu(imu(0.6 + (k - 2) * 0.1, 10))
        p.vo.t_wc = p.vo.t_wc + [0.01, 0, 0]
        out = push(p, k)
    assert out.info["at_rest"] and np.allclose(out.t_wc, first.t_wc)
    p.feed_imu(imu(1.2, 10, gyro=(0.5, 0, 0)))                          # moves again: no jump, then it follows the pictures
    p.vo.t_wc = p.vo.t_wc + [0.01, 0, 0]
    moved = push(p, 8)
    assert not moved.info["at_rest"] and np.linalg.norm(moved.t_wc - out.t_wc) < 0.02


def test_roll_drift_of_the_pictures_is_pulled_back_to_gravity_and_yaw_is_not():
    p = make()
    p.feed_imu(imu(0, 60))
    push(p, 1)
    p.vo.R_wc = R.from_euler("zxy", [10, 5, 0], degrees=True).as_matrix() @ p.vo.R_wc     # 10 deg of yaw and 5 deg of tilt the pictures believe
    t = 0.6
    for k in range(2, 400):
        p.feed_imu(imu(t, 10, gyro=(0.2, 0, 0))); t += 0.1              # a turn: accelerometer quiet, not at rest
        out = push(p, k)
    up_err = np.degrees(np.arccos(np.clip((out.R_wc.T @ [0, 0, 1]) @ [0, -1, 0], -1, 1)))
    assert up_err < 1.0                                                 # the tilt is back within a degree
    assert abs(R.from_matrix(out.R_wc).as_euler("zyx", degrees=True)[0] - R.from_matrix(p.R_w @ p.vo.R_wc).as_euler("zyx", degrees=True)[0]) < 3   # yaw untouched


def test_tilt_to_z_handles_the_poles():
    assert np.allclose(tilt_to_z([0, 0, 1]), np.eye(3)) and np.allclose(tilt_to_z([0, 0, -1]) @ [0, 0, -1], [0, 0, 1])


def test_a_picture_pose_that_jumps_faster_than_a_hand_can_move_is_not_trusted():
    p = make()
    p.feed_imu(imu(0, 60))
    good = push(p, 1)
    p.feed_imu(imu(0.6, 10, gyro=(0.5, 0, 0)))
    push(p, 2)
    p.vo.t_wc = p.vo.t_wc + [1.0, 0, 0]                                 # 1 m in 0.1 s
    p.feed_imu(imu(0.7, 10, gyro=(0.5, 0, 0)))
    bad = push(p, 3)
    assert not bad.ok and "not trusted" in bad.info["reason"] and np.allclose(bad.t_wc, good.t_wc, atol=0.02)
    assert p.vo.corrected is not None and np.allclose(p.vo.t_wc, good.t_wc, atol=0.02)      # the odometry continues from the last good pose


def test_without_tracking_the_position_stands_still_and_the_gyro_turns_the_pose_only_for_two_seconds():
    p = make()
    p.feed_imu(imu(0, 60))
    push(p, 1)
    p.vo.ok = False
    t, priors = 0.6, []
    for k in range(2, 40):                                              # 3.8 s without pictures, turning
        p.feed_imu(imu(t, 10, gyro=(0.5, 0, 0))); t += 0.1
        out = push(p, k)
        priors.append(p.vo.prior is not None)
        assert not out.ok and np.allclose(out.t_wc, 0)
    assert priors[0] and not priors[-1]                                  # the gyro prior is given at first and withdrawn after COAST_S


def test_after_a_stall_the_imu_covered_the_gyro_rotation_is_kept_and_only_the_translation_is_searched():
    p = make()
    p.feed_imu(imu(0, 60))
    push(p, 1)
    p.feed_imu(imu(0.6, 200, gyro=(0.3, 0, 0)))                         # 2 s of gyro for a pair that arrives 2 s later (the stall's samples came with it)
    p.push(int(2.1e9), None, None, None, None)
    assert p.vo.kw == {"prior_over_gap": True, "fix_rot": True} and p.vo.prior is not None
    p.feed_imu(imu(2.6, 6, gyro=(0.3, 0, 0)))                           # a pair 0.05 s later: an ordinary frame, the same rule
    p.push(int(2.15e9), None, None, None, None)
    assert p.vo.kw["fix_rot"] is True                                   # every pair: the gyro rotation, only the translation is searched


def test_a_stall_the_imu_did_not_cover_keeps_the_old_rule():
    p = make()
    p.feed_imu(imu(0, 60))
    push(p, 1)
    p.feed_imu(imu(0.6, 5, gyro=(0.3, 0, 0)))                           # 0.05 s of gyro for a 1.5 s stall
    p.push(int(2.1e9), None, None, None, None)
    assert "prior_over_gap" not in p.vo.kw


def test_a_gyro_that_disagrees_with_the_pictures_is_remembered_for_the_warning():
    p = make()
    p.feed_imu(imu(0, 60))
    push(p, 1)
    orig = p.vo.step

    def step(*a, **k):
        R_, t_, info = orig(*a, **k)
        info["free_vs_prior_deg"] = 25.0
        return R_, t_, info
    p.vo.step = step
    for k in range(2, 25):
        p.feed_imu(imu(0.6 + k * 0.1, 10, gyro=(0.5, 0, 0)))
        push(p, k)
    assert len(p.disagree) == 23 and float(np.median(p.disagree)) == 25.0


def test_gyro_bias_is_learned_from_the_pictures_rotation_error_since_the_keyframe(monkeypatch):
    monkeypatch.setattr(vio, "BIAS_GAIN", 0.05)                       # off by default, see vio.BIAS_GAIN
    p = make()
    p.feed_imu(imu(0, 60))
    push(p, 1)
    orig = p.vo.step

    def step(*a, **k):
        R_, t_, info = orig(*a, **k)
        info.update(free_rot_err=np.array([0.0, 0.02, 0.0]), free_inl=100, keyframe=(p.vo.calls == 0))
        p.vo.calls += 1
        return R_, t_, info
    p.vo.calls, p.vo.step = 0, step
    for k in range(2, 12):                                              # a keyframe on the first call, then pairs 0.5 s apart: 0.02 rad of error over 0.5 s = 0.04 rad/s too fast
        p.feed_imu(imu(0.6 + k * 0.5, 50, gyro=(0.3, 0, 0)))
        p.push(int(k * 5e8 + 1e8), None, None, None, None)
    assert 0.0 < p.bias[1] <= 0.04 and p.bias_known


def test_a_picture_rotation_error_that_big_is_not_taken_for_a_gyro_bias():
    p = make()
    p.feed_imu(imu(0, 60))
    push(p, 1)
    orig = p.vo.step

    def step(*a, **k):
        R_, t_, info = orig(*a, **k)
        info.update(free_rot_err=np.array([0.0, 0.5, 0.0]), free_inl=100, keyframe=True)
        return R_, t_, info
    p.vo.step = step
    for k in range(2, 8):
        p.feed_imu(imu(0.6 + k * 0.5, 50, gyro=(0.3, 0, 0)))
        p.push(int(k * 5e8 + 1e8), None, None, None, None)
    assert abs(p.bias[1]) < 1e-9


def test_a_graph_correction_is_eased_in_not_shown_as_a_jump():
    p = make()
    Rc, tc = R.from_euler("z", 10, degrees=True).as_matrix(), np.array([0.3, 0.0, 0.0])
    p._smooth_in(Rc, tc)
    Rs, ts = p._smooth_out(1 / 15)
    P_old_t = np.array([1.0, 0.0, 0.0])
    P_new_t = Rc @ P_old_t + tc                                                # where the corrected pose really is
    shown0 = Rs @ P_new_t + ts                                                # frame 0: still (nearly) the old pose
    assert np.linalg.norm(shown0 - P_old_t) < 1e-6
    gap = []
    for _ in range(60):                                                       # 4 s later it has eased onto the corrected pose
        Rs, ts = p._smooth_out(1 / 15)
        gap.append(np.linalg.norm(Rs @ P_new_t + ts - P_new_t))
    assert gap[0] < 0.35 and gap[10] < 0.6 * gap[0] and gap[-1] < 0.01


def test_two_corrections_in_a_row_keep_the_shown_pose_continuous():
    p = make()
    C1 = (R.from_euler("z", 5, degrees=True).as_matrix(), np.array([0.1, 0.0, 0.0]))
    C2 = (R.from_euler("z", -3, degrees=True).as_matrix(), np.array([0.0, 0.05, 0.0]))
    x = np.array([1.0, 0.5, 0.0])
    p._smooth_in(*C1)
    p._smooth_out(1 / 15)
    shown_before = (lambda S: S[0] @ (C1[0] @ x + C1[1]) + S[1])(p._S)         # what is on screen right now for the point
    p._smooth_in(*C2)
    new_raw = C2[0] @ (C1[0] @ x + C1[1]) + C2[1]
    shown_after = p._S[0] @ new_raw + p._S[1]
    assert np.linalg.norm(shown_after - shown_before) < 1e-9


# -- the Kalman filter inside the provider ---------------------------------------------------------------------------------------------------------------------
def make_filtered(enabled=True, **kw):
    from gabai.pipeline.config import FilterSection
    p = VisualInertialProvider(CAL, np.eye(3), filt=FilterSection(enabled=enabled, **kw))
    p.vo = FakeVO()
    p._at_rest = lambda: False                                          # the rest hold would freeze the pose; this is about the filter
    return p


def run_static(p, frames=160, noise=0.02, outlier_at=None, motion=False, seed=0):
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    p.feed_imu(imu(0, 60))
    base = p.vo.t_wc.copy()
    out, infos, t = [], [], 0.6
    for k in range(frames):
        p.feed_imu(imu(t, 7))
        t += 0.07
        p.vo.t_wc = base + rng.normal(0, noise, 3) + (np.array([0.15, 0, 0]) if k == outlier_at else 0.0)
        kl = SimpleNamespace(motion={"rvec": np.zeros(3), "t": np.zeros(3), "dt": 0.07, "n_inl": 80, "rms": 0.5}) if motion else None
        pose = p.push(int(t * 1e9), None, None, None, None, klt=kl)[0]
        out.append(pose.t_wc.copy())
        infos.append(pose.info)
    return np.array(out), infos


def test_the_filter_makes_a_noisy_picture_position_steadier():
    raw, _ = run_static(make_filtered(enabled=False))
    filt, _ = run_static(make_filtered())
    s = lambda x: float(np.mean(np.std(x[40:], axis=0)))
    assert s(filt) < 0.6 * s(raw), (s(filt), s(raw))


def test_a_picture_position_outlier_does_not_move_the_reported_pose():
    out, infos = run_static(make_filtered(), outlier_at=90)
    assert np.max(np.linalg.norm(np.diff(out[60:], axis=0), axis=1)) < 0.08           # no jump near the 15 cm outlier (past the speed check of the provider, so only the filter's gate stops it)
    assert infos[-1]["ekf_rejected"] >= 1


def test_the_pi_motion_reaches_the_filter_and_is_applied():
    out, infos = run_static(make_filtered(), motion=True)
    assert infos[-1]["klt_vel_applied"] and infos[-1]["klt_rate_applied"] is not None and "ekf_gyro_bias_dps" in infos[-1]
    assert make_filtered().wants_klt and not make_filtered(enabled=False).wants_klt


def test_the_filter_works_with_a_tracker_that_reports_a_keyframe_anchor():
    p = make_filtered()
    step = p.vo.step
    p.vo.step = lambda *a, **k: (lambda r: (r[0], r[1], dict(r[2], anchor=(1, np.eye(3), np.zeros(3)))))(step(*a, **k))
    out, infos = run_static(p, frames=60)
    assert infos[-1]["anchor"][0] == 1 and np.all(np.isfinite(out))


def test_a_corrupted_imu_read_does_not_wreck_the_filter_and_the_map_gets_a_sane_pose():
    p = make_filtered()
    p.feed_imu(imu(0, 60))
    base, t = p.vo.t_wc.copy(), 0.6
    rng = np.random.default_rng(3)
    worst = 0.0
    for k in range(200):
        rows = imu(t, 7)
        if k % 9 == 0:                                                   # an I2C error: a read far outside what the sensor can measure
            rows[3] = (rows[3][0], np.array([40.0, -30.0, 25.0]), np.array([900.0, -700.0, 500.0]))
        p.feed_imu(rows)
        t += 0.07
        p.vo.t_wc = base + rng.normal(0, 0.02, 3)
        pose = p.push(int(t * 1e9), None, None, None, None)[0]
        worst = max(worst, float(np.linalg.norm(pose.t_wc - base)))
    assert worst < 0.12 and p.ekf.stats["imu_rejected"] >= 20 and np.isfinite(pose.t_wc).all()
    assert np.linalg.norm(p.ekf.ba) <= 1.0 + 1e-9


def test_a_filter_that_has_blown_up_is_restarted_and_the_raw_pose_is_reported_meanwhile():
    p = make_filtered()
    out, _ = run_static(p, frames=60)
    p.ekf.v = np.array([1e6, 0.0, 0.0])                                  # whatever caused it
    p.ekf.ba = np.array([3.5e4, 0.0, 0.0])
    p.feed_imu(imu(5.0, 7))
    p.vo.t_wc = out[-1] * 0 + p.vo.t_wc
    pose = p.push(int(5.1 * 1e9), None, None, None, None)[0]
    assert pose.info["ekf_using_filter"] == 0 and p.ekf.stats["diverged"] == 1 and np.linalg.norm(p.ekf.v) < 1.0
    assert np.linalg.norm(pose.t_wc - out[-1]) < 0.3                     # the raw picture pose, not a million metres per second


def test_a_filter_that_drifts_from_the_pictures_reports_the_pictures():
    p = make_filtered(max_dev_cm=5.0)
    run_static(p, frames=60)
    p.ekf.p = p.ekf.p + np.array([0.0, 0.0, 0.5])                        # the state is 50 cm wrong and has not noticed yet
    p.feed_imu(imu(5.0, 7))
    pose = p.push(int(5.1 * 1e9), None, None, None, None)[0]
    assert pose.info["ekf_using_filter"] == 0 and np.linalg.norm(pose.t_wc - p.vo.t_wc) < 0.3
