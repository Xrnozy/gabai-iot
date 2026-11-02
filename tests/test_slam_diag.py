import numpy as np

from gabai.slam.diag import findings_for, rotation_consistency, summarize_trajectory

G = 9.80665


def walk(seconds=30, hz=14.0, speed=0.6, turn_dps=10.0):
    """A believable walk: a gentle curve at walking speed."""
    t = np.arange(0, seconds, 1 / hz)
    ang = np.radians(turn_dps) * t
    x = np.cumsum(speed / hz * np.cos(ang))
    y = np.cumsum(speed / hz * np.sin(ang))
    z = np.zeros_like(t)
    q = np.stack([np.zeros_like(t), np.zeros_like(t), np.sin(ang / 2), np.cos(ang / 2)], axis=1)
    return np.column_stack([t, x, y, z, q])


def names(fs):
    return {(f["level"], f["code"]) for f in fs}


def test_a_normal_walk_gets_a_clean_summary_and_no_findings():
    s = summarize_trajectory(walk())
    assert 13 <= s["pose_rate_hz"] <= 15 and s["n_jumps"] == 0 and s["n_gaps"] == 0
    assert 0.5 < s["speed_mps"]["p50"] < 0.7 and abs(s["path_m"] - 0.6 * 30) < 1.5
    assert findings_for(s) == []


def test_a_teleporting_pose_is_a_confirmed_jump_with_its_time():
    w = walk()
    w[200:, 1] += 3.0                                                    # the estimate jumps 3 m at frame 200
    s = summarize_trajectory(w)
    assert s["n_jumps"] == 1 and abs(s["jumps"][0]["t"] - w[200, 0]) < 0.1
    f = [x for x in findings_for(s) if x["code"] == "pose_jump"][0]
    assert f["level"] == "confirmed" and f["evidence"]["jumps"][0]["step_m"] > 2.5 and "accel" in " ".join(f["possible_causes"]).lower()


def test_a_missing_stretch_of_poses_is_reported_as_a_gap():
    w = walk()
    w = np.delete(w, np.s_[150:190], axis=0)                             # about 2.9 s with no pose
    s = summarize_trajectory(w)
    assert s["n_gaps"] == 1 and s["gaps"][0]["gap_s"] > 2.5
    assert ("confirmed", "pose_gap") in names(findings_for(s))


def test_runaway_scale_is_flagged_as_divergence_not_as_a_long_walk():
    t = np.arange(0, 20, 1 / 14)
    d = 0.5 * np.exp(t / 3.0)                                            # accelerates away
    w = np.column_stack([t, d, d * 0.2, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.ones_like(t)])
    f = findings_for(summarize_trajectory(w))
    assert any(x["code"] == "diverged" and x["level"] == "confirmed" for x in f)


def test_drift_while_the_rig_is_known_to_be_still_is_measured_against_the_spec():
    w = walk(seconds=12, speed=0.0, turn_dps=0.0)
    w[:, 1] = 0.03 * w[:, 0]                                             # creeps 3 cm per second while standing still
    imu = np.column_stack([np.arange(0, 12, 0.005), np.zeros((2400, 3)), np.zeros((2400, 2)), np.full(2400, G)])
    s = summarize_trajectory(w, imu=imu)
    assert s["still"]["seconds"] >= 10 and s["still"]["drift_cm_per_min"] > 100
    f = [x for x in findings_for(s) if x["code"] == "still_drift"][0]
    assert f["level"] == "confirmed" and "5 cm" in f["message"]


def test_still_and_stable_is_not_flagged():
    w = walk(seconds=12, speed=0.0, turn_dps=0.0)
    imu = np.column_stack([np.arange(0, 12, 0.005), np.zeros((2400, 3)), np.zeros((2400, 2)), np.full(2400, G)])
    s = summarize_trajectory(w, imu=imu)
    assert s["still"]["drift_cm_per_min"] < 1 and not [x for x in findings_for(s) if x["code"] == "still_drift"]


def test_rotation_from_poses_agrees_with_the_gyro_when_consistent_and_not_when_the_extrinsics_are_off():
    w = walk(seconds=20, turn_dps=20.0)
    t = np.arange(0, 20, 0.005)
    gyro = np.column_stack([np.zeros_like(t), np.zeros_like(t), np.full_like(t, np.radians(20.0))])    # rad/s about z
    imu = np.column_stack([t, gyro, np.zeros((len(t), 2)), np.full(len(t), G)])
    ok = rotation_consistency(w, imu)
    assert 0.9 < ok["ratio_median"] < 1.1
    bad = rotation_consistency(w, np.column_stack([t, gyro * 0.5, np.zeros((len(t), 2)), np.full(len(t), G)]))
    assert bad["ratio_median"] > 1.8
    s = summarize_trajectory(w, imu=np.column_stack([t, gyro * 0.5, np.zeros((len(t), 2)), np.full(len(t), G)]))
    assert ("possible", "rotation_mismatch") in names(findings_for(s))


def test_too_few_poses_is_reported_not_summarised_as_if_fine():
    s = summarize_trajectory(walk(seconds=0.2))
    assert s["n_poses"] < 5
    assert ("confirmed", "too_few_poses") in names(findings_for(s))


def test_findings_never_claim_accuracy_they_cannot_know():
    s = summarize_trajectory(walk())
    assert "no ground truth" in s["caveat"].lower()
