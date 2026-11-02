"""Frame calibration from two poses and two turns, on simulated sensors with a known truth."""
import json
import math
import os
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from gabai.frames import Frames, classify_mount, load_frames, nearest_rotation, triad
from gabai.framecal import solve, sweep_about
from gabai.fusion import SensorFusion
from gabai.kfvo import KeyframeVO
from gabai.mag import apply_cal
from gabai.mag_sync import rotations_24
from gabai.tools.frame_check import report

EARTH, DIP = 489.0, math.radians(55)
B_W = EARTH * np.array([math.cos(DIP), 0.0, -math.sin(DIP)])            # world x north, y west, z up; the field points north and down
HARD = np.array([-120.0, 900.0, 130.0])
R_TRUE_DEV_IMU = nearest_rotation(np.array([[0.141, -0.988, -0.063], [-0.988, -0.136, -0.077], [0.067, 0.073, -0.995]]))     # the camera-IMU map in data/imu_sync.json


def pose_R(upright, pitch_down_deg=0.0, theta_deg=0.0):
    """Device -> world rotation. Front-up: device z is world up. Upright (worn): device -y is world up, leaned so the camera looks pitch_down_deg below horizontal."""
    R0 = np.eye(3) if not upright else Rotation.from_euler("x", -90 - pitch_down_deg, degrees=True).as_matrix()
    return Rotation.from_euler("z", theta_deg, degrees=True).as_matrix() @ R0


def sense(R_wd, R_imu_mag, rng, noise_a=0.004, noise_m=2.5):
    """(accel in IMU axes in g, raw compass counts) for a device with device->world rotation R_wd."""
    up_dev = R_wd.T @ [0, 0, 1.0]
    b_dev = R_wd.T @ B_W
    accel = R_TRUE_DEV_IMU.T @ up_dev + rng.normal(0, noise_a, 3)
    mag = R_imu_mag.T @ (R_TRUE_DEV_IMU.T @ b_dev) + HARD + rng.normal(0, noise_m, 3)
    return accel, mag


def captures(R_imu_mag, upright_pitch=8.0, seed=0, turn_deg=360.0, upright_lean_side=0.0, spin_tilt_change=0.0, second_pose_upright=True):
    rng = np.random.default_rng(seed)
    out = {}
    for name, upright in (("front_up", False), ("upright", second_pose_upright)):
        pitch = upright_pitch if upright else 0.0
        A = np.array([sense(pose_R(upright, pitch, 17.0), R_imu_mag, rng)[0] for _ in range(150)])
        out["still_" + name] = {"accel": A, "gyro": rng.normal(0, 0.15, (150, 3))}
        acc, mg = [], []
        for th in np.linspace(0, turn_deg, 400):
            R = pose_R(upright, pitch + spin_tilt_change * (th / 360.0), th)
            a, m = sense(R, R_imu_mag, rng)
            acc.append(a), mg.append(m)
        out["spin_" + name] = {"accel": np.array(acc), "mag": np.array(mg)}
    return out


def test_two_gravity_poses_give_the_device_frame_even_when_the_chest_leans_forward():
    for pitch in (0.0, 8.0, 15.0, -10.0):
        c = captures(np.eye(3), upright_pitch=pitch)
        R = triad(c["still_front_up"]["accel"].mean(0), c["still_upright"]["accel"].mean(0))
        err = math.degrees(math.acos(np.clip((np.trace(R.T @ R_TRUE_DEV_IMU) - 1) / 2, -1, 1)))
        assert err < 1.0, (pitch, err)


def test_every_compass_axis_map_is_recovered_exactly_and_the_hard_iron_offset_to_a_few_counts():
    for k, P in enumerate(rotations_24()):
        c = captures(P, seed=k)
        r = solve(c["still_front_up"], c["still_upright"], c["spin_front_up"], c["spin_upright"])
        assert np.array_equal(r["R_imu_mag"], P), (k, P, r["R_imu_mag"], r["map_score"], r["map_score_next"])
        assert np.abs(r["offset"] - HARD).max() < 8.0, (k, r["offset"])
        assert not r["problems"], (k, r["problems"])
        assert abs(r["dip_deg"] - 55) < 6, r["dip_deg"]


def test_one_pose_alone_cannot_tell_the_axes_apart_and_the_solver_says_so():
    P = rotations_24()[7]
    c = captures(P, second_pose_upright=False)                                  # both turns about the same vertical axis
    try:
        r = solve(c["still_front_up"], c["still_upright"], c["spin_front_up"], c["spin_upright"])
    except ValueError as e:                                                        # the two gravity readings are identical: no device frame either
        assert "same gravity direction" in str(e)
        return
    assert any("not clearly better" in p or "tilt" in p for p in r["problems"]), r["problems"]


def test_a_partial_turn_and_a_changing_tilt_are_reported():
    P = rotations_24()[3]
    c = captures(P, turn_deg=120.0)
    r = solve(c["still_front_up"], c["still_upright"], c["spin_front_up"], c["spin_upright"])
    assert any("only" in p and "degrees" in p for p in r["problems"]), r["problems"]
    c = captures(P, spin_tilt_change=40.0)
    r = solve(c["still_front_up"], c["still_upright"], c["spin_front_up"], c["spin_upright"])
    assert any("tilt changed" in p for p in r["problems"]), r["problems"]


def test_a_sideways_lean_of_the_upright_pose_is_not_seen_by_gravity_and_becomes_an_error_of_the_same_size():
    """The documented limit of the two-pose method: it is why load_frames compares the result with the camera-gyro map."""
    c = captures(np.eye(3), upright_pitch=5.0)
    lean = 12.0
    L = R_TRUE_DEV_IMU.T @ Rotation.from_euler("z", lean, degrees=True).as_matrix() @ R_TRUE_DEV_IMU
    R = triad(c["still_front_up"]["accel"].mean(0), (L @ c["still_upright"]["accel"].mean(0)))
    err = math.degrees(math.acos(np.clip((np.trace(R.T @ R_TRUE_DEV_IMU) - 1) / 2, -1, 1)))
    assert abs(err - lean) < 1.5, err


def test_the_sweep_function_counts_a_full_turn_and_a_half_turn():
    t = np.linspace(0, 2 * math.pi, 200)
    ring = np.stack([np.cos(t), np.sin(t), np.full_like(t, 0.5)], 1)
    assert abs(sweep_about(ring, [0, 0, 1]) - 360) < 3 and abs(sweep_about(ring[:100], [0, 0, 1]) - 180) < 4


def test_pose_is_recognised_and_the_camera_pitch_is_read_from_gravity():
    assert classify_mount([0, -1.0, 0])[0] == "chest_upright" and classify_mount([0, 0, 1.0])[0] == "flat_front_up"
    assert classify_mount([0, 0, -1.0])[0] == "flat_front_down" and classify_mount([0, 1.0, 0])[0] == "upside_down" and classify_mount([1.0, 0, 0])[0] == "on_its_side"
    assert classify_mount([0.6, -0.6, 0.5])[0] == "tilted"
    p = math.radians(12)                                                         # camera looking 12 degrees below the horizon
    name, d = classify_mount([0, -math.cos(p), -math.sin(p)])
    assert name == "chest_upright" and abs(d["pitch_down_deg"] - 12) < 1e-6 and abs(d["roll_deg"]) < 1e-6
    name, d = classify_mount([math.sin(math.radians(9)), -math.cos(math.radians(9)), 0])
    assert abs(d["roll_deg"] - 9) < 1e-6


def test_the_frames_file_is_loaded_with_a_fallback_a_cross_check_and_notes():
    d = tempfile.mkdtemp()
    f = load_frames(os.path.join(d, "a.json"), os.path.join(d, "b.json"), os.path.join(d, "c.json"))
    assert np.allclose(f.R_dev_imu, np.eye(3)) and len(f.notes) == 2
    json.dump({"R_ci": R_TRUE_DEV_IMU.tolist()}, open(os.path.join(d, "imu.json"), "w"))
    json.dump({"R_im": np.eye(3).tolist()}, open(os.path.join(d, "mag.json"), "w"))
    f = load_frames(os.path.join(d, "imu.json"), os.path.join(d, "mag.json"), os.path.join(d, "none.json"))
    assert np.allclose(f.R_dev_imu, R_TRUE_DEV_IMU) and not f.notes and np.allclose(f.R_dev_mag, R_TRUE_DEV_IMU)
    json.dump({"R_dev_imu": (Rotation.from_euler("z", 30, degrees=True).as_matrix() @ R_TRUE_DEV_IMU).tolist()}, open(os.path.join(d, "dev.json"), "w"))
    f = load_frames(os.path.join(d, "imu.json"), os.path.join(d, "mag.json"), os.path.join(d, "dev.json"))
    assert any("differ by 30" in n for n in f.notes)
    assert np.allclose(f.forward_in_imu(), R_TRUE_DEV_IMU.T @ [0, 0, 1.0])


def test_the_fusion_uses_the_devices_forward_axis_for_the_heading_and_recognises_the_worn_pose():
    f = Frames(R_dev_imu=R_TRUE_DEV_IMU)
    fu = SensorFusion(frames=f)
    up_imu = R_TRUE_DEV_IMU.T @ np.array([0.0, -math.cos(0.17), -math.sin(0.17)])        # worn, camera 9.7 degrees below the horizon
    rng = np.random.default_rng(0)
    fu.add_imu([(i / 100, *rng.normal(0, 0.1, 3), *(up_imu + rng.normal(0, 0.003, 3))) for i in range(200)])
    fu.step()
    assert np.allclose(fu.heading.forward, R_TRUE_DEV_IMU.T @ [0, 0, 1.0])
    assert fu.mount is not None and fu.mount[0] == "chest_upright" and abs(fu.mount[1]["pitch_down_deg"] - math.degrees(0.17)) < 1.0, fu.mount


def test_the_heading_of_the_devices_forward_axis_is_the_true_heading_when_worn():
    from gabai.orient import q_from_axis_angle
    f = Frames(R_dev_imu=R_TRUE_DEV_IMU)
    fu = SensorFusion(frames=f, auto_bias=False)
    # a person wearing it turns on the spot at 30 deg/s for 12 s: the body turns about the vertical, which is device -y
    up_dev = np.array([0.0, -1.0, 0.0])
    up_imu = R_TRUE_DEV_IMU.T @ up_dev
    rate = 30.0
    smp = [(i / 100, *(up_imu * rate), *up_imu) for i in range(1200)]
    start = None
    for k in range(0, len(smp), 10):
        fu.add_imu(smp[k:k + 10])
        fu.step()
        start = fu.heading.unwrapped if start is None else start
    assert abs(abs(fu.heading.unwrapped - start) - 12 * 30) < 12, (start, fu.heading.unwrapped)       # 360 degrees turned, counted once


def test_the_slam_world_starts_with_the_camera_pitched_and_z_up():
    v = KeyframeVO(600, 600, 320, 240, initial_pitch_deg=12.0)
    fwd = v.R_wc @ [0, 0, 1.0]
    down = v.R_wc @ [0, 1.0, 0]
    assert abs(fwd[2] + math.sin(math.radians(12))) < 1e-9 and down[2] < 0                   # looks 12 degrees below the horizon; the camera's down still points down
    v0 = KeyframeVO(600, 600, 320, 240)
    assert abs((v0.R_wc @ [0, 0, 1.0])[2]) < 1e-9


def test_frame_check_reports_pose_pitch_and_a_compass_that_agrees_with_gravity():
    rng = np.random.default_rng(0)
    R_im = rotations_24()[11]
    f = Frames(R_dev_imu=R_TRUE_DEV_IMU, R_imu_mag=R_im)
    cfg = {"offset": HARD, "scale": np.ones(3), "radius_counts": EARTH, "W": np.eye(3)}
    R_wd = pose_R(True, 9.0, 40.0)
    pairs = [sense(R_wd, R_im, rng) for _ in range(100)]
    lines = report(f, np.array([p[0] for p in pairs]), cfg, np.array([p[1] for p in pairs]))
    text = "\n".join(lines)
    assert "chest_upright" in text and "initial_pitch_deg = 9" in text and "PROBLEM" not in text, text
    assert "uT" in text and "of magnetic north" in text
    bad = dict(cfg, offset=HARD + [0, 0, 900])                                           # a wrong offset must be called out
    text = "\n".join(report(f, np.array([p[0] for p in pairs]), bad, np.array([p[1] for p in pairs])))
    assert "PROBLEM" in text
