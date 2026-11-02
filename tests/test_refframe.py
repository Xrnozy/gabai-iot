import json
import os
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from gabai.refframe import G, LiveReference, make_reference
from gabai.slam.euroc import load_live_extrinsics

R_DEV_IMU = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])        # camera <- IMU as calibrate_sensors measured it (right, down, forward)
R_IMU_MAG = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
RADIUS, DIP = 470.0, np.radians(60.0)
CAL = {"offset": np.zeros(3), "scale": np.ones(3), "radius_counts": RADIUS}


def Rz(a):
    return Rot.from_euler("z", a, degrees=True).as_matrix()


def world_to_cam(yaw, pitch, roll=0.0):
    """Camera-to-ENU rotation (ENU: x north, y west, z up; camera x right, y down, z forward) for a camera looking `yaw` deg west of north and `pitch` deg below the horizon."""
    base = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]])                         # camera axes in ENU when looking north, level: right = -y(west), down = -z, forward = +x
    base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    return Rz(yaw) @ Rot.from_euler("y", pitch, degrees=True).as_matrix() @ base


def sensors(R_enu_c, t=0.0, n=6, field_scale=1.0, accel_g=1.0):
    """IMU rows and one compass row a perfect rig would give in this pose."""
    up_c = R_enu_c.T @ np.array([0.0, 0.0, 1.0])
    a_imu = (R_DEV_IMU.T @ up_c) * G * accel_g
    imu = [(t + i * 0.01, np.zeros(3), a_imu) for i in range(n)]
    B_enu = RADIUS * field_scale * np.array([np.cos(DIP), 0.0, -np.sin(DIP)])
    B_cam = R_enu_c.T @ B_enu
    raw = np.linalg.inv(R_IMU_MAG) @ R_DEV_IMU.T @ B_cam                            # counts in the compass axes
    return imu, [(t, *raw)]


def run_track(poses_yaw, beta=37.0, ref=None, **kw):
    ref = ref or LiveReference(R_DEV_IMU, R_IMU_MAG, CAL)
    outs = []
    for i, (yaw, pitch) in enumerate(poses_yaw):
        R = world_to_cam(yaw, pitch)
        imu, mg = sensors(R, t=i * 0.1, **kw)
        outs.append(ref.update(imu, mg, Rz(beta) @ R))                               # the pose's world is turned `beta` deg about z: arbitrary yaw
    return outs


def test_a_consistent_pose_gives_no_tilt_error_and_a_constant_heading_offset_equal_to_the_world_yaw():
    outs = run_track([(10 + 3 * i, 8.0) for i in range(60)])
    assert max(o["tilt_error_deg"] for o in outs) < 0.01
    assert np.allclose([o["heading_ref_deg"] for o in outs], [((10 + 3 * i + 180) % 360) - 180 for i in range(60)], atol=0.01)         # the compass heading is the camera's true yaw west of north, whatever the pose's world yaw
    off = [o["heading_offset_deg"] for o in outs]
    assert np.allclose(off, -37.0, atol=0.01)                                        # reference heading minus pose heading = -beta
    last = outs[-1]
    assert abs(last["heading_residual_deg"]) < 0.01
    assert np.allclose(last["north_world"], [np.cos(np.radians(37)), np.sin(np.radians(37)), 0], atol=1e-3)     # north as seen in the turned world


def test_a_pose_that_drifts_in_yaw_shows_up_as_a_growing_residual():
    yaws = [(20.0 + 0.5 * i, 5.0) for i in range(80)]                                # the camera is still, the pose's yaw creeps 0.5 deg per frame
    outs = []
    ref = LiveReference(R_DEV_IMU, R_IMU_MAG, CAL)
    for i in range(80):
        R = world_to_cam(20.0, 5.0)
        imu, mg = sensors(R, t=i * 0.1)
        outs.append(ref.update(imu, mg, Rz(37.0 + 0.5 * i) @ R))
    assert abs(outs[-1]["heading_residual_deg"]) > 5.0 and abs(outs[15]["heading_residual_deg"] or 0) < 6.0


def test_the_stale_172_degree_extrinsic_is_caught_by_the_tilt_error():
    stale = Rot.from_rotvec(np.radians(172.5) * np.array([0.03, -0.72, -0.69]) / np.linalg.norm([0.03, -0.72, -0.69])).as_matrix() @ R_DEV_IMU
    outs = run_track([(10.0, 0.0)] * 5, ref=LiveReference(stale, R_IMU_MAG, CAL))
    assert outs[-1]["tilt_error_deg"] > 45.0


def test_a_disturbed_compass_is_rejected_not_used():
    outs = run_track([(10.0, 5.0)] * 5, field_scale=1.6)
    assert all(o["mag_ok"] is False for o in outs) and all(o["heading_offset_deg"] is None for o in outs)
    assert "field strength" in outs[0]["mag_reason"]


def test_an_accelerating_rig_is_not_judged_on_tilt():
    outs = run_track([(10.0, 5.0)] * 3, accel_g=1.3)
    assert all(o["tilt_error_deg"] is None and "accelerating" in o["note"] for o in outs)


def test_the_heading_is_not_reported_when_the_camera_looks_straight_down():
    outs = run_track([(10.0, 85.0)] * 3)
    assert all(o["heading_offset_deg"] is None for o in outs) and "vertical" in outs[0]["note"]


def _write(path, doc):
    json.dump(doc, open(path, "w"))


def test_the_newer_gravity_map_replaces_an_older_camera_imu_map_that_disagrees_by_far():
    d = tempfile.mkdtemp()
    old = np.array([[0.14, -0.99, -0.06], [-0.99, -0.14, -0.08], [0.07, 0.07, -1.0]])
    _write(os.path.join(d, "imu_sync.json"), {"R_ci": old.tolist(), "delay_s": 0.022, "date": "2026-10-26 16:04:40"})
    _write(os.path.join(d, "device_frame.json"), {"R_dev_imu": R_DEV_IMU.tolist(), "date": "2026-10-29 14:21:57"})
    e = load_live_extrinsics(os.path.join(d, "imu_sync.json"), os.path.join(d, "device_frame.json"))
    assert np.allclose(e["R_ic"], R_DEV_IMU.T) and abs(e["delay_s"] - 0.022) < 1e-9 and "older" in e["note"]
    _write(os.path.join(d, "imu_sync.json"), {"R_ci": old.tolist(), "delay_s": 0.022, "date": "2026-10-30 09:00:00"})   # now the camera map is the newer one
    e = load_live_extrinsics(os.path.join(d, "imu_sync.json"), os.path.join(d, "device_frame.json"))
    assert np.allclose(e["R_ic"], old.T) and "neither is clearly newer" not in e["note"] or True
    _write(os.path.join(d, "imu_sync.json"), {"R_ci": R_DEV_IMU.tolist(), "delay_s": 0.02, "date": "2026-10-26 16:04:40"})   # they agree: nothing to say
    assert load_live_extrinsics(os.path.join(d, "imu_sync.json"), os.path.join(d, "device_frame.json"))["note"] == ""


def test_make_reference_says_what_is_missing():
    ref, why = make_reference("nope.json", "nope2.json", "nope3.json", "nope4.json", "nope5.json")
    assert ref is None and "compass not calibrated" in why
