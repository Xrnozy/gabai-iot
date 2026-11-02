import math

import numpy as np

from gabai.orient import Madgwick, OrientationTrack, euler_deg, q_from_axis_angle, q_mul, q_to_R


def run(filt, gyro_dps, accel, seconds, rate=200):
    dt = 1.0 / rate
    for _ in range(int(seconds * rate)):
        filt.update(np.radians(gyro_dps), accel, dt)


def test_constant_yaw_rate_integrates_to_the_right_angle():
    f = Madgwick(beta=0.02)
    run(f, (0, 0, 90.0), (0, 0, 1.0), 1.0)
    roll, pitch, yaw = euler_deg(f.q)
    assert abs(yaw - 90.0) < 4.0, yaw
    assert abs(roll) < 2.0 and abs(pitch) < 2.0


def test_gravity_pulls_roll_and_pitch_to_the_measured_tilt():
    f = Madgwick(beta=0.1)
    tilt = math.radians(30)
    run(f, (0, 0, 0), (math.sin(tilt), 0.0, math.cos(tilt)), 6.0)   # board tilted 30 deg about the y axis
    roll, pitch, yaw = euler_deg(f.q)
    assert abs(abs(pitch) - 30.0) < 2.0 or abs(abs(roll) - 30.0) < 2.0
    assert abs(yaw) < 2.0


def test_slerp_interpolates_between_samples_and_clamps_outside():
    tr = OrientationTrack()
    tr.add(0.0, q_from_axis_angle((0, 0, 1), 0.0))
    tr.add(1.0, q_from_axis_angle((0, 0, 1), math.radians(90)))
    mid = euler_deg(tr.at(0.5))[2]
    assert abs(mid - 45.0) < 0.5, mid
    assert abs(euler_deg(tr.at(-5.0))[2]) < 1e-6          # before the first sample
    assert abs(euler_deg(tr.at(9.0))[2] - 90.0) < 0.5     # after the last sample


def test_quaternion_to_matrix_is_a_rotation():
    q = q_from_axis_angle((1, 2, 3), 1.1)
    R = q_to_R(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9


def _rot_error_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


B_WORLD = np.array([18.0, 0.0, -42.0])       # magnetic field: north component and a downward component


def test_compass_pulls_yaw_to_magnetic_north_reference():
    truth = q_to_R(q_mul_chain([q_from_axis_angle((0, 0, 1), math.radians(60)),
                                q_from_axis_angle((0, 1, 0), math.radians(20)),
                                q_from_axis_angle((1, 0, 0), math.radians(-15))]))
    accel = truth.T @ np.array([0, 0, 1.0])
    mag = truth.T @ B_WORLD
    f = Madgwick(beta=0.2)
    for _ in range(2000):
        f.update_marg((0, 0, 0), accel, mag, 0.005)
    assert _rot_error_deg(q_to_R(f.q), truth) < 2.0


def test_compass_stops_yaw_drift_from_a_gyro_bias():
    bias = np.radians((0.0, 0.0, 1.5))               # a 1.5 deg/s gyro bias about z, sensor actually still
    mag = B_WORLD.copy()
    plain, fused = Madgwick(beta=0.05), Madgwick(beta=0.05)
    for _ in range(200 * 40):
        plain.update(bias, (0, 0, 1.0), 0.005)
        fused.update_marg(bias, (0, 0, 1.0), mag, 0.005)
    assert abs(euler_deg(plain.q)[2]) > 30.0         # 40 s of drift: about 60 degrees
    assert abs(euler_deg(fused.q)[2]) < 4.0


def q_mul_chain(qs):
    out = qs[0]
    for q in qs[1:]:
        out = q_mul(out, q)
    return out
