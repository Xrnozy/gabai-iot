import math

import numpy as np

from gabai.fusion import SensorFusion
from gabai.orient import q_to_R

B_WORLD = np.array([18.0, 0.0, -42.0])


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def scenario(delay=0.05, M=np.eye(3), seconds=30, rate=200, mag_rate=50, gyro_bias_dps=(0.0, 0.0, 1.5), seed=2, speed=1.2):
    """Truth: a body rotating in a fixed field. Returns IMU samples, compass samples (own frame, late by `delay`)
    and the true orientation (sensor-to-world rotation matrices) at each IMU sample."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / rate)
    def sm():
        x = np.convolve(rng.normal(0, 1, len(t)), np.ones(100) / 100, mode="same")
        return speed * x / x.std()
    w = np.stack([sm(), sm(), sm()], axis=1)                     # rad/s in the sensor frame
    R = np.eye(3)
    Rs, ms, accs = [], [], []
    for k in range(len(t)):
        Rs.append(R.copy())
        ms.append(R.T @ B_WORLD)
        accs.append(R.T @ np.array([0, 0, 1.0]))
        wx = np.array([[0, -w[k, 2], w[k, 1]], [w[k, 2], 0, -w[k, 0]], [-w[k, 1], w[k, 0], 0]])
        R = R @ (np.eye(3) + wx / rate)
        u, _, vt = np.linalg.svd(R)
        R = u @ vt
    ms, accs = np.array(ms), np.array(accs)
    bias = np.radians(gyro_bias_dps)
    imu = [(t[k], *np.degrees(w[k] + bias), *accs[k]) for k in range(len(t))]
    tm = np.arange(0.2, seconds - 0.2, 1 / mag_rate)
    true = np.stack([np.interp(tm - delay, t, ms[:, i]) for i in range(3)], axis=1)
    mag = [(tm[i], *(M @ true[i])) for i in range(len(tm))]
    return imu, mag, Rs, t


def feed(fu, imu, mag, batch=1 / 15):
    i = j = 0
    t_end = imu[-1][0]
    now = batch
    while now < t_end + batch:
        a = []
        while i < len(imu) and imu[i][0] < now:
            a.append(imu[i]); i += 1
        b = []
        while j < len(mag) and mag[j][0] < now:
            b.append(mag[j]); j += 1
        fu.add_imu(a); fu.add_mag(b); fu.step()
        now += batch


def final_error(fu, Rs, t):
    return rot_err_deg(q_to_R(fu.track.q[-1]), Rs[int(np.searchsorted(t, fu.track.t[-1]))])


def test_compass_with_right_delay_and_axes_keeps_orientation_true():
    M = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    imu, mag, Rs, t = scenario(delay=0.05, M=M)
    fu = SensorFusion(mag_cfg={"offset": np.zeros(3), "scale": np.ones(3), "R_im": M.T, "delay_s": 0.05})
    feed(fu, imu, mag)
    assert final_error(fu, Rs, t) < 8.0


def test_without_the_compass_a_gyro_bias_leaves_a_larger_error():
    M = np.eye(3)
    imu, mag, Rs, t = scenario(M=M, speed=0.6)                              # the compass counts for less in a fast spin (a stale reading is worst then), so use a moderate one
    plain = SensorFusion(mag_cfg=None)
    fused = SensorFusion(mag_cfg={"offset": np.zeros(3), "scale": np.ones(3), "R_im": M, "delay_s": 0.05})
    feed(plain, imu, mag)
    feed(fused, imu, mag)
    assert final_error(plain, Rs, t) > 2 * final_error(fused, Rs, t)
    assert final_error(plain, Rs, t) > 3.0


def test_a_wrong_compass_delay_costs_accuracy():
    M = np.eye(3)
    imu, mag, Rs, t = scenario(M=M, delay=0.05, speed=0.6)                  # the compass counts for less in a fast spin (a stale reading is worst then), so a delay shows at a moderate speed
    right = SensorFusion(mag_cfg={"offset": np.zeros(3), "scale": np.ones(3), "R_im": M, "delay_s": 0.05})
    wrong = SensorFusion(mag_cfg={"offset": np.zeros(3), "scale": np.ones(3), "R_im": M, "delay_s": -0.20})
    feed(right, imu, mag)
    feed(wrong, imu, mag)
    assert final_error(wrong, Rs, t) > 2 * final_error(right, Rs, t)


def test_wrong_axis_map_is_clearly_worse_than_the_right_one():
    M = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    imu, mag, Rs, t = scenario(M=M)
    good = SensorFusion(mag_cfg={"offset": np.zeros(3), "scale": np.ones(3), "R_im": M.T, "delay_s": 0.05})
    bad = SensorFusion(mag_cfg={"offset": np.zeros(3), "scale": np.ones(3), "R_im": np.eye(3), "delay_s": 0.05})
    feed(good, imu, mag)
    feed(bad, imu, mag)
    assert final_error(good, Rs, t) + 10.0 < final_error(bad, Rs, t)


def test_pending_samples_wait_for_the_compass_instead_of_guessing():
    fu = SensorFusion(mag_cfg={"offset": np.zeros(3), "scale": np.ones(3), "R_im": np.eye(3), "delay_s": 0.0})
    fu.add_imu([(1.0, 0, 0, 0, 0, 0, 1), (1.01, 0, 0, 0, 0, 0, 1)])
    fu.step()
    assert len(fu.track.t) == 0                      # no compass yet: wait
    fu.add_mag([(1.02, 18.0, 0.0, -42.0)])
    fu.step()
    assert len(fu.track.t) == 1                      # the first sample only sets the time base
