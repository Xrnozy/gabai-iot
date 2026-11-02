import numpy as np

from gabai.vioekf import GRAV, VioEkf, expm, logm

BG, BA = np.array([0.02, -0.015, 0.01]), np.array([0.05, -0.04, 0.03])


def rx(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def pos(t):
    return np.array([0.5 * np.sin(0.9 * t), 0.35 * np.sin(0.6 * t + 1.0), 0.12 * np.sin(0.4 * t)])


def rot(t):
    return rz(0.6 * np.sin(0.5 * t)) @ ry(0.15 * np.sin(0.7 * t)) @ rx(0.1 * np.sin(0.9 * t))


def imu(t, rng, h=1e-4):
    """What a gyro and an accelerometer on this camera would read at t, with the biases and noise of a cheap IMU."""
    w = logm(rot(t).T @ rot(t + h)) / h
    a_w = (pos(t + h) - 2 * pos(t) + pos(t - h)) / h ** 2
    f = rot(t).T @ (a_w - GRAV)
    return w + BG + rng.normal(0, 0.004, 3), f + BA + rng.normal(0, 0.1, 3)


def run(seconds=20, outliers=0.08, dropout=(10.0, 11.0), klt=True, seed=0, ekf=None, drift_step=0.002):
    rng = np.random.default_rng(seed)
    f = ekf or VioEkf()
    f.init(rot(0.0), pos(0.0), 0.0)
    dt_imu, dt_cam = 1 / 200, 1 / 15
    next_cam, drift, prev_cam = dt_cam, np.zeros(3), 0.0
    out = dict(t=[], ekf=[], raw=[], truth=[], att=[], n_out=0, n_out_applied=0)
    n_steps = int(seconds / dt_imu)
    for i in range(1, n_steps + 1):
        t = i * dt_imu
        g, a = imu(t, rng)
        f.predict([(t, g, a)])
        if t >= next_cam:
            drift += rng.normal(0, drift_step, 3)                                     # the pictures' own slow error (a random walk per frame)
            in_drop = dropout and dropout[0] <= t < dropout[1]
            if not in_drop:
                p_m = pos(t) + drift + rng.normal(0, 0.02, 3)
                bad = rng.random() < outliers
                if bad:
                    p_m = p_m + rng.uniform(-0.3, 0.3, 3)
                    out["n_out"] += 1
                R_m = rot(t) @ expm(rng.normal(0, np.radians(1.0), 3))
                before = f.stats["pose_rejected"]
                f.update_pose(R_m, p_m, n_inliers=80)
                if bad and f.stats["pose_rejected"] == before and np.linalg.norm(p_m - pos(t)) > 0.15:
                    out["n_out_applied"] += 1
                if klt:
                    Rp, Rc = rot(prev_cam), rot(t)
                    rv = logm(Rp.T @ Rc) + rng.normal(0, 0.002, 3)
                    tp = Rp.T @ (pos(t) - pos(prev_cam)) + rng.normal(0, 0.005, 3)
                    f.update_klt(rv, tp, t - prev_cam, n_inliers=80, rms=0.6)
                out["raw"].append(np.linalg.norm(p_m - pos(t)) if not bad else np.nan)
            prev_cam, next_cam = t, next_cam + dt_cam
        if i % 13 == 0:
            out["t"].append(t)
            out["ekf"].append(np.linalg.norm(f.p - pos(t)))
            out["att"].append(np.degrees(np.linalg.norm(logm(rot(t).T @ f.R))))
    out["filter"] = f
    return out


def rmse(x, t, lo):
    x = np.asarray(x)[np.asarray(t) >= lo]
    return float(np.sqrt(np.nanmean(x ** 2)))


def test_the_filter_is_steadier_than_the_raw_pictures_and_ignores_outliers():
    o = run(dropout=None, drift_step=0.0)                                         # pictures with noise and outliers only
    t = np.array(o["t"])
    ekf_rmse = rmse(o["ekf"], t, 3.0)
    raw_rmse = float(np.sqrt(np.nanmean(np.array(o["raw"]) ** 2)))                # raw pictures without the outliers: noise only
    assert ekf_rmse < 0.6 * raw_rmse and ekf_rmse < 0.025, (ekf_rmse, raw_rmse)
    assert o["n_out"] > 10 and o["n_out_applied"] <= 1                           # big outliers are refused, not averaged in
    drifting = run(dropout=None)                                                  # and a slow picture drift (which no filter can remove) is not made worse
    assert rmse(drifting["ekf"], drifting["t"], 3.0) <= 1.1 * float(np.sqrt(np.nanmean(np.array(drifting["raw"]) ** 2)))


def test_the_orientation_follows_the_imu_and_the_gyro_bias_is_learned():
    o = run(seconds=25)
    f = o["filter"]
    assert rmse(o["att"], o["t"], 3.0) < 1.0                                       # degrees
    assert np.linalg.norm(f.bg - BG) < 0.006                                       # rad/s: learned from the picture rotation and the Pi's rates, from a start of zero
    assert np.linalg.norm(f.ba - BA) < 0.08


def test_the_pi_motion_measurement_helps_when_the_pictures_are_poor():
    ekf_a = VioEkf()
    with_klt = run(seconds=20, outliers=0.3, klt=True, ekf=ekf_a)
    without = run(seconds=20, outliers=0.3, klt=False)
    assert rmse(with_klt["ekf"], with_klt["t"], 3.0) <= rmse(without["ekf"], without["t"], 3.0) * 1.02       # never worse
    assert ekf_a.stats["klt_rejected"] < 40


def test_through_a_one_second_picture_dropout_the_imu_and_filter_keep_the_position_close():
    o = run(seconds=14, outliers=0.0, dropout=(10.0, 11.0))
    t = np.array(o["t"])
    sel = (t >= 10.0) & (t < 11.0)
    assert np.max(np.array(o["ekf"])[sel]) < 0.25                                  # one second on the IMU alone
    assert np.mean(np.array(o["ekf"])[t > 12.5]) < 0.03                            # and back on track once the pictures return, without a jump past the gate


def test_a_filter_that_has_gone_wrong_is_reset_to_what_the_pictures_keep_saying():
    f = VioEkf()
    f.init(np.eye(3), np.zeros(3), 0.0)
    for k in range(7):
        f.update_pose(np.eye(3), np.array([5.0, 0, 0]))                            # the pictures say 5 m away, again and again
    assert f.stats["resets"] >= 1 and np.linalg.norm(f.p - [5.0, 0, 0]) < 0.5


def test_at_rest_the_velocity_and_the_bias_are_pinned():
    f = VioEkf()
    f.init(np.eye(3), np.zeros(3), 0.0)
    f.v = np.array([0.3, 0, 0])
    rng = np.random.default_rng(1)
    for i in range(1, 100):
        f.predict([(i / 100, BG + rng.normal(0, 0.004, 3), np.array([0, 0, 9.80665]) + rng.normal(0, 0.05, 3))])
    f.update_rest()
    assert np.linalg.norm(f.v) < 0.05 and np.linalg.norm(f.bg - BG) < 0.01
