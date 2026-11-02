"""Absolute heading from the compass. The reported symptom: lift the whole rig, put it back where it was, and the orientation does not return. Without a
compass the heading only integrates the gyro, so it cannot return; with a calibrated, axis-mapped, gated compass it must, and a magnet passing by must not move it.
The sensor data are simulated from a known true orientation, a known Earth field, and a known hard/soft-iron distortion."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from gabai.fusion import SensorFusion
from gabai.mag import MagGate, apply_cal, calibrate, calibrate_ellipsoid, tilt_heading_deg

HZ = 100
EARTH = 489.0                                       # counts, about 45 microtesla at 0.92 mG per count
DIP = math.radians(60)
B_WORLD = EARTH * np.array([math.cos(DIP), 0.0, -math.sin(DIP)])        # world: x north, y west, z up; the field points north and down
HARD = np.array([-120.0, 900.0, 130.0])             # like the real board: a hard-iron offset far larger than the Earth's field
_Q = Rotation.from_euler("xyz", [25, 30, 40], degrees=True).as_matrix()
SOFT = _Q @ np.diag([0.8, 1.15, 0.95]) @ _Q.T                     # soft iron is a symmetric stretch along some tilted axes, not along the sensor axes
R_IM = Rotation.from_euler("z", 90, degrees=True).as_matrix()          # compass axes -> IMU axes: a quarter turn, as a different mounting would give


def raw_compass(R_bw, rng, extra=0.0, noise=2.0):
    """What the HMC5883L would read. R_bw rotates body vectors into the world; the compass axes are R_IM^T of the body (IMU) axes."""
    b_imu = R_bw.T @ B_WORLD
    return SOFT @ (R_IM.T @ b_imu) + HARD + extra + rng.normal(0, noise, 3)


def make_cfg(seed=0, method=calibrate_ellipsoid):
    """Calibrate from a tumble through every direction, as sync_sensors does."""
    rng = np.random.default_rng(seed)
    pts = np.array([raw_compass(Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix(), rng) for _ in range(2500)])
    cal = method(pts)
    cfg = {"offset": cal["offset"], "scale": cal["scale"], "R_im": R_IM, "delay_s": 0.0, "radius_counts": cal["radius_counts"]}
    if "W" in cal:
        cfg["W"] = cal["W"]
    return cfg, cal


def trip(cfg, still_s=5.0, move_s=25.0, bias=(0.3, -0.2, 0.25), seed=1, disturb=None, auto_bias=True, mag_gate=True, R0=None):
    """Still for still_s, then tumble for move_s through large angles and come back to EXACTLY the starting orientation, then still for 3 s.
    Returns (fusion, headings at the end of the first still, at the end of the trip, true azimuth of the heading axis at the start and the end)."""
    rng = np.random.default_rng(seed)
    R0 = Rotation.from_euler("zyx", [35, 8, -5], degrees=True).as_matrix() if R0 is None else R0     # board slightly tilted, facing 35 degrees west of north
    n_still, n_move, n_end = int(still_s * HZ), int(move_s * HZ), 3 * HZ
    rv = np.zeros((n_still + n_move + n_end, 3))
    t_move = np.arange(n_move) / HZ
    s = np.sin(math.pi * t_move / move_s) ** 2                                  # 0 at both ends, smooth, so the rotation returns to zero
    rv[n_still:n_still + n_move] = np.stack([2.4 * s * np.sin(2 * math.pi * t_move / 7.0), 2.0 * s * np.sin(2 * math.pi * t_move / 5.0 + 1), 3.0 * s * np.sin(2 * math.pi * t_move / 11.0 + 2)], 1)
    Rs = [R0 @ Rotation.from_rotvec(v).as_matrix() for v in rv]
    f = SensorFusion(mag_cfg=cfg, auto_bias=auto_bias, mag_gate=mag_gate)
    mags, imus = [], []
    for i, R in enumerate(Rs):
        t = i / HZ
        w = Rotation.from_matrix(R.T @ Rs[min(i + 1, len(Rs) - 1)]).as_rotvec() * HZ if i + 1 < len(Rs) else np.zeros(3)         # body rate in rad/s
        gyro = np.degrees(w) + np.asarray(bias) + rng.normal(0, 0.15, 3)
        accel = R.T @ [0, 0, 1.0] + rng.normal(0, 0.004, 3)
        extra = 0.0 if disturb is None else disturb(t)
        imus.append((t, *gyro, *accel))
        mags.append((t, *raw_compass(R, rng, extra)))
    heads = []
    for k in range(0, len(imus), 5):
        if cfg is not None:
            f.add_mag(mags[k:k + 5])
        f.add_imu(imus[k:k + 5])
        f.step()
        heads.append(f.heading.wrapped)
    heads = np.array(heads)
    fwd = f.heading.forward if f.heading.forward is not None else np.array([1.0, 0, 0])
    az = lambda R: math.degrees(math.atan2((R @ fwd)[1], (R @ fwd)[0]))
    f.az_series = np.array([az(R) for R in Rs[::5]])
    return f, heads, n_still, az(R0), az(Rs[-1])


def angdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def test_with_a_calibrated_compass_the_heading_returns_after_the_rig_is_lifted_and_put_back_and_it_is_the_true_heading():
    cfg, cal = make_cfg()
    f, heads, n_still, az0, az1 = trip(cfg)
    before, after = heads[n_still // 5 - 1], heads[-1]
    assert abs(angdiff(after, before)) < 3.0, (before, after)                  # back where it was
    assert abs(angdiff(after, az1)) < 4.0, (after, az1)                         # and equal to the true heading (the compass supplies the absolute reference)
    assert f.mag_used_n > 0.9 * (f.mag_used_n + f.mag_gated_n)


def test_without_the_compass_the_same_trip_does_not_return_because_gyro_bias_and_noise_integrate():
    f, heads, n_still, az0, az1 = trip(None, auto_bias=False)
    before, after = heads[n_still // 5 - 1], heads[-1]
    assert abs(angdiff(after, before)) > 3.0, (before, after)                  # nothing holds the heading: the leftover bias integrates over the 33 s


def test_a_magnet_passing_by_does_not_move_the_heading_when_the_gate_is_on_and_does_when_it_is_off():
    cfg, _ = make_cfg()
    still = lambda t: np.array([320.0, -250.0, 90.0]) if 6.0 <= t <= 12.0 else 0.0     # during the still part and its end: the heading should simply stay put
    errs = {}
    for name, gate in (("on", True), ("off", False)):
        f, heads, n_still, az0, az1 = trip(cfg, disturb=still, mag_gate=gate, still_s=14.0, move_s=10.0)
        t = np.arange(len(heads)) * 5 / HZ
        win = (t >= 6.0) & (t <= 14.0)
        errs[name] = float(np.abs(angdiff(heads[win], f.az_series[win])).max())         # worst error against the TRUE heading during and just after the magnet
        if gate:
            assert f.mag_gated_n > 100, f.mag_gated_n
    assert errs["on"] < 6.0 and errs["off"] > 3 * errs["on"], errs


def test_the_gate_accepts_a_good_field_and_rejects_a_strong_one_or_a_changed_dip():
    g = MagGate(radius=489.0)
    up = np.array([0, 0, 1.0])
    good = lambda dip_deg, mag=489.0: mag * np.array([math.sin(math.radians(dip_deg)), 0, math.cos(math.radians(dip_deg))])
    for i in range(100):
        assert g.check(i * 0.01, good(30), up)                                  # learns that this place gives an angle of 30 degrees to gravity
    assert not g.check(2.0, good(30, 700.0), up) and "strength" in g.reason
    assert not g.check(2.01, good(70), up) and "angle" in g.reason
    after = [g.check(2.02 + i * 0.01, good(32), up) for i in range(5)]          # a small change is fine, but after a reject the gate wants 5 good readings in a row
    assert after == [False] * 4 + [True], after


def test_a_lasting_change_of_place_is_accepted_after_a_few_seconds_so_the_compass_is_not_locked_out_for_good():
    g = MagGate(radius=489.0, reset_s=3.0)
    up = np.array([0, 0, 1.0])
    f = lambda dip: 489.0 * np.array([math.sin(math.radians(dip)), 0, math.cos(math.radians(dip))])
    for i in range(100):
        g.check(i * 0.01, f(30), up)
    oks = [g.check(10 + i * 0.05, f(70), up) for i in range(100)]              # carried somewhere the dip is 70
    assert not oks[0] and oks[-1]


def test_a_full_soft_iron_matrix_beats_per_axis_scales_when_the_distortion_is_tilted():
    cfg_e, cal_e = make_cfg(method=calibrate_ellipsoid)
    cfg_s, cal_s = make_cfg(method=calibrate)
    rng = np.random.default_rng(5)
    errs = {}
    for name, c in (("ellipsoid", cfg_e), ("per-axis", cfg_s)):
        e = []
        for _ in range(300):
            R = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
            m = R_IM @ apply_cal(c, raw_compass(R, rng))
            up = R.T @ [0, 0, 1.0]
            hd, fr = tilt_heading_deg(m, up, [1.0, 0, 0])
            if hd is not None and fr > 0.7:
                truth = math.degrees(math.atan2((R @ [1.0, 0, 0])[1], (R @ [1.0, 0, 0])[0]))
                e.append(abs(angdiff(hd, truth)))
        errs[name] = float(np.median(e))
    assert cal_e["method"] == "ellipsoid" and cal_e["quality"] < 0.03 < cal_s["quality"]
    assert errs["ellipsoid"] < 3.0 and errs["per-axis"] > 2 * errs["ellipsoid"], errs


def test_calibration_with_too_few_directions_does_not_pretend_to_be_an_ellipsoid():
    rng = np.random.default_rng(6)
    pts = np.array([raw_compass(Rotation.from_euler("z", a, degrees=True).as_matrix(), rng) for a in rng.uniform(0, 360, 500)])     # turned flat only
    cal = calibrate_ellipsoid(pts)
    assert cal["method"] == "sphere" and cal["coverage"] < 0.75


def test_tilt_heading_is_the_same_whatever_the_tilt_and_undefined_when_the_axis_points_up():
    R0 = Rotation.from_euler("z", 50, degrees=True).as_matrix()
    heads = []
    for roll, pitch in ((0, 0), (20, 0), (0, -25), (-15, 30)):
        R = R0 @ Rotation.from_euler("xy", [roll, pitch], degrees=True).as_matrix()
        h, frac = tilt_heading_deg(R.T @ B_WORLD, R.T @ [0, 0, 1.0], [1.0, 0, 0])
        heads.append((h, math.degrees(math.atan2((R @ [1.0, 0, 0])[1], (R @ [1.0, 0, 0])[0]))))
    assert all(abs(angdiff(h, t)) < 1e-6 for h, t in heads)
    R = Rotation.from_euler("y", -90, degrees=True).as_matrix()                  # x axis now points straight up
    assert tilt_heading_deg(R.T @ B_WORLD, R.T @ [0, 0, 1.0], [1.0, 0, 0])[0] is None


def test_the_accelerometer_calibration_is_applied_before_the_filter_sees_it():
    cal = {"bias": [-0.294, 0.174, 0.139], "scale": [0.0036, 0.0008, -0.0012]}              # the values in data/accel_calib.json
    f = SensorFusion(accel_cal=cal, auto_bias=False)
    raw_up = np.array([0.0, 0.0, 0.95])                                                      # a board that reads 0.95 g flat
    seen = []
    orig = f.filt.update

    def spy(g, a, dt):
        seen.append(np.linalg.norm(a))
        return orig(g, a, dt)
    f.filt.update = spy
    f.add_imu([(i / HZ, 0, 0, 0, *raw_up) for i in range(50)])
    f.step()
    assert len(seen) > 10
    # the filter normalises the accelerometer, so what matters is its DIRECTION: with the bias removed the vector is closer to the true vertical than the raw one
    f2 = SensorFusion(accel_cal=cal, auto_bias=False)
    out = []
    orig2 = f2.filt.update
    f2.filt.update = lambda g, a, dt: (out.append(np.asarray(a, float)), orig2(g, a, dt))[1]
    f2.add_imu([(i / HZ, 0, 0, 0, 0.02, -0.01, 0.97) for i in range(50)])
    f2.step()
    corrected = np.mean(out, axis=0)
    raw = np.array([0.02, -0.01, 0.97])
    tilt = lambda v: math.degrees(math.acos(v[2] / np.linalg.norm(v)))
    assert not np.allclose(corrected, raw) and abs(np.linalg.norm(corrected) - 1.0) < abs(np.linalg.norm(raw) - 1.0) + 0.02


def test_the_debug_line_reports_the_compass_state_and_its_disagreement_with_the_filter():
    import gabai.fusion as fm
    cfg, _ = make_cfg()
    seen = []

    class Rec:
        def debug(self, ev, **f):
            seen.append((ev, f))
    old, fm.log = fm.log, Rec()
    try:
        trip(cfg, still_s=4.0, move_s=4.0)
    finally:
        fm.log = old
    assert seen
    _, fields = seen[-1]
    assert {"compass_ok", "compass_gated", "compass_used", "field_vs_calibrated", "compass_heading_deg", "compass_minus_filter_deg"} <= set(fields)
