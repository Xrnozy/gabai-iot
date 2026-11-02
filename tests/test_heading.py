"""Yaw / heading stability. A board standing on its side reads gravity on its x axis; the old Euler yaw (undefined when x is vertical) swept through +-180
every few seconds although nothing moved. These tests hold a board still for 60 s in several orientations, and spin it through +-180 in both directions."""
import math

import numpy as np

from gabai.fusion import SensorFusion
from gabai.orient import GyroBias, HeadingTracker, Madgwick, euler_deg, q_from_accel, q_from_axis_angle, q_to_R, wrap_deg

HZ = 100


def samples(seconds, up=(0, 0, 1.0), rate_about_up_dps=0.0, bias=(0, 0, 0), gyro_noise=0.15, accel_noise=0.004, seed=0, t0=0.0):
    """(t, gx, gy, gz, ax, ay, az) for a board whose 'up' (the accelerometer reading at rest, in g) is `up`, turning at rate_about_up_dps about the vertical.
    Turning about the vertical does not change what the accelerometer reads; the gyro sees that rate along the 'up' direction of the body."""
    r = np.random.default_rng(seed)
    up = np.asarray(up, float) / np.linalg.norm(up)
    out = []
    for i in range(int(seconds * HZ)):
        g = up * rate_about_up_dps + np.asarray(bias) + r.normal(0, gyro_noise, 3)
        a = up + r.normal(0, accel_noise, 3)
        out.append((t0 + i / HZ, *g, *a))
    return out


def run(fusion, smp):
    heads = []
    for k in range(0, len(smp), 10):
        fusion.add_imu(smp[k:k + 10])
        fusion.step()
        heads.append((fusion.heading.wrapped, fusion.heading.unwrapped, fusion.heading.valid))
    return np.array(heads)


ORIENTATIONS = {"flat": (0, 0, 1.0), "on its side, gravity on -x (as on the real board)": (-0.998, 0.01, -0.05), "gravity on +y": (0.01, 0.97, 0.1),
                "tilted 40 degrees": (0.5, 0.0, 0.64)}


def test_a_still_board_keeps_its_heading_for_60_seconds_in_every_orientation():
    for name, up in ORIENTATIONS.items():
        f = SensorFusion()
        h = run(f, samples(60, up, bias=(0.12, -0.08, 0.1)))               # a residual gyro bias the Pi's start-up calibration left behind
        valid = h[:, 2] > 0
        assert valid.all(), name
        assert np.ptp(h[valid, 1]) < 2.0, (name, np.ptp(h[:, 1]))              # the heading moves less than 2 degrees in 60 s
        steps = np.abs(np.diff(h[:, 1]))
        assert steps.max() < 0.5, (name, steps.max())                           # and never jumps


def test_the_old_chain_euler_yaw_from_a_filter_started_at_identity_does_fail_on_a_board_on_its_side_so_these_tests_would_have_caught_it():
    up = (-0.998, 0.01, -0.05)
    filt = Madgwick()                                                          # as before: starts at the identity and creeps onto gravity
    yaws = []
    for t, gx, gy, gz, ax, ay, az in samples(30, up, bias=(0.2, 0.0, 0.2)):
        filt.update(np.radians((gx, gy, gz)), (ax, ay, az), 1 / HZ)
        yaws.append(euler_deg(filt.q)[2])
    yaws = np.array(yaws)
    jumps = np.abs(np.diff(yaws))
    assert jumps.max() > 20 or np.ptp(yaws) > 100, (jumps.max(), np.ptp(yaws))   # the old yaw swept through a large angle with the board standing still
    f = SensorFusion()                                                         # the new chain on the same input
    h = run(f, samples(30, up, bias=(0.2, 0.0, 0.2)))
    assert np.ptp(h[:, 1]) < 2.0


def test_without_the_bias_follower_a_small_residual_bias_drifts_and_with_it_the_drift_stops():
    bias = (0.3, 0.0, 0.3)
    drift = {}
    for name, auto in (("off", False), ("on", True)):
        f = SensorFusion(auto_bias=auto)
        h = run(f, samples(60, (0, 0, 1.0), bias=bias))
        drift[name] = abs(h[-1, 1] - h[0, 1])
    assert drift["off"] > 8 and drift["on"] < 2.0, drift                        # 0.3 deg/s over 60 s is 18 degrees; followed, it is gone


def test_a_turn_through_plus_minus_180_is_smooth_and_the_right_size_in_both_directions():
    for rate in (45.0, -45.0):
        for up in ((0, 0, 1.0), (-0.95, 0.02, -0.05)):
            f = SensorFusion()
            h = run(f, samples(3, up) + samples(16, up, rate_about_up_dps=rate, t0=3.0, seed=1))
            wrapped, cont = h[:, 0], h[:, 1]
            turned = cont[-1] - cont[29]                                          # after the 3 s of standing still
            assert abs(abs(turned) - 16 * 45) < 20, (rate, up, turned)           # 720 degrees: two full turns, so the wrapped value crosses +-180 twice
            assert np.abs(np.diff(cont)).max() < 45 * 0.1 * 1.6 + 1.0, (rate, np.abs(np.diff(cont)).max())     # continuous: no step beyond what 0.1 s of turning gives
            folded = np.abs(np.diff(wrapped))
            assert (folded > 300).sum() == 2                                       # the wrapped value does jump at +-180 (that is what wrapping is) ...
            assert np.allclose(wrap_deg(cont), wrapped, atol=1e-6)                 # ... and is exactly the continuous one folded


def test_the_wrap_function():
    assert wrap_deg(190) == -170 and wrap_deg(-190) == 170 and wrap_deg(180) == -180 and wrap_deg(0) == 0 and abs(wrap_deg(540) + 180) < 1e-9


def test_heading_tracker_counts_straight_through_180_and_through_a_full_circle():
    h = HeadingTracker(forward=[1, 0, 0])
    prev = None
    for deg in range(0, 721, 5):
        q = q_from_axis_angle([0, 0, 1], math.radians(deg))
        _, un, valid = h.update(q)
        assert valid
        if prev is not None:
            assert abs(un - prev - 5) < 1e-6, (deg, un, prev)
        prev = un
    assert abs(prev - 720) < 1e-6


def test_the_heading_axis_is_never_the_vertical_one_and_is_held_when_the_axis_points_up():
    f = SensorFusion()
    run(f, samples(2, (-0.95, 0.02, -0.05)))
    assert f.heading.axis_name in ("y", "z")                                   # not x: gravity is on x
    h = HeadingTracker(forward=[1, 0, 0])
    h.update(q_from_axis_angle([0, 0, 1], 0.5))
    last = h.unwrapped
    _, un, valid = h.update(q_from_axis_angle([0, 1, 0], -math.pi / 2))        # pitched so that the chosen axis points straight up
    assert not valid and un == last                                            # undefined there: held, not a random angle


def test_gravity_alignment_at_start_puts_up_on_world_z_for_any_orientation():
    for up in ((0, 0, 1), (0, 0, -1), (-1, 0, 0), (0.3, -0.5, 0.8), (0, 1, 0)):
        up = np.array(up, float) / np.linalg.norm(up)
        assert np.allclose(q_to_R(q_from_accel(up)) @ up, [0, 0, 1], atol=1e-9), up


# ---- the bias follower -------------------------------------------------------------------------------------------------------------------------------------
def test_bias_follower_finds_the_residual_bias_while_still():
    b = GyroBias()
    for t, gx, gy, gz, ax, ay, az in samples(60, bias=(0.25, -0.15, 0.2)):
        b.feed(t, (gx, gy, gz), (ax, ay, az))
    assert np.allclose(b.bias, [0.25, -0.15, 0.2], atol=0.05), b.bias


def test_bias_follower_does_not_learn_a_real_turn_as_bias():
    b = GyroBias()
    for t, gx, gy, gz, ax, ay, az in samples(30, up=(0, 0, 1), rate_about_up_dps=20.0):
        b.feed(t, (gx, gy, gz), (ax, ay, az))
    assert np.abs(b.bias).max() < 0.05 and b.updates == 0
    b2 = GyroBias()                                                              # a turn that speeds up and slows down is not rest either
    r = np.random.default_rng(3)
    for i in range(3000):
        w = 1.2 * math.sin(i / 100 * 2)                                           # +-1.2 deg/s, under the rate limit, but changing
        b2.feed(i / HZ, (0, 0, w + r.normal(0, 0.05)), (0, 0, 1.0))
    assert np.abs(b2.bias).max() < 0.35


def test_bias_follower_is_ignored_while_the_board_is_being_shaken():
    b = GyroBias()
    r = np.random.default_rng(4)
    for i in range(3000):
        b.feed(i / HZ, r.normal(0, 3, 3), r.normal(0, 0.3, 3) + [0, 0, 1])
    assert b.updates == 0


# ---- debug log ---------------------------------------------------------------------------------------------------------------------------------------------
def test_the_debug_line_names_every_source_of_yaw():
    import gabai.fusion as fm
    seen = []

    class Rec:
        def debug(self, ev, **f):
            seen.append((ev, f))
    old, fm.log = fm.log, Rec()
    try:
        f = SensorFusion(log_every_s=1.0)
        run(f, samples(5, (-0.95, 0.02, -0.05), bias=(0.1, 0, 0.1)))
    finally:
        fm.log = old
    assert len(seen) >= 3
    ev, fields = seen[-1]
    assert ev == "fusion.yaw" and {"raw_gyro_mean_dps", "bias_dps", "heading_deg", "heading_unwrapped_deg", "gyro_only_heading_deg", "correction_deg", "axis", "visual"} <= set(fields)
