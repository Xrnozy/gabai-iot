"""The orientation filter must stay put when the board barely moves: no chatter at rest, no tilt from a shaking hand, no tilt from the compass."""
import math
import os

import numpy as np
import pytest

from gabai.mag import MagGate
from gabai.orient import Madgwick, accel_trust, euler_deg, q_from_accel, q_to_R

G_UP = np.array([0.0, 0.0, 1.0])
REC = "data/slam/rec_20261026_155458/mav0/imu0/data.csv"                       # your own 60 s recording; its first seconds are a resting board


def tilt_deg(f):
    return math.degrees(math.acos(max(-1.0, min(1.0, float((q_to_R(f.q).T @ G_UP) @ G_UP)))))


def test_the_accelerometer_is_believed_only_when_it_reads_one_g():
    assert accel_trust(1.0) == 1.0 and accel_trust(1.02) == 1.0 and accel_trust(0.98) == 1.0
    assert 0.0 < accel_trust(1.12) < 1.0 and accel_trust(1.3) == 0.0 and accel_trust(0.7) == 0.0


def test_at_rest_the_correction_dies_out_instead_of_chattering():
    rng = np.random.default_rng(0)
    f = Madgwick(beta=0.05)
    f.q = q_from_accel([0.02, -0.01, 1.0])
    moved = 0.0
    for _ in range(2000):
        q0 = f.q.copy()
        f.update(rng.normal(0, np.radians(0.1), 3), np.array([0.02, -0.01, 1.0]) + rng.normal(0, 0.004, 3), 0.006)
        moved += 2 * math.acos(min(1.0, abs(float(q0 @ f.q))))
    assert math.degrees(moved) / (2000 * 0.006) < 0.6                       # the old normalised step kept turning the board at 4.9 deg/s, whatever the error


def worst_tilt(accel_at, seconds=8.0, dt=0.006):
    f = Madgwick(beta=0.05)
    f.q = q_from_accel(G_UP)
    worst = 0.0
    for i in range(int(seconds / dt)):
        f.update(np.zeros(3), accel_at(i * dt), dt)
        worst = max(worst, tilt_deg(f))
    return worst


def test_a_hand_that_shakes_the_board_does_not_tilt_the_estimate():
    shake = lambda t: np.array([0.35 * math.sin(2 * math.pi * 4 * t) if 1.0 < t < 4.0 else 0.0, 0.0, 1.0])       # 0.35 g sideways at 4 Hz for 3 s
    assert worst_tilt(shake) < 0.5                                           # old filter: 0.72


def test_a_sustained_push_tilts_the_estimate_less_than_before():
    push = lambda t: np.array([0.30 if 1.0 < t < 2.5 else 0.0, 0.0, 1.0])    # accelerating forward for 1.5 s: gravity plus 0.3 g sideways (17 degrees of false tilt)
    assert worst_tilt(push) < 6.5                                            # old filter: 8.6; the rate is capped at 4 deg/s because the gyro would have shown a real tilt


def test_the_estimate_still_follows_a_real_tilt():
    f = Madgwick(beta=0.05)
    f.q = q_from_accel(G_UP)
    a = np.array([math.sin(math.radians(30)), 0.0, math.cos(math.radians(30))])      # the board is tipped 30 degrees and the gyro (wrongly) says nothing happened
    for _ in range(int(8 / 0.006)):
        f.update(np.zeros(3), a, 0.006)
    assert abs(tilt_deg(f) - 30.0) < 1.0                                    # gravity pulls it there within a few time constants (1 s each)


def test_the_compass_can_turn_the_heading_but_never_tilt_the_board():
    f = Madgwick(beta=0.05)
    f.q = q_from_accel(G_UP)
    tilt0 = tilt_deg(f)
    wrong = np.array([0.1, 0.9, -0.5])                                       # a disturbed field: wrong direction and a vertical part that does not fit
    for _ in range(3000):
        f.update_marg(np.zeros(3), G_UP, wrong, 0.006)
    assert abs(tilt_deg(f) - tilt0) < 0.3                                    # whatever the compass says, the horizon is the accelerometer's
    assert abs(euler_deg(f.q)[2]) > 10.0                                     # (and it did move the heading, so the test is not vacuous)


def test_a_still_board_beside_a_magnet_is_not_swung_round():
    f = Madgwick(beta=0.05)
    f.q = q_from_accel(G_UP)
    north = np.array([18.0, 0.0, -42.0])
    for _ in range(500):
        f.update_marg(np.zeros(3), G_UP, north, 0.006)                      # heading known: the compass agrees
    yaw0 = euler_deg(f.q)[2]
    magnet = np.array([-10.0, 30.0, -42.0])                                 # a 70 degree wrong heading, accepted for 4 s
    for _ in range(int(4 / 0.006)):
        f.update_marg(np.zeros(3), G_UP, magnet, 0.006)
    assert abs(euler_deg(f.q)[2] - yaw0) < 14.0                              # at most the 3 deg/s a drift could explain


def test_gate_does_not_flap_on_a_reading_that_hovers_at_the_limit():
    g = MagGate(radius=500.0, strength_tol=0.25)
    up = G_UP
    flips, prev = 0, True
    for i in range(400):
        strength = 500.0 * (1.25 + (0.01 if i % 2 else -0.01))              # alternates just inside and just outside the strength limit
        ok = g.check(i * 0.01, np.array([0.0, 0.0, strength]), up)
        flips += ok != prev
        prev = ok
    assert flips <= 1, flips


def test_gate_recovers_after_enough_good_readings():
    g = MagGate(radius=500.0)
    assert not g.check(0.0, [0, 0, 900.0], G_UP)
    for i in range(1, 5):
        assert not g.check(i * 0.01, [0, 0, 500.0], G_UP)                    # good, but not yet 5 in a row
    assert g.check(0.05, [0, 0, 500.0], G_UP)


def test_the_angle_to_gravity_is_not_judged_or_learned_while_the_board_is_shaken():
    g = MagGate(radius=500.0)
    good = lambda dip: 500.0 * np.array([math.sin(math.radians(dip)), 0, math.cos(math.radians(dip))])
    for i in range(100):
        g.check(i * 0.01, good(30), G_UP)
    assert not g.check(2.0, good(70), G_UP, trusted=True) and "angle" in g.reason
    g2 = MagGate(radius=500.0)
    for i in range(100):
        g2.check(i * 0.01, good(30), G_UP)
    assert g2.check(2.0, good(70), G_UP, trusted=False)                      # hand shaking: gravity is not where the accelerometer says, so the angle proves nothing
    assert max(g2.dips) < 31.0                                              # and the 70 was not learned as this place's normal


@pytest.mark.skipif(not os.path.exists(REC), reason="needs the recording in data/slam")
def test_on_a_real_resting_board_the_correction_is_small_and_the_tilt_stays_on_the_accelerometer():
    d = np.loadtxt(REC, delimiter=",", skiprows=1, max_rows=900)             # first ~5 s: the board rests
    t, g, a = d[:, 0] * 1e-9, d[:, 1:4], d[:, 4:7] / 9.80665
    f = Madgwick(beta=0.05)
    f.q = q_from_accel(a[0])
    still = np.linalg.norm(g.mean(0)) < 0.2
    moved, err = 0.0, []
    for i in range(1, len(t)):
        if np.linalg.norm(g[i - 3:i + 1].mean(0)) > 8.0:                     # stop at the first movement
            break
        q0 = f.q.copy()
        f.update(g[i] - g[:150].mean(0), a[i], t[i] - t[i - 1])
        moved += 2 * math.acos(min(1.0, abs(float(q0 @ f.q))))
        err.append(math.degrees(math.acos(max(-1.0, min(1.0, float((q_to_R(f.q).T @ G_UP) @ (a[i] / np.linalg.norm(a[i]))))))))
    n = len(err)
    assert n > 200, n
    assert math.degrees(moved) / (t[n] - t[0]) < 1.5                         # old filter: 4.8 deg/s on this very segment
    assert np.median(err) < 1.5
