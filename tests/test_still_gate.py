import numpy as np

from gabai.slam.still import StillGate

G = 9.80665


def feed(gate, seconds, gyro_dps=0.0, accel=(0.0, 0.0, G), hz=200, t0=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(int(seconds * hz)):
        t_ns = int((t0 + i / hz) * 1e9)
        g = np.radians([gyro_dps, 0.0, 0.0]) + rng.normal(0, noise, 3)
        out += gate.feed(t_ns, np.array(accel) + rng.normal(0, noise * 5, 3), g)
    return out


def test_becomes_ready_after_a_second_of_stillness_and_hands_over_that_second():
    gate = StillGate(window_s=1.0)
    out = feed(gate, 0.6, noise=0.002)
    assert out == [] and not gate.ready
    out += feed(gate, 0.8, t0=0.6, noise=0.002)
    assert gate.ready
    assert out[0][0] == 0                                           # the handover starts at the beginning of the still second
    assert len(out) >= 280                                          # that second (201 samples) plus what followed it
    assert [s[0] for s in out] == sorted(s[0] for s in out)


def test_after_ready_every_sample_passes_straight_through():
    gate = StillGate(window_s=0.5)
    feed(gate, 1.0, noise=0.002)
    assert gate.ready
    one = gate.feed(int(2e9), np.array([0, 0, G]), np.zeros(3))
    assert len(one) == 1 and one[0][0] == int(2e9)


def test_a_rig_that_keeps_rotating_never_becomes_ready():
    gate = StillGate(window_s=1.0)
    feed(gate, 5.0, gyro_dps=15.0)
    assert not gate.ready


def test_motion_resets_the_wait():
    gate = StillGate(window_s=1.0)
    feed(gate, 0.8, noise=0.002)
    feed(gate, 0.3, gyro_dps=30.0, t0=0.8)                          # bumped
    assert not gate.ready
    feed(gate, 0.7, t0=1.1, noise=0.002)
    assert not gate.ready                                           # only 0.7 s still since the bump
    feed(gate, 0.6, t0=1.8, noise=0.002)
    assert gate.ready


def test_accelerometer_not_reading_gravity_is_not_still():
    gate = StillGate(window_s=1.0)
    feed(gate, 3.0, accel=(0.0, 0.0, 2.0))                          # falling or a broken axis
    assert not gate.ready


def test_progress_runs_from_zero_to_one():
    gate = StillGate(window_s=1.0)
    assert gate.progress == 0.0
    feed(gate, 0.5, noise=0.002)
    assert 0.3 < gate.progress < 0.7
    feed(gate, 1.0, t0=0.5, noise=0.002)
    assert gate.progress == 1.0
