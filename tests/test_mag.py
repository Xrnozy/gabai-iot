import math
import struct

import numpy as np

from gabai.mag import calibrate, coverage, field_ut, fit_sphere, heading_deg, sweep_deg
from gabai.pi.mag import decode_mag


def sphere_points(n=600, radius=45.0, center=(10.0, -20.0, 5.0), scale=(1, 1, 1), noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    p = v * radius * np.array(scale) + np.array(center) + rng.normal(0, noise, (n, 3))
    return p


def test_decode_uses_the_chips_x_z_y_register_order():
    raw = struct.pack(">3h", 100, -200, 300)          # chip sends X, Z, Y
    assert decode_mag(raw) == (100, 300, -200)


def test_decode_flags_overflow():
    raw = struct.pack(">3h", 100, -4096, 300)         # -4096 means the sensor range was exceeded
    assert decode_mag(raw) is None


def test_field_strength_in_microtesla():
    # gain code 0x20: 0.92 milligauss per count; 10 milligauss = 1 microtesla
    assert math.isclose(field_ut((543.48, 0, 0)), 50.0, rel_tol=1e-3)


def test_sphere_fit_recovers_centre_and_radius():
    c, r = fit_sphere(sphere_points())
    assert np.allclose(c, (10, -20, 5), atol=0.5)
    assert abs(r - 45.0) < 0.5


def test_calibration_removes_offset_and_stretch():
    p = sphere_points(scale=(1.0, 0.8, 1.2), center=(30, -15, 12))
    raw_quality = np.std(np.linalg.norm(p, axis=1)) / np.mean(np.linalg.norm(p, axis=1))
    cal = calibrate(p)
    fixed = (p - cal["offset"]) * cal["scale"]
    q = np.std(np.linalg.norm(fixed, axis=1)) / np.mean(np.linalg.norm(fixed, axis=1))
    assert q < 0.03 and q < raw_quality / 3
    assert abs(cal["quality"] - q) < 0.01
    assert cal["coverage"] >= 0.95


def test_coverage_is_low_when_only_one_side_was_visited():
    p = sphere_points(center=(0, 0, 0))
    p = p[p[:, 2] > 20]                                   # board never tilted upside down
    assert coverage(p, (0, 0, 0)) < 0.7


def test_heading_runs_through_the_compass_points():
    assert abs(heading_deg(1, 0) - 0) < 1e-9
    assert abs(heading_deg(0, 1) - 90) < 1e-9
    assert abs(heading_deg(-1, 0) - 180) < 1e-9
    assert abs(heading_deg(0, -1) - 270) < 1e-9


def test_sweep_counts_a_full_turn_through_the_wraparound():
    h = [(i * 30) % 360 for i in range(13)]               # 0, 30, ... 330, 0 : one full turn
    assert abs(sweep_deg(h) - 360.0) < 1e-6
    assert abs(sweep_deg([350, 10, 30]) - 40.0) < 1e-6    # crosses 360 -> 0
