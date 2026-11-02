import numpy as np

from gabai.tools.noise_test import temporal_stats, plane_residual, depth_jitter_cm


def test_temporal_std_recovers_the_injected_noise():
    rng = np.random.default_rng(0)
    base = np.linspace(10, 40, 100 * 120).reshape(100, 120).astype(np.float32)
    stack = np.stack([base + rng.normal(0, 0.8, base.shape) for _ in range(60)]).astype(np.float32)
    ok = np.ones_like(stack, bool)
    s = temporal_stats(stack, ok)
    assert abs(s["temporal_std_px"] - 0.8) < 0.05
    assert abs(s["frame_to_frame_px"] - 0.8 * np.sqrt(2) * 0.8) < 0.15       # median |difference| of two noisy frames
    assert s["usable_fraction"] > 0.99


def test_pixels_that_are_often_untrusted_are_left_out():
    rng = np.random.default_rng(1)
    stack = (20 + rng.normal(0, 0.3, (40, 50, 50))).astype(np.float32)
    ok = np.ones_like(stack, bool)
    ok[:30, :, :25] = False                                    # left half untrusted in most frames
    stack[:, :, :25] += rng.normal(0, 5, (40, 50, 25))        # and it is very noisy
    s = temporal_stats(stack, ok)
    assert s["temporal_std_px"] < 0.5 and 0.4 < s["usable_fraction"] < 0.6


def test_plane_residual_of_a_tilted_plane_is_only_the_noise():
    rng = np.random.default_rng(2)
    y, x = np.mgrid[0:90, 0:120].astype(np.float32)
    d = 12 + 0.04 * x + 0.02 * y + rng.normal(0, 0.25, x.shape).astype(np.float32)
    assert abs(plane_residual(d, (10, 10, 110, 80)) - 0.25) < 0.03


def test_depth_jitter_in_cm_grows_with_the_square_of_distance():
    fb = 881.0 * 0.0506
    a, b = depth_jitter_cm(0.3, fb, 1.0), depth_jitter_cm(0.3, fb, 3.0)
    assert abs(b / a - 9.0) < 1e-6
    assert abs(depth_jitter_cm(0.5, fb, 3.0) - 10.1) < 0.2          # 0.5 px at 3 m is about 10 cm
