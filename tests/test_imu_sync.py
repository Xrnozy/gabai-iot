import cv2
import numpy as np

from gabai.tools.sync_imu_camera import visual_rates, estimate_delay, estimate_axes, resample


def textured(seed=0, w=160, h=120):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w), dtype=np.uint8), (0, 0), 2.0)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def test_visual_rates_follow_the_camera_axis_convention():
    a = textured()
    f, dt = 100.0, 0.1
    # Convention: dx = -f * w_y * dt and dy = +f * w_x * dt. Content moving right means w_y < 0.
    b = np.roll(a, 3, axis=1)
    t, wx, wy = visual_rates([a, b], [0.0, dt], f)[0]
    assert abs(t - 0.05) < 1e-9
    assert abs(wy - (-3 / (f * dt))) < 0.05 * abs(3 / (f * dt)) and abs(wx) < 0.05
    c = np.roll(a, 2, axis=0)                      # content moves down 2 px: w_x = +dy / (f dt)
    t, wx, wy = visual_rates([a, c], [0.0, dt], f)[0]
    assert abs(wx - 2 / (f * dt)) < 0.05 * abs(2 / (f * dt)) and abs(wy) < 0.05


def _signals(delay, seconds=20, rate=200, vis_rate=15):
    rng = np.random.default_rng(3)
    t = np.arange(0, seconds, 1 / rate)

    def smooth():
        x = np.convolve(rng.normal(0, 1, len(t)), np.ones(60) / 60, mode="same")
        return 1.5 * x / x.std()

    w = np.stack([smooth(), smooth(), smooth()])               # true camera angular velocity, rad/s
    tv = np.arange(0.5, seconds - 0.5, 1 / vis_rate)
    wv = np.stack([np.interp(tv - delay, t, w[i]) for i in range(3)])   # frame stamped late by `delay`
    return t, w, tv, wv


def test_delay_is_recovered():
    for true in (0.0, 0.045, 0.12, -0.03):
        t, w, tv, wv = _signals(true)
        est, quality = estimate_delay(tv, np.linalg.norm(wv[:2], axis=0), t, np.linalg.norm(w[:2], axis=0))
        assert abs(est - true) < 0.012, (true, est)
        assert quality > 0.8


def test_axis_map_is_recovered_with_signs():
    t, w, tv, wv = _signals(0.05)
    M = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float)     # imu = M @ cam  (a signed permutation)
    gyro = M @ w                                                   # what the IMU would report
    R_ci, corr = estimate_axes(tv, wv[0], wv[1], t, gyro, delay=0.05)
    assert np.allclose(R_ci, M.T), R_ci
    assert abs(np.linalg.det(R_ci) - 1) < 1e-9


def test_resample_interpolates():
    assert np.allclose(resample(np.array([0.0, 1.0]), np.array([0.0, 10.0]), np.array([0.25, 0.5])), [2.5, 5.0])


def test_signed_xy_delay_is_more_exact_than_speed_matching():
    from gabai.tools.sync_imu_camera import estimate_delay_xy
    M = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float)
    for true in (0.0, 0.07, 0.11):
        t, w, tv, wv = _signals(true)
        gyro = M @ w
        est, q = estimate_delay_xy(tv, wv[0], wv[1], t, gyro, M.T)
        assert abs(est - true) < 0.006, (true, est)
        assert q > 0.9
