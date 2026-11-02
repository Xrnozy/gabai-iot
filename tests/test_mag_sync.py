import numpy as np

from gabai.mag_sync import rotations_24, estimate_mag_alignment, estimate_mag_delay

FIELD = np.array([18.0, 0.0, -42.0])          # world magnetic field (uT): north component and a downward component


def smooth_omega(t, rng, peak=2.5):
    def one():
        x = np.convolve(rng.normal(0, 1, len(t)), np.ones(80) / 80, mode="same")
        return peak * x / x.std()
    return np.stack([one(), one(), one()], axis=1)           # rad/s, body frame


def simulate(delay, M, seconds=25, rate=200, mag_rate=50, noise=0.4, seed=1):
    """Rotate a body in a fixed field. Gyro = true rate (IMU frame). The compass reports the field in ITS own frame
    (m_mag = M @ m_imu) and its stamps are late by `delay`."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / rate)
    w = smooth_omega(t, rng)
    R = np.eye(3)
    ms = []
    for k in range(len(t)):
        ms.append(R.T @ FIELD)                                   # field in the body (IMU) frame
        wx = np.array([[0, -w[k, 2], w[k, 1]], [w[k, 2], 0, -w[k, 0]], [-w[k, 1], w[k, 0], 0]])
        R = R @ (np.eye(3) + wx / rate)
        u, _, vt = np.linalg.svd(R)
        R = u @ vt
    ms = np.array(ms)
    tm = np.arange(0.5, seconds - 0.5, 1 / mag_rate)
    true = np.stack([np.interp(tm - delay, t, ms[:, i]) for i in range(3)], axis=1)   # stamped late by `delay`
    mag = (M @ true.T).T + rng.normal(0, noise, true.shape)
    return t, w, tm, mag


def test_there_are_24_proper_rotations():
    R = rotations_24()
    assert len(R) == 24
    assert all(abs(np.linalg.det(r) - 1) < 1e-9 for r in R)


def test_axis_map_and_delay_are_recovered():
    for delay, M in ((0.04, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)),
                     (0.0, np.eye(3)),
                     (0.09, np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], float))):
        t, w, tm, mag = simulate(delay, M)
        d, R_im, score = estimate_mag_delay(tm, mag, t, w)
        assert abs(d - delay) < 0.012, (delay, d)
        assert np.allclose(R_im, M.T), (R_im, M)       # imu = R_im @ mag
        assert score > 0.8


def test_unrelated_compass_data_scores_low_and_a_wrong_delay_scores_lower():
    M = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    t, w, tm, mag = simulate(0.03, M)
    _, good = estimate_mag_alignment(tm, mag, t, w, delay=0.03)
    _, late = estimate_mag_alignment(tm, mag, t, w, delay=0.20)           # right axes, wrong time shift
    _, other = estimate_mag_alignment(tm, simulate(0.03, M, seed=9)[3], t, w, delay=0.03)   # a different motion
    assert good > 0.9
    assert late < good - 0.2
    assert other < 0.5
