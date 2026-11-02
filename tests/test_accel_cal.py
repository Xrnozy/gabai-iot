import numpy as np

from gabai.slam.accel_cal import (G, cluster_orientations, faces_covered, fit_bias_scale, still_windows,
                                  to_basalt_params)

TRUE_B = np.array([0.20, -0.15, 0.30])
TRUE_S = np.array([-0.030, 0.012, -0.045])


def raw_reading(direction, rng, noise=0.01):
    """What the accelerometer reports when gravity points along `direction`: inverse of calibrated = raw*(1+s) - b."""
    cal = np.asarray(direction, float) / np.linalg.norm(direction) * G
    return (cal + TRUE_B) / (1 + TRUE_S) + rng.normal(0, noise, 3)


def orientations(rng, n=30):
    d = rng.normal(size=(n, 3))
    faces = np.vstack([np.eye(3), -np.eye(3)])
    return np.vstack([faces, d])


def test_fit_recovers_bias_and_scale_from_many_orientations():
    rng = np.random.default_rng(1)
    means = np.array([raw_reading(d, rng) for d in orientations(rng)])
    fit = fit_bias_scale(means)
    assert fit["observable"]
    assert np.allclose(fit["bias"], TRUE_B, atol=0.03) and np.allclose(fit["scale"], TRUE_S, atol=0.004)
    assert fit["rms_after"] < 0.02 < fit["rms_before"]


def test_one_orientation_cannot_separate_bias_from_scale_and_says_so():
    rng = np.random.default_rng(2)
    means = np.array([raw_reading([0.7, -0.66, -0.17], rng, noise=0.005) for _ in range(40)])
    assert not fit_bias_scale(means)["observable"]


def test_faces_covered_names_the_missing_ones():
    u = np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0], [0.9, 0.3, 0.1]])
    c = faces_covered(u)
    assert c["+z"] and c["-z"] and c["+x"] and not c["-x"] and not c["+y"] and not c["-y"]


def test_still_windows_keep_only_quiet_stretches():
    rng = np.random.default_rng(3)
    t = np.arange(0, 6, 0.005)
    acc = np.tile([0.0, 0.0, G], (len(t), 1)) + rng.normal(0, 0.01, (len(t), 3))
    gyro = np.zeros((len(t), 3))
    shaken = (t > 2) & (t < 4)                                        # shaken for two seconds
    gyro[shaken] = np.radians(60)
    acc[shaken] += rng.normal(0, 2.0, (shaken.sum(), 3))
    w = still_windows(t, acc, gyro, win_s=0.25)
    assert 8 <= len(w) <= 16                                          # 4 s of quiet time, minus the shaken 2 s
    assert np.allclose(np.linalg.norm(w, axis=1), G, atol=0.05)


def test_nearby_windows_merge_into_one_orientation():
    a = np.array([[0.0, 0.0, 9.8], [0.0, 0.1, 9.8], [9.8, 0.0, 0.0], [9.7, 0.1, 0.1]])
    c = cluster_orientations(a, min_sep_deg=15)
    assert len(c) == 2


def test_parameters_match_basalts_layout():
    p = to_basalt_params(np.array([0.1, 0.2, 0.3]), np.array([0.01, 0.02, 0.03]))
    assert p == [0.1, 0.2, 0.3, 0.01, 0.0, 0.0, 0.02, 0.0, 0.03]
    # Basalt: calibrated = raw + S*raw - bias with S = [[p3,0,0],[p4,p6,0],[p5,p7,p8]]
    raw = np.array([1.0, 2.0, 3.0])
    S = np.array([[p[3], 0, 0], [p[4], p[6], 0], [p[5], p[7], p[8]]])
    assert np.allclose(raw + S @ raw - np.array(p[:3]), raw * (1 + np.array([0.01, 0.02, 0.03])) - [0.1, 0.2, 0.3])


def test_a_hold_is_captured_once_and_the_next_hold_after_motion_is_captured_again():
    from gabai.slam.accel_cal import StillCapture
    cap = StillCapture(min_s=1.0)
    got = []
    t = 0.0
    for phase, secs in (("still", 2.5), ("move", 0.5), ("still", 1.5)):
        for _ in range(int(secs * 200)):
            gyro = np.radians([40.0, 0, 0]) if phase == "move" else np.zeros(3)
            r = cap.feed(t, [0.0, 0.0, G], gyro)
            if r is not None:
                got.append((round(t, 2), r))
            t += 1 / 200
    assert len(got) == 2 and got[0][0] < 1.3 and got[1][0] > 3.0
    assert np.allclose(got[0][1], [0, 0, G])
