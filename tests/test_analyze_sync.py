import random

from gabai.spike.analyze_sync import detect_spikes, match_spikes, estimate_offset_ms


def simulate(offset_s, exposure_s, period_s=1 / 15, n_pulses=400, seed=1):
    """Pulses are instantaneous. A camera sees a pulse if it falls inside one of its exposure windows."""
    rng = random.Random(seed)
    pulses = [i * 0.5 + rng.random() * 0.3 for i in range(n_pulses)]
    left, right = [], []
    for p in pulses:
        k = int(p / period_s)
        for kk in (k, k + 1):
            start_l = kk * period_s
            start_r = kk * period_s + offset_s
            if start_l <= p <= start_l + exposure_s:
                left.append(kk * period_s)
            if start_r <= p <= start_r + exposure_s:
                right.append(kk * period_s)
    return left, right


def test_detect_spikes_finds_bright_frames():
    t = [i / 15 for i in range(300)]
    b = [100.0 + (i % 3) * 0.2 for i in range(300)]
    for i in (50, 120, 200):
        b[i] = 180.0
    assert detect_spikes(t, b) == [t[50], t[120], t[200]]


def test_identical_cameras_give_zero_offset():
    left, right = simulate(0.0, 0.030)
    both, nl, nr = match_spikes(left, right, tol=0.03)
    assert estimate_offset_ms(both, nl, nr, 0.030) < 2.0


def test_offset_estimate_close_to_truth():
    for true_ms in (10, 20):
        left, right = simulate(true_ms / 1000, 0.030)
        both, nl, nr = match_spikes(left, right, tol=0.03)
        est = estimate_offset_ms(both, nl, nr, 0.030)
        assert abs(est - true_ms) < 5.0, (true_ms, est)


def test_offset_beyond_exposure_is_flagged_as_lower_bound():
    left, right = simulate(0.050, 0.030)
    both, nl, nr = match_spikes(left, right, tol=0.03)
    assert estimate_offset_ms(both, nl, nr, 0.030) >= 28.0


def test_best_cell_finds_the_flashing_block():
    import numpy as np
    from gabai.spike.analyze_sync import best_cell
    rng = np.random.default_rng(0)
    g = 30 + rng.normal(0, 0.5, (300, 6, 8))
    for i in (20, 70, 120, 200, 260):
        g[i, 2, 5] = 200                      # light flashes in block (row 2, col 5) only
    g[:, 4, 1] += np.linspace(0, 15, 300)     # a slow drift elsewhere must not win
    (r, c), series = best_cell(g)
    assert (r, c) == (2, 5)
    assert series.max() > 150


def _strobe_series(offset_s, exposure_s=0.030, fps=15.0, period=0.37, seconds=60, lo=20.0, hi=220.0):
    """Brightness of two cameras watching a square-wave light (50% duty). Each frame = exposure-window overlap with 'on'."""
    import numpy as np
    n = int(seconds * fps)
    T = 1 / fps
    def cam(shift):
        out = []
        for k in range(n):
            s = k * T + 0.004 + shift          # window start
            # overlap of [s, s+E] with on-intervals [m*P, m*P + P/2]
            ov, m0 = 0.0, int(s // period)
            for m in (m0 - 1, m0, m0 + 1):
                a, b = m * period, m * period + period / 2
                ov += max(0.0, min(s + exposure_s, b) - max(s, a))
            out.append(lo + (hi - lo) * ov / exposure_s)
        return np.array(out)
    return cam(0.0), cam(offset_s)


def test_detect_pulses_ignores_long_runs():
    from gabai.spike.analyze_sync import detect_pulses
    t = [i / 15 for i in range(300)]
    b = [20.0] * 300
    for i in range(100, 140):
        b[i] = 80.0                      # a hand/lamp: 40 frames bright in a row, not a flash
    assert detect_pulses(t, b) == []
    for i in (20, 40, 60, 80):
        b[i] = 200.0                     # isolated flashes
    assert len(detect_pulses(t, b)) == 4


def test_edge_method_recovers_the_offset():
    from gabai.spike.analyze_sync import edge_offsets
    for true_ms in (0, 8, 15):
        bl, br = _strobe_series(true_ms / 1000)
        off = edge_offsets(bl, br, 0.030)
        assert len(off) >= 20, (true_ms, len(off))
        est = abs(float(sorted(off)[len(off) // 2])) * 1000
        assert abs(est - true_ms) < 4.0, (true_ms, est)


def test_edge_method_returns_nothing_without_contrast():
    import numpy as np
    from gabai.spike.analyze_sync import edge_offsets
    flat = 25 + np.random.default_rng(1).normal(0, 0.3, 900)
    assert len(edge_offsets(flat, flat.copy(), 0.030)) == 0


def test_best_cell_handles_a_fifty_percent_duty_strobe():
    import numpy as np
    from gabai.spike.analyze_sync import best_cell
    rng = np.random.default_rng(2)
    g = 10 + rng.normal(0, 0.05, (400, 6, 8))
    on = (np.arange(400) // 4) % 2 == 0           # 4 frames on, 4 frames off: median sits mid-way, MAD is huge
    g[:, 3, 6] = np.where(on, 150.0, 10.0)        # the strobe
    g[100:160, 0, 0] += 40                        # a hand/lamp: long bright run, few edges
    g[:, 5, 5] += rng.normal(0, 0.3, 400)         # noisy but dull
    (r, c), series = best_cell(g)
    assert (r, c) == (3, 6)
