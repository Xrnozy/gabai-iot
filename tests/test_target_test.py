import numpy as np

from gabai.tools.target_test import region_stats, summarize_targets

F, B = 881.0, 0.0507


def disp_for(z, shape=(480, 640)):
    return np.full(shape, F * B / z, np.float32)


def test_region_stats_report_depth_and_the_share_of_pixels_that_had_any():
    d = disp_for(2.0)
    trusted = np.ones(d.shape, bool)
    trusted[:, :320] = False                           # half the box is rejected (glass, a dark patch)
    s = region_stats(d, trusted, (280, 200, 360, 280), F, B)
    assert abs(s["est_m"] - 2.0) < 0.01 and abs(s["valid_pct"] - 50.0) < 1.0


def test_a_box_with_no_trusted_pixels_has_no_depth_not_zero_or_infinity():
    d = disp_for(2.0)
    s = region_stats(d, np.zeros(d.shape, bool), (280, 200, 360, 280), F, B)
    assert s["est_m"] is None and s["valid_pct"] == 0.0


def test_wrong_depth_on_a_thin_pole_shows_up_as_error_and_spread():
    d = disp_for(3.0)
    d[:, 300:340] = F * B / 6.0                        # the pole's box is mostly background 6 m behind it
    trusted = np.ones(d.shape, bool)
    s = region_stats(d, trusted, (280, 200, 360, 280), F, B)
    assert s["p90_m"] > 5.0 and s["p10_m"] < 3.5


def test_target_summary_groups_by_label_with_error_and_valid_share():
    rows = [dict(label="wall", true_m=1.0, est_m=1.01, valid_pct=99.0), dict(label="wall", true_m=2.0, est_m=2.06, valid_pct=98.0),
            dict(label="glass", true_m=1.5, est_m=None, valid_pct=0.0), dict(label="glass", true_m=2.5, est_m=4.0, valid_pct=10.0)]
    s = summarize_targets(rows)
    assert abs(s["wall"]["median_err_pct"] - 2.0) < 0.6 and s["wall"]["n"] == 2
    assert s["glass"]["no_depth"] == 1 and s["glass"]["n"] == 2 and abs(s["glass"]["worst_err_pct"] - 60.0) < 1e-6
