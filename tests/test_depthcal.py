import json
import os
import tempfile

import numpy as np

from gabai import depthcal


def test_the_correction_follows_the_measured_ratio_inside_the_range_and_fades_to_nothing_outside():
    c = depthcal.DisparityCorrection([30.0, 50.0, 70.0], [1.2, 1.0, 0.9])
    assert np.allclose(c(np.array([30.0, 50.0, 70.0])), [36.0, 50.0, 63.0])
    assert c(np.array([10.0]))[0] == 10.0 and c(np.array([100.0]))[0] == 100.0         # a far wall, or a very near hand, is nobody's data
    mid = float(c(np.array([27.0]))[0])
    assert 27.0 < mid < 27.0 * 1.2                                                    # in the taper between the two
    assert np.isnan(c(np.array([np.nan]))[0]) and c(np.array([-1.0]))[0] == -1.0


def test_a_fit_recovers_a_known_distance_dependent_bias_and_reduces_the_error():
    rng = np.random.default_rng(0)
    true = rng.uniform(34, 72, 4000)
    pred = 0.7 * true + 12.0 + rng.normal(0, 0.3, true.shape)                        # a model that compresses disparity
    c = depthcal.fit_knots(pred, true, 3)
    err_before = np.abs(pred / true - 1).mean()
    err_after = np.abs(c(pred.astype(np.float32)) / true - 1).mean()
    assert err_after < 0.35 * err_before, (err_before, err_after)


def test_load_returns_none_for_a_missing_file_or_a_model_that_was_not_fitted():
    assert depthcal.load("no/such/file.json", "h") is None
    p = os.path.join(tempfile.mkdtemp(), "c.json")
    json.dump({"models": {"h": {"knots_px": [30, 60], "ratio": [1.1, 0.95]}}}, open(p, "w"))
    assert depthcal.load(p, "h") is not None and depthcal.load(p, "s") is None
