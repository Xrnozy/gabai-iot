from gabai.tools.range_test import summarize, bucket


def test_bucket_edges():
    assert bucket(0.5) == "<0.8 m"
    assert bucket(1.2) == "0.8-1.5 m"
    assert bucket(2.4) == "1.5-3 m"
    assert bucket(4.0) == ">3 m"


def test_summarize_groups_by_distance_and_reports_bias_and_spread():
    samples = [  # (true distance m, estimated median m, spread: est p10 and p90)
        (0.6, 0.605, 0.60, 0.61), (0.7, 0.693, 0.69, 0.70),
        (2.0, 2.06, 2.00, 2.10), (2.5, 2.60, 2.50, 2.70), (2.2, 2.31, 2.20, 2.40),
    ]
    out = summarize(samples)
    assert out["<0.8 m"]["n"] == 2 and abs(out["<0.8 m"]["median_err_pct"]) < 1.0
    far = out["1.5-3 m"]
    assert far["n"] == 3 and 3.0 < far["median_err_pct"] < 5.0       # tends to overestimate at range
    assert "0.8-1.5 m" not in out                                       # no samples, no row
