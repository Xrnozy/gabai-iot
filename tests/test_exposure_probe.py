import numpy as np

from gabai.pi.exposure_probe import frame_stats, recommend, responds


def row(e, g, lm, rm=None, clip=0.0, mode="manual"):
    rm = lm if rm is None else rm
    s = lambda m: {"median": m, "clipped_pct": clip, "noise": 3.0}
    return {"mode": mode, "exposure_ms": e, "gain": g, "left": s(lm), "right": s(rm)}


def test_the_stats_see_brightness_clipping_and_temporal_noise():
    rng = np.random.default_rng(0)
    base = np.full((60, 80), 100.0)
    frames = [np.clip(base + rng.normal(0, 4, base.shape), 0, 255) for _ in range(6)]
    s = frame_stats(frames)
    assert abs(s["median"] - 100) < 2 and s["clipped_pct"] == 0 and abs(s["noise"] - 4) < 0.6
    assert frame_stats([np.full((10, 10), 255.0)] * 2)["clipped_pct"] == 100


def test_the_shortest_exposure_that_is_bright_enough_and_matching_wins_and_lowest_gain_breaks_ties():
    rows = [row(4, 0, 20), row(8, 0, 40), row(16, 0, 80), row(16, 100, 140), row(24, 0, 120), row(8, 100, 90)]
    best, why = recommend(rows)
    assert (best["exposure_ms"], best["gain"]) == (8, 100) and "shortest" in why     # 8 ms at full gain beats 16 ms at no gain: less smear
    rows = [row(8, 50, 90), row(8, 100, 90)]
    assert recommend(rows)[0]["gain"] == 50


def test_pictures_that_do_not_match_or_clip_are_not_recommended():
    rows = [row(8, 50, 100, 70), row(12, 50, 100, 100, clip=5.0), row(16, 50, 100, 100)]
    assert recommend(rows)[0]["exposure_ms"] == 16


def test_when_nothing_is_bright_enough_the_best_one_is_returned_with_the_reason():
    best, why = recommend([row(8, 100, 30), row(16, 100, 42), row(33, 100, 60)], cap_ms=16)
    assert best["exposure_ms"] == 16 and "add light" in why


def test_an_ignored_exposure_control_is_detected():
    ok = [row(4, 0, 20), row(8, 0, 40), row(16, 0, 80)]
    assert responds(ok) is True
    ignored = [row(4, 0, 41), row(8, 0, 40), row(16, 0, 42), row(33, 0, 43)]
    assert responds(ignored) is False
    assert responds([row(8, 0, 40)]) is None
