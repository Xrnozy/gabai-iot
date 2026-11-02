import numpy as np

from gabai.pc.receiver import _rotate
from gabai.tools.fake_pi import LOG_PACE_S, LOG_STUTTER, stutter_times, unturn

FPS = 1.0 / LOG_PACE_S


def test_stutter_profile_has_the_stalls_and_the_average_rate_of_the_log():
    t = np.array(stutter_times(LOG_STUTTER, FPS, 240.0))
    gaps = np.diff(t) * 1000
    assert abs(t[0] - 4.054) < 0.01 and abs(gaps.max() - 3797) < 5 and (gaps > 400).sum() >= 70      # the 4 s start-up stall, the 3.8 s stall, and the many 0.4 to 3 s ones
    assert 2.0 < len(t) / 240.0 < 2.6                                                 # the log delivered 551 pairs in 243 s (2.3 per second)


def test_stutter_repeats_to_fill_a_longer_recording():
    assert len(stutter_times(LOG_STUTTER, FPS, 600.0)) > 2 * len(stutter_times(LOG_STUTTER, FPS, 240.0)) * 0.9


def test_unturn_undoes_what_the_receiver_does():
    img = np.arange(6 * 4 * 3, dtype=np.uint8).reshape(4, 6, 3)
    for deg in (0, 90, 180, 270):
        assert np.array_equal(unturn(_rotate(img, deg), deg), img)
