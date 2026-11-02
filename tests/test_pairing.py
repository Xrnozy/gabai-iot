from gabai.pi.cams import choose_pair


def L(*ts):
    return [(t, f"L{t}") for t in ts]


def test_picks_the_right_frame_closest_in_time():
    left = L(0.900, 1.000)
    right = L(0.980, 1.010, 1.040)
    tl, fl, tr, fr = choose_pair(left, right, now=1.045)
    assert (tl, tr) == (1.000, 1.010)
    assert fr == "L1.01"


def test_waits_for_a_right_frame_that_may_be_closer():
    left = L(1.000)
    right = L(0.960)                       # right frame is older than left; a nearer one may be about to arrive
    assert choose_pair(left, right, now=1.010) is None
    tl, fl, tr, fr = choose_pair(left, right, now=1.050)   # waited long enough: use the best available
    assert tr == 0.960


def test_does_not_reuse_a_left_frame():
    left = L(1.000)
    right = L(1.005)
    assert choose_pair(left, right, now=1.01) is not None
    assert choose_pair(left, right, now=1.02, last_tl=1.000) is None


def test_empty_buffers_give_nothing():
    assert choose_pair([], L(1.0), now=1.0) is None
    assert choose_pair(L(1.0), [], now=1.0) is None
