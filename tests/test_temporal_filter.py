import numpy as np

from gabai.fuse_depth import TemporalFilter

H, W = 60, 80


def static_scene(n=40, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = 15 + 0.1 * x
    return d, [(d + rng.normal(0, noise, d.shape)).astype(np.float32) for _ in range(n)]


def flick(frames):
    return float(np.mean(np.std(np.stack(frames[10:]), axis=0)))


def test_static_scene_flicker_is_at_least_halved():
    d, frames = static_scene()
    tf = TemporalFilter()
    out = [tf.update(f, np.ones((H, W), bool)) for f in frames]
    assert flick(out) < flick(frames) / 2.0
    assert abs(float(np.nanmean(out[-1] - d))) < 0.2           # no bias


def test_a_real_change_passes_through_at_once():
    d, frames = static_scene(n=15, noise=0.3)
    tf = TemporalFilter()
    for f in frames:
        tf.update(f, np.ones((H, W), bool))
    near = frames[-1].copy()
    near[20:40, 20:60] += 25.0                                  # an obstacle appears: much closer (bigger disparity)
    out = tf.update(near, np.ones((H, W), bool))
    assert abs(float(out[30, 40] - near[30, 40])) < 0.5          # not smeared over several frames
    assert abs(float(out[5, 5] - frames[-1][5, 5])) < 1.0


def test_short_dropouts_are_bridged_but_long_ones_become_unknown():
    d, frames = static_scene(n=12, noise=0.2)
    tf = TemporalFilter(alpha=0.4, hold=2)
    ok = np.ones((H, W), bool)
    for f in frames:
        tf.update(f, ok)
    gone = ok.copy()
    gone[10:30, 10:30] = False
    out1 = tf.update(frames[0], gone)
    out2 = tf.update(frames[0], gone)
    out3 = tf.update(frames[0], gone)
    assert np.isfinite(out1[20, 20]) and np.isfinite(out2[20, 20])      # held for 2 frames
    assert np.isnan(out3[20, 20])                                       # then unknown, never stale
    assert np.isfinite(out3[50, 50])


def test_reset_gives_the_raw_frame_back():
    d, frames = static_scene(n=5)
    tf = TemporalFilter()
    for f in frames:
        tf.update(f, np.ones((H, W), bool))
    tf.reset()
    out = tf.update(frames[0], np.ones((H, W), bool))
    assert np.allclose(out, frames[0])


def test_a_moving_view_is_not_smeared_into_ghosts():
    from gabai.fuse_depth import TemporalFilter
    h, w = 120, 160
    x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    f = TemporalFilter()
    out = None
    for shift in range(0, 96, 12):                    # the picture slides sideways, a ramp of depth moves with it
        out = f.update(60 + 0.5 * np.roll(x, shift, axis=1), np.ones((h, w), bool))
    assert f.moving
    assert np.nanmax(np.abs(out - (60 + 0.5 * np.roll(x, 84, axis=1)))) < 1e-3     # exactly the newest frame


def test_a_steady_view_is_still_averaged_after_the_change():
    from gabai.fuse_depth import TemporalFilter
    rng = np.random.default_rng(0)
    f = TemporalFilter()
    base = np.full((120, 160), 80.0, np.float32)
    outs = [f.update(base + rng.normal(0, 0.3, base.shape).astype(np.float32), np.ones(base.shape, bool)) for _ in range(20)]
    assert not f.moving
    assert np.std(outs[-1]) < 0.7 * 0.3


def test_a_rejected_pixel_leaves_no_trail_while_moving_but_is_held_when_steady():
    from gabai.fuse_depth import TemporalFilter
    d = np.full((120, 160), 80.0, np.float32)
    ok = np.ones(d.shape, bool)
    gone = ok.copy()
    gone[:, 50:60] = False
    for alpha, expect_nan in ((None, False), (1.0, True)):          # steady: one rejected frame is bridged; moving: not
        f = TemporalFilter()
        f.update(d, ok)
        out = f.update(d, gone, alpha=alpha)
        assert np.isnan(out[:, 50:60]).all() == expect_nan
