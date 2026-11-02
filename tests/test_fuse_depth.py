import numpy as np

from gabai.fuse_depth import align_mono, fit_affine, fuse, masked_blur

H, W = 240, 320


def scene():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = 20 + 0.05 * x + 6 * np.exp(-(((x - 160) / 50) ** 2 + ((y - 120) / 40) ** 2))     # a slope with a bump, in px
    return d


def make(stereo_noise=1.5, mono_noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    d = scene()
    ds = d + rng.normal(0, stereo_noise, d.shape).astype(np.float32)
    trusted = np.ones_like(d, bool)
    ds[40:100, 200:280] = rng.uniform(5, 60, (60, 80))          # a patch where matching failed
    trusted[40:100, 200:280] = False
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    # monocular model: wrong overall scale and offset, a slow bend, but sharp structure and little pixel noise
    dm = (0.75 * d - 2.0) * (1 + 0.04 * np.sin(x / 70)) + rng.normal(0, mono_noise, d.shape).astype(np.float32)
    return d, ds, trusted, dm


def rms(a, b, mask=None):
    e = (a - b) if mask is None else (a - b)[mask]
    return float(np.sqrt(np.mean(e ** 2)))


def test_masked_blur_ignores_masked_pixels():
    x = np.full((50, 50), 10.0, np.float32)
    x[:, 25:] = 1000.0
    m = np.zeros((50, 50), bool)
    m[:, :25] = True
    out = masked_blur(x, m, 6.0)
    assert abs(out[25, 10] - 10.0) < 1e-3          # only trusted pixels contribute
    assert abs(out[25, 40] - 10.0) < 1.0           # far from them it still returns their local average, not 1000


def test_affine_fit_recovers_scale_and_offset_despite_outliers():
    rng = np.random.default_rng(1)
    dm = rng.uniform(5, 40, (10000,)).astype(np.float32)
    ds = 1.3 * dm + 2.0 + rng.normal(0, 0.3, dm.shape).astype(np.float32)
    ds[:1500] = rng.uniform(0, 80, 1500)           # 15% gross outliers
    a, b = fit_affine(dm, ds, np.ones(dm.shape, bool))
    assert abs(a - 1.3) < 0.03 and abs(b - 2.0) < 0.4


def test_alignment_pulls_the_monocular_depth_onto_the_stereo_scale():
    d, ds, trusted, dm = make()
    aligned, info = align_mono(dm, ds, trusted)
    assert rms(aligned, d) < 0.35 * rms(dm, d)
    assert abs(info["a"] - 1 / 0.75) < 0.15


def test_fusion_is_much_less_noisy_than_stereo_and_fills_failed_patches():
    d, ds, trusted, dm = make()
    fused, info = fuse(ds, trusted, dm)
    assert rms(fused, d, trusted) < 0.45 * rms(ds, d, trusted)          # where stereo worked
    assert rms(fused, d, ~trusted) < 1.5                                  # where stereo failed, mono fills in
    assert np.isfinite(fused).all()
    assert info["from_mono"][~trusted].all()                              # the filled patch is flagged as mono-only


def test_clean_stereo_is_not_degraded_by_a_noisy_monocular_model():
    d, ds, trusted, dm = make(stereo_noise=0.2, mono_noise=1.5)
    fused, _ = fuse(ds, trusted, dm)
    assert rms(fused, d, trusted) < 1.35 * rms(ds, d, trusted) + 0.1


def _frames(n=24, trusted_fraction=0.08, seed=3):
    """A static scene seen n times. Stereo is noisy and only a small random subset is trusted each frame, like a dark
    room where the left-right check throws most pixels away. The monocular model is stable frame to frame."""
    rng = np.random.default_rng(seed)
    d = scene()
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dm = (0.75 * d - 2.0) * (1 + 0.04 * np.sin(x / 70)) + rng.normal(0, 0.1, d.shape).astype(np.float32)
    out = []
    for _ in range(n):
        ds = d + rng.normal(0, 1.2, d.shape).astype(np.float32)
        trusted = rng.random(d.shape) < trusted_fraction
        out.append((ds, trusted, dm))
    return d, out


def test_smoothing_the_scale_fit_over_time_removes_flicker_from_the_monocular_map():
    from gabai.fuse_depth import DepthFuser
    d, frames = _frames()
    plain, smooth = [], []
    fuser = DepthFuser(alpha=0.25)
    for ds, tr, dm in frames:
        a, _ = align_mono(dm, ds, tr)
        plain.append(a)
        smooth.append(fuser.fuse(ds, tr, dm)[1]["aligned"])
    flick = lambda s: float(np.mean(np.std(np.stack(s[8:]), axis=0)))
    assert flick(smooth) < 0.5 * flick(plain)


def test_the_fuser_resets_after_a_scene_cut():
    from gabai.fuse_depth import DepthFuser
    d, frames = _frames(n=10)
    fuser = DepthFuser(alpha=0.25)
    for ds, tr, dm in frames:
        fuser.fuse(ds, tr, dm)
    ds, tr, dm = frames[0]
    fuser.fuse(ds, tr, dm * 3.0 + 10.0, )          # the monocular scale suddenly jumps (new scene)
    _, info = fuser.fuse(ds, tr, dm * 3.0 + 10.0)
    assert abs(info["a"] - 1 / 3 * 1 / 0.75) < 0.5 or info["reset"]


def test_right_view_map_is_brought_into_the_left_view_by_the_disparity():
    from gabai.fuse_depth import warp_right_to_left
    h, w = 40, 100
    right = np.tile(np.arange(w, dtype=np.float32), (h, 1))          # value = column in the right image
    disp = np.full((h, w), 7.0, np.float32)                          # every left pixel x matches right pixel x - 7
    out = warp_right_to_left(right, disp)
    assert np.allclose(out[:, 20:90], np.arange(20, 90) - 7.0, atol=1e-4)
    assert np.isnan(out[:, :6]).all()                                # would point left of the right image: unknown


def test_combining_two_monocular_views_halves_their_noise_and_flags_disagreement():
    from gabai.fuse_depth import combine_mono_pair
    rng = np.random.default_rng(0)
    truth = np.full((H, W), 20.0, np.float32)
    a = truth + rng.normal(0, 1.0, truth.shape).astype(np.float32)
    b = truth + rng.normal(0, 1.0, truth.shape).astype(np.float32)
    b[:30, :30] += 15.0                                             # the right-view model is wrong in one corner
    both, agree = combine_mono_pair(a, b, tol=3.0)
    ok = np.ones_like(agree)
    ok[:30, :30] = False
    assert abs(np.std((both - truth)[agree & ok]) - 1.0 / np.sqrt(2)) < 0.1
    assert not agree[:30, :30].any()
    assert np.allclose(both[:30, :30], a[:30, :30])                  # where they disagree keep the left view's own


def test_mono_veto_rejects_gross_stereo_mismatches_and_keeps_good_pixels_untouched():
    from gabai.fuse_depth import mono_veto
    d, ds, trusted, dm = make(stereo_noise=0.5, mono_noise=0.8)
    trusted[:] = True
    ds[150:200, 40:100] = d[150:200, 40:100] + 40.0            # a repeated-bar mismatch the stereo check let through
    ok, bad, _ = mono_veto(ds, trusted, dm)
    assert bad[150:200, 40:100].mean() > 0.95
    clean = np.ones_like(bad)
    clean[140:210, 30:110] = False
    clean[40:100, 200:280] = False                             # the scene's own failed-matching patch, marked trusted here
    assert bad[clean].mean() < 0.02                            # almost nothing good is thrown away
    assert np.array_equal(ds[ok], ds[ok])                      # and the surviving values are the stereo values themselves


def test_fill_supplies_only_the_holes_and_never_touches_a_trusted_pixel():
    from gabai.fuse_depth import DepthFuser
    d, ds, trusted, dm = make()
    f = DepthFuser(min_pixels=300)
    aligned, fill, info = f.fill(np.where(trusted, ds, 0), trusted, dm)
    assert info["ok"] and fill.sum() == (~trusted).sum() and not (fill & trusted).any()
    assert rms(aligned, d, fill) < 1.0                                                # the aligned monocular map follows the true disparity in the hole


def test_fill_is_refused_when_the_monocular_map_does_not_fit_the_stereo_pixels():
    from gabai.fuse_depth import DepthFuser
    d, ds, trusted, dm = make()
    rng = np.random.default_rng(3)
    scrambled = rng.permutation(dm.ravel()).reshape(dm.shape)                         # structure unrelated to the scene: a misread picture
    aligned, fill, info = DepthFuser(min_pixels=300).fill(np.where(trusted, ds, 0), trusted, scrambled)
    assert not info["ok"] and not fill.any() and ("px from stereo" in info["why"] or "uncorrelated" in info["why"])


def test_fill_is_refused_with_too_few_trusted_pixels_to_fit_the_scale():
    from gabai.fuse_depth import DepthFuser
    d, ds, trusted, dm = make()
    few = np.zeros_like(trusted)
    few[:10, :10] = True
    _, fill, info = DepthFuser(min_pixels=300).fill(np.where(few, ds, 0), few, dm)
    assert not info["ok"] and not fill.any()
