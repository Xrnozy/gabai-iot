import cv2
import numpy as np

from rtabmap_lab.depthfusion import FusionParams, StereoIn, fuse

CAL = {"f": 600.0, "B": 0.04}
H, W = 360, 640
FB = CAL["f"] * CAL["B"]


def scene():
    """A wall 3 m away, a box 1.5 m away in the middle, a thin pole 1.2 m away (8 px wide): the true depth in metres."""
    d = np.full((H, W), 3.0, np.float32)
    d[110:260, 220:420] = 1.5
    d[60:330, 560:568] = 1.2
    return d


def stereo_noise(rng, z, sigma_px):
    disp = FB / z + rng.normal(0, sigma_px, z.shape)
    return (FB / np.maximum(disp, 0.5)).astype(np.float32)


def sources(seed=0):
    rng = np.random.default_rng(seed)
    z = scene()
    ffs = stereo_noise(rng, z, 0.2)
    las = stereo_noise(rng, z, 0.3)
    out = rng.random(z.shape) < 0.08
    las = np.where(out, rng.uniform(0.5, 3.9, z.shape), las).astype(np.float32)                    # LAS2 gives 8% wild values
    rtab = (z * 1.04 + rng.normal(0, 0.03, z.shape)).astype(np.float32)                              # RTAB-Map: 4% too far
    rtab[rng.random(z.shape) < 0.55] = np.nan                                                        # and sparse
    for s in (ffs, las, rtab):                                                                       # nobody sees the stripe x 440..540 (a dark patch)
        s[:, 440:540] = np.nan
    ffs[300:340, 20:120] = np.nan                                                                    # FFS alone loses a corner: the others still have it
    return z, ffs, las, rtab, out


def mono_of(z, shift=0):
    """What a monocular model says: the right structure, the wrong scale (0.6 z + 0.25 here), a slow wave, a little noise."""
    wave = 0.08 * np.sin(np.linspace(0, 6, W))[None, :]
    m = (0.6 * z + 0.25 + wave).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 1.0)
    return m


def run(seed=0, with_mono=True, **kw):
    z, ffs, las, rtab, out = sources(seed)
    ml = mono_of(z)
    disp = FB / z
    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    ys = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    mr = cv2.remap(ml, xs + disp.astype(np.float32), ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)     # the right view of the same scene
    f = fuse([StereoIn(ffs, 0.2, "ffs"), StereoIn(las, 0.3, "las2"), StereoIn(rtab, 0.5, "rtabmap")], ml if with_mono else None, mr if with_mono else None, CAL, FusionParams(**kw))
    return z, ffs, las, rtab, out, f


def rel_err(a, z):
    ok = np.isfinite(a)
    return np.abs(a[ok] - z[ok]) / z[ok]


def test_fusion_beats_every_single_source_and_does_not_average_in_the_outliers_or_the_bias():
    z, ffs, las, rtab, out, f = run()
    meas = (f.cls == 1) | (f.cls == 2)
    fused_err, las_err, rtab_err = rel_err(np.where(meas, f.z, np.nan), z), rel_err(las, z), rel_err(rtab, z)
    assert np.median(fused_err) < np.median(rtab_err) / 2 and np.mean(fused_err) < np.mean(las_err) / 3
    wild = out & meas
    assert wild.sum() > 1000 and np.mean(rel_err(np.where(wild, f.z, np.nan), z) > 0.10) < 0.05           # where LAS2 was wild the fused depth is not
    assert abs(np.nanmedian(f.z[meas] / z[meas]) - 1.0) < 0.01                                          # RTAB-Map's 4% bias did not pull the result


def test_monocular_depth_only_fills_holes_never_replaces_a_measurement_and_is_marked_and_weak():
    z, ffs, las, rtab, out, f = run()
    est = f.cls == 3
    assert est.sum() > 500 and np.all(f.conf[est] <= 0.3 + 1e-6)
    assert np.mean(rel_err(np.where(est, f.z, np.nan), z) < 0.12) > 0.8                                  # the fill, scaled to the stereo pixels, is close to the truth
    agreed = f.cls == 1
    assert np.median(f.conf[agreed]) > f.conf[est].max()                                              # an estimate never counts as much as a typical measurement
    meas = (f.cls == 1) | (f.cls == 2)
    mono_only_err = np.nanmean(np.abs(mono_of(z) - z) / z)
    assert mono_only_err > 0.3 and np.nanmean(rel_err(np.where(meas, f.z, np.nan), z)) < 0.05           # measured pixels carry the stereo accuracy, not the monocular map's 30% error


def test_nothing_is_invented_far_from_any_measurement():
    z, ffs, las, rtab, out, f = run()
    middle = f.cls[:, 478:502]                                                                          # the middle of the 100 px stripe: more than 40 px from measured pixels
    assert np.mean(middle == 3) < 0.05 and np.mean(middle == 0) > 0.9
    f0 = run(with_mono=False)[5]
    assert (f0.cls == 3).sum() == 0                                                                     # without monocular maps there is no estimate at all


def test_a_thin_object_survives():
    z, ffs, las, rtab, out, f = run()
    pole = np.zeros((H, W), bool)
    pole[80:310, 560:568] = True
    kept = pole & ((f.cls == 1) | (f.cls == 2))
    assert kept.sum() > 0.6 * pole.sum() and np.mean(rel_err(np.where(kept, f.z, np.nan), z) < 0.06) > 0.9


def test_conflicting_stereo_sources_are_settled_by_the_monocular_depth_or_rejected():
    z = scene()
    a, b = z.copy(), z.copy()
    a[:, :200], b[:, :200] = z[:, :200] * 1.25, z[:, :200]                                              # two sources disagree on the left third, no third source
    ml = mono_of(z)
    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    ys = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    mr = cv2.remap(ml, xs + (FB / z).astype(np.float32), ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    f = fuse([StereoIn(a, 0.2, "a"), StereoIn(b, 0.2, "b")], ml, mr, CAL)
    left = (slice(20, 340), slice(30, 190))
    won = f.cls[left] == 2
    assert won.mean() > 0.5 and np.nanmedian(np.abs(f.z[left][won] - z[left][won]) / z[left][won]) < 0.05    # the one that matches the scaled monocular depth won: b, the true one
    f2 = fuse([StereoIn(a, 0.2, "a"), StereoIn(b, 0.2, "b")], None, None, CAL)
    assert f2.rejected[left].mean() > 0.8 and (f2.cls[left] > 0).mean() < 0.1                          # no arbiter: rejected, not averaged


def test_completion_keeps_rtabmap_and_fills_only_holes_that_fit():
    from rtabmap_lab.completion import complete
    h, w = 120, 160
    fb = 30.0
    r = np.full((h, w), 2.0, np.float32)
    g = np.full((h, w), 2.0, np.float32)
    r[40:70, 40:80] = np.nan                                  # a hole RTAB-Map could not match, LAS2 sees a wall that fits the surroundings
    r[:, 120:] = np.nan                                       # a big hole far from any RTAB-Map depth
    g[10:20, 10:20] = 3.5                                     # LAS2 disagrees where RTAB-Map has depth: RTAB-Map wins
    r[100:110, 20:30] = np.nan
    g[100:110, 20:30] = 3.5                                   # LAS2 value that does not fit the RTAB-Map depth around the hole: not taken
    g[0:2, 0:2] = np.nan
    z, cls, info, conflict = complete(r, g, fb)
    assert np.allclose(z[10:20, 10:20], 3.5) and conflict[10:20, 10:20].all(), "where the sources conflict the less noisy one (LAS2) wins"
    z_old = complete(r, g, fb, conflict_policy="rtab")[0]
    assert np.allclose(z_old[10:20, 10:20], 2.0), "the old policy keeps RTAB-Map"
    assert np.allclose(z[45:65, 45:75], 2.0) and (cls[45:65, 45:75] >= 2).all()
    assert (cls[41:45, 45:75] == 2).all(), "next to RTAB-Map depth the fill is corroborated (class 2); deep inside a big hole it is class 3"
    assert np.isnan(z[100:110, 20:30]).all()
    assert (cls[:, 130:150] == 3).any() and np.isfinite(z[:, 130:150]).any()
    assert (z_old[np.isfinite(r)] == r[np.isfinite(r)]).all(), "policy 'rtab': a pixel RTAB-Map measured is never replaced"
    assert info["valid_after"] > info["valid_rtab"]


def test_completion_mono_layer_fills_what_stereo_left_empty_and_keeps_measured_pixels():
    from rtabmap_lab.completion import complete
    h, w = 180, 240
    fb = 30.0
    yy = np.mgrid[0:h, 0:w][0]
    truth = (1.5 + 0.004 * yy).astype(np.float32)               # a tilted floor-like surface
    r = truth.copy()
    r[60:120, 80:160] = np.nan                                  # a hole both stereo sources missed
    g = r.copy()
    mono = (truth * 2.0 + 0.3).astype(np.float32)               # monocular depth: right shape, wrong scale
    z, cls, info, _ = complete(r, g, fb, mono_left=mono, mono_right=mono)
    assert info["mono_filled"] > 0.05 and (cls[60:120, 80:160] == 3).any()
    f = np.isfinite(z[60:120, 80:160])
    assert np.nanmedian(np.abs(z[60:120, 80:160][f] - truth[60:120, 80:160][f]) / truth[60:120, 80:160][f]) < 0.05
    assert np.allclose(z[np.isfinite(r)], r[np.isfinite(r)], atol=1e-4), "measured pixels stay (here both sources agree, so the average is the same value)"
