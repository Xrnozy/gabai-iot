import numpy as np

from gabai.pipeline.run import remove_speckles


def _plane():
    d = np.tile(np.linspace(20.0, 24.0, 160, dtype=np.float32), (120, 1))        # a smooth slanted surface
    return d, np.ones(d.shape, bool)


def test_a_small_island_that_jumps_from_its_surroundings_is_dropped_and_the_surface_stays():
    d, t = _plane()
    d[40:46, 60:66] += 9.0                                                       # 36 px blob 9 px off: a speckle
    t2 = remove_speckles(d, t, 100, 1.0)
    assert not t2[40:46, 60:66].any()
    assert t2.sum() == t.sum() - 36 and t2[0, 0]


def test_a_large_region_and_a_smooth_surface_are_kept_and_zero_switches_it_off():
    d, t = _plane()
    d[:, 80:] += 6.0                                                             # a big step is a real edge, not a speckle
    assert remove_speckles(d, t, 100, 1.0).all()
    d2, t2 = _plane()
    d2[10:14, 10:14] += 9.0
    assert (remove_speckles(d2, t2, 0, 1.0) == t2).all()


def test_untrusted_pixels_stay_untrusted():
    d, t = _plane()
    t[50:60, 50:60] = False
    assert not remove_speckles(d, t, 100, 1.0)[50:60, 50:60].any()


def test_mono_modes_are_validated_and_off_by_default():
    from gabai.pipeline.config import PipelineConfig, validate
    c = PipelineConfig()
    assert c.stereo.mono == "" and not any("stereo.mono" in e for e in validate(c, check_files=False)[0])
    c.stereo.mono, c.stereo.mono_mode = "midas", "mix"
    errs = validate(c, check_files=False)[0]
    assert any("stereo.mono 'midas'" in e for e in errs) and any("mono_mode 'mix'" in e for e in errs)
