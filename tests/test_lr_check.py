import cv2
import numpy as np

from gabai.spike.las2_runner import lr_consistency


def test_consistent_disparities_pass_and_occlusion_fails():
    h, w = 40, 100
    dl = np.full((h, w), 10.0, np.float32)
    dr = np.full((h, w), 10.0, np.float32)
    ok = lr_consistency(dl, dr)
    assert ok[:, 10:].all()           # right image agrees everywhere it can be compared
    assert not ok[:, :10].any()       # pixels whose match falls off the right image's left edge are untrusted


def test_disagreeing_right_estimate_is_rejected():
    h, w = 40, 100
    dl = np.full((h, w), 10.0, np.float32)
    dr = np.full((h, w), 10.0, np.float32)
    dr[:, 40:60] = 25.0               # right camera sees something else there (occluder / mismatch)
    ok = lr_consistency(dl, dr)
    assert not ok[:, 50:70].any() or ok[:, 50:70].mean() < 0.5
    assert ok[:, 80:].all()


def test_tolerance_scales_with_disparity():
    dl = np.full((10, 200), 100.0, np.float32)
    dr = np.full((10, 200), 103.0, np.float32)   # 3% off at 100 px is within 5%
    assert lr_consistency(dl, dr)[:, 120:].all()
    dr[:] = 110.0                                 # 10% off is not
    assert not lr_consistency(dl, dr)[:, 120:].any()


def test_batched_left_right_pass_matches_the_two_separate_passes():
    import glob
    import cv2
    import numpy as np
    from gabai.spike.las2_runner import Las2Depth, lr_consistency
    m = Las2Depth("s")
    fn = sorted(glob.glob("data/calib_pairs_swapped/*_L.png"))[3]
    l = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
    r = cv2.cvtColor(cv2.imread(fn[:-6] + "_R.png"), cv2.COLOR_BGR2RGB)
    l, r = cv2.resize(l, (640, 480)), cv2.resize(r, (640, 480))
    d1, ok1, _ = m.disparity_lr(l, r)
    d2, ok2, _ = m.disparity_lr_batched(l, r)
    assert np.median(np.abs(d1 - d2)) < 0.1                 # same answer (fp16 batching may differ slightly)
    assert (ok1 == ok2).mean() > 0.97


def test_near_pass_replaces_saturated_full_resolution_disparity():
    import numpy as np
    from gabai.spike.las2_runner import merge_near
    h, w = 80, 120
    disp = np.full((h, w), 60.0, np.float32)
    trusted = np.ones((h, w), bool)
    disp[:, 40:80] = 150.0                                   # the full-size model's wrong answer at the 192 px limit
    trusted[:, 40:80] = False
    dh = np.full((h // 2, w // 2), 30.0, np.float32)         # half-resolution view: background 30 px = 60 full-size
    dh[:, 20:40] = 110.0                                     # the near object: 220 px full-size
    th = np.ones(dh.shape, bool)
    d, t, near = merge_near(disp, trusted, dh, th)
    assert np.allclose(d[:, 45:75], 220.0, atol=1.0) and t[:, 45:75].all()
    assert np.allclose(d[:, :35], 60.0) and not near[:, :35].any()    # far parts keep their full-resolution value
