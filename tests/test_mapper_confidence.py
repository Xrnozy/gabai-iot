import torch

from mapper.confidence import confidence

H, W, FB = 120, 160, 29.0


def _scene(d=20.0):
    g = torch.Generator().manual_seed(0)
    gray = torch.rand(H, W, generator=g) * 255                 # textured
    dl = torch.full((H, W), d)
    dr = torch.full((H, W), d)                                  # constant disparity: right view agrees everywhere
    return dl, dr, gray


def test_consistent_textured_plane_is_confident_and_metric():
    dl, dr, gray = _scene(20.0)
    z, c, sz = confidence(dl, dr, gray, FB)
    assert abs(z[60, 80].item() - FB / 20.0) < 1e-5
    assert c[60, 80] > 0.9 and sz[60, 80] > 0


def test_lr_disagreement_lowers_confidence():
    dl, dr, gray = _scene()
    dr[:, :80] += 3.0                                           # right view contradicts the left on its left half
    _, c, _ = confidence(dl, dr, gray, FB)
    assert c[60, 130] > 0.9 and c[60, 20] < 0.05


def test_blank_patch_is_less_confident_than_textured():
    dl, dr, gray = _scene()
    gray[:, :80] = 100.0
    _, c, _ = confidence(dl, dr, gray, FB)
    assert c[60, 130] > 2 * c[60, 20] > 0


def test_flying_pixel_at_depth_jump():
    dl, dr, gray = _scene()
    dl[:, 80:] = 40.0
    dr[:, 80:] = 40.0
    _, c, _ = confidence(dl, dr, gray, FB)
    assert c[60, 80] < 0.5 * c[60, 40]


def test_zero_past_max_range_and_no_disparity():
    dl, dr, gray = _scene(5.0)                                  # z = 5.8 m
    z, c, _ = confidence(dl, dr, gray, FB, max_range=2.5)
    assert z.max() == 0 and c.max() == 0
