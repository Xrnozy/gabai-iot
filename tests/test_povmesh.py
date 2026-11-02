import numpy as np

from gabai import povmesh

CAL = {"f": 500.0, "B": 0.1, "cx": 160.0, "cy": 120.0}
SHAPE = (240, 320)


def plane(z):
    return np.full(SHAPE, CAL["f"] * CAL["B"] / z, np.float32)


def test_a_flat_wall_becomes_one_connected_surface_at_the_right_distance():
    v, t, uv = povmesh.build(plane(2.0), CAL)
    assert np.allclose(v[:, 2], 2.0, atol=1e-4)
    assert len(t) > 1500 and t.max() < len(v)
    assert uv.min() >= 0 and uv.max() <= 1


def test_a_hole_is_not_bridged():
    d = plane(2.0)
    d[60:180, 100:220] = np.nan
    v, t, _ = povmesh.build(d, CAL)
    cx = v[t].mean(1)                                                # triangle centres, camera frame
    px = cx[:, 0] * CAL["f"] / cx[:, 2] + CAL["cx"]
    py = cx[:, 1] * CAL["f"] / cx[:, 2] + CAL["cy"]
    assert not ((px > 110) & (px < 210) & (py > 70) & (py < 170)).any()


def test_a_depth_jump_is_not_a_stretched_sheet():
    d = plane(1.0)
    d[:, 160:] = CAL["f"] * CAL["B"] / 4.0                           # near wall left, far wall right
    v, t, _ = povmesh.build(d, CAL)
    span = (v[t][:, :, 2].max(1) - v[t][:, :, 2].min(1))
    assert span.max() < 0.2


def test_nothing_to_show_gives_none():
    assert povmesh.build(np.full(SHAPE, np.nan, np.float32), CAL) is None


def test_the_fused_view_adds_the_fill_only_in_the_holes():
    d = plane(2.0)
    d[:, :50] = np.nan
    fill = (np.full(SHAPE, 9.0, np.float32), np.isnan(d), {})
    f = povmesh.fused_for_view(d, fill)
    assert (f[:, 50:] == d[:, 50:]).all() and (f[:, :50] == 9.0).all()
    assert povmesh.fused_for_view(d, None) is d


def two_walls():
    d = plane(1.0)
    d[:, 160:] = CAL["f"] * CAL["B"] / 4.0
    return d


def test_the_surface_mask_keeps_a_wall_and_drops_flying_pixels_between_two_surfaces():
    d = two_walls()
    d[100:140, 158:162] = CAL["f"] * CAL["B"] / 2.5                  # stereo's in-between pixels along the edge
    m = povmesh.surface_mask(d, CAL)
    assert m[:, 20:100].all() and m[:, 220:300].all()                 # the walls themselves stay
    assert not m[100:140, 158:162].any()                              # nothing floats between them


def test_the_surface_mask_drops_a_small_island():
    d = plane(2.0)
    d[:, :60] = np.nan
    d[100:110, 200:210] = CAL["f"] * CAL["B"] / 1.5                   # 10x10 px speck
    m = povmesh.surface_mask(d, CAL)
    assert not m[100:110, 200:210].any() and m[:, 100:300].sum() > 0.8 * m[:, 100:300].size
