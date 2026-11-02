import numpy as np

from gabai.tools.wall_flatness import flatness
from gabai.tsdf import TsdfMap, depth_limit

CAL = dict(f=300.0, B=0.1, cx=160.0, cy=120.0)
W, H = 320, 240


def wall_frame(rng, z=2.0, noise_px=0.3):
    """Disparity image of a wall facing the camera, with stereo-like noise."""
    disp = np.full((H, W), CAL["f"] * CAL["B"] / z, np.float32) + rng.normal(0, noise_px, (H, W)).astype(np.float32)
    return np.full((H, W, 3), 120, np.uint8), disp, np.ones((H, W), bool)


def test_depth_limit_follows_the_voxel_map_formula():
    f, B = 618.0, 0.0417
    assert abs(depth_limit(4.0, 0.05, f, B) - 2.07) < 0.02 and depth_limit(3.0, 0.0, f, B) == 3.0 and depth_limit(2.0, 0.5, f, B) == 2.0


def test_many_noisy_looks_fuse_into_a_thinner_wall_than_one_look():
    rng = np.random.default_rng(0)
    one, many = TsdfMap(CAL, (W, H), voxel=0.02, trunc=0.08), TsdfMap(CAL, (W, H), voxel=0.02, trunc=0.08)
    R, t = np.eye(3), np.zeros(3)
    img, disp, ok = wall_frame(rng)
    one.add(img, disp, ok, R, t)
    for _ in range(30):
        img, disp, ok = wall_frame(rng)
        many.add(img, disp, ok, R, t)
    spread = []
    for m in (one, many):
        v, _, _ = m.mesh()
        z = v[(abs(v[:, 0]) < 0.4) & (abs(v[:, 1]) < 0.3), 2]
        spread.append(z.std() * 100)
    assert spread[1] < spread[0] * 0.5 and abs(z.mean() - 2.0) < 0.03          # the wall is where it should be, and thinner after 30 looks


def test_flatness_tells_one_layer_from_two():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, (4000, 2))
    one = np.c_[xy, rng.normal(0, 0.01, 4000)]
    two = np.c_[xy, rng.normal(0, 0.01, 4000) + np.where(rng.random(4000) < 0.5, 0.0, 0.07)]
    a, b = flatness(one), flatness(two)
    assert a["thickness_cm"] < 1.5 and not a["two_layers"]
    assert b["thickness_cm"] > 2.5 and b["two_layers"]
