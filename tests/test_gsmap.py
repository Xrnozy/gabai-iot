"""Incremental Gaussian map on a synthetic textured scene (floor, box, wall) seen from a moving camera."""
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
sys.path.insert(0, "tests")
from test_navmap import CX, CY, F, H, W, pose, scene_depth                       # noqa: E402

FB = F * 0.047


def picture(T_gc, rng, noise=True):
    z = scene_depth(T_gc)
    v, u = np.mgrid[0:H, 0:W]
    pc = np.stack([(u - CX) / F * z, (v - CY) / F * z, z], -1)
    pw = pc @ T_gc[:3, :3].T + T_gc[:3, 3]
    rgb = np.stack([0.5 + 0.5 * np.sin(9 * pw[..., 0]), 0.5 + 0.5 * np.sin(9 * pw[..., 1] + 1), 0.5 + 0.5 * np.sin(7 * pw[..., 2] + 2)], -1)
    zn = z + (rng.normal(0, 1, z.shape) * z ** 2 * 0.3 / FB if noise else 0)
    return np.where(z > 0, zn, 0), (rgb * 255).astype(np.uint8)[:, :, ::-1]          # depth with noise, BGR


def to_world_pose(T_gc):
    return T_gc                                                                         # the "world" here is the gravity frame; the sink wants a base pose, GaussianMap takes optical poses directly


def feed(gm, T_gc, nid, rng, depth=None):
    d, bgr = picture(T_gc, rng)
    if depth is not None:
        d = depth
    gm.add_node(nid, (d * 1000).astype(np.uint16), bgr)
    gm.poses[nid] = T_gc                                                                # already camera -> world (optical)
    gm.step()


def test_map_grows_where_needed_refines_and_stays_bounded():
    from rtabmap_lab.gsmap import GaussianMap
    rng = np.random.default_rng(0)
    gm = GaussianMap(F, CX, CY, (W, H), FB, max_gaussians=60000, max_new=4000, iters=8, threaded=False)
    counts = []
    for k in range(10):
        feed(gm, pose(0.04 * k, 15), k, rng)
        counts.append(len(gm.P))
    assert counts[0] > 1000 and len(gm.P) <= 60000
    assert counts[-1] < 5 * counts[0], "later pictures add little where the map already explains the depth: %s" % counts
    T = pose(0.2, 15)
    rgb, depth, alpha = gm.render(T)
    true = scene_depth(T)
    near = (true > 0) & (true < 2.5)                                                    # farther than this the stereo noise (> 8 cm) keeps those pixels out of the map by design
    covered = near & (alpha > 0.7)
    assert covered.sum() / near.sum() > 0.9, "the part of the picture with usable depth is covered"
    assert np.median(np.abs(depth[covered] - true[covered]) / true[covered]) < 0.03, "rendered depth matches the scene within 3%"


def test_an_outlier_seen_once_does_not_stay_in_the_map():
    from rtabmap_lab.gsmap import GaussianMap
    rng = np.random.default_rng(1)
    gm = GaussianMap(F, CX, CY, (W, H), FB, max_gaussians=60000, max_new=4000, iters=10, min_views=4, threaded=False)
    for k in range(4):
        feed(gm, pose(0.04 * k, 15), k, rng)
    d, _ = picture(pose(0.16, 15), rng)
    d[100:108, 300:308] = 1.0                                                           # one picture claims a surface in the air
    feed(gm, pose(0.16, 15), 4, rng, depth=d)
    for k in range(5, 14):
        feed(gm, pose(0.04 * k, 15), k, rng)
    pts, _, _ = gm.points(min_opacity=0.35, min_seen=1)
    T = pose(0.16, 15)
    ray = np.array([(304 - CX) / F, (104 - CY) / F, 1.0])
    speck = T[:3, :3] @ (ray * 1.0) + T[:3, 3]
    assert (np.linalg.norm(pts - speck, axis=1) < 0.05).sum() < 5


def test_loop_closure_moves_the_keyframes_gaussians_without_a_rebuild():
    from rtabmap_lab.gsmap import GaussianMap
    rng = np.random.default_rng(2)
    gm = GaussianMap(F, CX, CY, (W, H), FB, max_gaussians=60000, max_new=3000, iters=2, threaded=False)
    feed(gm, pose(0.0, 15), 0, rng)
    feed(gm, pose(0.1, 15), 1, rng)
    sel = gm.owner == 0
    before = gm.P[sel, :3].mean(0).cpu().numpy()
    other = gm.P[gm.owner == 1, :3].mean(0).cpu().numpy()
    T = pose(0.0, 15).copy()
    T[:3, 3] += [0.0, 0.05, 0.0]                                                        # the graph moved node 0 by 5 cm
    gm.poses[0] = T
    gm._reanchor()
    after = gm.P[gm.owner == 0, :3].mean(0).cpu().numpy()
    assert np.allclose(after - before, [0, 0.05, 0], atol=1e-3) and gm.stats["moved_kf"] == 1
    assert np.allclose(gm.P[gm.owner == 1, :3].mean(0).cpu().numpy(), other, atol=1e-6), "other keyframes stay"
