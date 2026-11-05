import numpy as np

from mapper.cloud import OPTICAL, PointMap

K = (500.0, 500.0, 160.0, 120.0)
H, W = 240, 320


def _wall(z=2.0, noise=0.0, seed=0):
    """A fronto-parallel textured wall at depth z seen by a camera with intrinsics K."""
    rng = np.random.default_rng(seed)
    depth = np.full((H, W), z, np.float32) + rng.normal(0, noise, (H, W)).astype(np.float32)
    conf = np.ones((H, W), np.float32)
    sz = np.full((H, W), 0.01, np.float32)
    bgr = rng.integers(0, 255, (H, W, 3)).astype(np.uint8)
    return depth, conf, sz, bgr


def _pose(x=0.0, y=0.0):
    T = np.eye(4); T[0, 3], T[1, 3] = x, y
    return T


def test_two_views_fuse_into_one_surface_and_single_view_is_hidden():
    m = PointMap(K, min_views=2)
    d, c, s, b = _wall()
    m.add_node(0, d, c, s, b); m.add_node(1, d, c, s, b)
    m.set_poses({0: _pose(), 1: _pose()})
    P, C, cf, views = m.cloud()
    assert len(P) > 1000 and (views >= 2).all()
    # base frame x is forward: the wall is at world x = 2
    assert abs(np.median(P[:, 0]) - 2.0) < 0.01 and P[:, 0].std() < 0.01
    m1 = PointMap(K, min_views=2)
    m1.add_node(0, d, c, s, b); m1.set_poses({0: _pose()})
    assert len(m1.cloud()[0]) == 0


def test_loop_closure_moves_points_without_duplicating_them():
    m = PointMap(K)
    d, c, s, b = _wall()
    for i in (0, 1):
        m.add_node(i, d, c, s, b)
    m.set_poses({0: _pose(), 1: _pose(0.0, 0.3)})            # node 1 starts 30 cm to the side: two sheets side by side
    n_apart = len(m.cloud()[0])
    assert m.set_poses({0: _pose(), 1: _pose()}) == 1           # the optimiser pulls node 1 back; one node moved
    P, *_ = m.cloud()
    assert len(P) > n_apart                                      # now they overlap, so more voxels are seen by both nodes
    assert m.set_poses({0: _pose(), 1: _pose()}) == 0 and m.cloud()[0] is P    # unchanged poses: nothing rebuilt


def test_low_confidence_and_blank_pixels_are_not_stored():
    m = PointMap(K)
    d, c, s, b = _wall()
    c[:, :160] = 0.1                                             # left half untrusted
    b[:, 200:] = 100                                             # right part blank: no texture
    m.add_node(0, d, c, s, b)
    xyz = m.nodes[0][0]
    u = xyz[:, 0] * K[0] / xyz[:, 2] + K[2]
    assert len(xyz) > 500 and u.min() >= 159 and u.max() <= 201


def test_noisy_points_use_coarser_voxels():
    def count(sigma):
        m = PointMap(K, min_views=2)
        d, c, s, b = _wall(z=2.0)
        s[:] = sigma
        m.add_node(0, d, c, s, b); m.add_node(1, d, c, s, b)
        m.set_poses({0: _pose(), 1: _pose()})
        return len(m.cloud()[0])
    assert count(0.06) * 5 < count(0.005)                      # 3 cm voxels hold far fewer points than 5 mm ones


def test_gpu_fusion_matches_the_cpu_fusion():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    d, c, s, b = _wall(noise=0.004)
    poses = {0: _pose(), 1: _pose(0.0, 0.02), 2: _pose(0.0, -0.03)}
    res = {}
    for dev in ("cpu", "cuda"):
        m = PointMap(K, min_views=2, device=dev)
        for i in poses:
            m.add_node(i, d, c, s, b)
        m.set_poses(poses)
        P, C, cf, views = m.cloud()
        res[dev] = (P, C, cf, views)
    n_c, n_g = len(res["cpu"][0]), len(res["cuda"][0])
    assert abs(n_c - n_g) <= 0.001 * n_c                                           # float32 vs float64 voxel borders: a handful of points land in the neighbouring voxel
    assert np.allclose(res["cpu"][0].mean(0), res["cuda"][0].mean(0), atol=1e-3) and np.allclose(res["cpu"][0].std(0), res["cuda"][0].std(0), atol=2e-3)
    assert abs(res["cpu"][1].mean() - res["cuda"][1].mean()) < 0.5 and abs(res["cpu"][3].mean() - res["cuda"][3].mean()) < 0.01
