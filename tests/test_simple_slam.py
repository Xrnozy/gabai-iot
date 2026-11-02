import numpy as np

from gabai.tools.simple_slam import VoxelMap, depth_points


def test_voxel_map_merges_and_stays_bounded():
    m = VoxelMap(voxel=0.1, max_voxels=50, min_hits=2)
    rng = np.random.default_rng(0)
    plane = np.column_stack([rng.uniform(0, 1, 2000), rng.uniform(0, 1, 2000), np.full(2000, 2.0)])
    col = np.full((2000, 3), 200, np.uint8)
    m.add(plane, col, np.zeros(3))
    assert len(m) <= 100
    m2 = VoxelMap(voxel=0.1, max_voxels=50, min_hits=2)
    for k in range(40):                                   # a camera moving along x, new points far from the old ones
        pts = np.column_stack([np.full(300, k * 1.0) + rng.uniform(0, .5, 300), rng.uniform(0, 1, 300), np.full(300, 2.0)])
        m2.add(pts, np.full((300, 3), 90, np.uint8), np.array([k * 1.0, 0, 0]))
        assert len(m2) <= 50
    p, c = m2.points()
    assert p[:, 0].min() > 30, "the voxels near the camera are kept, the old far ones dropped"
    assert (c == 90).all()


def test_depth_points_geometry():
    cal = {"f": 500.0, "B": 0.05, "cx": 320.0, "cy": 180.0}
    disp = np.full((360, 640), 12.5, np.float32)          # z = 500*0.05/12.5 = 2 m
    ok = np.ones((360, 640), bool)
    pts, cols = depth_points(np.zeros((360, 640, 3), np.uint8), disp, ok, cal, 40, 0.3, 4.0)
    assert np.allclose(pts[:, 2], 2.0) and len(pts) == len(cols)
    far, _ = depth_points(np.zeros((360, 640, 3), np.uint8), disp, ok, cal, 40, 0.3, 1.5)
    assert len(far) == 0
