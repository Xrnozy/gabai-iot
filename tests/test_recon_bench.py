import numpy as np

from rtabmap_lab.tools import recon_bench as rb


def test_ground_truth_distance_and_renderer_agree():
    p = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.03], [rb.WALL_X - 0.02, 0.5, 1.0], [1.6, 0.0, 0.6], [1.6, 0.0, 0.65], [1.6, 0.0, 0.3]])
    d = rb.gt_distance(p)
    assert np.allclose(d, [0.0, 0.03, 0.02, 0.0, 0.05, 0.2], atol=1e-6)               # the last point is inside the box, 0.2 m from its nearest face
    T = rb.cam_pose([0.0, 0.0, 1.3], 0.0, 15.0)
    z = rb.scene_depth(T)
    v, u = np.mgrid[0:rb.H, 0:rb.W]
    pw = np.stack([(u - rb.CX) / rb.F * z, (v - rb.CY) / rb.F * z, z], -1) @ T[:3, :3].T + T[:3, 3]
    hit = z > 0
    assert hit.mean() > 0.95 and np.percentile(rb.gt_distance(pw[hit]), 99) < 1e-3, "every rendered pixel lies on a true surface"
