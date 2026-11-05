import numpy as np

from mapper.semantics import K, NONE, expand, instances, plane_fit
from mapper.surfel_fuse import SemanticSurfelFusion, SemanticSurfelMap
from rtabmap_lab.tsdf import OPTICAL

F, CX, CY, H, W = 500.0, 160.0, 120.0, 240, 320
FB = F * 0.047


def _wall_depth(z=2.0, noise_mm=0.0, seed=0):
    rng = np.random.default_rng(seed)
    d = np.full((H, W), z * 1000.0) + rng.normal(0, noise_mm, (H, W))
    bgr = rng.integers(0, 255, (H, W, 3)).astype(np.uint8)
    return d.astype(np.uint16), bgr


def _map(**kw):
    return SemanticSurfelMap((F, F, CX, CY), FB, K, stride=2, max_range=5.0, **kw)


def _pose_cam(x=0.0, y=0.0):
    """Optical-frame camera pose for a base pose translated by (x, y) (RTAB-Map base: x forward)."""
    T = np.eye(4)
    T[0, 3], T[1, 3] = x, y
    return T @ OPTICAL


def test_repeated_views_update_surfels_instead_of_stacking_layers():
    m = _map()
    d, bgr = _wall_depth(noise_mm=8)
    m.integrate(d, bgr, _pose_cam())
    n1 = len(m)
    for s in range(1, 6):
        d, bgr = _wall_depth(noise_mm=8, seed=s)
        m.integrate(d, bgr, _pose_cam())
    assert len(m) < 1.25 * n1                                   # six views of one wall: one layer, not six
    P, N, _ = m.points()
    assert len(P) > 0.6 * n1 and P[:, 0].std() < 0.01 + 8e-3   # base x is the wall's depth: surfels average the noise away
    assert abs(np.median(P[:, 0]) - 2.0) < 0.01


def test_bayesian_label_fusion_converges_and_survives_one_bad_picture():
    m = _map()
    d, bgr = _wall_depth()
    hg, wg = H // 2, W // 2

    def probs(c, p):
        out = np.full((hg, wg, K), (1 - p) / (K - 1), np.float32)
        out[..., c] = p
        return out
    for i in range(6):
        m.integrate(d, bgr, _pose_cam(), probs=probs(1 if i != 3 else 2, 0.8))      # class 1 five times, class 2 once
    k, p = m.labels()
    assert (k == 1).mean() > 0.95 and p.mean() > 0.9


def test_new_surfels_start_uniform_and_arrays_stay_aligned():
    m = _map(min_views=2)
    d, bgr = _wall_depth()
    m.integrate(d, bgr, _pose_cam())
    assert np.allclose(m.L, 1.0 / K)
    d2, bgr2 = _wall_depth(z=1.0)                              # nearer wall in front: the old one is blocked, singles expire
    for _ in range(6):
        m.integrate(d2, bgr2, _pose_cam())
    assert len(m.L) == len(m.P) == len(m.W) == len(m.N) == len(m.V)


def test_rebuild_after_a_loop_closure_merges_overlapping_nodes():
    f = SemanticSurfelFusion((F, F, CX, CY), FB, K, threaded=False, stride=2, max_range=5.0)
    d, bgr = _wall_depth()
    f.add_node(0, d, bgr)
    f.add_node(1, d, bgr)
    base = lambda y: np.array([[1, 0, 0, 0], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    f.set_poses({0: base(0.0), 1: base(0.30)})                  # node 1 wrongly 30 cm to the side: two staggered copies
    m = f.flush()
    n_apart = int(m.stable().sum())
    f.set_poses({0: base(0.0), 1: base(0.0)})                   # the optimiser puts it back
    m = f.flush()
    assert f.stats["rebuilds"] == 1
    assert int(m.stable().sum()) > 1.5 * n_apart and len(m) < 1.3 * (H // 2) * (W // 2)    # one fused layer, not two


def test_expand_and_plane_fit():
    cls = np.zeros((4, 4), np.uint8)
    cls[0, 0], cls[1, 1] = 1, NONE
    conf = np.full((4, 4), 0.8, np.float16)
    p = expand(cls, conf)
    assert np.allclose(p.sum(-1), 1, atol=1e-5) and p[0, 0, 1] > 0.79 and p[0, 1, 0] == 0.5 and np.allclose(p[1, 1], 1.0 / K)
    rng = np.random.default_rng(0)
    Pm = np.c_[rng.uniform(0, 0.6, 3000), rng.uniform(0, 0.35, 3000), rng.normal(0, 0.002, 3000)]      # a 60 x 35 cm monitor face, 2 mm noise
    n, rms, ext = plane_fit(Pm)
    assert abs(abs(n[2]) - 1) < 0.01 and rms < 0.003 and abs(max(ext) - 0.59) < 0.03


def test_instances_groups_a_class_into_separate_objects():
    rng = np.random.default_rng(0)
    A = rng.normal([0, 0, 0], 0.03, (400, 3))
    B = rng.normal([2, 0, 0], 0.03, (400, 3))
    P = np.concatenate([A, B]).astype(np.float32)
    labels = np.full(len(P), 1)
    out = instances(P, labels, np.full(len(P), 0.9))
    assert len(out) == 2 and {o["label"] for o in out} == {"screen"}
