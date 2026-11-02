"""nvblox-style GPU TSDF on a synthetic scene: a wall at 2 m and a sphere, noisy stereo-like depth from 12 camera positions, plus a speck of noise seen by one picture only."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from rtabmap_lab.gpu_tsdf import GpuVolume

W, H, F, CX, CY = 640, 360, 620.0, 320.0, 180.0
FB = F * 0.047


def render(T_wc, rng):
    v, u = np.mgrid[0:H, 0:W]
    ray = np.stack([(u - CX) / F, (v - CY) / F, np.ones((H, W))], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    dw = ray @ R.T
    d = np.full((H, W), np.inf)
    tw = (2.0 - t[2]) / np.where(dw[..., 2] > 1e-6, dw[..., 2], 1)
    d = np.where((dw[..., 2] > 1e-6) & (tw > 0), tw, d)
    oc = t - np.array([0, 0, 1.5])
    b = dw @ oc
    disc = b * b - (oc @ oc - 0.16)
    ts = -b - np.sqrt(np.where(disc > 0, disc, 0))
    d = np.where((disc > 0) & (ts > 0) & (ts < d), ts, d)
    z = d * ray[..., 2]
    z = z + rng.normal(0, 1, z.shape) * z ** 2 * 0.3 / FB
    z[~np.isfinite(z)] = 0
    return z.astype(np.float32)


def build():
    rng = np.random.default_rng(0)
    vol = GpuVolume(0.02, 0.08, 4.0, F, F, CX, CY, (W, H), FB, free_space=True, fade=True)          # carving is opt-in: it is what removes the speck
    for k in range(12):
        a = np.radians(-8 + k * 1.5)
        T = np.eye(4)
        T[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        T[:3, 3] = [-0.3 + 0.05 * k, 0, 0]
        d = render(T, rng)
        if k == 5:
            d[150:156, 300:306] = 1.0                                   # one picture sees a surface in the air
        vol.integrate(d, np.full((H, W, 3), 150, np.uint8), T)
    return vol


def test_wall_and_sphere_are_reconstructed_and_the_speck_is_carved_away():
    mesh = build().extract()
    v = np.asarray(mesh.vertices)
    wall = v[np.abs(v[:, 2] - 2.0) < 0.15]
    assert len(wall) > 5000 and abs(wall[:, 2].mean() - 2.0) < 0.03
    assert (v[:, 2] > 2.08).sum() == 0, "no second skin behind the wall (a cube with an unobserved corner must not make a surface)"
    sph = v[np.linalg.norm(v - [0, 0, 1.5], axis=1) < 0.6]
    assert len(sph) > 1000 and abs(np.linalg.norm(sph - [0, 0, 1.5], axis=1).mean() - 0.4) < 0.02
    assert ((np.abs(v[:, 2] - 1.0) < 0.05) & (np.abs(v[:, 0]) < 0.1)).sum() == 0, "the one-picture speck must not become geometry"
    mesh.compute_triangle_normals()
    n = np.asarray(mesh.triangle_normals)
    on_wall = np.abs(v[np.asarray(mesh.triangles)[:, 0], 2] - 2.0) < 0.05
    assert n[on_wall][:, 2].mean() < -0.2, "normals face the camera"


def test_pool_is_bounded():
    vol = GpuVolume(0.02, 0.08, 4.0, F, F, CX, CY, (W, H), FB, max_blocks=50)
    rng = np.random.default_rng(1)
    vol.integrate(render(np.eye(4), rng), np.zeros((H, W, 3), np.uint8), np.eye(4))
    assert vol.n_blocks <= 50
