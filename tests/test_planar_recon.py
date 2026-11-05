"""Plane-based reconstruction on a synthetic room (exact walls, known size, noisy stereo depth) and the gyro integration the RTAB-Map bridge uses."""
import cv2
import numpy as np
import pytest

from rtabmap_lab import planar_recon as PR
from rtabmap_lab.orientation import gyro_rotvec
from rtabmap_lab.tsdf import OPTICAL

F, CX, CY, B, W, H = 400.0, 160.0, 120.0, 0.05, 320, 240
CAL = {"f": F, "cx": CX, "cy": CY, "B": B}
ROOM = [(np.array([1.0, 0, 0]), 1.2), (np.array([-1.0, 0, 0]), 1.0), (np.array([0, 0, 1.0]), 1.8), (np.array([0, 1.0, 0]), 0.3)]   # n . X = d in the optical world: x = +1.2 / -1.0 walls, far wall z = 1.8, floor 0.3 m below
# within ~2 m: further away two walls meeting in a corner differ by less than 1 px of disparity near the corner and one picture cannot tell them apart (see planar_recon)


def render(T_wc, sigma_px=0.5, seed=0):
    """Depth (mm) of the box room seen from camera T_wc (optical), with stereo disparity noise."""
    np.seterr(divide="ignore")
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - CX) / F, (v - CY) / F, np.ones_like(u, float)], -1) @ T_wc[:3, :3].T
    o = T_wc[:3, 3]
    z = np.full((H, W), np.inf)
    for n, d in ROOM:
        den = rays @ n
        t = np.where(np.abs(den) > 1e-6, (d - o @ n) / den, np.inf)
        z = np.where((t > 0) & (t < z), t, z)
    disp = F * B / z + np.random.default_rng(seed).normal(0, sigma_px, z.shape)
    return np.where(np.isfinite(z) & (disp > 1), F * B / disp * 1000, 0).astype(np.uint16)


def pose(yaw_deg, x=0.0):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array([0, np.radians(yaw_deg), 0]))[0]
    T[0, 3] = x
    return T


def test_frame_planes_recover_a_wall_in_disparity_space():
    d = render(pose(0))
    planes = PR.frame_planes(d, F, CX, CY, F * B)
    best = max(planes, key=lambda p: p[3])
    n, dd = best[0], best[1]
    assert abs(abs(n @ np.array([0, 0, 1.0])) - 1) < 0.01 and dd == pytest.approx(1.8, abs=0.02)


def test_room_reconstruction_keeps_angles_and_sizes():
    nodes = {}
    for k, yaw in enumerate(range(-40, 41, 10)):
        T_wc = pose(yaw, 0.05 * k)
        nodes[k] = (render(T_wc, seed=k), np.full((H, W, 3), 128, np.uint8), T_wc @ np.linalg.inv(OPTICAL))
    r = PR.reconstruct(nodes, CAL, (W, H), residual=False)
    big = sorted(r.planes, key=lambda p: -p.S0)[:4]
    for n0, d0 in ROOM:                                                            # every wall of the room is found where it is, flat
        err = [abs(np.sign(p.n @ n0) * p.d - d0) for p in big if abs(p.n @ n0) > 0.999]
        assert err and min(err) < 0.02, (n0, d0)                                   # 1-14 mm off with 0.5 px of disparity noise
    x = [p for p in big if abs(p.n[0]) > 0.99]
    assert len(x) == 2 and abs(abs(x[0].d) + abs(x[1].d) - 2.2) < 0.02             # room width 2.2 m
    V = np.asarray(r.mesh.vertices)
    on = np.min(np.stack([np.abs(V @ n - d) for n, d in ROOM]), 0)
    assert np.percentile(on, 99) < 0.025                                           # the surfaces lie on the walls (as well as the planes do), not a slab around them


def test_denoise_only_replaces_plane_pixels():
    clean = render(pose(0), sigma_px=0.0)
    d = render(pose(0), sigma_px=0.5)
    d[60:100, 140:180] = 900                                                        # a box in front of the far wall: not on any large plane
    out, m = PR.denoise(d, F, CX, CY, F * B)
    assert np.array_equal(out[60:100, 140:180], d[60:100, 140:180]) and m.mean() > 0.6
    err_before = np.median(np.abs(d[m].astype(float) - clean[m]))
    err_after = np.median(np.abs(out[m].astype(float) - clean[m]))
    assert err_after < 0.2 * err_before                                             # replaced pixels are on the true surface, not just smoother


def test_gyro_rotvec_integrates_and_flips():
    R_ic = np.eye(3)
    t = np.linspace(0, 1, 101)[:-1]
    w = np.tile([0, 0.5, 0], (100, 1))                                             # 0.5 rad/s about the camera y axis for 1 s
    assert gyro_rotvec(t, w, R_ic, False, 1.0) == pytest.approx([0, 0.5, 0], abs=1e-6)
    assert gyro_rotvec(t, w, R_ic, True, 1.0) == pytest.approx([0, -0.5, 0], abs=1e-6)  # pictures turned 180 deg about the optical axis
    assert gyro_rotvec([], [], R_ic, False, 1.0) is None
