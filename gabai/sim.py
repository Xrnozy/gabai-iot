"""A tiny deterministic scene renderer: ground-truth disparity for a floor and axis-aligned boxes seen from a known camera pose.

Used by tests (and the self-check) so that depth, zone, grid and alert logic can be checked against geometry whose true answer is
known, with no camera. It renders what a PERFECT stereo model would output; noise_px adds repeatable Gaussian disparity noise.

World frame: x forward at yaw 0, y left, z up (floor at z = 0). Camera frame: x right, y down, z forward. look_pose() returns
R_wc (camera-to-world rotation) and t_wc (camera position in the world).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    f: float
    B: float
    cx: float
    cy: float
    size: tuple          # (width, height)


@dataclass
class Box:
    lo: tuple            # (x, y, z) of the lower corner, metres
    hi: tuple


def look_pose(position, yaw_deg=0.0, pitch_deg=0.0):
    """Camera at `position` looking along the world direction yaw (about z, 0 = +x), tilted down by pitch_deg."""
    yaw, pit = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.cos(yaw) * np.cos(pit), np.sin(yaw) * np.cos(pit), -np.sin(pit)])
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    up = np.cross(fwd, left)
    right = -left
    down = -up
    return np.column_stack([right, down, fwd]), np.asarray(position, float)


def render_disparity(cam, R_wc, t_wc, boxes=(), floor=True, max_depth=12.0, noise_px=0.0, seed=0):
    w, h = cam.size
    u = (np.arange(w, dtype=np.float64)[None, :] - cam.cx) / cam.f
    v = (np.arange(h, dtype=np.float64)[:, None] - cam.cy) / cam.f
    rays_c = np.stack([np.broadcast_to(u, (h, w)), np.broadcast_to(v, (h, w)), np.ones((h, w))], axis=-1)
    d = rays_c @ np.asarray(R_wc).T                                  # ray directions in the world, with unit camera-z (so s == depth Z)
    o = np.asarray(t_wc, float)
    z = np.full((h, w), np.inf)
    if floor:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -o[2] / d[..., 2]
        z = np.where((d[..., 2] < 0) & (s > 0), s, z)
    for b in boxes:
        lo, hi = np.asarray(b.lo, float), np.asarray(b.hi, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / d, (hi - o) / d
        tmin = np.minimum(t1, t2)
        tmax = np.maximum(t1, t2)
        tmin = np.where(np.isnan(tmin), -np.inf, tmin)                # a ray parallel to a slab inside it never leaves it
        tmax = np.where(np.isnan(tmax), np.inf, tmax)
        inside_slab = np.where(np.abs(d) < 1e-12, (o >= lo) & (o <= hi), True)
        enter, leave = tmin.max(axis=-1), tmax.min(axis=-1)
        hit = (enter <= leave) & (leave > 0) & inside_slab.all(axis=-1)
        s = np.where(enter > 0, enter, leave)
        z = np.where(hit & (s > 0) & (s < z), s, z)
    disp = np.where(np.isfinite(z) & (z < max_depth), cam.f * cam.B / np.maximum(z, 1e-6), np.nan).astype(np.float32)
    if noise_px:
        rng = np.random.default_rng(seed)
        disp = disp + (rng.normal(0, noise_px, disp.shape) * np.isfinite(disp)).astype(np.float32)
    return disp
