"""Camera-POV mesh: the current fused depth (stereo, plus monocular where stereo has holes) turned into a textured surface in front of the camera.

One vertex per `step` pixels, two triangles per cell, texture coordinates into the left picture, so the picture keeps its full detail and the geometry stays light.
A cell becomes triangles only where the depth can be trusted: all four corners have depth, no corner is a depth jump (an edge between a near and a far surface would
be a stretched sheet), and the surface is not seen almost edge-on (a long thin sliver). A hole in the depth stays a hole; nothing is bridged over it.
It only covers what the camera sees now; the world map (SlamCloud) is a separate layer and is not touched.
"""
import warnings

import cv2
import numpy as np


def surface_mask(disp, cal, max_m=6.0, ring=1, jump_rel=0.06, jump_abs_m=0.04, max_slope=16.0, min_blob_px=50, half=True, hole_border=False):
    """Pixels that lie on a real, connected surface: valid depth, not at a depth edge, not edge-on, not a small isolated island.
    Both the mesh and the point cloud take only these pixels, so they show the same geometry. A depth edge (near object against a far wall) is where stereo
    produces 'flying' pixels between the two surfaces, and (hole_border=True) so is a pixel next to a hole; both are dropped.
    Computed at half size (4 times cheaper, ring and blob sizes are in those pixels) and brought back by nearest neighbour. Returns a bool mask the size of disp."""
    h, w = disp.shape
    f = cal["f"]
    if half:
        disp, f = cv2.resize(disp, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST), f / 2.0
    valid = np.isfinite(disp) & (disp > 0.5)
    z = np.where(valid, cal["f"] * cal["B"] / np.where(valid, disp, 1.0), 0.0).astype(np.float32)
    valid &= z < max_m
    z = np.where(valid, z, 0.0).astype(np.float32)
    k = np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)
    hi = cv2.dilate(z, k)
    lo = cv2.erode(np.where(valid, z, 1e3).astype(np.float32), k)
    smooth = valid & (hi - lo <= jump_rel * z + jump_abs_m)           # a depth jump within `ring` pixels removes the pixel
    if hole_border:
        smooth &= cv2.erode(valid.astype(np.uint8), k).astype(bool)     # so does a hole right next to it
    wv = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 1.0)
    zs = cv2.GaussianBlur(z, (0, 0), 1.0) / np.maximum(wv, 1e-3)       # blur that ignores the holes, so a hole border is not read as a steep slope
    gx, gy = cv2.Sobel(zs, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(zs, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    slope = np.hypot(gx, gy) * f / np.maximum(zs, 1e-3)                # depth change per metre sideways: large = seen edge-on
    smooth &= slope <= max_slope
    n, lab, stats, _ = cv2.connectedComponentsWithStats(smooth.astype(np.uint8), connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_blob_px
    m = big[lab]
    return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) if half else m


def smooth_disparity(disp):
    """Edge-preserving smoothing of the valid disparity (bilateral: noise of about a pixel is averaged out, a real step is kept). Holes stay holes."""
    valid = np.isfinite(disp) & (disp > 0.5)
    d = np.where(valid, disp, 0.0).astype(np.float32)
    out = cv2.bilateralFilter(d, 9, 1.0, 4.0)
    return np.where(valid, out, np.nan).astype(np.float32)


def build(disp, cal, step=4, max_m=6.0, jump_rel=0.06, jump_abs_m=0.04, min_cos=0.12):
    """disp: HxW disparity in pixels (NaN or <= 0.5 where unknown; pass it through surface_mask first), cal: dict with f, B, cx, cy.
    Returns (verts_cam, triangles, uv) or None. verts_cam are OpenCV camera coordinates (x right, y down, z forward) in metres; uv is in 0..1 over the picture (u right, v down)."""
    h, w = disp.shape
    disp = smooth_disparity(disp)
    ys, xs = np.arange(0, h, step), np.arange(0, w, step)
    d = disp[np.ix_(ys, xs)].astype(np.float64)
    ok = np.isfinite(d) & (d > 0.5)
    z = np.where(ok, cal["f"] * cal["B"] / np.where(ok, d, 1.0), np.nan)
    ok &= z < max_m
    gx, gy = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64))
    X = (gx - cal["cx"]) * z / cal["f"]
    Y = (gy - cal["cy"]) * z / cal["f"]
    P = np.stack([X, Y, z], -1)
    rows, cols = ok.shape
    a, b, c, e = (slice(0, rows - 1), slice(0, cols - 1)), (slice(0, rows - 1), slice(1, cols)), (slice(1, rows), slice(0, cols - 1)), (slice(1, rows), slice(1, cols))

    def tri(i, j, k):
        """Keep mask for triangle (i, j, k): all corners valid, no depth jump, not edge-on."""
        keep = ok[i] & ok[j] & ok[k]
        zs = np.stack([z[i], z[j], z[k]])
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)           # all-NaN cells are the holes, dropped by `keep`
            spread = np.nanmax(zs, 0) - np.nanmin(zs, 0)
            keep &= spread <= jump_rel * np.nanmin(zs, 0) + jump_abs_m
            n = np.cross(P[j] - P[i], P[k] - P[i])
            centre = (P[i] + P[j] + P[k]) / 3.0
            cos = np.abs((n * centre).sum(-1)) / (np.linalg.norm(n, axis=-1) * np.linalg.norm(centre, axis=-1) + 1e-12)
            keep &= cos >= min_cos
        return keep

    idx = np.arange(rows * cols).reshape(rows, cols)
    tris = []
    for (i, j, k) in ((a, c, b), (b, c, e)):
        keep = tri(i, j, k)
        tris.append(np.stack([idx[i][keep], idx[j][keep], idx[k][keep]], -1))
    tris = np.concatenate(tris)
    if not len(tris):
        return None
    used = np.unique(tris)                                          # drop the vertices no triangle uses, and renumber
    remap = np.full(rows * cols, -1, np.int64)
    remap[used] = np.arange(len(used))
    uv = np.stack([gx / (w - 1), gy / (h - 1)], -1).reshape(-1, 2)
    return P.reshape(-1, 3)[used].astype(np.float32), remap[tris].astype(np.uint32), uv[used].astype(np.float32)


def fused_for_view(disp, fill):
    """Stereo disparity with the monocular fill added in the holes (the pixels `fill` supplies; trusted stereo pixels are never changed). fill = (aligned, mask, info) or None."""
    if fill is None or not fill[1].any():
        return disp
    return np.where(fill[1], fill[0], disp).astype(np.float32)
