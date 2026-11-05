"""Plane-based room reconstruction: walls, floor, ceiling and other large flat surfaces become fitted planes with measured boundaries; everything else stays a TSDF (the hybrid).

Why planes and not more smoothing: one stereo pixel has 0.5 px of disparity noise (7 cm at 2 m), so any per-voxel or per-surfel average of a few views stays wavy. A plane is fitted to thousands of
pixels from many views at once, so its position is known to a few millimetres even when no single pixel is, and a wall cannot come out wavy because it is one equation. Nothing is moved onto a plane
that the measurements do not support: a pixel joins a plane only when it lies within its own noise of it, and the plane only exists where pixels were measured (its support grid).

  1. PER PICTURE, IN DISPARITY SPACE. A flat surface is exactly a plane in (u, v, disparity), and there the noise is the same for every pixel (about 0.5 px), so an ordinary least-squares fit is
     unbiased (in metres the far pixels are much noisier and would pull it). Sequential RANSAC, connected-component check, least-squares refit.
  2. GLOBAL PLANES. Each picture's planes go to world coordinates with the RTAB-Map pose and join a global plane when normal (< 8 deg) and offset (< 5 cm) agree, else start a new one. A global plane is
     the inverse-variance weighted fit of all its member points (sigma_z = z^2 * 0.5 px / fB).
  3. OPTIONAL MANHATTAN SNAP: planes within `snap_deg` of parallel / perpendicular to the room's main axes are rotated (about their own centre) onto them. Only small corrections; a real slanted
     surface is further away than that and is left alone.
  4. SECOND PASS, EVERY PIXEL. A pixel whose world point is within 2 sigma_z (clipped to 2..6 cm) of a global plane and inside that plane's support (dilated 6 cm) belongs to it, even in pictures where the
     plane was not detected on its own. The plane's mesh is its 2 cm support cells seen by at least `min_views` pictures, at their mean colour.
  5. RESIDUAL: the pixels that belong to no plane are fused by the GPU TSDF (objects, clutter); its triangles that lie on a plane's support are dropped, so a wall is never there twice.

reconstruct(nodes, cal, size) -> Result (planes, plane mesh, boundaries, residual mesh, timings). nodes = {id: (depth_mm, bgr, base pose, ...)}, as surface_lab and the calibration lab use.
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import open3d as o3d

from rtabmap_lab.tsdf import OPTICAL

SIGMA_PX = 0.5                      # measured disparity noise of this rig (docs/rtabmap-noise.md)


@dataclass
class Plane:
    n: np.ndarray                   # unit normal, world
    d: float                        # n . X = d
    S0: float = 0.0                 # weighted moments of the member points: the fit is the smallest eigenvector of their covariance
    S1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    S2: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    frames: set = field(default_factory=set)
    pts: list = field(default_factory=list)       # subsampled member points (pass 1), to place the support grid

    def add(self, P, w):
        self.S0 += w.sum()
        self.S1 += w @ P
        self.S2 += (P * w[:, None]).T @ P
        c = self.S1 / self.S0
        C = self.S2 / self.S0 - np.outer(c, c)
        n = np.linalg.eigh(C)[1][:, 0]
        self.n = n if n @ self.n >= 0 else -n
        self.d = float(self.n @ c)

    @property
    def centre(self):
        return self.S1 / max(self.S0, 1e-12)


@dataclass
class Result:
    planes: list
    mesh: o3d.geometry.TriangleMesh             # all plane surfaces
    boundaries: list                            # [(K,3) closed polylines], one or more per plane
    residual: o3d.geometry.TriangleMesh          # TSDF of the pixels on no plane (may be empty)
    labels: dict                                 # node id -> (h, w) int8 plane index per pixel, -1 none
    timings: dict
    snapped: list                                # [(plane index, degrees moved)]

    def combined(self):
        return self.mesh + self.residual


# ---- 1. planes of one picture ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def frame_planes(depth_mm, f, cx, cy, fB, stride=4, thr_px=1.0, min_cells=300, max_planes=8, iters=200, rng=None):
    """Planes of one depth image, fitted in disparity space. Returns [(n, d, mask on the stride grid, cells, (a, b, c) with disparity = a u + b v + c)]; n, d in CAMERA coordinates (optical: x right, y down, z forward)."""
    rng = rng or np.random.default_rng(0)
    z = depth_mm[::stride, ::stride].astype(np.float64) / 1000.0
    gh, gw = z.shape
    v, u = np.mgrid[0:gh, 0:gw] * stride
    valid = z > 0
    disp = np.where(valid, fB / np.maximum(z, 1e-6), 0.0)
    free = valid.copy()
    out = []
    for _ in range(max_planes):
        idx = np.flatnonzero(free)
        if len(idx) < min_cells:
            break
        U, V, Dd = u.ravel()[idx], v.ravel()[idx], disp.ravel()[idx]
        A = np.stack([U, V, np.ones_like(U)], 1)
        s = rng.integers(0, len(idx), (iters, 3))
        M = A[s]                                                           # (iters, 3, 3)
        ok = np.abs(np.linalg.det(M)) > 1e-6
        if not ok.any():
            break
        coef = np.linalg.solve(M[ok], Dd[s[ok]][..., None])[..., 0]         # (k, 3): disp = a u + b v + c
        cnt = (np.abs(A @ coef.T - Dd[:, None]) < thr_px).sum(0)
        best, piece = coef[np.argmax(cnt)], None
        for _ in range(2):                                                 # inliers -> largest connected piece -> least squares, twice
            inl = np.zeros(gh * gw, bool)
            inl[idx[np.abs(A @ best - Dd) < thr_px]] = True
            n_cc, cc = cv2.connectedComponents(inl.reshape(gh, gw).astype(np.uint8), connectivity=8)
            piece = cc == (1 + np.argmax(np.bincount(cc.ravel())[1:])) if n_cc > 1 else None
            if piece is None or piece.ravel()[idx].sum() < min_cells:
                piece = None
                break
            sel = piece.ravel()[idx]
            best = np.linalg.lstsq(A[sel], Dd[sel], rcond=None)[0]
        if piece is None:
            break
        free &= ~piece
        out.append([piece, best])
    # the first plane found takes every pixel within thr_px, including the neighbouring surface's pixels along their shared corner, which tilts it: give every pixel to the plane it fits
    # BEST (among the planes whose piece it touches), then refit each plane on its own pixels only. Twice.
    U, V, Dd = u.ravel(), v.ravel(), disp.ravel()
    A = np.stack([U, V, np.ones_like(U)], 1)
    for _ in range(2):
        if len(out) < 2:
            break
        R = np.full((len(out), gh * gw), np.inf)
        for k, (piece, coef) in enumerate(out):
            near = cv2.dilate(piece.astype(np.uint8), np.ones((3, 3), np.uint8)).ravel() > 0
            r = np.abs(A @ coef - Dd)
            R[k] = np.where(near & valid.ravel() & (r < thr_px), r, np.inf)
        owner = np.argmin(R, 0)
        owned = np.isfinite(R.min(0))
        for k in range(len(out)):
            sel = owned & (owner == k)
            if sel.sum() >= 3:
                out[k] = [sel.reshape(gh, gw), np.linalg.lstsq(A[sel], Dd[sel], rcond=None)[0]]
    res = []
    for piece, (a, b, c) in out:
        if piece.sum() < min_cells:
            continue
        nrm = np.array([a * f, b * f, a * cx + b * cy + c])                 # disp = a u + b v + c with u = f x / z + cx:  fB = a f x + b f y + (a cx + b cy + c) z  ->  n . X = fB
        L = np.linalg.norm(nrm)
        res.append((nrm / L, fB / L, piece, int(piece.sum()), np.array([a, b, c])))
    return res


def denoise(depth_mm, f, cx, cy, fB, thr_px=1.0, **kw):
    """The same depth with every pixel of a detected plane replaced by the plane's own disparity (fitted to thousands of pixels of THIS picture, no pose involved).
    A pixel is replaced only when it is within thr_px of the plane and next to the plane's piece; everything else (objects, edges, pixels off the plane) is returned exactly as measured.
    Returns (depth_mm, mask of replaced pixels)."""
    h, w = depth_mm.shape
    z = depth_mm.astype(np.float64) / 1000.0
    v, u = np.mgrid[0:h, 0:w]
    disp = np.where(z > 0, fB / np.maximum(z, 1e-6), 0.0)
    out, done = depth_mm.copy(), np.zeros((h, w), bool)
    stride = kw.get("stride", 4)
    for n, d, piece, cells, (a, b, c) in frame_planes(depth_mm, f, cx, cy, fB, thr_px=thr_px, **kw):
        near = cv2.dilate(piece.astype(np.uint8), np.ones((3, 3), np.uint8))
        near = cv2.resize(near, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        pd = a * u + b * v + c
        m = near & ~done & (z > 0) & (np.abs(disp - pd) < thr_px) & (pd > 0.5)
        out[m] = np.round(fB / pd[m] * 1000.0).astype(out.dtype)
        done |= m
    return out, done


def _points(depth_mm, f, cx, cy, stride=1):
    z = depth_mm[::stride, ::stride].astype(np.float64) / 1000.0
    v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]] * stride
    return np.stack([(u - cx) * z / f, (v - cy) * z / f, z], -1), z > 0


def _basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


# ---- the whole reconstruction ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def reconstruct(nodes, cal, size, cell=0.02, max_angle=8.0, max_offset=0.05, min_area=0.10, min_views=2, snap_deg=0.0, residual=True, max_range=4.0, plane_range=3.0, band=(0.02, 0.06), thr_px=1.0,
                voxel=0.02, trunc=0.08):
    f, cx, cy, fB = cal["f"], cal["cx"], cal["cy"], cal["f"] * cal["B"]
    T = {i: nd[2] @ OPTICAL for i, nd in nodes.items()}
    tm = {}
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    planes = []
    cos_max = np.cos(np.radians(max_angle))
    # pass 1: per-picture planes -> global planes
    for i, (dmm, *_) in nodes.items():
        dmm = np.where(dmm.astype(np.float64) / 1000.0 <= plane_range, dmm, 0)       # past ~3 m one pixel is +-15 cm: too loose to define a wall
        P, _ = _points(dmm, f, cx, cy, 4)
        R, t = T[i][:3, :3], T[i][:3, 3]
        for n, d, piece, cells, _ in frame_planes(dmm, f, cx, cy, fB, thr_px=thr_px, rng=rng):
            Xc = P[piece]
            Xw = Xc @ R.T + t
            nw = R @ n
            dw = d + nw @ t
            w = 1.0 / (Xc[:, 2] ** 2 * SIGMA_PX / fB) ** 2
            best, bscore = None, None
            for k, pl in enumerate(planes):
                c = pl.n @ nw
                if abs(c) < cos_max:
                    continue
                off = abs(np.median(Xw @ pl.n) - pl.d)
                if off < max_offset and (bscore is None or off < bscore):
                    best, bscore = k, off
            if best is None:
                planes.append(Plane(nw, dw))
                best = len(planes) - 1
            pl = planes[best]
            pl.add(Xw, w)
            pl.frames.add(i)
            pl.pts.append(Xw)
    tm["detect_s"] = time.perf_counter() - t0
    # merge global planes that converged onto each other, drop small ones
    planes = _merge(planes, cos_max, max_offset)
    planes = [p for p in planes if len(p.frames) >= min_views and _area(p, cell) >= min_area]
    snapped = _manhattan(planes, snap_deg) if snap_deg > 0 else []
    # support grids from pass-1 points, dilated 10 cm: where a plane may claim pixels in pass 2
    grids = [_grid(p, cell) for p in planes]
    t1 = time.perf_counter()
    # pass 2: every pixel of every picture (every 2nd, labels upsampled) against the global planes
    labels, resid = {}, {}
    for i, (dmm_full, bgr_full, *_) in nodes.items():
        bgr = bgr_full[::2, ::2]
        P, ok = _points(dmm_full, f, cx, cy, 2)
        ok &= P[..., 2] <= plane_range
        R, t = T[i][:3, :3], T[i][:3, 3]
        Xw = P @ R.T + t
        sig = P[..., 2] ** 2 * SIGMA_PX / fB
        lab = np.full(ok.shape, -1, np.int8)
        bestd = np.full(ok.shape, np.inf)
        for k, (p, g) in enumerate(zip(planes, grids)):
            dist = np.abs(Xw @ p.n - p.d)
            cand = ok & (dist < np.clip(2.0 * sig, *band)) & (dist < bestd)
            if not cand.any():
                continue
            inside = g.contains(Xw[cand], dilate=True)
            sel = np.zeros_like(cand)
            sel[cand] = inside
            lab[sel] = k
            bestd[sel] = dist[sel]
        for k, g in enumerate(grids):
            m = lab == k
            if m.any():
                g.hit(Xw[m], bgr[m], i)
        lab = np.repeat(np.repeat(lab, 2, 0), 2, 1)[: dmm_full.shape[0], : dmm_full.shape[1]]
        labels[i] = lab
        resid[i] = np.where(lab < 0, dmm_full, 0).astype(dmm_full.dtype)
    tm["assign_s"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    mesh, bounds = o3d.geometry.TriangleMesh(), []
    for p, g in zip(planes, grids):
        m, b = g.mesh(min_views)
        mesh += m
        bounds += b
    tm["plane_mesh_s"] = time.perf_counter() - t2
    res_mesh = o3d.geometry.TriangleMesh()
    if residual:
        t3 = time.perf_counter()
        from rtabmap_lab.gpu_tsdf import GpuTsdfMap
        tsdf = GpuTsdfMap((f, f, cx, cy), size, voxel=voxel, trunc=trunc, max_range=max_range, fb=fB, threaded=False)
        for i, nd in nodes.items():
            tsdf.add_node(i, resid[i], nd[1])
            tsdf.set_poses({i: nd[2]})
        res_mesh = tsdf.flush() or o3d.geometry.TriangleMesh()
        tsdf.close()
        res_mesh = _drop_on_planes(res_mesh, planes, grids)
        tm["residual_s"] = time.perf_counter() - t3
    tm["total_s"] = time.perf_counter() - t0
    return Result(planes, mesh, bounds, res_mesh, labels, tm, snapped)


def _merge(planes, cos_max, max_offset):
    out = []
    for p in sorted(planes, key=lambda q: -q.S0):
        for q in out:
            if abs(q.n @ p.n) >= cos_max and abs(q.n @ p.centre - q.d) < max_offset:
                q.S0 += p.S0; q.S1 += p.S1; q.S2 += p.S2
                q.frames |= p.frames; q.pts += p.pts
                q.add(np.zeros((0, 3)), np.zeros(0))
                break
        else:
            out.append(p)
    return out


def _area(p, cell):
    e1, e2 = _basis(p.n)
    X = np.concatenate(p.pts)
    uv = np.floor(np.stack([X @ e1, X @ e2], 1) / cell).astype(np.int64)
    return len(np.unique(uv, axis=0)) * cell * cell                # pass-1 points are every 4th pixel: a lower bound of the area, only for dropping specks


def _manhattan(planes, snap_deg):
    """Rotate planes within snap_deg of the room's main axes onto them. Axes: the normal of the plane with the most weight, then the heaviest plane roughly perpendicular to it."""
    if not planes:
        return []
    order = sorted(range(len(planes)), key=lambda k: -planes[k].S0)
    a1 = planes[order[0]].n
    a2 = next((planes[k].n for k in order[1:] if abs(planes[k].n @ a1) < np.sin(np.radians(15))), None)
    if a2 is None:
        return []
    a2 = a2 - (a2 @ a1) * a1
    a2 /= np.linalg.norm(a2)
    axes = [a1, a2, np.cross(a1, a2)]
    moved = []
    for k, p in enumerate(planes):
        best = max(axes, key=lambda a: abs(a @ p.n))
        ang = np.degrees(np.arccos(min(1.0, abs(best @ p.n))))
        if 0 < ang <= snap_deg:
            c = p.centre
            p.n = best if best @ p.n > 0 else -best
            p.d = float(p.n @ c)
            moved.append((k, float(ang)))
    return moved


class _Grid:
    """A plane's support: 2D cells in the plane's own coordinates, with the pictures that saw each cell and their mean colour."""

    def __init__(self, p, cell, seed_pts, dilate_m=0.06):
        self.p, self.cell = p, cell
        self.e1, self.e2 = _basis(p.n)
        uv = self._uv(seed_pts)
        self.o = uv.min(0) - 8
        sz = uv.max(0) - self.o + 9
        self.shape = (int(sz[1]), int(sz[0]))
        seed = np.zeros(self.shape, np.uint8)
        ij = uv - self.o
        seed[ij[:, 1], ij[:, 0]] = 1
        r = max(1, int(round(dilate_m / cell)))
        self.allowed = cv2.dilate(cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        self.views = np.zeros(self.shape, np.int32)
        self.last = np.full(self.shape, -1, np.int64)
        self.col = np.zeros(self.shape + (3,), np.float64)
        self.ncol = np.zeros(self.shape, np.float64)

    def _uv(self, X):
        return np.floor(np.stack([X @ self.e1, X @ self.e2], 1) / self.cell).astype(np.int64)

    def _ij(self, X):
        ij = self._uv(X) - self.o
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[1]) & (ij[:, 1] < self.shape[0])
        return ij, ok

    def contains(self, X, dilate=True):
        ij, ok = self._ij(X)
        out = np.zeros(len(X), bool)
        out[ok] = self.allowed[ij[ok, 1], ij[ok, 0]]
        return out

    def hit(self, X, bgr, frame):
        ij, ok = self._ij(X)
        ij, c = ij[ok], bgr[ok].astype(np.float64)
        r, q = ij[:, 1], ij[:, 0]
        new = self.last[r, q] != frame
        rr, qq = r[new], q[new]
        self.views[rr, qq] += 1                     # duplicate (r, q) in one call add once (fancy indexing): one view per picture, as intended
        self.last[r, q] = frame
        np.add.at(self.col, (r, q), c)
        np.add.at(self.ncol, (r, q), 1)

    def cells(self, min_views):
        m = (self.views >= min_views).astype(np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0     # fills single-cell gaps between seen cells, nothing larger

    def to3d(self, ij):
        uv = (ij + self.o) * self.cell
        return uv[:, :1] * self.e1 + uv[:, 1:] * self.e2 + self.p.d * self.p.n

    def mesh(self, min_views):
        m = self.cells(min_views)
        r, q = np.nonzero(m)
        if not len(r):
            return o3d.geometry.TriangleMesh(), []
        H, W = self.shape
        vid = np.full((H + 1, W + 1), -1, np.int64)
        corners = np.unique(np.concatenate([np.stack([r + dr, q + dq], 1) for dr in (0, 1) for dq in (0, 1)]), axis=0)
        vid[corners[:, 0], corners[:, 1]] = np.arange(len(corners))
        V = self.to3d(corners[:, ::-1].astype(np.float64))
        a, b, c, d = vid[r, q], vid[r, q + 1], vid[r + 1, q + 1], vid[r + 1, q]
        tri = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
        col = np.where(self.ncol[..., None] > 0, self.col / np.maximum(self.ncol, 1)[..., None], 128.0)
        cc = np.zeros((len(corners), 3))
        cn = np.zeros(len(corners))
        for dr in (0, 1):
            for dq in (0, 1):
                np.add.at(cc, vid[r + dr, q + dq], col[r, q])
                np.add.at(cn, vid[r + dr, q + dq], 1)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(tri.astype(np.int32)))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(cc / cn[:, None] / 255.0, 0, 1)[:, ::-1])
        mesh.compute_vertex_normals()
        bounds = []
        for cnt in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if len(cnt) >= 3 and cv2.contourArea(cnt) >= 4:
                ij = cnt[:, 0, :].astype(np.float64) + 0.5               # (x=q, y=r)
                bounds.append(self.to3d(ij))
        return mesh, bounds


def _grid(p, cell):
    return _Grid(p, cell, np.concatenate(p.pts))


def _drop_on_planes(mesh, planes, grids, band=0.04):
    """Remove residual triangles that lie on a plane's support: that surface already exists as the plane."""
    if not len(mesh.triangles):
        return mesh
    V = np.asarray(mesh.vertices)
    on = np.zeros(len(V), bool)
    for p, g in zip(planes, grids):
        near = np.abs(V @ p.n - p.d) < band
        if near.any():
            ins = np.zeros(len(V), bool)
            ins[near] = g.contains(V[near])
            on |= ins
    tri = np.asarray(mesh.triangles)
    mesh.remove_triangles_by_mask(on[tri].all(1))
    mesh.remove_unreferenced_vertices()
    return mesh
