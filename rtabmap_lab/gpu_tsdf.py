"""GPU TSDF mapping after nvblox (NVIDIA, github.com/nvidia-isaac/nvblox), written in PyTorch so it runs native on Windows. Same API as tsdf.TsdfMap: a drop-in for it.

What is taken from nvblox (Millane et al., "nvblox: GPU-Accelerated Incremental Signed Distance Field Mapping", 2023):
  - block-sparse voxel storage: 8x8x8 voxel blocks in a pool on the GPU, found by a hash (sorted keys + searchsorted). Only blocks near a surface are allocated.
  - PROJECTIVE integration: every voxel of every block in view is projected into the depth picture; its distance is (picture depth - voxel depth), clamped to +-truncation.
  - a weight per measurement and a weighted running average per voxel, with a maximum weight.
  - FREE-SPACE carving: voxels in front of the surface (inside the frustum, in an allocated block) are pulled toward +truncation. A speck of noise that one picture put in the air is erased by the
    next pictures that see through it. This is what removes floating rubble; an Open3D ScalableTSDFVolume only ever adds near-surface values.
  - voxels behind the surface by more than the truncation are not touched.
  - the mesh contains only voxels whose weight reached a minimum (nvblox: min weight gates mesh and ESDF), so a surface seen once, through one noisy picture, is not drawn. The weight of a picture falls with
    1/z^2, so the minimum is small (0.1: one picture at 1 m, or about 3 at 2 m, 9 at 3 m); 1.0 made the far walls of a real room vanish.

What is different, because the input is STEREO, not an RGB-D sensor (nvblox assumes a depth sensor with small, near-constant noise):
  - stereo depth noise grows with the square of the distance: sigma_z = z^2 * sigma_px / (f * B). The truncation distance is therefore PER PIXEL, max(trunc, k * sigma_z), and the weight falls with distance (1/z^power).
  - pixels at depth edges (flying pixels between a near and a far surface) and pixels seen at a grazing angle (stretched "skirts") are dropped or down-weighted before they reach the volume.
  - after the marching cubes, small disconnected pieces of mesh are removed.
Not taken: the ESDF (nothing here plans a path), dynamic/people layers, occupancy decay. Marching cubes runs on the CPU (scikit-image) over the bounding box of the allocated blocks.
"""
import time

import numpy as np
import open3d as o3d
import torch

from rtabmap_lab.tsdf import OPTICAL, TsdfMap

B = 8                                                         # voxels per block side
OFF = 1 << 20
CLS_LUT = np.array([0.0, 1.0, 0.66, 0.33], np.float32)         # weight by fused-depth class (depthfusion.py): none, measured and agreed, one source, estimated


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GpuVolume:
    def __init__(self, voxel, trunc, max_range, fx, fy, cx, cy, size, fb, sigma_px=0.3, k_trunc=3.0, max_weight=60.0, min_weight=0.1, power=2.0, max_blocks=24000,
                 max_std=0.12, edge_rel=0.06, min_cos=0.0, free_space=True, fade=True, refine=False):
        self.dev = _dev()
        self.v, self.tmin, self.max_range, self.size, self.fb = voxel, trunc, max_range, size, fb
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.sigma_px, self.k_trunc, self.max_w, self.min_w, self.power = sigma_px, k_trunc, max_weight, min_weight, power
        self.max_std, self.edge_rel, self.min_cos, self.free_space, self.fade = max_std, edge_rel, min_cos, free_space, fade
        n = max_blocks * B ** 3
        d = self.dev
        self.tsdf = torch.zeros(n, device=d)
        self.w = torch.zeros(n, device=d)
        self.wc = torch.zeros(n, device=d)
        self.col = torch.zeros(n, 3, device=d, dtype=torch.float16)
        self.coords = torch.zeros(max_blocks, 3, device=d, dtype=torch.long)
        self.keys = torch.empty(0, device=d, dtype=torch.long)
        self.slot_of = torch.empty(0, device=d, dtype=torch.long)
        self.max_blocks, self.n_blocks = max_blocks, 0
        g = torch.arange(B, device=d)
        self.local = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)      # x slowest: the order of a (B,B,B) block
        self.n_in = 0
        self.refine_on, self.corrections = refine, []

    # -- block hash ------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _enc(c):
        return ((c[..., 0] + OFF) << 42) | ((c[..., 1] + OFF) << 21) | (c[..., 2] + OFF)

    def _slots(self, coords, allocate):
        """Pool slot of every block coordinate (M,3); allocates the missing ones when asked (-1 when missing and not allocated, or the pool is full)."""
        k = self._enc(coords)
        pos = torch.searchsorted(self.keys, k) if len(self.keys) else torch.zeros_like(k)
        have = (pos < len(self.keys)) & (self.keys[pos.clamp(max=max(len(self.keys) - 1, 0))] == k) if len(self.keys) else torch.zeros_like(k, dtype=torch.bool)
        if allocate and (~have).any():
            new = ~have
            room = self.max_blocks - self.n_blocks
            if room <= 0:
                new = torch.zeros_like(new)
            else:
                idx = new.nonzero().ravel()[:room]
                new = torch.zeros_like(new)
                new[idx] = True
                slots = torch.arange(self.n_blocks, self.n_blocks + len(idx), device=self.dev)
                self.coords[slots] = coords[idx]
                allk, alls = torch.cat([self.keys, k[idx]]), torch.cat([self.slot_of, slots])
                o = torch.argsort(allk)
                self.keys, self.slot_of, self.n_blocks = allk[o], alls[o], self.n_blocks + len(idx)
            pos = torch.searchsorted(self.keys, k)
            have = (pos < len(self.keys)) & (self.keys[pos.clamp(max=len(self.keys) - 1)] == k)
        out = torch.full_like(k, -1)
        if have.any():
            out[have] = self.slot_of[pos[have]]
        return out

    # -- one picture -> per-pixel truncation and weight --------------------------------------------------------------------------------------------
    def _pixels(self, d, cls_mult):
        """d (H,W) metres, 0 = none. Returns (depth cleaned, truncation per pixel, weight per pixel)."""
        H, W = d.shape
        sig = d * d * self.sigma_px / self.fb
        ok = (d > 0.1) & (d <= self.max_range) & (sig <= self.max_std)
        dz = torch.where(ok, d, torch.zeros_like(d))
        big = torch.where(ok, d, torch.full_like(d, 99.0))
        hi = torch.nn.functional.max_pool2d(dz[None, None], 5, 1, 2)[0, 0]
        lo = -torch.nn.functional.max_pool2d(-big[None, None], 5, 1, 2)[0, 0]
        ok &= ~((hi - lo) > self.edge_rel * d + 5.0 * sig)                             # flying pixels: next to a surface much nearer or farther than the depth noise explains
        # normals from the smoothed depth: a surface seen at a grazing angle is stretched, its depth unreliable
        ds = torch.nn.functional.avg_pool2d(torch.where(ok, d, torch.zeros_like(d))[None, None], 9, 1, 4)[0, 0]       # wide smoothing and wide differences: the noise of one pixel must not decide the normal
        cnt = torch.nn.functional.avg_pool2d(ok.float()[None, None], 9, 1, 4)[0, 0].clamp(min=1e-3)
        ds = ds / cnt
        v, u = torch.meshgrid(torch.arange(H, device=d.device, dtype=torch.float32), torch.arange(W, device=d.device, dtype=torch.float32), indexing="ij")
        P = torch.stack([(u - self.cx) / self.fx * ds, (v - self.cy) / self.fy * ds, ds], -1)
        du = torch.zeros_like(P)
        dv = torch.zeros_like(P)
        du[:, 5:-5] = P[:, 10:] - P[:, :-10]
        dv[5:-5] = P[10:] - P[:-10]
        n = torch.cross(du, dv, dim=-1)
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        ray = P / P.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        cosang = (n * ray).sum(-1).abs()
        ok &= cosang >= self.min_cos
        wn = (cosang / 0.5).clamp(0.15, 1.0)                                          # a grazing surface counts less, it is not thrown away: far walls are often seen edge-on
        wd = (1.0 / d.clamp(min=0.5)).pow(self.power)
        w = torch.where(ok, wd * wn * cls_mult, torch.zeros_like(d))
        tr = torch.maximum(torch.full_like(d, self.tmin), self.k_trunc * sig)
        return torch.where(ok, d, torch.zeros_like(d)), tr, w

    # -- pose refinement against the volume ---------------------------------------------------------------------------------------------------------
    def _fetch(self, vox):
        """(tsdf, weight) of voxel indices (N,3); 0, 0 where no block exists."""
        blk = torch.div(vox, B, rounding_mode="floor")
        loc = vox - blk * B
        sl = self._slots(blk, False)
        ok = sl >= 0
        flat = sl.clamp(min=0) * B ** 3 + loc[:, 0] * B * B + loc[:, 1] * B + loc[:, 2]
        return torch.where(ok, self.tsdf[flat], torch.zeros_like(flat, dtype=torch.float32)), torch.where(ok, self.w[flat], torch.zeros_like(flat, dtype=torch.float32))

    def _sample(self, pw):
        """Trilinear signed distance (m) and its gradient (unit-ish, towards the camera side) at world points (N,3); ok = all 8 voxels measured and near the surface."""
        q = pw / self.v - 0.5
        i0 = torch.floor(q).long()
        f = q - i0
        corners = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        T, W = [], []
        for a, b, c in corners:
            t, w = self._fetch(i0 + torch.tensor([a, b, c], device=pw.device))
            T.append(t), W.append(w)
        T, W = torch.stack(T, 1), torch.stack(W, 1)
        ok = (W >= self.min_w).all(1) & (T.abs() < 0.9 * self.tmin).all(1)
        wx, wy, wz = (torch.stack([1 - f[:, i], f[:, i]], 1) for i in range(3))
        sdf = torch.zeros(len(q), device=pw.device)
        g = torch.zeros(len(q), 3, device=pw.device)
        for k, (a, b, c) in enumerate(corners):
            sdf += T[:, k] * wx[:, a] * wy[:, b] * wz[:, c]
            g[:, 0] += T[:, k] * (1 if a else -1) * wy[:, b] * wz[:, c]
            g[:, 1] += T[:, k] * (1 if b else -1) * wx[:, a] * wz[:, c]
            g[:, 2] += T[:, k] * (1 if c else -1) * wx[:, a] * wy[:, b]
        return sdf, g / self.v, ok

    def refine(self, depth_m, T_wc, stride=6, iters=6, max_t=0.06, max_deg=3.0, huber=0.02, min_pts=400):
        """Moves a new picture's camera pose so its depth lies on the surface already in the volume (KinectFusion-style TSDF registration, Gauss-Newton on the signed distance of the picture's points).
        Levenberg damping leaves directions the volume does not constrain (one wall: sliding along it) where they were. Returns T_wc unchanged, and says why in self.corrections, when the volume
        is empty, too few points land on it, the move would exceed max_t / max_deg, or the robust cost did not drop."""
        import cv2
        dev = self.dev
        D, _, WP = self._pixels(torch.as_tensor(depth_m, device=dev, dtype=torch.float32), torch.ones(depth_m.shape, device=dev))
        vv, uu = torch.nonzero(D[::stride, ::stride] > 0, as_tuple=True)
        if len(vv) < min_pts or self.n_blocks == 0:
            self.corrections.append(None)
            return T_wc
        vv, uu = vv * stride, uu * stride
        z, wpx = D[vv, uu], WP[vv, uu]
        P = torch.stack([(uu - self.cx) / self.fx * z, (vv - self.cy) / self.fy * z, z], -1)
        T0 = torch.as_tensor(T_wc, device=dev, dtype=torch.float32)
        R, t = T0[:3, :3].clone(), T0[:3, 3].clone()
        cost0 = None
        for _ in range(iters):
            pw = P @ R.T + t
            sdf, g, ok = self._sample(pw)
            if int(ok.sum()) < min_pts:
                self.corrections.append(None)
                return T_wc
            r, n, pwo = sdf[ok], g[ok], pw[ok]
            n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            w = wpx[ok] / wpx[ok].mean() * torch.where(r.abs() <= huber, torch.ones_like(r), huber / r.abs().clamp(min=1e-9))
            if cost0 is None:
                cost0 = float((w * r * r).sum() / w.sum())
            J = torch.cat([torch.cross(pwo, n, dim=-1), n], 1).double()
            A = J.T @ (w.double()[:, None] * J)
            b = -J.T @ (w.double() * r.double())
            A += torch.eye(6, device=dev, dtype=torch.float64) * (1e-3 * A.diagonal().mean() + 1e-9)
            d = torch.linalg.solve(A, b).cpu().numpy()
            dR = torch.as_tensor(cv2.Rodrigues(d[:3])[0], device=dev, dtype=torch.float32)
            R, t = dR @ R, dR @ t + torch.as_tensor(d[3:], device=dev, dtype=torch.float32)
            if np.linalg.norm(d[:3]) < 1e-4 and np.linalg.norm(d[3:]) < 1e-4:
                break
        pw = P @ R.T + t
        sdf, _, ok = self._sample(pw)
        w = (wpx / wpx.mean())[ok] * torch.where(sdf[ok].abs() <= huber, torch.ones_like(sdf[ok]), huber / sdf[ok].abs().clamp(min=1e-9))
        cost1 = float((w * sdf[ok] ** 2).sum() / w.sum()) if int(ok.sum()) >= min_pts else float("inf")
        dt = float((t - T0[:3, 3]).norm())
        deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues((R @ T0[:3, :3].T).cpu().numpy().astype(np.float64))[0])))
        if dt > max_t or deg > max_deg or not cost1 < 0.95 * cost0:
            self.corrections.append((dt * 1000.0, deg, cost0, cost1, False))
            return T_wc
        self.corrections.append((dt * 1000.0, deg, cost0, cost1, True))
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = R.cpu().numpy(), t.cpu().numpy()
        return out

    # -- integration ---------------------------------------------------------------------------------------------------------------------------------
    def integrate(self, depth_m, rgb, T_wc, cls_mult=None):
        """depth_m (H,W) float32 numpy metres (0 = none), rgb (H,W,3) uint8, T_wc 4x4 camera -> world (camera axes x right, y down, z forward)."""
        dev = self.dev
        D0 = torch.as_tensor(depth_m, device=dev, dtype=torch.float32)
        cm = torch.ones_like(D0) if cls_mult is None else torch.as_tensor(cls_mult, device=dev, dtype=torch.float32)
        D, TR, WP = self._pixels(D0, cm)
        if not (D > 0).any():
            return
        RGB = torch.as_tensor(rgb, device=dev)
        T = torch.as_tensor(T_wc, device=dev, dtype=torch.float32)
        R, t = T[:3, :3], T[:3, 3]
        H, W = D.shape
        # 1. allocate the blocks around the surface the picture sees (a few samples along every 4th ray, from truncation in front to behind)
        sv, su = torch.nonzero(D[::4, ::4] > 0, as_tuple=True)
        if len(sv):
            vv, uu = sv * 4, su * 4
            z, tr = D[vv, uu], TR[vv, uu]
            pts = []
            for f in (-1.0, -0.5, 0.0, 0.5):
                zz = z + f * tr
                pts.append(torch.stack([(uu - self.cx) / self.fx * zz, (vv - self.cy) / self.fy * zz, zz], -1) @ R.T + t)
            blk = torch.floor(torch.cat(pts) / (B * self.v)).long()
            self._slots(torch.unique(blk, dim=0), True)
        if self.n_blocks == 0:
            return
        # 2. every allocated block that falls inside the frustum is updated (surface blocks, and the blocks in front of the surface that carve free space)
        cen = (self.coords[:self.n_blocks].float() + 0.5) * (B * self.v)
        pc = (cen - t) @ R
        zc = pc[:, 2]
        rad = 0.87 * B * self.v
        ub, vb = self.fx * pc[:, 0] / zc.clamp(min=1e-3) + self.cx, self.fy * pc[:, 1] / zc.clamp(min=1e-3) + self.cy
        m = rad * self.fx / zc.clamp(min=1e-3)
        inview = (zc > -rad) & (zc < self.max_range + 2 * rad) & (ub > -m) & (ub < W + m) & (vb > -m) & (vb < H + m)
        slots = inview.nonzero().ravel()
        for s in torch.split(slots, 1500):
            self._update(s, D, TR, WP, RGB, R, t)
        self.n_in += 1

    def _update(self, slots, D, TR, WP, RGB, R, t):
        H, W = D.shape
        vox = self.coords[slots][:, None, :] * B + self.local[None]                       # (n,512,3) voxel index
        ctr = (vox.float() + 0.5) * self.v
        pc = (ctr - t) @ R
        z = pc[..., 2]
        zs = z.clamp(min=1e-3)
        ui, vi = torch.round(self.fx * pc[..., 0] / zs + self.cx).long(), torch.round(self.fy * pc[..., 1] / zs + self.cy).long()
        ok = (z > 0.1) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi = ui.clamp(0, W - 1), vi.clamp(0, H - 1)
        dd, tr, wp = D[vi, ui], TR[vi, ui], WP[vi, ui]
        sdf = dd - z
        ok &= (dd > 0) & (wp > 0) & (sdf > -tr)
        if not self.free_space:
            ok &= sdf < tr
        if not ok.any():
            return
        flat = (slots[:, None] * (B ** 3) + torch.arange(B ** 3, device=self.dev)[None])[ok]
        sdf, tr, wp, dd = sdf[ok], tr[ok], wp[ok], dd[ok]
        drop = torch.where(sdf < -self.v, (1.0 + (sdf + self.v) / (tr - self.v).clamp(min=1e-3)).clamp(0, 1), torch.ones_like(sdf))      # weight fades behind the surface
        wn = wp * drop if self.fade else wp
        val = torch.minimum(torch.maximum(sdf, -tr), tr)
        w0 = self.w[flat]
        self.tsdf[flat] = (self.tsdf[flat] * w0 + val * wn) / (w0 + wn).clamp(min=1e-9)
        self.w[flat] = (w0 + wn).clamp(max=self.max_w)
        near = sdf.abs() < tr
        if near.any():
            fi, c = flat[near], RGB[vi[ok][near], ui[ok][near]].float()
            wc0, wcn = self.wc[fi], wn[near]
            self.col[fi] = ((self.col[fi].float() * wc0[:, None] + c * wcn[:, None]) / (wc0 + wcn).clamp(min=1e-9)[:, None]).to(torch.float16)
            self.wc[fi] = (wc0 + wcn).clamp(max=self.max_w)

    # -- mesh ------------------------------------------------------------------------------------------------------------------------------------------
    def extract(self, min_cluster_area=0.04):
        from skimage.measure import marching_cubes
        n = self.n_blocks
        if n == 0:
            return o3d.geometry.TriangleMesh()
        dev = self.dev
        co = self.coords[:n]
        mn, mx = co.min(0).values, co.max(0).values
        dims = ((mx - mn + 1) * B).tolist()
        vol = torch.full(dims, 1.0, device=dev)
        msk = torch.zeros(dims, device=dev, dtype=torch.bool)
        g = torch.arange(B, device=dev)
        gx, gy, gz = ((co[:, i:i + 1] - mn[i]) * B + g[None] for i in range(3))
        ix, iy, iz = gx[:, :, None, None], gy[:, None, :, None], gz[:, None, None, :]
        t3 = self.tsdf[:n * B ** 3].view(n, B, B, B)
        w3 = self.w[:n * B ** 3].view(n, B, B, B)
        good = w3 >= self.min_w
        vol[ix, iy, iz] = torch.where(good, t3, torch.ones_like(t3))
        msk[ix, iy, iz] = good
        vol, msk = vol.cpu().numpy(), msk.cpu().numpy()
        if msk.sum() < 8:
            return o3d.geometry.TriangleMesh()
        try:
            verts, faces, _, _ = marching_cubes(vol, 0.0, mask=msk, gradient_direction="descent")
        except (ValueError, RuntimeError):                                              # no zero crossing
            return o3d.geometry.TriangleMesh()
        world = (verts + (mn.cpu().numpy() * B + 0.5)) * self.v
        # colour of the voxel each vertex falls in
        vg = torch.floor(torch.as_tensor(world / self.v, device=dev)).long()
        sl = self._slots(torch.div(vg, B, rounding_mode="floor"), False)
        loc = vg - torch.div(vg, B, rounding_mode="floor") * B
        flat = sl.clamp(min=0) * B ** 3 + loc[:, 0] * B * B + loc[:, 1] * B + loc[:, 2]
        col = (self.col[flat].float() / 255.0).clamp(0, 1)
        col[sl < 0] = 0.5
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(world.astype(np.float64)), o3d.utility.Vector3iVector(faces.astype(np.int32)))
        mesh.vertex_colors = o3d.utility.Vector3dVector(col.cpu().numpy().astype(np.float64))
        if min_cluster_area > 0 and len(faces):
            cl, ntri, area = mesh.cluster_connected_triangles()
            small = np.asarray(area)[np.asarray(cl)] < min_cluster_area
            if small.any():
                mesh.remove_triangles_by_mask(small)
                mesh.remove_unreferenced_vertices()
        return mesh


class GpuTsdfMap(TsdfMap):
    """tsdf.TsdfMap with the volume replaced by GpuVolume (same worker thread, rebuild-on-moved-poses, flush, save_mesh). fb = focal * baseline (px * m), for the stereo noise model."""

    def __init__(self, intrinsics, size, voxel=0.02, trunc=0.08, max_range=4.0, fb=30.0, volume_opts=None, min_cluster_area=0.04, **kw):
        self._vopts = dict(volume_opts or {})
        self._vgeom = (intrinsics, fb)
        self.min_cluster_area = min_cluster_area
        super().__init__(intrinsics, size, voxel=voxel, trunc=trunc, max_range=max_range, **kw)

    def _new_volume(self):
        (fx, fy, cx, cy), fb = self._vgeom
        return GpuVolume(self.voxel, self.trunc, self.max_range, fx, fy, cx, cy, (self.K.width, self.K.height), fb, **self._vopts)

    def _integrate_from(self, vol, frame, T_base):
        depth_mm, bgr, cls = frame
        mult = None if cls is None else CLS_LUT[cls]
        depth_m, T_wc = depth_mm.astype(np.float32) / 1000.0, T_base @ OPTICAL
        if vol.refine_on and vol.n_in > 0:
            T_wc = vol.refine(depth_m, T_wc)
        vol.integrate(depth_m, np.ascontiguousarray(bgr[:, :, ::-1]), T_wc, mult)

    def _extract(self):
        t0 = time.perf_counter()
        with self._lock:
            mesh = self.vol.extract(self.min_cluster_area)
        if len(mesh.vertices):
            mesh.compute_vertex_normals()
        self.mesh, self._last_mesh = mesh, time.monotonic()
        self.stats.update(mesh_ms=(time.perf_counter() - t0) * 1000.0, vertices=len(mesh.vertices), triangles=len(mesh.triangles), gpu_blocks=self.vol.n_blocks)
        if self.on_mesh:
            self.on_mesh(mesh, dict(self.stats))
        return True
