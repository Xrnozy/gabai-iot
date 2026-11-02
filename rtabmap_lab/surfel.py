"""Surfel fusion: the map is a set of small oriented surface elements, and every new depth image is CHECKED against them before it is believed.

Each surfel has a position, a normal, a colour, a confidence (the sum of the weights of the measurements that agreed) and the number of pictures that saw it. A new depth image is handled like this:
  1. Its pixels become points with normals (from the neighbouring pixels; pixels on depth edges or seen at a grazing angle are dropped) and a weight 1 / sigma_z^2 * cos^2(angle), sigma_z = z^2 * 0.3 px / (f * B):
     near, frontal pixels count a lot, far or slanted ones very little.
  2. The surfels already in the map are projected into the picture. A surfel that lands on a pixel with about the same depth AND the same normal is that pixel's surfel: it is updated (a weighted mean of
     position, normal and colour), and its view count goes up.
  3. A pixel that matches nothing becomes a NEW surfel, unless a confident surfel sits in front of it (a measurement behind a surface the map is sure of is rejected).
  4. A surfel that is in front of what the picture measured behind it was seen through: the camera looked at that spot and found empty space. Its confidence drops, and a surfel that has been seen through often enough goes.
  5. A surfel only seen once is kept on probation for a few pictures; if no second picture agrees with it, it is thrown away.
Only surfels seen by at least `min_views` pictures are reported (`points`): a single view's noise (the fan of points along the viewing rays) never reaches the output.

A surfel map cannot take a picture back out either: when the poses change (a loop closure) every node is fused again with its new pose, on a worker thread (SurfelFusion), like the TSDF.
"""
import threading
import time

import numpy as np

COS_GRAZING = 0.2                                       # a surface seen more edge-on than ~78 deg gives no usable normal
COS_NORMAL = np.cos(np.radians(45.0))                   # two normals closer than this are the same surface
PROBATION = 3                                           # pictures a single-view surfel may wait for a second look


class SurfelMap:
    def __init__(self, intrinsics, fB, stride=2, max_range=4.0, sigma_px=0.2, min_views=2, tau_min=0.02, tau_sigmas=2.5, see_through=0.5):
        """intrinsics = (fx, fy, cx, cy); fB = focal length (px) x baseline (m). stride: use every Nth pixel. tau: how far two depths may differ to be the same surface: max(tau_min, tau_sigmas * sigma_z)."""
        self.fx, self.fy, self.cx, self.cy = intrinsics
        self.fB, self.stride, self.max_range, self.sig, self.min_views, self.tau_min, self.tau_sig, self.thru = fB, stride, max_range, sigma_px, min_views, tau_min, tau_sigmas, see_through
        self.clear()

    def clear(self):
        self.P, self.N, self.C = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        self.W, self.V, self.born, self.n_pics = np.zeros(0, np.float32), np.zeros(0, np.int32), np.zeros(0, np.int32), 0

    def __len__(self):
        return len(self.W)

    # -- one depth picture -> points, normals, weights in the CAMERA frame -------------------------------------------------------------------------------------
    def _measure(self, depth_mm, bgr, conf=None):
        s = self.stride
        z = depth_mm[::s, ::s].astype(np.float32) / 1000.0
        z[(z <= 0) | (z > self.max_range)] = np.nan
        cg = None if conf is None else conf[::s, ::s].astype(np.float32)
        if cg is not None:
            z[cg[:z.shape[0], :z.shape[1]] <= 0] = np.nan                             # no confidence, no measurement
        hg, wg = z.shape
        v, u = np.mgrid[0:hg, 0:wg]
        u, v = u * s, v * s
        P = np.stack([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z], -1)
        import cv2
        # normals from a SMOOTHED depth and a wide baseline: raw depth is noisy by about as much as the distance between neighbouring pixels, so differences of raw neighbours give random normals
        valid = np.isfinite(z)
        z0 = np.where(valid, z, 0).astype(np.float32)
        vf = valid.astype(np.float32)
        cnt = cv2.boxFilter(vf, -1, (5, 5), normalize=False)
        zs = np.where(cnt >= 12, cv2.boxFilter(z0, -1, (5, 5), normalize=False) / np.maximum(cnt, 1), np.nan)
        big = np.where(valid, z, 99.0).astype(np.float32)
        rng = cv2.dilate(z0, np.ones((5, 5), np.uint8)) - cv2.erode(big, np.ones((5, 5), np.uint8))          # depth range in the 5x5 neighbourhood: an edge if large
        Ps = np.stack([(u - self.cx) * zs / self.fx, (v - self.cy) * zs / self.fy, zs], -1)
        k = 2
        sl = (slice(k, -k), slice(k, -k))
        dx = Ps[k:-k, 2 * k:] - Ps[k:-k, :-2 * k]
        dy = Ps[2 * k:, k:-k] - Ps[:-2 * k, k:-k]
        n = np.cross(dx, dy)
        ln = np.linalg.norm(n, axis=-1)
        zc = z[k:-k, k:-k]
        smooth = rng[k:-k, k:-k] < 0.05 * np.nan_to_num(zc, nan=1.0) + 0.04
        with np.errstate(invalid="ignore", divide="ignore"):
            n = n / ln[..., None]
            Pc = P[k:-k, k:-k]
            ray = Pc / np.linalg.norm(Pc, axis=-1, keepdims=True)
            cos = np.sum(n * -ray, axis=-1)
            flip = cos < 0
            n[flip], cos[flip] = -n[flip], -cos[flip]
            ok = np.isfinite(zc) & np.isfinite(ln) & (ln > 1e-12) & smooth & (cos > COS_GRAZING)
            sig = zc * zc * self.sig / self.fB
            w = (cos ** 2) / (sig / 0.01) ** 2
            if cg is not None:
                w = w * cg[:hg, :wg][sl]                                                  # a measurement the fusion trusts less moves a surfel less
        grid = {"z": np.full((hg, wg), np.nan, np.float32), "w": np.zeros((hg, wg), np.float32), "n": np.zeros((hg, wg, 3), np.float32), "p": np.zeros((hg, wg, 3), np.float32), "c": np.zeros((hg, wg, 3), np.float32)}
        grid["z"][sl] = np.where(ok, zc, np.nan)
        grid["w"][sl] = np.where(ok, w, 0)
        grid["n"][sl] = np.where(ok[..., None], n, 0)
        grid["p"][sl] = np.where(ok[..., None], Pc, 0)
        grid["zs"] = zs.astype(np.float32)                                          # the smoothed depth: what a see-through test is judged on, never a single pixel
        grid["flat"] = np.zeros((hg, wg), bool)
        grid["flat"][sl] = rng[sl] < 0.05 * np.nan_to_num(zs[sl], nan=1.0) + 0.04
        grid["c"] = bgr[::s, ::s][:hg, :wg].astype(np.float32)
        grid["ok"] = np.zeros((hg, wg), bool)
        grid["ok"][sl] = ok
        return grid

    # -- fuse ------------------------------------------------------------------------------------------------------------------------------------------------
    def integrate(self, depth_mm, bgr, T_wc, conf=None):
        """T_wc: 4x4 pose of the camera (optical frame) in the world. conf: optional per-pixel confidence 0..1 (the fusion's)."""
        g = self._measure(depth_mm, bgr, conf)
        R, t = T_wc[:3, :3].astype(np.float32), T_wc[:3, 3].astype(np.float32)
        hg, wg = g["z"].shape
        self.n_pics += 1
        matched_cell = np.zeros((hg, wg), bool)
        blocked = np.zeros((hg, wg), bool)
        if len(self):
            pc = (self.P - t) @ R                                           # world -> camera: R^T (P - t)
            nc = self.N @ R
            z = pc[:, 2]
            with np.errstate(invalid="ignore", divide="ignore"):
                uf, vf = self.fx * pc[:, 0] / z + self.cx, self.fy * pc[:, 1] / z + self.cy
            ci, cj = np.rint(uf / self.stride).astype(np.int64), np.rint(vf / self.stride).astype(np.int64)
            inside = (z > 0.2) & np.isfinite(uf) & np.isfinite(vf) & (ci >= 0) & (ci < wg) & (cj >= 0) & (cj < hg)
            idx = np.nonzero(inside)[0]
            cell_ok = g["ok"][cj[idx], ci[idx]]
            idx = idx[cell_ok]
            zm = g["z"][cj[idx], ci[idx]]
            tau = np.maximum(self.tau_min, self.tau_sig * zm * zm * self.sig / self.fB)
            dz = z[idx] - zm
            agree_n = np.sum(nc[idx] * g["n"][cj[idx], ci[idx]], axis=1)
            match = (np.abs(dz) < tau) & (agree_n > COS_NORMAL)
            # one surfel per pixel: the closest in depth
            cand = idx[match]
            if len(cand):
                cell = cj[cand] * wg + ci[cand]
                order = np.lexsort((np.abs(dz[match]), cell))
                cand, cell = cand[order], cell[order]
                first = np.r_[True, cell[1:] != cell[:-1]]
                cand, cell = cand[first], cell[first]
                cj_, ci_ = cell // wg, cell % wg
                w = g["w"][cj_, ci_][:, None]
                W = self.W[cand][:, None]
                self.P[cand] = (W * self.P[cand] + w * (g["p"][cj_, ci_] @ R.T + t)) / (W + w)
                nn = W * self.N[cand] + w * (g["n"][cj_, ci_] @ R.T)
                self.N[cand] = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
                self.C[cand] = (W * self.C[cand] + w * g["c"][cj_, ci_]) / (W + w)
                self.W[cand] = (W + w)[:, 0]
                self.V[cand] += 1
                matched_cell[cj_, ci_] = True
            # SEEING THROUGH: a facing surfel clearly in front of the smoothed, flat measured surface behind it (a single noisy pixel never counts). A confirmed surfel only blocks the measurement
            # behind it and loses a little; an unconfirmed one is deleted.
            ia = np.nonzero(inside)[0]
            zsm = g["zs"][cj[ia], ci[ia]]
            flat = g["flat"][cj[ia], ci[ia]] & np.isfinite(zsm)
            ia, zsm = ia[flat], zsm[flat]
            tau2 = np.maximum(self.tau_min, self.tau_sig * zsm * zsm * self.sig / self.fB)
            face = np.sum(nc[ia] * -pc[ia], axis=1) > 0
            thru = (z[ia] - zsm < -tau2) & face
            fi = ia[thru]
            if len(fi):
                confident = self.V[fi] >= self.min_views
                blocked[cj[fi][confident], ci[fi][confident]] = True
                self.W[fi[~confident]] = 0
                self.W[fi[confident]] -= 0.1 * self.thru * g["w"][cj[fi][confident], ci[fi][confident]]      # a confirmed surface is seen through only slowly: one bad picture must not erase a wall
        new = g["ok"] & ~matched_cell & ~blocked
        nj, ni = np.nonzero(new)
        if len(nj):
            self.P = np.concatenate([self.P, g["p"][nj, ni] @ R.T + t])
            self.N = np.concatenate([self.N, g["n"][nj, ni] @ R.T])
            self.C = np.concatenate([self.C, g["c"][nj, ni]])
            self.W = np.concatenate([self.W, g["w"][nj, ni]])
            self.V = np.concatenate([self.V, np.ones(len(nj), np.int32)])
            self.born = np.concatenate([self.born, np.full(len(nj), self.n_pics, np.int32)])
        keep = (self.W > 0) & ~((self.V < self.min_views) & (self.n_pics - self.born > PROBATION))
        if not keep.all():
            self.P, self.N, self.C, self.W, self.V, self.born = self.P[keep], self.N[keep], self.C[keep], self.W[keep], self.V[keep], self.born[keep]

    def gate_depth(self, depth_mm, T_wc, window=11, min_support=2):
        """The depth picture with every pixel removed that the confirmed surfels do not agree with: a pixel is kept when the stable surfels that project near it (within `window` pixels) have about its depth.
        This is what turns a map of surfels into a clean TSDF input: a pixel that no other picture confirmed (noise, a flying edge, the fan of points along a viewing ray) never gets into the volume."""
        import cv2
        h, w = depth_mm.shape
        m = self.stable()
        if not m.any():
            return np.zeros_like(depth_mm)
        P = self.P[m]
        pc = (P - T_wc[:3, 3].astype(np.float32)) @ T_wc[:3, :3].astype(np.float32)
        z = pc[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            u, v = np.rint(self.fx * pc[:, 0] / z + self.cx).astype(np.int64), np.rint(self.fy * pc[:, 1] / z + self.cy).astype(np.int64)
        ok = (z > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        zs = np.zeros((h, w), np.float32)
        order = np.argsort(-z[ok])                                         # the nearest surfel of a pixel is written last and wins
        zs[v[ok][order], u[ok][order]] = z[ok][order]
        have = (zs > 0).astype(np.float32)
        k = (window, window)
        cnt, tot = cv2.boxFilter(have, -1, k, normalize=False), cv2.boxFilter(zs, -1, k, normalize=False)
        zl = np.where(cnt >= min_support, tot / np.maximum(cnt, 1), np.nan)
        zm = depth_mm.astype(np.float32) / 1000.0
        tau = np.maximum(self.tau_min, self.tau_sig * zm * zm * self.sig / self.fB)
        keep = np.isfinite(zl) & (zm > 0) & (np.abs(zm - zl) < tau)
        return np.where(keep, depth_mm, 0).astype(np.uint16)

    def stable(self):
        return self.V >= self.min_views

    def points(self):
        """(points, normals, colours bgr 0..255) of the surfels that at least `min_views` pictures agree on."""
        m = self.stable()
        return self.P[m], self.N[m], self.C[m]


class SurfelFusion:
    """The surfel map fed by nodes with their optimised poses, rebuilt when the poses move (same contract as TsdfMap): a worker thread, on_update(points, normals, colours bgr) after each round."""

    def __init__(self, intrinsics, fB, on_update=None, threaded=True, max_nodes=600, **kw):
        self.args, self.kw, self.on_update, self.max_nodes = (intrinsics, fB), kw, on_update, max_nodes
        self.map = SurfelMap(intrinsics, fB, **kw)
        self.frames, self.poses, self.used, self.stats = {}, {}, {}, {"rebuilds": 0, "surfels": 0, "stable": 0, "ms": 0.0}
        self._lock, self._step_lock, self._wake, self._stop = threading.RLock(), threading.Lock(), threading.Event(), False
        self._thread = threading.Thread(target=self._work, daemon=True) if threaded else None
        if self._thread:
            self._thread.start()

    def add_node(self, node_id, depth_mm, bgr, conf=None, cls=None):
        with self._lock:
            self.frames[node_id] = (depth_mm, bgr, conf)
            while len(self.frames) > self.max_nodes:
                old = min(self.frames)
                self.frames.pop(old), self.poses.pop(old, None), self.used.pop(old, None)
        self._wake.set()

    def set_poses(self, poses):
        with self._lock:
            self.poses.update({i: T for i, T in poses.items() if T is not None})
        self._wake.set()

    def step(self):
        from rtabmap_lab.tsdf import OPTICAL, _moved
        with self._step_lock:
            with self._lock:
                todo = {i: self.poses[i].copy() for i in self.frames if i in self.poses}
                stale = any(i in self.used and _moved(self.used[i], T) for i, T in todo.items()) or any(i not in todo for i in self.used)
                order = sorted(todo) if stale else sorted(i for i in todo if i not in self.used)
                frames = {i: self.frames[i] for i in order}
            if not order:
                return False
            t0 = time.perf_counter()
            if stale:
                self.map.clear()
                self.stats["rebuilds"] += 1
            for i in order:
                d, bgr, conf = frames[i]
                self.map.integrate(d, bgr, todo[i] @ OPTICAL, conf)
            with self._lock:
                self.used.update({i: todo[i] for i in order})
                for i in [k for k in self.used if k not in todo]:
                    del self.used[i]
            self.stats.update(surfels=len(self.map), stable=int(self.map.stable().sum()), ms=(time.perf_counter() - t0) * 1000.0)
            if self.on_update:
                self.on_update(*self.map.points(), dict(self.stats))
            return True

    def _work(self):
        while not self._stop:
            self._wake.wait(0.5)
            self._wake.clear()
            try:
                self.step()
            except Exception as e:
                self.stats["error"] = f"{type(e).__name__}: {e}"

    def flush(self):
        self.step()
        return self.map.points()

    def close(self):
        self._stop = True
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)
