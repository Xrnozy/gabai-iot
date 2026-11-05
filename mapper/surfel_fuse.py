"""Semantic surfel fusion in the manner of SemanticFusion (McCormac et al. 2017), on RTAB-Map's optimised node poses and FoundationStereo depth.

Geometry: rtabmap_lab/surfel.py's SurfelMap (kept as it is) already does what ElasticFusion's map does: every surfel is projected into each new picture, a measurement with the same depth and normal UPDATES it
(confidence-weighted mean of position, normal and colour), anything else becomes a new surfel, a surfel seen through is removed, a surfel only one picture saw is dropped. `SemanticSurfelMap` is that map with one
more column per surfel, a class distribution, and the paper's recursive Bayesian update:   P(l | 1..k) = P(l | 1..k-1) * P(pixel = l | picture k) / Z
using the same projective correspondence that decides which pixel updates which surfel. A new surfel starts uniform. The pixel probabilities are tempered (alpha < 1) because pictures of the same surface are not
independent, and floored so one wrong picture cannot zero a class for good (the paper's own independence caveat).
Loop closure: ElasticFusion bends its map with a deformation graph. Here the surfels are not tied to nodes; when the optimiser moves poses, `SemanticSurfelFusion` re-integrates the nodes with their new poses, so
overlapping surfels from two nodes merge instead of stacking, and each surfel's class evidence is rebuilt from the same pictures.
"""
import threading
import time

import numpy as np

from mapper.semantics import expand
from rtabmap_lab.surfel import COS_NORMAL, PROBATION, SurfelMap
from rtabmap_lab.tsdf import OPTICAL, _moved


class SemanticSurfelMap(SurfelMap):
    def __init__(self, intrinsics, fB, n_classes, alpha=0.5, floor=1e-3, **kw):
        self.K, self.alpha, self.floor = n_classes, alpha, floor
        super().__init__(intrinsics, fB, **kw)

    def clear(self):
        super().clear()
        self.L = np.zeros((0, self.K), np.float32)

    def _bayes(self, prior, p):
        post = prior * np.maximum(p, self.floor) ** self.alpha
        return post / post.sum(axis=1, keepdims=True)

    def integrate(self, depth_mm, bgr, T_wc, conf=None, probs=None):
        """probs: optional (hg, wg, K) class probabilities on the surfel grid (every `stride`-th pixel; rows sum to 1)."""
        g = self._measure(depth_mm, bgr, conf)
        pg = None if probs is None else probs[:g["z"].shape[0], :g["z"].shape[1]].astype(np.float32)
        R, t = T_wc[:3, :3].astype(np.float32), T_wc[:3, 3].astype(np.float32)
        hg, wg = g["z"].shape
        self.n_pics += 1
        matched_cell = np.zeros((hg, wg), bool)
        blocked = np.zeros((hg, wg), bool)
        if len(self):
            pc = (self.P - t) @ R
            nc = self.N @ R
            z = pc[:, 2]
            with np.errstate(invalid="ignore", divide="ignore"):
                uf, vf = self.fx * pc[:, 0] / z + self.cx, self.fy * pc[:, 1] / z + self.cy
            ci, cj = np.rint(uf / self.stride).astype(np.int64), np.rint(vf / self.stride).astype(np.int64)
            inside = (z > 0.2) & np.isfinite(uf) & np.isfinite(vf) & (ci >= 0) & (ci < wg) & (cj >= 0) & (cj < hg)
            idx = np.nonzero(inside)[0]
            idx = idx[g["ok"][cj[idx], ci[idx]]]
            zm = g["z"][cj[idx], ci[idx]]
            tau = np.maximum(self.tau_min, self.tau_sig * zm * zm * self.sig / self.fB)
            dz = z[idx] - zm
            match = (np.abs(dz) < tau) & (np.sum(nc[idx] * g["n"][cj[idx], ci[idx]], axis=1) > COS_NORMAL)
            cand = idx[match]
            if len(cand):
                cell = cj[cand] * wg + ci[cand]
                order = np.lexsort((np.abs(dz[match]), cell))
                cand, cell = cand[order], cell[order]
                first = np.r_[True, cell[1:] != cell[:-1]]                       # one surfel per pixel: the closest in depth
                cand, cell = cand[first], cell[first]
                cj_, ci_ = cell // wg, cell % wg
                w, W = g["w"][cj_, ci_][:, None], self.W[cand][:, None]
                self.P[cand] = (W * self.P[cand] + w * (g["p"][cj_, ci_] @ R.T + t)) / (W + w)
                nn = W * self.N[cand] + w * (g["n"][cj_, ci_] @ R.T)
                self.N[cand] = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
                self.C[cand] = (W * self.C[cand] + w * g["c"][cj_, ci_]) / (W + w)
                self.W[cand] = (W + w)[:, 0]
                self.V[cand] += 1
                if pg is not None:
                    self.L[cand] = self._bayes(self.L[cand], pg[cj_, ci_])
                matched_cell[cj_, ci_] = True
            ia = np.nonzero(inside)[0]
            zsm = g["zs"][cj[ia], ci[ia]]
            flat = g["flat"][cj[ia], ci[ia]] & np.isfinite(zsm)
            ia, zsm = ia[flat], zsm[flat]
            tau2 = np.maximum(self.tau_min, self.tau_sig * zsm * zsm * self.sig / self.fB)
            thru = (z[ia] - zsm < -tau2) & (np.sum(nc[ia] * -pc[ia], axis=1) > 0)
            fi = ia[thru]
            if len(fi):
                confident = self.V[fi] >= self.min_views
                blocked[cj[fi][confident], ci[fi][confident]] = True
                self.W[fi[~confident]] = 0
                self.W[fi[confident]] -= 0.1 * self.thru * g["w"][cj[fi][confident], ci[fi][confident]]
        new = g["ok"] & ~matched_cell & ~blocked
        nj, ni = np.nonzero(new)
        if len(nj):
            self.P = np.concatenate([self.P, g["p"][nj, ni] @ R.T + t])
            self.N = np.concatenate([self.N, g["n"][nj, ni] @ R.T])
            self.C = np.concatenate([self.C, g["c"][nj, ni]])
            self.W = np.concatenate([self.W, g["w"][nj, ni]])
            self.V = np.concatenate([self.V, np.ones(len(nj), np.int32)])
            self.born = np.concatenate([self.born, np.full(len(nj), self.n_pics, np.int32)])
            uniform = np.full((len(nj), self.K), 1.0 / self.K, np.float32)
            self.L = np.concatenate([self.L, uniform if pg is None else self._bayes(uniform, pg[nj, ni])])
        keep = (self.W > 0) & ~((self.V < self.min_views) & (self.n_pics - self.born > PROBATION))
        if not keep.all():
            self.P, self.N, self.C, self.W, self.V, self.born, self.L = (a[keep] for a in (self.P, self.N, self.C, self.W, self.V, self.born, self.L))

    def labels(self, min_prob=0.5):
        """(class index, probability) of every STABLE surfel; class -1 where the best class has less than min_prob (unknown)."""
        L = self.L[self.stable()]
        k, p = L.argmax(axis=1), L.max(axis=1)
        return np.where(p >= min_prob, k, -1), p


class SemanticSurfelFusion:
    """Feeds nodes to the map in node order and re-integrates ALL of them when the optimiser moved a pose (same contract as rtabmap_lab.surfel.SurfelFusion, plus class probabilities)."""

    def __init__(self, intrinsics, fB, n_classes, on_update=None, threaded=True, max_nodes=800, **kw):
        self.map = SemanticSurfelMap(intrinsics, fB, n_classes, **kw)
        self.on_update, self.max_nodes = on_update, max_nodes
        self.frames, self.poses, self.used, self.stats = {}, {}, {}, {"rebuilds": 0, "surfels": 0, "stable": 0, "ms": 0.0}
        self._lock, self._step_lock, self._wake, self._stop = threading.RLock(), threading.Lock(), threading.Event(), False
        self._thread = threading.Thread(target=self._work, daemon=True) if threaded else None
        if self._thread:
            self._thread.start()

    def add_node(self, node_id, depth_mm, bgr, conf=None, seg=None):
        """seg: (class index per grid cell uint8, 255 = no evidence; confidence per grid cell), from semantics.Segmenter.grid; stored compactly and expanded to probabilities at integration."""
        with self._lock:
            self.frames[node_id] = (depth_mm, bgr, conf, seg)
            while len(self.frames) > self.max_nodes:
                old = min(self.frames)
                self.frames.pop(old), self.poses.pop(old, None), self.used.pop(old, None)
        self._wake.set()

    def set_poses(self, poses):
        with self._lock:
            self.poses.update({i: T for i, T in poses.items() if T is not None})
        self._wake.set()

    def step(self, min_rebuild_s=0.0):
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
                d, bgr, conf, seg = frames[i]
                self.map.integrate(d, bgr, todo[i] @ OPTICAL, conf, None if seg is None else expand(*seg, self.map.K))
            with self._lock:
                self.used.update({i: todo[i] for i in order})
                for i in [k for k in self.used if k not in todo]:
                    del self.used[i]
            self.stats.update(surfels=len(self.map), stable=int(self.map.stable().sum()), ms=(time.perf_counter() - t0) * 1000.0)
            if self.on_update:
                self.on_update(self.map, dict(self.stats))
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
        return self.map

    def close(self):
        self._stop = True
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)
