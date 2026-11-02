"""Clean 3D points from stereo depth, and a voxel map that fuses many views (spec phase 4, "Accurate 3D points").

Steps from the spec, in the order they run here:
  1. keep only trusted pixels (left-right consistent; the caller passes `trusted`),
  2. drop depth edges (flying pixels between a near and a far surface),
  3. drop everything past `max_range` (4 m for the map; the obstacle warnings keep using farther points),
  4. drop pixels in `exclude` (people, hands, held objects) so moving things leave no ghost trail,
  5. fuse many views into 5 cm voxels by averaging.

Step 5 is a CPU implementation (sorted voxel keys, running mean per voxel). The spec names nvblox on the GPU with an Open3D CUDA grid as the
fallback; neither is used here, so speed on the real rig is not established. A voxel needs `min_obs` observations before it counts as
surface, because one noisy observation is not evidence. Every drop is counted so a thin cloud can be traced to the step that removed points.
"""
import numpy as np

MAX_MAP_RANGE_M = 4.0
EDGE_REL = 0.05             # depth jump between neighbours, relative to the depth, above which a pixel is an edge
OFFSET = 1 << 20            # voxel index offset so packed keys stay positive
SPAN = 1 << 21


def backproject(disp, f, B, cx, cy, mask=None, stride=1):
    """Camera-frame points (x right, y down, z forward, metres) for valid disparity pixels; also returns their (u, v)."""
    d = np.asarray(disp)
    h, w = d.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    dd = d[::stride, ::stride]
    ok = np.isfinite(dd) & (dd > 0.5)
    if mask is not None:
        ok &= np.asarray(mask)[::stride, ::stride]
    u, v, dd = us[ok].astype(np.float64), vs[ok].astype(np.float64), dd[ok].astype(np.float64)
    z = f * B / dd
    return np.stack([(u - cx) / f * z, (v - cy) / f * z, z], axis=1), np.stack([u, v], axis=1)


def depth_edge_mask(disp, f, B, rel=EDGE_REL):
    """True where depth jumps by more than `rel` of the depth to a neighbour (dilated by one pixel so the whole seam is covered).
    |z1 - z2| > rel * min(z1, z2) is the same as |d1 - d2| > rel * min(d1, d2) for z = f*B/d, so it is computed on the disparity directly, in float32:
    no division, half the memory, about three times faster. f and B are kept in the signature for the callers."""
    d = np.asarray(disp, np.float32)
    d = np.where(np.isfinite(d) & (d > 0.5), d, np.float32(np.nan))
    e = np.zeros(d.shape, bool)
    with np.errstate(invalid="ignore"):
        dx = np.abs(d[:, 1:] - d[:, :-1]) > rel * np.minimum(d[:, 1:], d[:, :-1])
        dy = np.abs(d[1:] - d[:-1]) > rel * np.minimum(d[1:], d[:-1])
    e[:, 1:] |= dx
    e[:, :-1] |= dx
    e[1:] |= dy
    e[:-1] |= dy
    return e


DISP_SIGMA_PX = 0.3         # assumed disparity noise of the stereo network. An assumption, not a measurement of this rig.


def depth_std(z, f, B, sigma_px=DISP_SIGMA_PX):
    """Standard deviation of a stereo depth z (m): sigma_z = z^2 * sigma_d / (f * B). It grows with the square of the distance."""
    return np.asarray(z, float) ** 2 * sigma_px / (f * B)


def depth_weight(z, ref_z=1.0):
    """Relative weight 1 / sigma_z^2 of a point at depth z, equal to 1 at ref_z and below. Needs no f and B: they cancel in the ratio."""
    return np.minimum(1.0, (ref_z / np.maximum(np.asarray(z, float), 1e-3)) ** 4)


def map_points(disp, trusted, f, B, cx, cy, max_range=MAX_MAP_RANGE_M, exclude=None, drop_edges=True, stride=2, max_std_m=0.0):
    """Steps 1-4, plus an optional uncertainty limit like Aria's dist_std threshold: a point whose depth standard deviation exceeds max_std_m is dropped.
    Returns (points in the camera frame, counts of what each step removed)."""
    d = np.asarray(disp)
    valid = np.isfinite(d) & (d > 0.5)
    keep = valid & np.asarray(trusted, bool)
    info = {"valid": int(valid[::stride, ::stride].sum()), "dropped_untrusted": int((valid & ~keep)[::stride, ::stride].sum())}
    if drop_edges:
        edge = depth_edge_mask(d, f, B)
        info["dropped_edges"] = int((keep & edge)[::stride, ::stride].sum())
        keep &= ~edge
    if exclude is not None:
        ex = np.asarray(exclude, bool)
        info["dropped_excluded"] = int((keep & ex)[::stride, ::stride].sum())
        keep &= ~ex
    p, _ = backproject(d, f, B, cx, cy, mask=keep, stride=stride)
    far = p[:, 2] > max_range
    info["dropped_range"] = int(far.sum())
    p = p[~far]
    if max_std_m and len(p):
        unsure = depth_std(p[:, 2], f, B) > max_std_m
        info["dropped_uncertain"] = int(unsure.sum())
        p = p[~unsure]
    info["kept"] = int(len(p))
    return p, info


def transform(points, R_wc, t_wc):
    return np.asarray(points, float) @ np.asarray(R_wc, float).T + np.asarray(t_wc, float)


class VoxelMap:
    def __init__(self, voxel=0.05, window_m=None):
        self.voxel, self.window_m = voxel, window_m
        self.keys = np.zeros(0, np.int64)
        self.sum = np.zeros((0, 3))                      # sum of weight * point
        self.w = np.zeros(0)                             # sum of weights; equals n when every point has weight 1
        self.n = np.zeros(0, np.int64)                   # how many points fell in the voxel (the evidence count that min_obs is checked against)
        self.evicted = 0

    def __len__(self):
        return len(self.keys)

    def _pack(self, ijk):
        i = ijk + OFFSET
        return (i[:, 0] * SPAN + i[:, 1]) * SPAN + i[:, 2]

    def integrate(self, pts_w, weights=None):
        """Fuse world points. With `weights` (e.g. 1 / depth variance) a precise point counts for more than a noisy one in the voxel's mean."""
        p = np.asarray(pts_w, float)
        if not len(p):
            return
        wt = np.ones(len(p)) if weights is None else np.asarray(weights, float)
        k = self._pack(np.floor(p / self.voxel).astype(np.int64))
        u, inv = np.unique(k, return_inverse=True)
        s = np.zeros((len(u), 3))
        for a in range(3):
            s[:, a] = np.bincount(inv, weights=p[:, a] * wt, minlength=len(u))
        ws = np.bincount(inv, weights=wt, minlength=len(u))
        c = np.bincount(inv).astype(np.int64)
        idx = np.searchsorted(self.keys, u)
        idx_c = np.minimum(idx, max(len(self.keys) - 1, 0))
        hit = (idx < len(self.keys)) & (self.keys[idx_c] == u) if len(self.keys) else np.zeros(len(u), bool)
        self.sum[idx[hit]] += s[hit]
        self.w[idx[hit]] += ws[hit]
        self.n[idx[hit]] += c[hit]
        new = ~hit
        if new.any():
            pos = idx[new]
            self.keys = np.insert(self.keys, pos, u[new])
            self.sum = np.insert(self.sum, pos, s[new], axis=0)
            self.w = np.insert(self.w, pos, ws[new])
            self.n = np.insert(self.n, pos, c[new])

    def points(self, min_obs=3, core=0.0, faces_only=False):
        """Voxel centres of mass seen at least `min_obs` times. With core > 0 a voxel is also dropped when a neighbour was seen more than
        1/core times as often (all 26 neighbours, or only the 6 face neighbours with faces_only: 4x faster, a little less strict): noisy depth smears one surface over several layers, and this keeps the dense core layer of it."""
        ok = self.n >= min_obs
        if core > 0 and len(self.keys):
            best = self.n.copy()
            offs = (((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)) if faces_only else
                    tuple((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)))
            for dx, dy, dz in offs:
                k = self.keys + (dx * SPAN + dy) * SPAN + dz
                i = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
                hit = self.keys[i] == k
                best = np.maximum(best, np.where(hit, self.n[i], 0))
            ok &= self.n >= core * best
        return self.sum[ok] / self.w[ok, None]

    def counts(self, min_obs=3):
        return self.n[self.n >= min_obs]

    def trim(self, center, window_m=None):
        """Rolling window: forget voxels farther than window_m from `center`, so memory stays flat on a long walk."""
        r = window_m or self.window_m
        if not r or not len(self.keys):
            return 0
        far = np.linalg.norm(self.sum / self.w[:, None] - np.asarray(center, float), axis=1) > r
        k = int(far.sum())
        if k:
            self.keys, self.sum, self.w, self.n = self.keys[~far], self.sum[~far], self.w[~far], self.n[~far]
            self.evicted += k
        return k

    def prune(self, center, window_m=None, carve_m=0.0):
        """trim() and the camera carve in one pass over the voxels: forget what is farther than window_m from the camera, and what the camera itself is now inside
        (within carve_m): a camera cannot be inside a solid, so such a voxel is a moved object or an early error. Returns how many voxels went."""
        r = window_m or self.window_m
        if not len(self.keys) or not (r or carve_m):
            return 0
        d = np.linalg.norm(self.sum / self.w[:, None] - np.asarray(center, float), axis=1)
        gone = np.zeros(len(d), bool)
        if r:
            gone |= d > r
        if carve_m:
            gone |= d < carve_m
        k = int(gone.sum())
        if k:
            keep = ~gone
            self.keys, self.sum, self.w, self.n = self.keys[keep], self.sum[keep], self.w[keep], self.n[keep]
            self.evicted += k
        return k

    def contradict(self, idx, factor=0.6):
        """The camera looked through these voxels (indices into keys) and saw a surface farther away: weaken their evidence by `factor` and forget the ones left
        with none. One look from a solid voxel costs it part of its count, so a wall that is really there recovers at once from a stray frame, and one that is gone
        disappears after a few looks. Returns how many voxels were removed."""
        if not len(idx):
            return 0
        self.sum[idx] *= factor
        self.w[idx] *= factor
        self.n[idx] = np.floor(self.n[idx] * factor).astype(self.n.dtype)
        gone = self.n < 1
        k = int(gone.sum())
        if k:
            keep = ~gone
            self.keys, self.sum, self.w, self.n = self.keys[keep], self.sum[keep], self.w[keep], self.n[keep]
            self.evicted += k
        return k

    def cap(self, max_voxels, center):
        """Hard size limit. Over max_voxels, drop voxels seen under 3 times first (noise and one-off points), farthest from `center` first, then, if that is
        not enough, the farthest solid voxels. Returns how many were dropped. Memory stays bounded whatever the pose does."""
        over = len(self.keys) - int(max_voxels or 0)
        if not max_voxels or over <= 0:
            return 0
        dist = np.linalg.norm(self.sum / self.w[:, None] - np.asarray(center, float), axis=1)
        order = np.lexsort((-dist, self.n >= 3))                                  # weak voxels first, and within each group the farthest first
        keep = np.ones(len(self.keys), bool)
        keep[order[:over]] = False
        self.keys, self.sum, self.w, self.n = self.keys[keep], self.sum[keep], self.w[keep], self.n[keep]
        self.evicted += over
        return over

    def clear_frustum(self, R_wc, t_wc, f, cx, cy, max_range):
        """Remove the voxels the camera sees now (inside its picture and within max_range), keep everything behind and beside it. What a pose that disagrees with the map
        has made wrong is what is in view; the rest of the map was built while the pose was good."""
        if not len(self.keys):
            return 0
        Xc = (self.sum / self.w[:, None] - np.asarray(t_wc, float)) @ np.asarray(R_wc, float)
        z = Xc[:, 2]
        ok = (z > 0.2) & (z < max_range)
        zs = np.where(ok, z, 1.0)
        u, v = f * Xc[:, 0] / zs + cx, f * Xc[:, 1] / zs + cy
        inside = ok & (u >= 0) & (u <= 2 * cx) & (v >= 0) & (v <= 2 * cy)
        k = int(inside.sum())
        if k:
            self.keys, self.sum, self.w, self.n = self.keys[~inside], self.sum[~inside], self.w[~inside], self.n[~inside]
        return k

    def clear_sphere(self, center, radius):
        """Remove voxels inside a sphere, e.g. where a person just walked and left debris."""
        if not len(self.keys):
            return 0
        inside = np.linalg.norm(self.sum / self.w[:, None] - np.asarray(center, float), axis=1) <= radius
        k = int(inside.sum())
        if k:
            self.keys, self.sum, self.w, self.n = self.keys[~inside], self.sum[~inside], self.w[~inside], self.n[~inside]
        return k
