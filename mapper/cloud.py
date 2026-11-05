"""Persistent semi-dense point cloud, anchored to map nodes.

Each node keeps its own points in its own camera frame, once. The world cloud is a function of the node poses, so a loop closure that moves poses only changes a matrix per node: nothing is
re-fused from pictures and no point is lost or doubled. The world cloud is rebuilt from the stored points when poses change, with a voxel hash:
  semi-dense  a pixel is stored only if its depth confidence is high AND the picture has texture there (blank-wall pixels carry no measurement worth keeping)
  matched     the voxel size follows the depth noise (sigma_z): 5 mm where the depth is good, up to 30 mm where it is not. Finer voxels would only keep the noise as detail
  fused       points in one voxel join into one: a confidence / sigma_z^2 weighted centroid and mean colour, and a count of the DISTINCT nodes that saw it
  stable      a point is shown only when at least `min_views` nodes agree on it: the fan of points along a viewing ray that one picture produced never appears
Point size is not stored; it belongs to the viewer.
The fusion runs on the GPU when there is one (torch: sort, scatter-add; the rebuild of a thousand nodes took 10 s on the CPU and starved everything else of cores), on the CPU with numpy otherwise or with device="cpu".
Not done: surface normals (a new point is not checked for a compatible normal before it joins a voxel).
"""
import cv2
import numpy as np

OPTICAL = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])    # base frame -> optical frame, as rtabmap_lab/tsdf.py
LEVELS = np.array([0.005, 0.01, 0.02, 0.03])                                       # voxel sizes, m


def _gradient(gray):
    g = gray.astype(np.float32)
    gx = np.abs(g[1:-1, 2:] - g[1:-1, :-2])
    gy = np.abs(g[2:, 1:-1] - g[:-2, 1:-1])
    out = np.zeros_like(g)
    out[1:-1, 1:-1] = gx + gy
    return out


def select(depth, conf, sigma_z, gray, k, conf_min=0.5, grad_min=6.0, max_points=40000):
    """Pixels worth keeping from one node. depth m (0 = none), conf 0..1, sigma_z m, gray 0..255, k = (fx, fy, cx, cy).
    Returns (xyz in the camera frame, colour index (v, u), weight, sigma_z)."""
    ok = (depth > 0) & (conf >= conf_min) & (_gradient(gray) >= grad_min)
    v, u = np.nonzero(ok)
    if len(v) > max_points:                                                       # keep the most reliable ones (partition: O(n), a full sort of 150k pixels cost ms per node at a low conf_min)
        keep = np.argpartition(-conf[v, u], max_points)[:max_points]
        v, u = v[keep], u[keep]
    z = depth[v, u].astype(np.float32)
    fx, fy, cx, cy = k
    xyz = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1).astype(np.float32)
    sz = sigma_z[v, u].astype(np.float32)
    w = (conf[v, u] / np.maximum(sz, 1e-3) ** 2).astype(np.float32)
    return xyz, (v, u), w, sz


class PointMap:
    def __init__(self, k, min_views=2, device=None):
        self.k, self.min_views = k, min_views
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self._g = {}             # id -> (the CPU tuple it mirrors, the same arrays as GPU tensors)
        self.nodes = {}          # id -> (xyz camera frame, rgb u8, weight, sigma_z, conf)
        self.poses = {}          # id -> 4x4 base-frame pose in the world (RTAB-Map's optimised one)
        self._cache, self._ver, self._built = None, 0, {}

    def add_node(self, nid, depth, conf, sigma_z, bgr, **kw):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        xyz, (v, u), w, sz = select(depth, conf, sigma_z, gray, self.k, **kw)
        self.nodes[nid] = (xyz, bgr[v, u][:, ::-1].copy() if bgr.ndim == 3 else np.repeat(bgr[v, u][:, None], 3, 1), w, sz, conf[v, u].astype(np.float32))
        self._cache, self._ver = None, self._ver + 1

    def set_poses(self, poses, tol_m=0.01, tol_rad=np.radians(0.5)):
        """Adopt the graph's optimised poses. The cloud is rebuilt next time it is asked for only when a node sits more than tol_m / tol_rad away from where the LAST BUILD put it: the optimiser nudges
        every pose by a millimetre with every node, which used to throw the cache away at every node (and the rebuild walks every stored point). Returns the number of nodes that moved that much."""
        moved = 0
        for i, T in poses.items():
            T = np.asarray(T, float)
            self.poses[i] = T
            old = self._built.get(i)
            if old is None or np.linalg.norm(old[:3, 3] - T[:3, 3]) > tol_m or np.arccos(np.clip((np.trace(old[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)) > tol_rad:
                moved += 1
        if moved:
            self._cache, self._ver = None, self._ver + 1
        return moved

    def cloud(self):
        """(points (N,3), rgb (N,3) u8, confidence (N,), views (N,)) of the world cloud, seen by min_views or more nodes."""
        if self._cache is not None:
            return self._cache
        ver, nodes, poses = self._ver, dict(self.nodes), dict(self.poses)             # a snapshot: the live view builds this on its own thread while the main loop keeps adding nodes
        ids = [i for i in nodes if i in poses and len(nodes[i][0])]
        if not ids:
            self._cache = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros(0, np.float32), np.zeros(0, np.int32))
            return self._cache
        if self.device != "cpu":
            res = self._cloud_gpu(ids, nodes, poses)
            self._built = {i: poses[i] for i in ids}
            if ver == self._ver:
                self._cache = res
            return res
        P, C, W, S, F, N = [], [], [], [], [], []
        for n, i in enumerate(ids):
            xyz, rgb, w, sz, cf = nodes[i]
            T = poses[i] @ OPTICAL
            P.append(xyz @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32))
            C.append(rgb); W.append(w); S.append(sz); F.append(cf); N.append(np.full(len(xyz), n, np.int32))
        P, C, W, S, F, N = (np.concatenate(a) for a in (P, C, W, S, F, N))
        level = np.clip(np.searchsorted(LEVELS, S * 0.5), 0, len(LEVELS) - 1)   # smallest voxel at least half the depth noise
        out = []
        for li, vox in enumerate(LEVELS):
            s = level == li
            if not s.any():
                continue
            q = np.floor(P[s] / vox).astype(np.int64)
            q -= q.min(axis=0)
            m = q.max(axis=0) + 1
            key = (q[:, 0] * m[1] + q[:, 1]) * m[2] + q[:, 2]
            uk, inv = np.unique(key, return_inverse=True)
            w = W[s].astype(np.float64)
            sw = np.bincount(inv, w)
            pts = np.stack([np.bincount(inv, w * P[s][:, a]) / sw for a in range(3)], 1)
            rgb = np.stack([np.bincount(inv, w * C[s][:, a]) / sw for a in range(3)], 1)
            cf = np.bincount(inv, w * F[s]) / sw
            pair = np.unique(inv.astype(np.int64) * len(ids) + N[s])              # distinct (voxel, node) pairs
            views = np.bincount(pair // len(ids), minlength=len(uk))
            keep = views >= self.min_views
            out.append((pts[keep], rgb[keep], cf[keep], views[keep]))
        res = (np.concatenate([o[0] for o in out]).astype(np.float32), np.concatenate([o[1] for o in out]).round().astype(np.uint8),
               np.concatenate([o[2] for o in out]).astype(np.float32), np.concatenate([o[3] for o in out]).astype(np.int32))
        self._built = {i: poses[i] for i in ids}                                    # what the caller now draws
        if ver == self._ver:                                                         # nothing changed while this was built: it stays valid
            self._cache = res
        return res

    def _gpu(self, i, node):
        import torch
        g = self._g.get(i)
        if g is None or g[0] is not node:                                          # a node is uploaded once
            xyz, rgb, w, sz, cf = node
            g = (node, tuple(torch.as_tensor(a).to(self.device) for a in (xyz, rgb, w, sz, cf)))
            self._g[i] = g
        return g[1]

    def _cloud_gpu(self, ids, nodes, poses):
        """The same fusion as the numpy path below, on the GPU: transform every node's points, then per voxel level one sort (torch.unique) and scatter-adds for the weighted means."""
        import torch
        dev = self.device
        n_ids = len(ids)
        P, C, W, S, F, N = [], [], [], [], [], []
        for n, i in enumerate(ids):
            xyz, rgb, w, sz, cf = self._gpu(i, nodes[i])
            T = torch.as_tensor(poses[i] @ OPTICAL, dtype=torch.float32, device=dev)
            P.append(xyz @ T[:3, :3].T + T[:3, 3]); C.append(rgb); W.append(w); S.append(sz); F.append(cf); N.append(torch.full((len(xyz),), n, dtype=torch.int32, device=dev))
        P, C, W, S, F, N = (torch.cat(a) for a in (P, C, W, S, F, N))
        lv = torch.as_tensor(LEVELS, dtype=torch.float32, device=dev)
        level = torch.clamp(torch.searchsorted(lv, S * 0.5), 0, len(LEVELS) - 1)
        out = []
        for li, vox in enumerate(LEVELS):
            s = level == li
            if not bool(s.any()):
                continue
            Ps, Cs, Ws, Fs, Ns = P[s], C[s].float(), W[s], F[s], N[s]
            q = torch.floor(Ps / float(vox)).long()
            q -= q.min(dim=0).values
            m = q.max(dim=0).values + 1
            key = (q[:, 0] * m[1] + q[:, 1]) * m[2] + q[:, 2]
            uk, inv = torch.unique(key, return_inverse=True)
            k = len(uk)
            sw = torch.zeros(k, device=dev).index_add_(0, inv, Ws)
            o = Ps.min(dim=0).values                                               # sums of offsets from one corner: float32 stays exact enough at any map size
            pts = torch.stack([torch.zeros(k, device=dev).index_add_(0, inv, Ws * (Ps[:, a] - o[a])) / sw + o[a] for a in range(3)], 1)
            rgb = torch.stack([torch.zeros(k, device=dev).index_add_(0, inv, Ws * Cs[:, a]) / sw for a in range(3)], 1)
            cf = torch.zeros(k, device=dev).index_add_(0, inv, Ws * Fs) / sw
            pair = torch.unique(inv * n_ids + Ns.long())                               # distinct (voxel, node) pairs
            views = torch.bincount(pair // n_ids, minlength=k)
            keep = views >= self.min_views
            out.append((pts[keep], rgb[keep], cf[keep], views[keep]))
        return (torch.cat([o[0] for o in out]).float().cpu().numpy(), torch.cat([o[1] for o in out]).round().to(torch.uint8).cpu().numpy(),
                torch.cat([o[2] for o in out]).float().cpu().numpy(), torch.cat([o[3] for o in out]).int().cpu().numpy())

    def save_ply(self, path):
        P, C, cf, views = self.cloud()
        with open(path, "wb") as f:
            f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
                     "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float confidence\nproperty uchar views\nend_header\n").encode())
            rec = np.empty(len(P), np.dtype([("p", "<f4", 3), ("c", "u1", 3), ("q", "<f4"), ("v", "u1")]))
            rec["p"], rec["c"], rec["q"], rec["v"] = P, C, cf, np.minimum(views, 255)
            f.write(rec.tobytes())
