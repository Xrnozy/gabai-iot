"""Egocentric navigation map for the rtabmap_lab pipeline: depth + camera pose in, a levelled rolling obstacle grid, a distance-to-obstacle field, a traversable map and a path out.

It reuses gabai.grid.RollingGrid (the tested log-odds free/occupied/unknown grid of the navigation viewer, CPU, a few ms per picture) and adds what that grid assumes but this pipeline lacks:
  - GRAVITY AND FLOOR: the odometry world has no gravity. The floor plane is fitted (RANSAC) to the lower part of every picture's depth; its normal, kept within 45 deg of the camera's own up axis,
    gives the up direction (averaged over the first pictures, then fixed: gravity does not move) and its offset gives the floor height (averaged, so a few bad pictures do not shift it).
    The grid then lives in a frame with z up and the floor at 0, which is what RollingGrid expects.
  - DISTANCE FIELD (a 2D ESDF): metres to the nearest occupied cell, by a distance transform (OpenCV, under 1 ms for 120 x 120 cells).
  - TRAVERSABLE MAP: a cell is traversable when it was seen as floor/free, is at least `radius` from any obstacle, and unknown cells are NEVER traversable (the "can't see ahead" rule of the main navigation code).
  - PATH: a least-cost route (scikit-image MCP) over a cost that rises as the clearance shrinks, to a goal you give, or by default to the farthest reachable cell in front of the camera.
Cost per picture is the grid update plus a few small array operations; no GPU, no mesh, no wait for the TSDF.
"""
import time

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

from gabai.grid import FREE, OCCUPIED, UNKNOWN, RollingGrid


class FloorEstimator:
    """Floor plane from the depth points of one picture (camera frame: x right, y down, z forward)."""

    def __init__(self, iters=150, tol=0.03, min_points=120, max_tilt_deg=45.0, seed=0):
        self.iters, self.tol, self.min_points, self.cos_max = iters, tol, min_points, np.cos(np.radians(max_tilt_deg))
        self.rng = np.random.default_rng(seed)

    def fit(self, P):
        """P (N,3) camera-frame points. Returns (n, d) with n a unit normal pointing UP (toward the camera side) and the camera d metres above the plane, or None."""
        if len(P) < self.min_points:
            return None
        up_c = np.array([0.0, -1.0, 0.0])
        best, best_n = None, 0
        idx = self.rng.integers(0, len(P), (self.iters, 3))
        a, b, c = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n, axis=1)
        ok = norm > 1e-6
        n = n[ok] / norm[ok, None]
        a = a[ok]
        n = np.where((n @ up_c)[:, None] < 0, -n, n)
        ok = n @ up_c > self.cos_max                                       # a floor is not a wall
        n, a = n[ok], a[ok]
        if not len(n):
            return None
        d = -(n * a).sum(1)                                                # plane: n.p + d = 0, the camera (origin) is d above it when d > 0
        tol = self.tol + 0.02 * P[:, 2]
        counts = np.array([(np.abs(P @ n[i] + d[i]) < tol).sum() for i in range(len(n))])
        i = int(counts.argmax())
        if counts[i] < self.min_points or d[i] <= 0.05:
            return None
        inl = np.abs(P @ n[i] + d[i]) < tol                                 # refit on the inliers
        Q = P[inl]
        m = Q.mean(0)
        nn = np.linalg.svd(Q - m, full_matrices=False)[2][-1]
        nn = nn if nn @ up_c > 0 else -nn
        return nn, float(-(nn @ m)), int(inl.sum())


class NavMap:
    def __init__(self, half_size_m=3.0, cell_m=0.05, radius_m=0.30, clearance_m=0.8, min_hits=3, forget_s=60.0, stride=4, up_frames=6, blind_m=2.2):
        self.grid = RollingGrid(half_size_m, cell_m, forget_s, min_hits)
        self.radius, self.clearance, self.stride, self.up_frames, self.blind_m = radius_m, clearance_m, stride, up_frames, blind_m
        self.floor = FloorEstimator()
        self.up_w, self._ups, self.R_gw, self.floor_s, self._floors = None, [], None, None, []
        self.stats = {"ms": 0.0, "floor_ok": 0, "floor_missing": 0}

    # -- gravity and floor ------------------------------------------------------------------------------------------------------------------------
    def _levelled(self, T_wc, P_c):
        R, t = T_wc[:3, :3], T_wc[:3, 3]
        fit = self.floor.fit(P_c)
        if fit is not None:
            n, d, _ = fit
            self.stats["floor_ok"] += 1
            if self.R_gw is None:
                self._ups.append(R @ n)
                if len(self._ups) >= self.up_frames:
                    up = np.mean(self._ups, 0)
                    up /= np.linalg.norm(up)
                    fwd = R[:, 2] - (R[:, 2] @ up) * up
                    x = fwd / max(np.linalg.norm(fwd), 1e-9) if np.linalg.norm(fwd) > 1e-6 else np.cross(up, R[:, 0])
                    self.up_w, self.R_gw = up, np.stack([x, np.cross(up, x), up])
            up = self.up_w if self.up_w is not None else R @ n
            self._floors.append(float(t @ up) - d)
            del self._floors[:-15]
            self.floor_s = float(np.median(self._floors))
        else:
            self.stats["floor_missing"] += 1
        if self.R_gw is None:                                              # no floor seen yet: level the grid on the camera's own up axis, update nothing
            return None
        return self.R_gw @ R, self.R_gw @ t

    # -- one picture -----------------------------------------------------------------------------------------------------------------------------
    def update(self, now, depth_m, f, cx, cy, T_wc, goal_w=None):
        """depth_m (H,W) metres, 0/NaN = none. T_wc: camera -> world 4x4, camera axes x right, y down, z forward. goal_w: world point to reach, or None for 'ahead'. Returns a dict."""
        t0 = time.perf_counter()
        d = np.nan_to_num(np.asarray(depth_m, np.float32))
        s = self.stride
        v, u = np.mgrid[0:d.shape[0]:s, 0:d.shape[1]:s]
        z = d[v, u]
        ok = z > 0.2
        P_c = np.stack([(u[ok] - cx) * z[ok] / f, (v[ok] - cy) * z[ok] / f, z[ok]], 1)
        low = P_c[v[ok] > cy]                                              # the floor is in the lower half of the picture
        lv = self._levelled(np.asarray(T_wc, float), low[::2])
        if lv is None or self.floor_s is None:
            return {"ready": False}
        R_g, t_g = lv
        info = self.grid.update(now, np.where(d > 0.2, f / np.maximum(d, 1e-3), 0.0), f, 1.0, cx, cy, R_g, t_g, floor_z=self.floor_s, stride=s)
        out = self.plan(R_g, t_g, goal_w, np.asarray(T_wc, float))
        out.update(ready=True, update=info)
        self.stats["ms"] = (time.perf_counter() - t0) * 1000.0
        out["ms"] = self.stats["ms"]
        return out

    # -- distance field, traversable map, path ----------------------------------------------------------------------------------------------------
    def fields(self):
        st = self.grid.state_grid()
        occ = st == OCCUPIED
        esdf = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 3) * self.grid.cell if occ.any() else np.full(st.shape, 99.0, np.float32)
        return st, esdf

    def plan(self, R_g, t_g, goal_w, T_wc):
        g = self.grid
        st, esdf = self.fields()
        cam = g.index(t_g[0], t_g[1])
        ii, jj = np.meshgrid(np.arange(g.n), np.arange(g.n), indexing="ij")
        wx, wy = g.origin[0] + (ii + 0.5) * g.cell - t_g[0], g.origin[1] + (jj + 0.5) * g.cell - t_g[1]
        heading = R_g[:, 2]
        heading = heading[:2] / max(np.linalg.norm(heading[:2]), 1e-9)
        fwd = wx * heading[0] + wy * heading[1]
        lat = np.abs(-wx * heading[1] + wy * heading[0])
        seen_free = (st == FREE) & (esdf >= self.radius)
        # the ground right in front of a camera that looks forward is never in view (a camera 1.3 m up, pitched 15 deg down, first sees the floor 1.9 m ahead): the path may cross that
        # blind strip, at a high cost, so it can start at the walker. Unknown ground is not safe, only crossable: it is counted in unseen_path_m, and a goal is never placed in it.
        blind = (st == UNKNOWN) & (fwd > -0.3) & (fwd < self.blind_m) & (lat < 0.8) & (esdf >= self.radius)
        trav = seen_free | blind
        if 0 <= cam[0] < g.n and 0 <= cam[1] < g.n:
            ci, cj = cam
            trav[max(ci - 2, 0):ci + 3, max(cj - 2, 0):cj + 3] = True      # the cell under the camera is where the walker stands
        cost = np.where(trav, 1.0 + 6.0 * np.clip((self.clearance - esdf) / self.clearance, 0, 1) ** 2 + np.where(seen_free, 0.0, 3.0), np.inf).astype(np.float64)
        path_g, goal_cell = [], None
        if trav.sum() > 1 and 0 <= cam[0] < g.n and 0 <= cam[1] < g.n:
            mcp = MCP_Geometric(cost)
            costs, _ = mcp.find_costs([cam])
            reach = np.isfinite(costs) & seen_free
            if goal_w is not None:
                gg = self.R_gw @ np.asarray(goal_w, float)
                goal_cell = g.index(gg[0], gg[1])
                if not (0 <= goal_cell[0] < g.n and 0 <= goal_cell[1] < g.n and reach[goal_cell]):
                    goal_cell = None
            else:
                score = np.where(reach & (fwd > 0.3) & (lat < fwd * 1.2) & (fwd < 3.0), fwd - 0.15 * costs, -np.inf)
                if np.isfinite(score).any():
                    goal_cell = tuple(int(k) for k in np.unravel_index(int(score.argmax()), score.shape))
            if goal_cell is not None:
                path_g = [tuple(p) for p in mcp.traceback(goal_cell)]
        path_w = [self.R_gw.T @ np.array([*g.centre(i, j), self.floor_s or 0.0]) for i, j in path_g]
        cor = g.corridor(heading, (t_g[0], t_g[1]), half_width=self.radius, max_range=3.0)
        steer = None
        if len(path_g) > 6:
            tx, ty = g.centre(*path_g[min(len(path_g) - 1, 10)])
            steer = float(np.degrees(np.arctan2(heading[0] * (ty - t_g[1]) - heading[1] * (tx - t_g[0]), heading @ np.array([tx - t_g[0], ty - t_g[1]]))))       # + = the path bends left
        unseen = sum(1 for i, j in path_g if not seen_free[i, j]) * g.cell
        return {"unseen_path_m": float(unseen), "state": st, "esdf": esdf, "traversable": trav, "path_world": np.array(path_w), "path_cells": path_g, "cam_cell": cam, "goal_cell": goal_cell,
                "nearest_obstacle_ahead_m": cor["nearest_occupied_m"], "unknown_ahead": cor["unknown_fraction"], "steer_deg": steer, "heading": heading}

    def image(self, res, scale=4):
        """Top-down picture: grey unknown, green free, red occupied, blue inflated obstacle band, yellow path, white camera. grid x (the way the rig faced when the floor was first found) is up the picture, grid y to the left."""
        g = self.grid
        st, esdf = res["state"], res["esdf"]
        img = np.zeros((g.n, g.n, 3), np.uint8)
        img[st == UNKNOWN] = (60, 60, 60)
        img[st == FREE] = (90, 170, 90)
        img[(st != OCCUPIED) & (esdf < self.radius)] = (200, 120, 40)
        img[st == OCCUPIED] = (40, 40, 220)
        for i, j in res["path_cells"]:
            img[i, j] = (0, 230, 255)
        img = cv2.resize(np.ascontiguousarray(img[::-1, ::-1]), (g.n * scale, g.n * scale), interpolation=cv2.INTER_NEAREST)
        ci, cj = res["cam_cell"]
        cv2.circle(img, ((g.n - 1 - cj) * scale + scale // 2, (g.n - 1 - ci) * scale + scale // 2), 2 * scale, (255, 255, 255), -1)
        return img


class NavSink:
    """The dashboard sink interface (add_node / set_poses) around NavMap: each node's depth is folded into the map once, in order, as soon as the node has a pose. A map this short-lived
    (rolling window, forgets after a minute) does not need the later pose corrections of loop closures. on_update(result, navmap) gets every result."""

    def __init__(self, navmap, f, cx, cy, on_update=None, optical=None):
        import threading
        from rtabmap_lab.tsdf import OPTICAL
        self.nm, self.f, self.cx, self.cy, self.on_update, self.optical = navmap, f, cx, cy, on_update, OPTICAL if optical is None else optical
        self.pending, self.done, self.lock, self.last = {}, set(), threading.Lock(), None

    def add_node(self, node_id, depth_mm, bgr=None, cls=None, conf=None):
        with self.lock:
            self.pending[node_id] = depth_mm

    def set_poses(self, poses):
        with self.lock:
            ready = sorted(i for i in self.pending if poses.get(i) is not None and i not in self.done)
            work = [(i, self.pending.pop(i), poses[i]) for i in ready]
            self.done.update(ready)
        for i, d, T in work:
            res = self.nm.update(time.monotonic(), d.astype(np.float32) / 1000.0, self.f, self.cx, self.cy, np.asarray(T, float) @ self.optical)
            if res.get("ready"):
                self.last = res
                if self.on_update:
                    self.on_update(res, self.nm)
