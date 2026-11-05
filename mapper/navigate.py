"""Walking guidance from the current depth map and RTAB-Map pose: where can the user walk, what is in the way, what to do. A few milliseconds per node, no map needed.

Per call: depth (m) + confidence + camera pose -> points in a gravity-levelled body frame (x forward, y left, z up; up comes from the caller, the vision-only world alignment of mapper/align.py, not an IMU).
  floor      a PLANE (height + slope) fitted to the lowest points of 0.3 m cells ahead, cells far above the fit (a desk, a bed) trimmed, smoothed over time. The "up" from mapper/align.py is the mean
             camera up, which is off by the rig's tilt: with a flat floor height only, a 7 deg error lifted the floor 2 m ahead above the 0.15 m threshold and the floor itself became the obstacle
  obstacles  points between 0.15 m above that plane and 0.25 m above the head
  drops      floor points more than 0.15 m BELOW the floor plane (a step down, stairs going down, a hole): the walk stops at their edge like at an obstacle
  overhead   obstacle points more than 1.2 m above the floor (a shelf, a sign at head height) are kept apart so the view can show them
  grid       0.1 m cells over EVERYTHING the picture sees (not only the heading fan): 1 floor, 2 obstacle, 3 overhead, 4 drop, 0 unknown
  free path  for each heading (every 10 deg, only those the camera actually sees), how far a user 0.7 m wide can walk before an obstacle point enters the corridor
  cue        a lexicon cue: stop / slow down / careful / go straight / path is clear / move left|right (veer) / turn left|right / path is blocked / turn around; kept until a new one repeats twice
Not done: stairs going UP are an obstacle, not a staircase; moving people are tracked in mapper/dynamic.py, not here; doors.
"""
import time

import numpy as np

from mapper.align import OPTICAL

HEADINGS = np.radians(np.arange(-60, 61, 10))
CLEAR_M, STOP_M, SLOW_M = 3.0, 0.7, 1.5


class Navigator:
    def __init__(self, f, cx, cy, half_width=0.35, step=8, max_range=3.0, conf_min=0.5, floor_h=-1.3):
        self.f, self.cx, self.cy, self.hw, self.step, self.max_range, self.conf_min = f, cx, cy, half_width, step, max_range, conf_min
        self.floor_h, self._cue, self._pend, self._n = floor_h, None, None, 0
        self.slope = np.zeros(2)                                              # floor rise per metre forward and per metre left, in the level frame
        self._fwd = None

    def level_frame(self, T_wc, up_w):
        """Rows (forward, left, up) in world coordinates: up as given, forward = the camera's forward made horizontal."""
        u = np.asarray(up_w, float) / np.linalg.norm(up_w)
        fwd = (T_wc @ OPTICAL)[:3, :3] @ np.array([0.0, 0.0, 1.0])
        f = fwd - (fwd @ u) * u
        n = np.linalg.norm(f)
        f = self._fwd if (n < 0.2 and self._fwd is not None) else f / max(n, 1e-9)
        self._fwd = f
        return np.stack([f, np.cross(u, f), u])

    def points(self, depth, conf, T_wc, up_w):
        h, w = depth.shape
        v, u = np.mgrid[self.step // 2:h:self.step, self.step // 2:w:self.step]
        z = depth[v, u]
        ok = (z > 0.2) & (z < self.max_range + 0.5) & (conf[v, u] >= self.conf_min)
        z, u, v = z[ok], u[ok], v[ok]
        opt = np.stack([(u - self.cx) * z / self.f, (v - self.cy) * z / self.f, z], axis=1)
        Tc = T_wc @ OPTICAL
        d = (Tc[:3, :3] @ opt.T).T                                            # offsets from the camera in world axes
        return d @ self.level_frame(T_wc, up_w).T                             # (x forward, y left, h up)

    def update(self, depth, conf, T_wc, up_w):
        t0 = time.perf_counter()
        p = self.points(depth, conf, T_wc, up_w)
        if len(p) < 50:
            return self._result(None, None, p, "lost tracking", t0)
        self._fit_floor(p)
        above = p[:, 2] - (self.floor_h + p[:, :2] @ self.slope)
        is_obs = (above > 0.15) & (p[:, 2] < 0.25) & (p[:, 0] > 0.15)
        is_drop = (above < -0.15) & (p[:, 0] > 0.3)
        obs, drops = p[is_obs], p[is_drop]
        stop = np.r_[obs, drops]
        az = np.arctan2(p[:, 1], p[:, 0])
        free = np.full(len(HEADINGS), -1.0)                                   # -1: not seen
        for i, th in enumerate(HEADINGS):
            if np.sum(np.abs(az - th) < np.radians(8)) < 15:
                continue
            c, s = np.cos(th), np.sin(th)
            along, lat = stop[:, 0] * c + stop[:, 1] * s, np.abs(-stop[:, 0] * s + stop[:, 1] * c)
            hit = along[(lat < self.hw) & (along > 0.15)]
            free[i] = min(float(hit.min()), self.max_range) if len(hit) >= 3 else self.max_range
        kind = np.where(np.abs(above) < 0.08, 1, 0)
        kind = np.where(is_obs, np.where(above > 1.2, 3, 2), kind)
        kind = np.where(is_drop, 4, kind)
        res = self._result(free, obs, p, None, t0)
        g = self.grid(p, kind)
        ix, iy = (p[:, 0] / g["cell"]).astype(int), ((p[:, 1] + g["half"]) / g["cell"]).astype(int)
        inside = (ix >= 0) & (ix < g["kind"].shape[0]) & (iy >= 0) & (iy < g["kind"].shape[1])
        hang = (kind == 3) & inside
        hang[hang] = g["kind"][ix[hang], iy[hang]] == 3                       # overhead only where nothing stands under it (a wall's top is a wall)
        res.update(drops=drops, overhead=p[hang], grid=g)
        return res

    def grid(self, p, kind, cell=0.1, half=3.0):
        """Egocentric cells (x 0..max_range+0.5 forward, y -half..half left) of what the whole picture shows; the most dangerous kind in a cell wins: drop, then obstacle (anything standing in the body range), then overhead (only above 1.2 m), then floor."""
        nx, ny = int((self.max_range + 0.5) / cell), int(2 * half / cell)
        ix, iy = (p[:, 0] / cell).astype(int), ((p[:, 1] + half) / cell).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (kind > 0)
        g, low = np.zeros(nx * ny, np.uint8), np.zeros(nx * ny, bool)
        np.maximum.at(g, ix[ok] * ny + iy[ok], kind[ok].astype(np.uint8))
        low[ix[ok & (kind == 2)] * ny + iy[ok & (kind == 2)]] = True
        g[low & (g == 3)] = 2                                                 # something standing from the floor up is an obstacle, even if it also reaches overhead
        return {"cell": cell, "half": half, "kind": g.reshape(nx, ny)}

    def _fit_floor(self, p, cell=0.3, max_slope=0.3):
        """Floor plane z = h + b x + c y from the lowest point of each 0.3 m cell ahead; twice: fit, drop cells more than 0.1 m above it (furniture tops), fit again. Slow EMA, so one bad frame cannot move it."""
        a = p[(p[:, 0] > 0.6) & (p[:, 0] < 4.0) & (np.abs(p[:, 1]) < 1.5) & (p[:, 2] < -0.5)]
        if len(a) < 80:
            return
        key = np.floor(a[:, 0] / cell).astype(int) * 1000 + np.floor(a[:, 1] / cell).astype(int)
        order = np.lexsort((a[:, 2], key))
        first = np.r_[True, key[order][1:] != key[order][:-1]]
        low = a[order][first]                                                 # the lowest point of every cell
        if len(low) < 6:
            return
        keep = np.ones(len(low), bool)
        for _ in range(2):
            X = np.c_[np.ones(keep.sum()), low[keep, :2]]
            coef = np.linalg.lstsq(X, low[keep, 2], rcond=None)[0]
            keep = low[:, 2] - np.c_[np.ones(len(low)), low[:, :2]] @ coef < 0.10
            if keep.sum() < 6:
                return
        h, sl = float(coef[0]), np.clip(coef[1:], -max_slope, max_slope)
        if -1.9 < h < -0.5:
            self.floor_h = 0.9 * self.floor_h + 0.1 * h
            self.slope = 0.9 * self.slope + 0.1 * sl

    def _result(self, free, obs, p, forced, t0):
        cue, head = forced, None
        if cue is None:
            i0 = len(HEADINGS) // 2
            ahead = free[i0] if free[i0] >= 0 else free[free >= 0].max() if (free >= 0).any() else -1
            if ahead < 0:
                cue = "careful"
            elif ahead >= CLEAR_M - 1e-6:
                cue = "path is clear"
            elif ahead < STOP_M:
                cue = "stop"
            else:
                # a better heading: clearly further than straight ahead. Capped at max_range, because free never exceeds it: with ahead > 2.2 m the old need (ahead + 0.8) was unreachable
                need = min(max(SLOW_M, ahead + 0.5), self.max_range)
                seen = np.nonzero(free >= need)[0]
                if len(seen):
                    j = seen[np.argmin(np.abs(HEADINGS[seen]))]
                    head = float(np.degrees(HEADINGS[j]))
                    cue = ("move left" if head > 0 else "move right") if abs(head) < 30 else ("turn left" if head > 0 else "turn right")
                    if ahead < SLOW_M and cue in ("move left", "move right"):
                        cue = "slow down"
                elif ahead >= SLOW_M:
                    cue = "go straight"                                       # nothing better, but 1.5+ m ahead is free: walking on is fine (this used to be "path is blocked")
                elif free[free >= 0].max() < SLOW_M:
                    cue = "path is blocked"                                   # under 1.5 m free in EVERY direction the camera sees
                else:
                    cue = "slow down"
        if cue == self._cue:
            self._pend = None
        elif cue in ("stop", "lost tracking") or self._cue is None or self._pend == cue:
            self._cue, self._pend = cue, None
        else:
            self._pend = cue                                                  # a change must repeat once before it is spoken
        return {"cue": self._cue or cue, "heading_deg": head, "free": free, "obstacles": obs, "points": p, "floor_h": self.floor_h, "drops": None, "overhead": None, "grid": None,
                "ms": (time.perf_counter() - t0) * 1e3}
