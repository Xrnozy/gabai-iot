"""Rolling obstacle grid (spec phase 2a): free / occupied / UNKNOWN cells around the camera, in the world frame.

  * Built from a disparity map and the camera pose (R_wc, t_wc; world z up, floor at floor_z). With no pose estimate the caller may
    pass the IMU orientation and a fixed position: the grid then only describes what is in view now.
  * A cell is OCCUPIED when points between 0.15 m and 2.2 m above the floor fall in it (a person or a box, not the floor, not an
    overhead shelf), FREE when the floor itself was seen in it, and UNKNOWN when nothing was seen, when the evidence is mixed, or
    when the last observation is older than `forget_s`. Unknown is NEVER free: the "can't see ahead" rule relies on that.
  * Evidence is a log-odds count per cell, so a person who walks away is cleared after a few frames of seeing the floor again, and a
    single noisy frame cannot create an obstacle (a cell needs `min_hits` points in one frame).
  * People found by a detector can be passed as a pixel mask: those pixels are left out of the map (no ghost trails); an approaching
    person is a separate, higher-priority alert, not a grid cell.
The grid drives warnings by occupancy, not by label.
"""
import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
OBSTACLE_MIN_H, OBSTACLE_MAX_H = 0.15, 2.2       # metres above the floor
FLOOR_MAX_H, FLOOR_MIN_H = 0.10, -0.25
# Fast to detect, slower to clear: an obstacle that appears over floor seen as clear for a long time must show within two frames
# (2 x +2.0 from the floor of -2.0 gives +2.0, above the +1.0 threshold), while a person who left takes about nine frames (0.65 s at
# 14 fps) of seeing floor again before their cells clear.
L_OCC, L_FREE, L_MAX, L_MIN = 2.0, -0.5, 4.0, -2.0
OCC_ABOVE, FREE_BELOW = 1.0, -0.2


class RollingGrid:
    def __init__(self, half_size_m=3.0, cell_m=0.10, forget_s=30.0, min_hits=2):
        self.half, self.cell, self.forget, self.min_hits = half_size_m, cell_m, forget_s, min_hits
        self.n = int(round(2 * half_size_m / cell_m))
        self.L = np.zeros((self.n, self.n), np.float32)
        self.seen = np.zeros((self.n, self.n), bool)
        self.last = np.full((self.n, self.n), -1e9, np.float64)
        self.origin = np.array([-half_size_m, -half_size_m], float)      # world xy of the lower corner of cell (0, 0)
        self._init = False
        self.frames = 0                    # how many disparity maps this grid has taken in

    # -- geometry ---------------------------------------------------------------------------------------------------------------
    def index(self, x, y):
        return int(np.floor((x - self.origin[0]) / self.cell)), int(np.floor((y - self.origin[1]) / self.cell))

    def centre(self, ix, iy):
        return self.origin[0] + (ix + 0.5) * self.cell, self.origin[1] + (iy + 0.5) * self.cell

    def _recentre(self, cam_xy):
        want = np.round((np.asarray(cam_xy, float) - self.half) / self.cell) * self.cell
        if not self._init:
            self.origin, self._init = want, True
            return 0, 0
        d = np.round((want - self.origin) / self.cell).astype(int)
        if d[0] == 0 and d[1] == 0:
            return 0, 0
        if np.abs(d).max() >= self.n:                       # moved farther than the whole window: nothing in it can still be in view
            self.L[...], self.seen[...], self.last[...] = 0.0, False, -1e9
            self.origin = self.origin + d * self.cell
            return int(d[0]), int(d[1])
        for arr, fill in ((self.L, 0.0), (self.seen, False), (self.last, -1e9)):
            for ax, k in ((0, d[0]), (1, d[1])):
                if k:
                    arr[...] = np.roll(arr, -k, axis=ax)
                    sl = [slice(None)] * 2
                    sl[ax] = slice(self.n - k, self.n) if k > 0 else slice(0, -k)
                    arr[tuple(sl)] = fill
        self.origin = self.origin + d * self.cell
        return int(d[0]), int(d[1])

    # -- update -----------------------------------------------------------------------------------------------------------------
    def update(self, now, disp, f, B, cx, cy, R_wc, t_wc, floor_z=0.0, person_mask=None, stride=4):
        """Fold one disparity map into the grid. Returns counts for the diagnostics."""
        t_wc = np.asarray(t_wc, float)
        self.frames += 1
        shift = self._recentre(t_wc[:2])
        d = np.asarray(disp)[::stride, ::stride]
        valid = np.isfinite(d) & (d > 0.5)
        if person_mask is not None:
            valid &= ~np.asarray(person_mask, bool)[::stride, ::stride]
        h, w = d.shape
        u = ((np.arange(w) * stride)[None, :] - cx) / f
        v = ((np.arange(h) * stride)[:, None] - cy) / f
        Z = np.where(valid, f * B / np.where(valid, d, 1.0), 0.0)
        P = np.stack([u * Z, v * Z, Z], axis=-1)[valid] @ np.asarray(R_wc).T + t_wc
        height = P[:, 2] - floor_z
        ix = np.floor((P[:, 0] - self.origin[0]) / self.cell).astype(int)
        iy = np.floor((P[:, 1] - self.origin[1]) / self.cell).astype(int)
        inside = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        occ = inside & (height > OBSTACLE_MIN_H) & (height < OBSTACLE_MAX_H)
        free = inside & (height < FLOOR_MAX_H) & (height > FLOOR_MIN_H)
        flat = lambda m: np.bincount(ix[m] * self.n + iy[m], minlength=self.n * self.n).reshape(self.n, self.n)
        n_occ, n_free = flat(occ), flat(free)
        hit_occ, hit_free = n_occ >= self.min_hits, (n_free >= self.min_hits) & (n_occ < self.min_hits)
        self.L[hit_occ] = np.minimum(self.L[hit_occ] + L_OCC, L_MAX)
        self.L[hit_free] = np.maximum(self.L[hit_free] + L_FREE, L_MIN)
        touched = hit_occ | hit_free
        self.seen |= touched
        self.last[touched] = now
        return {"points": int(valid.sum()), "obstacle_points": int(occ.sum()), "floor_points": int(free.sum()),
                "cells_occupied_now": int(hit_occ.sum()), "cells_free_now": int(hit_free.sum()), "shift_cells": shift}

    def expire(self, now):
        """Cells not observed for `forget_s` go back to unknown."""
        old = self.seen & (now - self.last > self.forget)
        self.seen[old] = False
        self.L[old] = 0.0

    # -- read -------------------------------------------------------------------------------------------------------------------
    def state_grid(self):
        st = np.full((self.n, self.n), UNKNOWN, np.uint8)
        st[self.seen & (self.L >= OCC_ABOVE)] = OCCUPIED
        st[self.seen & (self.L <= FREE_BELOW)] = FREE
        return st

    def state_at(self, x, y):
        ix, iy = self.index(x, y)
        if not (0 <= ix < self.n and 0 <= iy < self.n):
            return UNKNOWN
        return int(self.state_grid()[ix, iy])

    def corridor(self, heading_xy, cam_xy=(0.0, 0.0), half_width=0.5, max_range=3.0, min_range=0.2):
        """What is straight ahead: nearest occupied distance and how much of the corridor is unknown or free."""
        hx, hy = np.asarray(heading_xy, float) / max(np.linalg.norm(heading_xy), 1e-9)
        st = self.state_grid()
        ii, jj = np.meshgrid(np.arange(self.n), np.arange(self.n), indexing="ij")
        cxw, cyw = self.origin[0] + (ii + 0.5) * self.cell - cam_xy[0], self.origin[1] + (jj + 0.5) * self.cell - cam_xy[1]
        fwd, lat = cxw * hx + cyw * hy, -cxw * hy + cyw * hx
        sel = (fwd > min_range) & (fwd <= max_range) & (np.abs(lat) <= half_width)
        s = st[sel]
        if not s.size:
            return {"nearest_occupied_m": None, "unknown_fraction": 1.0, "free_fraction": 0.0, "cells": 0}
        occ = sel & (st == OCCUPIED)
        return {"nearest_occupied_m": float(fwd[occ].min()) if occ.any() else None, "unknown_fraction": float((s == UNKNOWN).mean()),
                "free_fraction": float((s == FREE).mean()), "cells": int(s.size)}

    def to_image(self, scale=4, camera_xy=None, heading_xy=None):
        """Top-down picture for the diagnostics: grey unknown, green free, red occupied; x is up the picture, y is to the left."""
        import cv2
        st = self.state_grid()
        img = np.zeros((self.n, self.n, 3), np.uint8)
        img[st == UNKNOWN] = (60, 60, 60)
        img[st == FREE] = (90, 170, 90)
        img[st == OCCUPIED] = (40, 40, 220)
        img = np.rot90(img)                                  # grid[ix (x), iy (y)] -> rows = -x, cols = -y... keep x up
        img = cv2.resize(np.ascontiguousarray(img), (self.n * scale, self.n * scale), interpolation=cv2.INTER_NEAREST)
        if camera_xy is not None:
            ix, iy = self.index(*camera_xy)
            px, py = (self.n - 1 - iy) * scale + scale // 2, (self.n - 1 - ix) * scale + scale // 2
            cv2.circle(img, (px, py), 2 * scale, (255, 255, 255), -1)
            if heading_xy is not None:
                hx, hy = np.asarray(heading_xy, float) / max(np.linalg.norm(heading_xy), 1e-9)
                cv2.arrowedLine(img, (px, py), (int(px - hy * 8 * scale), int(py - hx * 8 * scale)), (255, 255, 255), 1)
        return img
