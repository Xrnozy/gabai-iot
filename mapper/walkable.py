"""Walkable surface, doorways and a path to follow, from the map points (semantic surfels or the semi-dense cloud) in the gravity-aligned world of mapper/align.py (z up).

  floor      the lowest strong horizontal layer of the points (z histogram, 2 cm bins)
  walkable   2D grid (5 cm cells): a cell is floor when >= 2 points lie within 7 cm of the floor height, an obstacle when >= 3 points lie between 0.2 m and 1.8 m above it.
             An obstacle blob under 0.06 m2 with fewer than 40 points is noise and does not block. Walkable = measured floor, plus INFERRED floor: where the map has points within 0.75 m
             (holes filled) but no floor under them and nothing standing there, the floor is assumed (reported separately, drawn in a different colour). "safe" = walkable and at least half a
             body width (HALF_WIDTH) from any obstacle: what a path may use.
  doorways   narrow passages between two rooms: floor cells with 0.3..0.62 m clearance to obstacles whose neighbourhood (1 m) touches two different room cores (clearance > 0.7 m).
             Width = twice the clearance at the middle of the passage, kept when 0.6..1.5 m. Geometry only: a closed door leaf looks like a wall and is not found (no door class in the detector).
  path       A* over the safe cells from the user to a goal (nearest doorway, else the nearest frontier = safe cell next to unknown), 8-connected, waypoints every 0.5 m.
"""
import heapq

import numpy as np
from scipy import ndimage as ndi

CELL = 0.05
OBSTACLE_PTS = 5   # points above the floor in one 5 cm cell for it to be an obstacle (was 3: the permissive depth confidence made specks into obstacles and the dark band around them)
HALF_WIDTH = 0.15         # m kept free beside an obstacle before a path may use a cell: the dark green band of the viewer (was 0.3, then 0.2); the cell next to an obstacle is blocked too, so the path stays >= 0.2 m from it


def floor_height(Pa, bin_m=0.02, frac=0.15, min_pts=30):
    z = Pa[:, 2]
    h, e = np.histogram(z, np.arange(z.min(), z.max() + bin_m, bin_m))
    ok = np.nonzero(h >= max(frac * h.max(), min_pts))[0]
    z0 = e[ok[0]] + bin_m / 2
    near = z[np.abs(z - z0) < 0.04]
    return float(np.median(near)) if len(near) else float(z0)


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def build(Pa, cell=CELL, half_width=HALF_WIDTH, max_pts=400_000, min_blob_m2=0.10, blob_pts=60, infer_m=0.75):
    """Grids from aligned points (n,3). Returns a dict: origin (x0, y0), cell, floor_z, floor (supported by points) / inferred (no floor points, assumed walkable) / obstacle / walkable (floor or inferred) / safe (bool),
    clearance (m), stats. An obstacle blob smaller than min_blob_m2 with fewer than blob_pts points is noise and does not block. Where the map has points near (within infer_m of any point, holes filled)
    but no floor under them and nothing standing there, the floor is assumed: walkable but marked inferred, because nothing measured it."""
    if len(Pa) > max_pts:
        Pa = Pa[np.linspace(0, len(Pa) - 1, max_pts).astype(int)]
    fz = floor_height(Pa)
    x0, y0 = Pa[:, 0].min() - 0.5, Pa[:, 1].min() - 0.5
    shape = (int((Pa[:, 1].max() + 0.5 - y0) / cell) + 1, int((Pa[:, 0].max() + 0.5 - x0) / cell) + 1)
    ix, iy = ((Pa[:, 0] - x0) / cell).astype(int), ((Pa[:, 1] - y0) / cell).astype(int)
    flat = iy * shape[1] + ix
    dz = Pa[:, 2] - fz
    fl = np.bincount(flat[np.abs(dz) < 0.07], minlength=shape[0] * shape[1]).reshape(shape) >= 2
    n_all = shape[0] * shape[1]
    cnt = np.bincount(flat[(dz > 0.2) & (dz < 1.8)], minlength=n_all).reshape(shape)
    ob = cnt >= OBSTACLE_PTS
    lab, nb = ndi.label(ob, structure=np.ones((3, 3)))
    if nb:
        area, pts = ndi.sum(ob, lab, range(1, nb + 1)) * cell * cell, ndi.sum(cnt, lab, range(1, nb + 1))
        small = np.nonzero((area < min_blob_m2) & (pts < blob_pts))[0] + 1       # a few scattered points: noise or a speck, not a thing to walk around
        ob = ob & ~np.isin(lab, small)
    tall = np.bincount(flat[(dz > 1.3) & (dz < 2.4)], minlength=n_all).reshape(shape) >= 3     # something reaching above 1.3 m: a wall, not furniture
    seen = np.bincount(flat, minlength=n_all).reshape(shape) > 0
    r = max(int(round(0.10 / cell / 2)), 1)
    floor = ndi.binary_closing(fl, _disk(r)) & ~ob
    footprint = ndi.binary_fill_holes(ndi.binary_dilation(seen, _disk(max(int(infer_m / cell), 1))))
    inferred = footprint & ~floor & ~ob
    lab, nl = ndi.label(floor | inferred)                                    # inferred floor only where it joins measured floor without crossing an obstacle: the band outside a wall is not floor
    if nl:
        keep = np.unique(lab[floor])
        inferred = inferred & np.isin(lab, keep[keep > 0])
    walk = floor | inferred
    clear = ndi.distance_transform_edt(~ndi.binary_dilation(ob, _disk(1))) * cell
    safe = walk & (clear > half_width)
    return {"origin": (x0, y0), "cell": cell, "floor_z": fz, "floor": floor, "inferred": inferred, "obstacle": ob, "tall": tall, "walkable": walk, "safe": safe, "clearance": clear,
            "stats": {"walkable_m2": float(walk.sum() * cell * cell), "measured_floor_m2": float(floor.sum() * cell * cell), "inferred_m2": float(inferred.sum() * cell * cell),
                      "obstacle_m2": float(ob.sum() * cell * cell), "floor_z": fz}}


def cell_xy(g, rows, cols):
    return g["origin"][0] + (np.asarray(cols) + 0.5) * g["cell"], g["origin"][1] + (np.asarray(rows) + 0.5) * g["cell"]


def to_cell(g, x, y):
    return int((y - g["origin"][1]) / g["cell"]), int((x - g["origin"][0]) / g["cell"])


def doorways(g, min_w=0.6, max_w=1.5, wall_len_m=0.3):
    """[{center (x, y), width, axis (unit vector through the opening)}]: a narrow passage of MEASURED floor between two rooms, with at least 0.3 m of wall (points above 1.3 m) on both sides of it. Furniture gaps, inferred floor
    and gaps between low objects are not doorways."""
    cl = np.where(g["floor"], g["clearance"], 0.0)
    cores, _ = ndi.label(cl > 0.7)
    cand, m = ndi.label((cl >= 0.3) & (cl <= 0.62))
    out = []
    reach = max(int(round(1.0 / g["cell"])), 1)
    tall_r, tall_c = np.nonzero(g["tall"])
    tx, ty = cell_xy(g, tall_r, tall_c)
    for k in range(1, m + 1):
        comp = cand == k
        if comp.sum() < 4:
            continue
        near = ndi.binary_dilation(comp, _disk(reach))
        if len(set(np.unique(cores[near])) - {0}) < 2:
            continue
        r, c = np.nonzero(comp)
        x, y = cell_xy(g, r, c)
        pts = np.stack([x, y], 1)
        mid = np.hypot(*(pts - pts.mean(0)).T) < 0.25                         # the throat is in the middle of the passage; its ends open into the rooms
        w = 2 * float(cl[r[mid], c[mid]].max())
        if not min_w <= w <= max_w:
            continue
        center = pts.mean(0)
        axis = np.linalg.eigh(np.cov((pts - center).T))[1][:, -1]
        side = np.array([-axis[1], axis[0]])                                  # across the passage: where the two sides of the wall are
        rel = np.stack([tx, ty], 1) - center
        close = (np.hypot(*rel.T) < 0.9) & (np.abs(rel @ axis) < 0.5)
        left, right = (rel[close] @ side) > 0, (rel[close] @ side) < 0
        if min(left.sum(), right.sum()) * g["cell"] < wall_len_m:                 # a wall is a thin line of cells: at least 0.3 m of it each side
            continue
        out.append({"center": center, "width": w, "axis": axis})
    return out


def astar(safe, start, goal):
    """8-connected A* on a bool grid (row, col); returns the list of (row, col) or None."""
    H, W = safe.shape
    if not (safe[start] and safe[goal]):
        return None
    hq, g, par = [(0.0, start)], {start: 0.0}, {}
    nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    while hq:
        _, u = heapq.heappop(hq)
        if u == goal:
            path = [u]
            while u in par:
                u = par[u]
                path.append(u)
            return path[::-1]
        for dr, dc, w in nb:
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < H and 0 <= v[1] < W and safe[v] and g[u] + w < g.get(v, 1e18):
                g[v] = g[u] + w
                par[v] = u
                heapq.heappush(hq, (g[v] + np.hypot(v[0] - goal[0], v[1] - goal[1]), v))
    return None


def _nearest(mask, g, xy, rmax):
    r0, c0 = to_cell(g, *xy)
    n = int(rmax / g["cell"])
    r_lo, c_lo = max(r0 - n, 0), max(c0 - n, 0)
    sub = mask[r_lo:r0 + n + 1, c_lo:c0 + n + 1]
    rr, cc = np.nonzero(sub)
    if not len(rr):
        return None
    k = np.argmin((rr + r_lo - r0) ** 2 + (cc + c_lo - c0) ** 2)
    return int(rr[k] + r_lo), int(cc[k] + c_lo)


def guide(g, user_xy, user_fwd, doors):
    """Path to the nearest doorway, else the nearest frontier. Returns {target, path (n,2) metres, dist, bearing_deg (left positive), goal_xy}, or None when the user is not on safe floor."""
    user_xy = np.asarray(user_xy, float)
    start = _nearest(g["safe"], g, user_xy, 0.6)
    if start is None:
        return None
    lab, _ = ndi.label(g["safe"], structure=np.ones((3, 3)))
    reach = lab == lab[start]
    goal, name = None, None
    for d in sorted(doors, key=lambda d: np.hypot(*(d["center"] - user_xy))):
        goal = _nearest(reach, g, d["center"], 0.8)
        if goal is not None:
            name = "doorway"
            break
    if goal is None:
        unknown = ndi.binary_dilation(~(g["walkable"] | g["obstacle"]), _disk(2)) & reach     # reachable cells touching the unmapped outside
        fr, fc = np.nonzero(unknown)
        if not len(fr):
            return None
        k = np.argmin((fr - start[0]) ** 2 + (fc - start[1]) ** 2)
        goal, name = (int(fr[k]), int(fc[k])), "frontier"
    path = astar(reach, start, goal)
    if path is None:
        return None
    x, y = cell_xy(g, [p[0] for p in path], [p[1] for p in path])
    full = np.stack([x, y], 1)
    xy = full[::10]
    length = float(np.sum(np.hypot(*np.diff(full, axis=0).T)))
    ahead = next((p for p in xy if np.hypot(*(p - user_xy)) > 1.0), xy[-1])
    v = ahead - user_xy
    f = np.asarray(user_fwd[:2], float) / max(np.linalg.norm(user_fwd[:2]), 1e-9)
    bearing = float(np.degrees(np.arctan2(f[0] * v[1] - f[1] * v[0], f @ v)))
    return {"target": name, "path": xy, "dist": length, "bearing_deg": bearing, "goal_xy": xy[-1]}
