"""Plane flattening of the surfel map that also removes stacked layers.

rtabmap_lab/planes.py snaps surfels that are already within 2.5 cm of a dominant plane. That cannot help when the poses of the nodes that saw a wall or a monitor disagree by more than that: the surface
then exists as several parallel sheets 5-30 cm apart, and every sheet is "a plane" to a distance test. This module decides which sheet is the surface:
  1. planes are found with a WEIGHTED RANSAC: a candidate's score is the summed surfel weight (confidence / sigma_z^2) of its inliers, so the sheet made of the most reliable measurements wins, not the biggest
  2. each plane is refined by weighted least squares on its inliers, twice, with the tolerance tightened to the sheet's own scatter
  3. surfels on the sheet (within tol, normal agreeing) are projected onto it. Surfels that lie inside the sheet's in-plane footprint, face the same way, but sit off it by more than tol and less than
     `layer_gap` are a second copy of the same surface seen through inconsistent poses: they are dropped from the result (never moved onto the plane: that would invent geometry where nothing was measured)
  4. edges stay: only surfels with a plane-agreeing normal move, and nothing outside the footprint is touched, so a corner is the intersection of two flat sheets and small objects keep their shape
Class-aware: planes are searched first inside each planar-class object (screen, keyboard, table, book, bed) so a monitor gets its own plane even when it is small next to a wall; then the dominant planes of
what is left (walls, floor, ceiling). Points of classes that are not planar (person, cup, bottle, plant) are never touched.
"""
import numpy as np

from mapper.semantics import CLASSES

PLANAR = {"screen", "keyboard", "dining table", "book", "bed", "couch"}
PLANAR_IDX = {i for i, n in enumerate(CLASSES) if n in PLANAR}
MEMBER_MIN = 0.03                                                              # m: a surfel this close to a sheet is on it (depth noise at 2 m is 7 cm / sqrt(views))


def _wfit(P, w):
    c = (P * w[:, None]).sum(0) / w.sum()
    X = (P - c) * np.sqrt(w)[:, None]
    n = np.linalg.svd(X, full_matrices=False)[2][-1]
    return n, -float(n @ c)


def _inliers(P, N, n, d, tol, cos_n):
    return (np.abs(P @ n + d) < tol) & (np.abs(N @ n) > cos_n)


def find_plane(P, N, w, tol, cos_n=0.85, iters=300, max_sample=20000, seed=0):
    """Weighted RANSAC. Returns (normal, d, inlier mask) of the heaviest sheet, or None."""
    if len(P) < 50:
        return None
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(P), min(len(P), max_sample), replace=False) if len(P) > max_sample else np.arange(len(P))
    Ps, Ns, ws = P[sel], N[sel], w[sel]
    best, best_score = None, 0.0
    for _ in range(iters):
        i = rng.choice(len(Ps), 3, replace=False)
        n = np.cross(Ps[i[1]] - Ps[i[0]], Ps[i[2]] - Ps[i[0]])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        d = -float(n @ Ps[i[0]])
        score = ws[_inliers(Ps, Ns, n, d, tol, cos_n)].sum()
        if score > best_score:
            best, best_score = (n, d), score
    if best is None:
        return None
    n, d = best
    t = tol
    for _ in range(2):                                                         # refine on the whole set, tolerance tightened to the sheet's own scatter
        m = _inliers(P, N, n, d, t, cos_n)
        if m.sum() < 30:
            return None
        n, d = _wfit(P[m], w[m])
        rms = float(np.sqrt(np.average((P[m] @ n + d) ** 2, weights=w[m])))
        t = max(0.01, min(tol, 2.5 * rms))
    t = max(t, MEMBER_MIN)                                                      # noise tails of the sheet are members, not a second layer
    return n, d, _inliers(P, N, n, d, t, cos_n), t


def _footprint(P, n, margin):
    """Axis-aligned box of the member points in the plane's own 2-D frame (2nd-98th percentile + margin): (origin, u, v, lo, hi)."""
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    c = P.mean(0)
    a, b = (P - c) @ u, (P - c) @ v
    return c, u, v, np.array([np.percentile(a, 2), np.percentile(b, 2)]) - margin, np.array([np.percentile(a, 98), np.percentile(b, 98)]) + margin


def flatten(P, N, w, labels=None, min_points=400, tol=0.04, layer_gap=0.20, n_planes=8, class_tol=None):
    """P (n,3), N (n,3), w (n,) surfel weights, labels (n,) class index or -1/0 for none. Returns (flat points, keep mask, planes), planes = list of dicts
    (normal, d, label, members, dropped, rms_before_mm, rms_after_mm, extent). `keep` is False for the stacked-layer surfels that were dropped."""
    P = np.asarray(P, np.float64).copy()
    N = np.asarray(N, np.float64)
    w = np.maximum(np.asarray(w, np.float64), 1e-9)
    n_all = len(P)
    free = np.ones(n_all, bool)                                                # not yet on a plane
    keep = np.ones(n_all, bool)
    planes = []
    labels = np.zeros(n_all, int) if labels is None else np.asarray(labels)

    def take(region, label):
        idx = np.nonzero(region & free)[0]
        if len(idx) < min_points:
            return False
        r = find_plane(P[idx], N[idx], w[idx], tol)
        if r is None:
            return False
        n, d, m, t = r
        mem = idx[m]
        if len(mem) < min_points:
            return False
        before = float(np.sqrt(np.average((P[mem] @ n + d) ** 2, weights=w[mem])))
        c, u, v, lo, hi = _footprint(P[mem], n, margin=0.03)
        # stacked layers: inside the footprint, same facing, off the sheet by more than its tolerance and less than layer_gap
        cand = np.nonzero(free & ~np.isin(np.arange(n_all), mem) & (np.abs(N @ n) > 0.85))[0]
        dist = P[cand] @ n + d
        a, b = (P[cand] - c) @ u, (P[cand] - c) @ v
        inside = (a >= lo[0]) & (a <= hi[0]) & (b >= lo[1]) & (b <= hi[1])
        if label_is_planar(label):
            same = labels[cand] == label                                       # a monitor's layers are monitor-labelled; the wall behind it is not
            inside &= same
        layer = cand[inside & (np.abs(dist) >= t) & (np.abs(dist) < layer_gap)]
        P[mem] -= (P[mem] @ n + d)[:, None] * n
        after = float(np.sqrt(np.average((P[mem] @ n + d) ** 2, weights=w[mem])))
        free[mem] = False
        keep[layer] = False
        free[layer] = False
        planes.append({"normal": n, "d": d, "label": CLASSES[label] if label > 0 else "scene", "members": int(len(mem)), "dropped": int(len(layer)),
                       "rms_before_mm": before * 1000, "rms_after_mm": after * 1000, "tol_mm": t * 1000, "extent": (hi - lo - 0.06).round(2).tolist()})
        return True

    def label_is_planar(label):
        return label > 0

    for k in sorted(PLANAR_IDX):                                               # planar objects first, each in its own class
        for _ in range(2):                                                     # a second sheet of the class (two monitors) if there is one
            if not take(labels == k, k):
                break
    for _ in range(n_planes):                                                  # then the dominant planes of the rest: walls, floor, ceiling
        if not take(labels <= 0, 0):                                             # unlabelled and background only: people, cups and plants are never moved
            break
    return P.astype(np.float32), keep, planes
