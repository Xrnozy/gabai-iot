"""How flat are the flat things? The room's big planes (walls, floor, ceiling) should come out as thin sheets. A plane is found by RANSAC on the surface's points, refitted by least squares,
and its THICKNESS is the RMS distance of the points near it (within `band`) from the fitted plane: a perfect wall is a few millimetres, a smeared one (poses that do not agree, depth noise) is centimetres.
Lower is flatter. The share of points that belong to a plane says how much of the surface is planar at all.

  report(points)  ->  {"planes": [ {points, thickness_mm, normal, d} ... ], "plane_share": 0..1, "mean_thickness_mm": ..}
"""
import numpy as np
import open3d as o3d


def _fit(p):
    c = p.mean(axis=0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    return n, -float(n @ c)


def report(points, n_planes=4, thresh=0.02, band=0.05, min_points=800, seed=0):
    pts = np.asarray(points, np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    out = {"planes": [], "points": len(pts)}
    left = pts
    o3d.utility.random.seed(seed)
    for _ in range(n_planes):
        if len(left) < min_points:
            break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(left))
        model, inl = pc.segment_plane(distance_threshold=thresh, ransac_n=3, num_iterations=500)
        if len(inl) < min_points:
            break
        n, d = _fit(left[inl])
        dist = left @ n + d
        near = np.abs(dist) < band
        out["planes"].append({"points": int(near.sum()), "thickness_mm": float(np.sqrt(np.mean(dist[near] ** 2)) * 1000.0), "normal": n.tolist(), "d": d})
        left = left[~near]
    inl = sum(p["points"] for p in out["planes"])
    out["plane_share"] = inl / max(len(pts), 1)
    w = np.array([p["points"] for p in out["planes"]], float)
    out["mean_thickness_mm"] = float(np.average([p["thickness_mm"] for p in out["planes"]], weights=w)) if len(w) else float("nan")
    return out


def line(name, r):
    t = ", ".join(f"{p['thickness_mm']:.0f}" for p in r["planes"]) or "-"
    return f"{name:34s} {r['points']:7d} pts   planar {r['plane_share'] * 100:3.0f}%   thickness mm per plane [{t}]   weighted mean {r['mean_thickness_mm']:.1f} mm"
