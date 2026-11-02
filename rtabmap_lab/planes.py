"""Flat walls: the dominant planes of the confirmed surfels are found (RANSAC), and the surfels that already lie on one (within `thresh` and with a normal that agrees) are moved onto it.

Only points that are nearly flat already move, and by at most `thresh` (2.5 cm): a curved or small object, or a surface that is not a plane, is left exactly as measured. What it removes is the few centimetres of
thickness a wall gets from depth noise and from poses that disagree by a little: the result is a sheet, not a slab.
"""
import numpy as np
import open3d as o3d


def _fit(p):
    c = p.mean(axis=0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    return n, -float(n @ c)


def planarize(points, normals, n_planes=6, thresh=0.025, min_points=600, cos_normal=0.9, iterations=400):
    """Returns (points with the plane members snapped onto their plane, bool mask of the members, [(normal, d, count)])."""
    P = np.asarray(points, np.float64).copy()
    N = np.asarray(normals, np.float64)
    used = np.zeros(len(P), bool)
    planes = []
    for _ in range(n_planes):
        idx = np.nonzero(~used)[0]
        if len(idx) < min_points:
            break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[idx]))
        _, inl = pc.segment_plane(distance_threshold=thresh, ransac_n=3, num_iterations=iterations)
        if len(inl) < min_points:
            break
        n, d = _fit(P[idx[inl]])                                              # least squares on the RANSAC members, then the members again by distance AND normal
        member = idx[(np.abs(P[idx] @ n + d) < thresh) & (np.abs(N[idx] @ n) > cos_normal)]
        if len(member) < min_points:
            break
        n, d = _fit(P[member])
        member = idx[(np.abs(P[idx] @ n + d) < thresh) & (np.abs(N[idx] @ n) > cos_normal)]
        P[member] -= (P[member] @ n + d)[:, None] * n
        used[member] = True
        planes.append((n, d, int(len(member))))
    return P.astype(np.float32), used, planes
