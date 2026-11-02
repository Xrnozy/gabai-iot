"""Pose graph optimisation: spread a loop correction over the whole path.

Nodes are keyframe poses (camera in the world). Edges are relative-pose measurements: consecutive keyframes from the odometry, and, when a revisit is
verified, a loop edge between the current keyframe and an old one. Visual odometry drifts, so the odometry edges add up to a path that does not close;
the loop edge says it should. Optimisation finds the poses that disagree least with ALL edges, so the error is shared along the path instead of being put
into the last step (which is what moving only the current pose does, and what leaves the earlier path and map bent).

Residual of an edge (i, j) with measured relative pose (R_ij, t_ij) of node j in node i's frame:
    rotation:     log(R_ij^T R_i^T R_j) / sigma_rot
    translation:  (R_i^T (t_j - t_i) - t_ij) / sigma_t
Plain least squares. A robust loss was tried (Huber) and did not converge on these graphs, because a real drift of several sigma is exactly what a loop edge
reports. A wrong loop edge is therefore kept out BEFORE this step: the caller verifies the match geometrically and limits how far it may disagree with the odometry. Node 0 is fixed (gauge). scipy's trust-region
least squares with a sparse Jacobian pattern; the work is a few hundred nodes, tens of milliseconds.

Not done: scale (Sim(3)) nodes, which a monocular system needs and a stereo one does not; information matrices from the actual feature covariances (the
sigmas are fixed numbers, see ODO_* and LOOP_*); marginalisation.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ODO_SIGMA_ROT = np.radians(0.4)       # per odometry edge, assumed
ODO_SIGMA_T = 0.01                    # metres, plus ODO_SIGMA_T_FRAC of the edge length
ODO_SIGMA_T_FRAC = 0.02
LOOP_SIGMA_ROT = np.radians(1.0)
LOOP_SIGMA_T = 0.03


def odometry_edge(i, j, R_i, t_i, R_j, t_j):
    """The measured relative pose of j in i's frame, from the two poses as the odometry had them."""
    R_ij, t_ij = R_i.T @ R_j, R_i.T @ (t_j - t_i)
    return (i, j, R_ij, t_ij, ODO_SIGMA_ROT, ODO_SIGMA_T + ODO_SIGMA_T_FRAC * float(np.linalg.norm(t_ij)))


def loop_edge(i, j, R_ij, t_ij):
    return (i, j, R_ij, t_ij, LOOP_SIGMA_ROT, LOOP_SIGMA_T)


def optimize(R, t, edges, fixed=(0,), max_nfev=40, lm_obs=None, lm_x=None, robust_c=3.0):
    """R (N,3,3), t (N,3), edges [(i, j, R_ij, t_ij, sigma_rot, sigma_t)]. Returns (R_new, t_new, info).

    Landmarks (optional): lm_x (L,3) start positions of L fixed objects in the world, lm_obs [(node, landmark, p_k (3,), sigma)] = "node's camera saw landmark at p_k in its own frame".
    Each observation is a residual  (R_k^T (X_l - t_k) - p_k) / sigma  that ties the landmark and that keyframe pose together; several keyframes seeing one landmark are tied to each other
    through it, like a loop edge that needs no feature match between them. The landmark rows are solved twice: after the first pass an observation more than robust_c sigma off is
    down-weighted (c / its size), because one wrong box must not bend the path; the odometry and loop rows stay plain least squares (see the note above). info["landmarks"] = (L,3) result."""
    R, t = np.asarray(R, float), np.asarray(t, float)
    n = len(R)
    L = 0 if lm_x is None else len(lm_x)
    lm_obs = [] if not L else list(lm_obs or [])
    free = np.array([k for k in range(n) if k not in set(fixed)], int)
    if (not len(edges) and not lm_obs) or not len(free):
        return R.copy(), t.copy(), {"nodes": n, "edges": len(edges), "iterations": 0, "cost_before": 0.0, "cost_after": 0.0, "max_shift_m": 0.0, "max_rot_deg": 0.0,
                                    "landmarks": None if not L else np.asarray(lm_x, float).copy(), "lm_obs": len(lm_obs), "lm_outliers": 0, "lm_cm_before": 0.0, "lm_cm_after": 0.0}
    slot = -np.ones(n, int)
    slot[free] = np.arange(len(free))
    ne = len(edges)
    if ne:
        ei = np.array([e[0] for e in edges])
        ej = np.array([e[1] for e in edges])
        Rm = np.stack([e[2] for e in edges])
        tm = np.stack([e[3] for e in edges])
        sr = np.array([e[4] for e in edges])
        st = np.array([e[5] for e in edges])
        Rm_inv = Rotation.from_matrix(Rm).inv()
    rv0 = Rotation.from_matrix(R).as_rotvec()
    no = len(lm_obs)
    if no:
        ok_ = np.array([o[0] for o in lm_obs], int)
        ol_ = np.array([o[1] for o in lm_obs], int)
        op = np.stack([np.asarray(o[2], float) for o in lm_obs])
        osg = np.array([o[3] for o in lm_obs], float)
    w_obs = np.ones(no)
    nfree6 = 6 * len(free)

    def unpack(x):
        P = x[:nfree6].reshape(-1, 6)
        rv, tt = rv0.copy(), t.copy()
        rv[free], tt[free] = P[:, :3], P[:, 3:]
        return Rotation.from_rotvec(rv), tt, x[nfree6:].reshape(-1, 3)

    def lm_res(rot, tt, X):
        return rot[ok_].inv().apply(X[ol_] - tt[ok_]) - op                                       # metres, (no, 3)

    def residual(x):
        rot, tt, X = unpack(x)
        parts = []
        if ne:
            Ri, Rj = rot[ei], rot[ej]
            r_rot = (Rm_inv * Ri.inv() * Rj).as_rotvec() / sr[:, None]
            r_t = (Ri.inv().apply(tt[ej] - tt[ei]) - tm) / st[:, None]
            parts.append(np.concatenate([r_rot, r_t], axis=1).ravel())
        if no:
            parts.append((lm_res(rot, tt, X) * (w_obs / osg)[:, None]).ravel())
        return np.concatenate(parts)

    def sparsity():
        sp = lil_matrix((6 * ne + 3 * no, nfree6 + 3 * L), dtype=int)
        for k in range(ne):
            for node in (ei[k], ej[k]):
                if slot[node] >= 0:
                    sp[6 * k:6 * k + 6, 6 * slot[node]:6 * slot[node] + 6] = 1
        for k in range(no):
            r0_ = 6 * ne + 3 * k
            if slot[ok_[k]] >= 0:
                sp[r0_:r0_ + 3, 6 * slot[ok_[k]]:6 * slot[ok_[k]] + 6] = 1
            sp[r0_:r0_ + 3, nfree6 + 3 * ol_[k]:nfree6 + 3 * ol_[k] + 3] = 1
        return sp

    sp = sparsity()
    x0 = np.concatenate([np.concatenate([rv0[free], t[free]], axis=1).ravel(), np.zeros(0) if not L else np.asarray(lm_x, float).ravel()])
    r0 = residual(x0)
    cm_before = float(np.median(np.linalg.norm(lm_res(*unpack(x0)), axis=1)) * 100) if no else 0.0
    sol = least_squares(residual, x0, jac_sparsity=sp, method="trf", x_scale="jac", max_nfev=max_nfev, xtol=1e-8, ftol=1e-8)
    n_out = 0
    if no:
        size = np.linalg.norm(lm_res(*unpack(sol.x)), axis=1) / osg                                  # in sigmas
        w_obs = np.where(size > robust_c, robust_c / np.maximum(size, 1e-9), 1.0)
        n_out = int((size > robust_c).sum())
        if n_out:
            sol = least_squares(residual, sol.x, jac_sparsity=sp, method="trf", x_scale="jac", max_nfev=max_nfev, xtol=1e-8, ftol=1e-8)
    rot, tt, X = unpack(sol.x)
    R_new = rot.as_matrix()
    shift = np.linalg.norm(tt - t, axis=1)
    ang = np.degrees((Rotation.from_matrix(R).inv() * rot).magnitude())
    cm_after = float(np.median(np.linalg.norm(lm_res(rot, tt, X), axis=1)) * 100) if no else 0.0
    return R_new, tt, {"nodes": n, "edges": ne, "iterations": int(sol.nfev), "cost_before": float(0.5 * r0 @ r0), "cost_after": float(sol.cost * 2),
                       "max_shift_m": float(shift.max()), "max_rot_deg": float(ang.max()), "landmarks": X if L else None, "lm_obs": no, "lm_outliers": n_out,
                       "lm_cm_before": cm_before, "lm_cm_after": cm_after}
