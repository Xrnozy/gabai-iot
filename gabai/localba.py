"""Sliding-window bundle adjustment for stereo keyframes.

Poses of the last few keyframes and the 3D positions of the landmarks that several of them see are refined together. Each observation gives two kinds of
residual: the landmark's reprojection in the keyframe's left picture, and (where the stereo network gave a depth there) the disparity it predicts against the
disparity that was measured. Stereo alone fixes each landmark's depth with a persistent, region-wide error that no amount of re-measuring from the same viewpoint
removes; the same landmark seen from keyframes some tens of centimetres apart is constrained by parallax, which does not share that error. The bundle adjustment
trades the two against each other by their assumed noise (sigma_uv, sigma_d).

The oldest `n_fixed` poses of the window are held (they fix the position, orientation and scale of the problem and carry the earlier path). Outliers: a first
pass of plain least squares, then every observation whose residual is above OUTLIER_SIGMA is dropped and a second pass runs on the rest (a Huber loss was tried and
stalled on poor starting values). scipy trust-region least squares with a sparse Jacobian pattern.

Not done: marginalisation of the keyframes that leave the window (they are just frozen, so their information about the landmarks is dropped), and analytic
Jacobians. Cost is a few hundred milliseconds for a window of 5-6 keyframes and a few hundred landmarks: run it off the tracking thread.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

SIGMA_UV = 1.0            # pixels, assumed tracking noise
SIGMA_D = 0.7             # pixels of disparity, assumed stereo noise (kept above the nominal 0.3 because the stereo error is persistent, not independent)


OUTLIER_SIGMA = 4.0


def optimize(K, B, R, t, n_fixed, obs_node, obs_lm, obs_uv, obs_d, lm_xyz, max_nfev=15, sigma_uv=None, sigma_d=None):
    """R (M,3,3), t (M,3) camera-in-world poses; the first n_fixed are held. Observation k: node obs_node[k], landmark obs_lm[k], pixel obs_uv[k], measured
    disparity obs_d[k] (nan = none). lm_xyz (L,3) world. Returns (R_new, t_new, lm_new, info); info["dropped"] counts the observations removed as outliers."""
    sigma_uv, sigma_d = sigma_uv or SIGMA_UV, sigma_d or SIGMA_D
    R, t, lm_xyz = np.asarray(R, float), np.asarray(t, float), np.asarray(lm_xyz, float)
    obs_node, obs_lm, obs_uv, obs_d = np.asarray(obs_node), np.asarray(obs_lm), np.asarray(obs_uv, float), np.asarray(obs_d, float)
    M, L, O = len(R), len(lm_xyz), len(obs_node)
    empty = {"cost_before": 0.0, "cost_after": 0.0, "nfev": 0, "observations": O, "landmarks": L, "dropped": 0}
    if n_fixed >= M or O == 0 or L == 0:
        return R.copy(), t.copy(), lm_xyz.copy(), empty
    R1, t1, X1, info1, res1 = _solve(K, B, R, t, n_fixed, obs_node, obs_lm, obs_uv, obs_d, lm_xyz, max_nfev, sigma_uv, sigma_d)
    bad = (res1 > OUTLIER_SIGMA).reshape(-1, 3).any(axis=1)                  # a residual of a few sigma in the picture or in the disparity
    if not bad.any() or bad.all():
        return R1, t1, X1, dict(info1, dropped=0)
    keep = ~bad
    R2, t2, X2, info2, _ = _solve(K, B, R1, t1, n_fixed, obs_node[keep], obs_lm[keep], obs_uv[keep], obs_d[keep], X1, max_nfev, sigma_uv, sigma_d)
    return R2, t2, X2, dict(info2, cost_before=info1["cost_before"], nfev=info1["nfev"] + info2["nfev"], observations=O, dropped=int(bad.sum()))


def _solve(K, B, R, t, n_fixed, obs_node, obs_lm, obs_uv, obs_d, lm_xyz, max_nfev, sigma_uv, sigma_d):
    M = len(R)
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    used = np.unique(obs_lm)                                                  # landmarks that still have an observation
    remap = -np.ones(len(lm_xyz), int)
    remap[used] = np.arange(len(used))
    olm = remap[obs_lm]
    O, L = len(obs_node), len(used)
    free = np.arange(n_fixed, M)
    nf = len(free)
    has_d = np.isfinite(obs_d)
    rv0 = Rotation.from_matrix(R).as_rotvec()

    def unpack(x):
        P = x[:6 * nf].reshape(-1, 6)
        rv, tt = rv0.copy(), t.copy()
        rv[free], tt[free] = P[:, :3], P[:, 3:]
        return Rotation.from_rotvec(rv), tt, x[6 * nf:].reshape(-1, 3)

    def residual(x):
        rot, tt, X = unpack(x)
        Xc = rot[obs_node].inv().apply(X[olm] - tt[obs_node])
        z = np.maximum(Xc[:, 2], 1e-3)
        r_uv = np.stack([f * Xc[:, 0] / z + cx - obs_uv[:, 0], f * Xc[:, 1] / z + cy - obs_uv[:, 1]], 1) / sigma_uv
        r_d = np.where(has_d, (f * B / z - np.where(has_d, obs_d, 0.0)) / sigma_d, 0.0)
        return np.concatenate([r_uv, r_d[:, None]], axis=1).ravel()

    sp = lil_matrix((3 * O, 6 * nf + 3 * L), dtype=int)
    slot = -np.ones(M, int)
    slot[free] = np.arange(nf)
    for k in range(O):
        rows = slice(3 * k, 3 * k + 3)
        if slot[obs_node[k]] >= 0:
            sp[rows, 6 * slot[obs_node[k]]:6 * slot[obs_node[k]] + 6] = 1
        sp[rows, 6 * nf + 3 * olm[k]:6 * nf + 3 * olm[k] + 3] = 1
    x0 = np.concatenate([np.concatenate([rv0[free], t[free]], axis=1).ravel(), lm_xyz[used].ravel()])
    r0 = residual(x0)
    sol = least_squares(residual, x0, jac_sparsity=sp, method="trf", x_scale="jac", max_nfev=max_nfev, tr_solver="lsmr", ftol=1e-10, xtol=1e-10, gtol=1e-10)
    rot, tt, Xu = unpack(sol.x)
    X = lm_xyz.copy()
    X[used] = Xu
    res = np.abs(sol.fun)
    info = {"cost_before": float(0.5 * r0 @ r0), "cost_after": float(2 * sol.cost), "nfev": int(sol.nfev), "observations": O, "landmarks": L,
            "max_pose_shift_m": float(np.linalg.norm(tt - t, axis=1).max())}
    return rot.as_matrix(), tt, X, info, res
