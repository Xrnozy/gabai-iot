"""Target-free stereo self-calibration: the geometry.

Unknowns (nothing taken from a previous calibration):
  per camera  f (square pixels), cx, cy, k1, k2  (OpenCV radial model; tangential and k3 are not observable enough from natural scenes and stay 0)
  stereo      rotation of the right camera and the DIRECTION of the baseline (2 angles). Its LENGTH is not observable from pictures: every solution here is in baseline units; metres need a measured
              baseline (see __main__ --baseline-mm), and without one the result is saved as not metric.
  per frame   the pose of the left camera (the first frame is the origin), and one 3D point per track.

Pipeline
  1. observations: tracks of features seen in the left and right picture of still, sharp frames and across frames (matching.py builds them)
  2. focal length: from fundamental matrices alone, by the essential-matrix criterion (Mendonca and Cipolla): with the right f, E = K^T F K has two equal singular values. Principal point at the picture
     centre for this step only. A flat cost curve means the views do not constrain f: refused, not guessed.
  3. initial stereo rotation and baseline direction from the pooled left-right essential matrix; frame poses incrementally by PnP on stereo-triangulated points
  4. bundle adjustment of everything with a robust (Huber) reprojection loss, sparse Jacobian
  5. uncertainty: covariance of the camera parameters from the Schur complement of J^T J (points marginalised), scaled by the residual variance
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

CAM = ["f", "cx", "cy", "k1", "k2"]
NC = len(CAM)


# ---- camera model ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def project(Xc, c):
    """Xc (n,3) in the camera frame, c = (f, cx, cy, k1, k2) -> (n,2) pixels."""
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    r2 = x * x + y * y
    d = 1 + c[3] * r2 + c[4] * r2 * r2
    return np.stack([c[0] * x * d + c[1], c[0] * y * d + c[2]], 1)


def K_of(c):
    return np.array([[c[0], 0, c[1]], [0, c[0], c[2]], [0, 0, 1.0]])


def D_of(c):
    return np.array([c[3], c[4], 0.0, 0.0, 0.0])


def tdir(a, e):
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


# ---- observations ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Obs:
    """Flat arrays, one row per observation: frame index, side (0 left, 1 right), pixel, track (point) index."""

    def __init__(self, frame, side, xy, point, n_frames, size):
        self.frame, self.side, self.xy, self.point = (np.asarray(frame, int), np.asarray(side, int), np.asarray(xy, float), np.asarray(point, int))
        self.n_frames, self.n_points, self.size = int(n_frames), int(self.point.max() + 1) if len(self.point) else 0, tuple(size)

    def subset(self, keep):
        """Keep some observations; points and frames are re-indexed compactly. Returns (Obs, old frame index of each new frame)."""
        f, s, xy, p = self.frame[keep], self.side[keep], self.xy[keep], self.point[keep]
        uf, f2 = np.unique(f, return_inverse=True)
        _, p2 = np.unique(p, return_inverse=True)
        return Obs(f2, s, xy, p2, len(uf), self.size), uf

    def pairs(self, a_frame, a_side, b_frame, b_side):
        """Pixel pairs of points seen in both (a_frame, a_side) and (b_frame, b_side)."""
        ma = (self.frame == a_frame) & (self.side == a_side)
        mb = (self.frame == b_frame) & (self.side == b_side)
        pa = dict(zip(self.point[ma], self.xy[ma]))
        common = [(pa[p], q) for p, q in zip(self.point[mb], self.xy[mb]) if p in pa]
        if not common:
            return np.zeros((0, 2)), np.zeros((0, 2))
        a, b = zip(*common)
        return np.array(a), np.array(b)


# ---- 2. focal length from fundamental matrices ------------------------------------------------------------------------------------------------------------------------------------------------
def focal_from_fundamentals(Fs, size, lo=200.0, hi=2000.0, weights=None):
    """Mendonca-Cipolla: the f that makes every K^T F K closest to an essential matrix. Returns (f, cost curve (f values, costs), well_determined)."""
    w, h = size
    fs = np.geomspace(lo, hi, 300)
    wts = np.ones(len(Fs)) if weights is None else np.asarray(weights, float)

    def cost(f):
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        c = 0.0
        for F, wt in zip(Fs, wts):
            s = np.linalg.svd(K.T @ F @ K, compute_uv=False)
            c += wt * (s[0] - s[1]) / max(s[0] + s[1], 1e-12)
        return c / wts.sum()
    cs = np.array([cost(f) for f in fs])
    i = int(np.argmin(cs))
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lf: cost(np.exp(lf)), bounds=(np.log(fs[max(i - 3, 0)]), np.log(fs[min(i + 3, len(fs) - 1)])), method="bounded")
    f = float(np.exp(r.x))
    # well determined: the minimum is inside the range and clearly lower than the cost 25% away on both sides
    side = [cost(f * 0.75), cost(f * 1.25)]
    well = (fs[0] * 1.05 < f < fs[-1] / 1.05) and min(side) > 1.5 * cost(f) + 1e-4
    return f, (fs, cs), bool(well)


# ---- 3. initialisation ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _undist(xy, c):
    return cv2.undistortPoints(xy.reshape(-1, 1, 2).astype(np.float64), K_of(c), D_of(c)).reshape(-1, 2)


def stereo_init(obs, cL, cR, seed=0):
    """Pooled left-right correspondences of all frames -> (R, unit t, inlier ratio). The rig is rigid, so all frames share one geometry.
    The baseline (5 cm) is small against the scene (1-4 m), so an essential matrix leaves its direction almost free (tried: 55 deg off on a synthetic rig) and the bundle adjustment then starts in the wrong
    basin. Instead: the rotation from a robust rotation-only fit of the bearing vectors (parallax biases it by about baseline / depth, which the adjustment removes), and the baseline along the camera's
    x axis, the one fact of the rig's design used here; its SIGN comes from the data (where right-picture points sit relative to left ones), so swapped cameras are found, not assumed."""
    A, B = [], []
    for i in range(obs.n_frames):
        a, b = obs.pairs(i, 0, i, 1)
        A.append(a); B.append(b)
    A, B = np.concatenate(A), np.concatenate(B)
    na, nb = (np.c_[_undist(p, c), np.ones(len(p))] for p, c in ((A, cL), (B, cR)))
    na /= np.linalg.norm(na, axis=1, keepdims=True)
    nb /= np.linalg.norm(nb, axis=1, keepdims=True)
    rng, best, tol = np.random.default_rng(seed), None, 0.03                       # 0.03 rad: covers parallax up to 5 cm at 1.7 m
    for _ in range(300):
        i = rng.choice(len(na), 3, replace=False)
        U, _, Vt = np.linalg.svd(nb[i].T @ na[i])
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        inl = np.linalg.norm(nb - na @ R.T, axis=1) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    U, _, Vt = np.linalg.svd(nb[best].T @ na[best])
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    # then rotation and baseline direction together by the epipolar (Sampson) error, starting from that rotation and a baseline along x: the parallax goes back into t
    a, b = na[best] / na[best, 2:3], nb[best] / nb[best, 2:3]

    def sampson(x):
        Rx, t = cv2.Rodrigues(x[:3])[0], np.array([1.0, 0.0, 0.0])             # direction held on the rig's x axis here: with a still-wrong focal length a free direction slides toward
                                                                                    # forward motion (tried: 45 deg off); the bundle adjustment frees it once f is right
        E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ Rx
        Ea, Etb = a @ E.T, b @ E
        return np.sum(b * Ea, 1) / np.sqrt(Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2 + 1e-18)
    r = least_squares(sampson, cv2.Rodrigues(R)[0].ravel(), loss="huber", f_scale=1.0 / cL[0])
    R, t = cv2.Rodrigues(r.x[:3])[0], np.array([1.0, 0.0, 0.0])
    # the epipolar error cannot tell t from -t; the side where points land in front of both cameras can
    front = []
    for s in (1.0, -1.0):
        X = cv2.triangulatePoints(np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R, s * t.reshape(3, 1)]), a[:, :2].T, b[:, :2].T)
        X = (X[:3] / X[3]).T
        front.append(np.mean((X[:, 2] > 0) & ((X @ R.T + s * t)[:, 2] > 0)))
    t = t if front[0] >= front[1] else -t
    return R, t, float(best.mean())


def poses_init(obs, cL, cR, Rs, ts, min_inliers=20, pnp_px=8.0, min_disp_px=4.0, gate_px=15.0):
    """Left-camera poses (rvec, tvec, world -> camera) by greedy PnP on points triangulated from each posed frame's stereo pair. Returns (poses dict, points dict)."""
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([Rs, ts.reshape(3, 1)])

    def stereo_points(i):
        ml, mr = (obs.frame == i) & (obs.side == 0), (obs.frame == i) & (obs.side == 1)
        pl = dict(zip(obs.point[ml], obs.xy[ml]))
        ids = [p for p in obs.point[mr] if p in pl]
        if not ids:
            return {}
        pr = dict(zip(obs.point[mr], obs.xy[mr]))
        a, b = _undist(np.array([pl[p] for p in ids]), cL), _undist(np.array([pr[p] for p in ids]), cR)
        X = cv2.triangulatePoints(P1, P2, a.T, b.T)
        X = (X[:3] / X[3]).T
        ok = (X[:, 2] > 0.5) & (X[:, 2] < cL[0] / min_disp_px)                    # in front, and with enough stereo parallax to have a depth worth using (baseline units)
        return {p: x for p, x, o in zip(ids, X, ok) if o}

    counts = np.bincount(obs.frame, minlength=obs.n_frames)
    ref = int(np.argmax(counts))
    poses = {ref: (np.zeros(3), np.zeros(3))}
    pts = dict(stereo_points(ref))
    todo = set(range(obs.n_frames)) - {ref}
    while todo:
        best, best_n = None, 0
        for i in todo:
            m = (obs.frame == i) & (obs.side == 0)
            n = sum(1 for p in obs.point[m] if p in pts)
            if n > best_n:
                best, best_n = i, n
        if best is None or best_n < min_inliers:
            break
        m = (obs.frame == best) & (obs.side == 0)
        ids = [p for p in obs.point[m] if p in pts]
        xy = dict(zip(obs.point[m], obs.xy[m]))
        ok, rv, tv, inl = cv2.solvePnPRansac(np.array([pts[p] for p in ids]), np.array([xy[p] for p in ids]), K_of(cL), D_of(cL), reprojectionError=pnp_px, iterationsCount=500)
        todo.discard(best)
        if not ok or inl is None or len(inl) < min_inliers:
            continue
        poses[best] = (rv.ravel(), tv.ravel())
        R = cv2.Rodrigues(rv)[0]
        for p, x in stereo_points(best).items():                                # new points, camera -> world
            if p not in pts:
                pts[p] = R.T @ (x - tv.ravel())
    # every point again from ALL its observations in placed frames (frame-to-frame baselines are many times the stereo one): DLT on undistorted rays, kept if in front of every camera
    Rs_, ts_ = Rs, ts.ravel()
    cams = {}
    for f, (rv, tv) in poses.items():
        R = cv2.Rodrigues(rv)[0]
        cams[(f, 0)] = np.hstack([R, tv.reshape(3, 1)])
        cams[(f, 1)] = np.hstack([Rs_ @ R, (Rs_ @ tv + ts_).reshape(3, 1)])
    sel = np.isin(obs.frame, list(poses))
    und = np.zeros_like(obs.xy)
    for side, c in ((0, cL), (1, cR)):
        m = sel & (obs.side == side)
        und[m] = _undist(obs.xy[m], c)
    order = np.argsort(obs.point[sel], kind="stable")
    idx = np.flatnonzero(sel)[order]
    pid = obs.point[idx]
    starts = np.r_[0, np.flatnonzero(np.diff(pid)) + 1, len(pid)]
    out = {}
    for a, b in zip(starts[:-1], starts[1:]):
        rows = idx[a:b]
        if len(rows) < 2:
            continue
        A = []
        for r in rows:
            Pm = cams[(obs.frame[r], obs.side[r])]
            u, v = und[r]
            A += [u * Pm[2] - Pm[0], v * Pm[2] - Pm[1]]
        Xh = np.linalg.svd(np.array(A))[2][-1]
        if abs(Xh[3]) < 1e-12:
            continue
        X = Xh[:3] / Xh[3]
        Xc = np.array([cams[(obs.frame[r], obs.side[r])] @ np.r_[X, 1] for r in rows])
        if (Xc[:, 2] <= 0.3).any():
            continue
        side = obs.side[rows]
        proj = np.where(side[:, None] == 1, project(Xc, cR), project(Xc, cL))
        if np.linalg.norm(proj - obs.xy[rows], axis=1).max() < gate_px:            # a nearly-parallel-ray point can be anywhere along the ray: kept only if it explains every view
            out[int(pid[a])] = X
    return ref, poses, out


# ---- 4. bundle adjustment -----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Problem:
    """x = [cam L (5), cam R (5), stereo rvec (3), baseline angles (2), poses of frames except ref (6 each), points (3 each)]."""

    def __init__(self, obs, ref):
        self.obs, self.ref = obs, ref
        self.frames = [i for i in range(obs.n_frames) if i != ref]
        self.fidx = {f: k for k, f in enumerate(self.frames)}
        self.o_pose = 2 * NC + 5
        self.o_pts = self.o_pose + 6 * len(self.frames)
        self.n = self.o_pts + 3 * obs.n_points

    def pack(self, cL, cR, Rs, ts, poses, pts):
        a = np.arctan2(ts[1], ts[0])
        e = np.arcsin(np.clip(ts[2], -1, 1))
        x = np.zeros(self.n)
        x[:NC], x[NC:2 * NC] = cL, cR
        x[2 * NC:2 * NC + 3] = cv2.Rodrigues(Rs)[0].ravel()
        x[2 * NC + 3:2 * NC + 5] = a, e
        for f, k in self.fidx.items():
            x[self.o_pose + 6 * k:self.o_pose + 6 * k + 6] = np.r_[poses[f][0], poses[f][1]]
        x[self.o_pts:] = np.asarray(pts).ravel()
        return x

    def unpack(self, x):
        cL, cR = x[:NC], x[NC:2 * NC]
        Rs = cv2.Rodrigues(x[2 * NC:2 * NC + 3])[0]
        ts = tdir(*x[2 * NC + 3:2 * NC + 5])
        poses = {self.ref: (np.zeros(3), np.zeros(3))}
        for f, k in self.fidx.items():
            v = x[self.o_pose + 6 * k:self.o_pose + 6 * k + 6]
            poses[f] = (v[:3], v[3:])
        return cL, cR, Rs, ts, poses, x[self.o_pts:].reshape(-1, 3)

    def residuals(self, x):
        cL, cR, Rs, ts, poses, X = self.unpack(x)
        o = self.obs
        out = np.empty((len(o.frame), 2))
        for f in range(o.n_frames):
            m = o.frame == f
            if not m.any():
                continue
            R = cv2.Rodrigues(poses[f][0])[0]
            Xc = X[o.point[m]] @ R.T + poses[f][1]
            right = o.side[m] == 1
            Xr = Xc @ Rs.T + ts
            proj = np.where(right[:, None], project(Xr, cR), project(Xc, cL))
            out[m] = proj - o.xy[m]
        return out.ravel()

    def sparsity(self):
        o, n_obs = self.obs, len(self.obs.frame)
        A = lil_matrix((2 * n_obs, self.n), dtype=np.int8)
        rows = np.arange(n_obs)
        for r in (2 * rows, 2 * rows + 1):
            for k in range(NC):
                A[r[o.side == 0], k] = 1
                A[r[o.side == 1], NC + k] = 1
            for k in range(5):
                A[r[o.side == 1], 2 * NC + k] = 1
            for f, kf in self.fidx.items():
                m = o.frame == f
                for k in range(6):
                    A[r[m], self.o_pose + 6 * kf + k] = 1
            for k in range(3):
                A[r, self.o_pts + 3 * o.point + k] = 1
        return A


def bundle_adjust(obs, ref, cL, cR, Rs, ts, poses, pts, huber_px=1.5, max_nfev=300, fix=()):
    """Returns dict(cL, cR, Rs, ts, poses, points, rms_px, inlier_rms_px, x, problem, result). fix: names of camera parameters held at their current value in both cameras
    (e.g. ("cx", "cy", "k1", "k2") for the first, better-conditioned stage)."""
    pr = Problem(obs, ref)
    x0 = pr.pack(cL, cR, Rs, ts, poses, pts)
    free = np.ones(pr.n, bool)
    for name in fix:
        free[CAM.index(name)] = free[NC + CAM.index(name)] = False
    sp = pr.sparsity().tocsc()[:, free]

    def fun(xf):
        x = x0.copy()
        x[free] = xf
        return pr.residuals(x)
    r = least_squares(fun, x0[free], jac_sparsity=sp, loss="huber", f_scale=huber_px, x_scale="jac", method="trf", max_nfev=max_nfev, verbose=0)
    xfull = x0.copy()
    xfull[free] = r.x
    r.x = xfull
    cL, cR, Rs, ts, poses, X = pr.unpack(r.x)
    res = r.fun.reshape(-1, 2)
    e = np.linalg.norm(res, axis=1)
    return {"cL": cL, "cR": cR, "Rs": Rs, "ts": ts, "poses": poses, "points": X, "err_px": e, "rms_px": float(np.sqrt(np.mean(e ** 2))),
            "inlier_rms_px": float(np.sqrt(np.mean(e[e < 3 * huber_px] ** 2))), "x": r.x, "problem": pr, "result": r}


# ---- 5. uncertainty ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def camera_std(ba, inlier_px=4.5):
    """1-sigma of [cam L (5), cam R (5), stereo rvec (3, rad), baseline angles (2, rad)] from the Schur complement of J^T J over the inlier observations, points marginalised."""
    pr, r = ba["problem"], ba["result"]
    J = r.jac.tocsr()
    keep_obs = ba["err_px"] < inlier_px
    rows = np.flatnonzero(np.repeat(keep_obs, 2))
    J = J[rows]
    nc = pr.o_pts
    Jc, Jp = J[:, :nc].toarray(), J[:, nc:].tocsc()
    S = Jc.T @ Jc
    pt_of_row = np.repeat(pr.obs.point[keep_obs], 2)
    order = np.argsort(pt_of_row, kind="stable")
    bounds = np.searchsorted(pt_of_row[order], np.arange(pr.obs.n_points + 1))
    for p in range(pr.obs.n_points):
        rr = order[bounds[p]:bounds[p + 1]]
        if len(rr) < 3:
            continue
        Jpp = Jp[rr][:, 3 * p:3 * p + 3].toarray()
        if Jpp.size == 0:
            continue
        C = Jc[rr]
        H = Jpp.T @ Jpp
        if np.linalg.cond(H) > 1e12:
            continue
        W = C.T @ Jpp
        S -= W @ np.linalg.solve(H, W.T)
    sigma2 = float(np.mean(ba["err_px"][keep_obs] ** 2))
    cov = sigma2 * np.linalg.pinv(S)
    return np.sqrt(np.maximum(np.diag(cov)[:2 * NC + 5], 0))


# ---- the whole solve --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Insufficient(RuntimeError):
    """The observations do not determine the calibration: nothing is returned rather than a guess."""


def fundamentals(obs, min_pairs=40):
    """F of the pooled stereo correspondences and of every left-left / right-right frame pair with enough shared points. Returns [(F, inliers)]."""
    out = []
    A, B = zip(*[obs.pairs(i, 0, i, 1) for i in range(obs.n_frames)])
    A, B = np.concatenate(A), np.concatenate(B)
    if len(A) >= min_pairs:
        F, m = cv2.findFundamentalMat(A, B, cv2.FM_RANSAC, 1.0, 0.999)
        if F is not None and F.shape == (3, 3):
            out.append((F, int(m.sum())))
    for side in (0, 1):
        for i in range(obs.n_frames):
            for j in range(i + 1, obs.n_frames):
                a, b = obs.pairs(i, side, j, side)
                if len(a) < min_pairs:
                    continue
                F, m = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.0, 0.999)
                if F is not None and F.shape == (3, 3) and m.sum() >= min_pairs:
                    out.append((F, int(m.sum())))
    return out


def calibrate(obs, log=print, rounds=4, outlier_px=3.0, huber=(30.0, 8.0, 3.0, 1.5)):
    """The full solve. Raises Insufficient when the data do not determine it."""
    w, h = obs.size
    Fs = fundamentals(obs)
    if len(Fs) < 5:
        raise Insufficient(f"only {len(Fs)} usable fundamental matrices (need 5+): too few overlapping still frames")
    f0, curve, well = focal_from_fundamentals([F for F, _ in Fs], obs.size, weights=[n for _, n in Fs])
    log(f"focal initialisation from {len(Fs)} fundamental matrices: f = {f0:.1f} px ({'well determined' if well else 'POORLY DETERMINED'})")
    if not well:
        raise Insufficient("the focal length is not determined by these views (flat essential-matrix cost): they need more rotation between frames")
    cL = cR = np.array([f0, w / 2.0, h / 2.0, 0.0, 0.0])
    Rs, ts, ratio = stereo_init(obs, cL, cR)
    log(f"stereo initialisation: rotation {np.degrees(np.linalg.norm(cv2.Rodrigues(Rs)[0])):.2f} deg, baseline direction {np.round(ts, 3)} ({'right camera to the right' if ts[0] < 0 else 'CAMERAS SWAPPED: right picture comes from the left camera'}), {ratio * 100:.0f}% rotation inliers")
    for k in range(rounds):
        ref, poses, pts = poses_init(obs, cL, cR, Rs, ts)
        stage_fix = ("cx", "cy", "k1", "k2") if k == 0 else ()
        # graduated robust loss: a tight Huber scale on a still-wrong model treats the focal-length error at the picture edges as outliers and never corrects it (seen: f stuck 6% off)
        hk = huber[min(k, len(huber) - 1)]                # first round: focal, rig and structure only; the weakly observed parameters join once those are right
        if len(poses) < 3:
            raise Insufficient(f"only {len(poses)} frames could be placed: too few points shared between frames")
        keep = np.isin(obs.frame, list(poses)) & np.isin(obs.point, list(pts))
        sub, old_f = obs.subset(keep)
        old_p = np.unique(obs.point[keep])
        fmap = {int(f): i for i, f in enumerate(old_f)}
        ba = bundle_adjust(sub, fmap[ref], cL, cR, Rs, ts, {fmap[f]: v for f, v in poses.items() if f in fmap}, np.array([pts[p] for p in old_p]), fix=stage_fix, huber_px=hk)
        log(f"bundle adjustment round {k + 1}: {sub.n_frames} frames, {sub.n_points} points, {len(sub.frame)} observations, rms {ba['rms_px']:.3f} px (inliers {ba['inlier_rms_px']:.3f} px)")
        cL, cR, Rs, ts = ba["cL"], ba["cR"], ba["Rs"], ba["ts"]
        bad = ba["err_px"] > max(outlier_px, 3 * hk)
        if (not bad.any() and k >= 1 and hk <= huber[-1]) or k == rounds - 1:       # done only with every parameter free and the final loss scale
            break
        if not bad.any():
            continue
        drop = np.zeros(len(obs.frame), bool)
        drop[np.flatnonzero(keep)[bad]] = True                                   # drop the outlying observations; points left with < 3 observations go at the next subset
        obs, _ = obs.subset(~drop)
        cnt = np.bincount(obs.point, minlength=obs.n_points)
        obs, _ = obs.subset(cnt[obs.point] >= 3)
    ba["std"] = camera_std(ba)
    ba["obs"] = sub
    return ba
