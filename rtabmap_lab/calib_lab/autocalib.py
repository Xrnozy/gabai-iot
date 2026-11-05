"""Automatic calibration check and refinement from the footage itself: no board, no printed pattern.

What the pictures can and cannot tell. Matched features of an ordinary room constrain the EPIPOLAR geometry of the pair: which rows match. From that, three things are observable
and are estimated here: the pitch and roll of the right camera relative to the left (rx, rz), the direction of the baseline (ty, tz, small) and a focal-length ratio of the right camera.
Three things are NOT observable and are never changed by this module, whatever the data: the baseline LENGTH (a scene has no known size: set it with a tape-measured distance),
the yaw / toe-in (ry: it shifts disparity, not rows) and the principal point x. The solver reports a standard deviation for every estimate from the curvature of the cost and
refuses to propose a number whose uncertainty is too large, so weak data gives 'not enough information', not a confident wrong value.

  stats = epipolar_stats(left_rectified, right_rectified)     live quality: median row error, tilt (roll) and height (scale) slope, edge-vs-centre
  r = AutoCalibration(params, get_pair).run(frames)           dict: verdict, per-parameter proposals with std, the checks that passed and failed, row error before and after (held-out frames)

Moving camera: only matches inside ONE left/right pair are used (no frame-to-frame matching), pairs captured with a large left/right time offset or a blurred picture are skipped.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

from gabai.spike.autorect import match
from rtabmap_lab.calib_lab.params import CalibParams

STD_LIMIT = {"R.rx_deg": 0.05, "R.rz_deg": 0.05, "dir_y_deg": 0.5, "dir_z_deg": 0.5, "s": 0.003}   # largest standard deviation at which an estimate may be proposed
MIN_FRAMES, MIN_POINTS, MIN_CELLS, MIN_DISP_STD, MAX_DT_MS, MAX_RESID_PX = 10, 800, 7, 3.0, 3.0, 0.8     # 0.8 px: SIFT matches of this footage scatter about 0.5 px by themselves


def _gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im


def epipolar_stats(left, right, max_dy=12.0):
    """Row alignment of a RECTIFIED pair from its own features (SIFT + ratio test + fundamental-matrix RANSAC). Returns None when there are too few matches, else a dict:
    n, row_err_px (median |dy|), dy_offset_px (median signed dy: right picture lower is positive), roll_slope (dy per 100 px of x), scale_slope (dy per 100 px of y),
    edge_px / centre_px (median |dy| in the outer / inner thirds of the width), neg_disp_share (matches whose right point is to the right of the left point)."""
    p1, p2 = match(left, right)
    if len(p1) < 30:
        return None
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.5, 0.99)
    if F is None:
        return None
    k = mask.ravel() > 0
    p1, p2 = p1[k], p2[k]
    dy, dx = p2[:, 1] - p1[:, 1], p1[:, 0] - p2[:, 0]
    k = (np.abs(dy) < max_dy) & (np.abs(dx) < 200)
    p1, p2, dy, dx = p1[k], p2[k], dy[k], dx[k]
    if len(p1) < 30:
        return None
    w, h = left.shape[1], left.shape[0]
    A = np.stack([np.ones(len(p1)), (p1[:, 0] - w / 2) / 100.0, (p1[:, 1] - h / 2) / 100.0], 1)
    wgt = np.ones(len(p1))
    for _ in range(6):                                                       # Huber-weighted least squares: a few bad matches do not move the slopes
        coef = np.linalg.lstsq(A * np.sqrt(wgt)[:, None], dy * np.sqrt(wgt), rcond=None)[0]
        r = dy - A @ coef
        s = max(1.4826 * np.median(np.abs(r)), 0.05)
        wgt = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-9))
    edge = np.abs(p1[:, 0] - w / 2) > w / 3
    return {"n": int(len(p1)), "row_err_px": float(np.median(np.abs(dy))), "dy_offset_px": float(coef[0]), "roll_slope": float(coef[1]), "scale_slope": float(coef[2]),
            "edge_px": float(np.median(np.abs(r[edge]))) if edge.sum() > 10 else None, "centre_px": float(np.median(np.abs(r[~edge]))) if (~edge).sum() > 10 else None,
            "neg_disp_share": float((dx < 0).mean()), "disp_median_px": float(np.median(dx))}


def explain_stats(s):
    """Plain-language reading of epipolar_stats: which parameter to touch."""
    if s is None:
        return ["too few matched features in this picture to judge the rows"]
    out = []
    if abs(s["dy_offset_px"]) > 0.25:
        out.append(f"the right picture sits {abs(s['dy_offset_px']):.2f} px {'lower' if s['dy_offset_px'] > 0 else 'higher'} than the left: right-camera pitch (R.rx_deg) or K2.cy")
    if abs(s["roll_slope"]) > 0.15:
        out.append(f"rows drift {s['roll_slope']:+.2f} px per 100 px sideways: right-camera roll (R.rz_deg)")
    if abs(s["scale_slope"]) > 0.15:
        out.append(f"rows drift {s['scale_slope']:+.2f} px per 100 px downward: the two focal lengths (K1.fy / K2.fy) differ")
    if s["neg_disp_share"] > 0.05:
        out.append(f"{s['neg_disp_share'] * 100:.0f}% of the matches have a NEGATIVE disparity: cameras swapped, a big yaw error or a wrong layout")
    if s["edge_px"] is not None and s["centre_px"] is not None and s["edge_px"] > 2 * s["centre_px"] + 0.15:
        out.append(f"the picture edges are worse ({s['edge_px']:.2f} px) than the centre ({s['centre_px']:.2f} px): lens distortion (k1, k2, k3)")
    return out or ["rows line up (no parameter stands out)"]


def _rot(v):
    return cv2.Rodrigues(np.asarray(v, float))[0]


def _skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


class AutoCalibration:
    """get_pair(i) -> (left, right, dt_ms) RAW pictures in the orientation the calibration expects (dt_ms may be None)."""

    def __init__(self, params, get_pair, gate=None):
        """gate(i) -> None, or the reason pair i must not be used (the lab passes the rig-motion check)."""
        self.p, self.get_pair, self.gate = params, get_pair, gate

    # -- 1. matches --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def collect(self, frames, progress=None):
        """Per usable frame: the matches that agree with SOME epipolar geometry (essential-matrix RANSAC on the normalised points, whatever the current calibration says), as
        normalised left / right points, and their pixel place in the left picture. Returns (list of dicts, list of skip reasons)."""
        K1, D1, K2, D2, R, T = self.p.arrays()
        fm = 0.5 * (K1[0, 0] + K2[0, 0])
        out, skipped = [], []
        sharp = {}
        for n, i in enumerate(frames):
            if progress:
                progress(n / max(len(frames), 1), f"matching pair {n + 1}/{len(frames)}")
            why = self.gate(i) if self.gate else None
            if why:
                skipped.append((i, why))
                continue
            l, r, dt = self.get_pair(i)
            if dt is not None and abs(dt) > MAX_DT_MS:
                skipped.append((i, f"left/right capture offset {dt:.1f} ms"))
                continue
            sharp[i] = cv2.Laplacian(_gray(l), cv2.CV_64F).var()
            a, b = match(l, r, 1500)
            if len(a) < 40:
                skipped.append((i, f"only {len(a)} matches"))
                continue
            na = cv2.undistortPoints(a.reshape(-1, 1, 2).astype(np.float64), K1, D1).reshape(-1, 2)
            nb = cv2.undistortPoints(b.reshape(-1, 1, 2).astype(np.float64), K2, D2).reshape(-1, 2)
            E, m = cv2.findEssentialMat(na, nb, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=1.5 / fm)
            if E is None or m is None or m.sum() < 30:
                skipped.append((i, "no consistent geometry among the matches"))
                continue
            k = m.ravel() > 0
            out.append({"i": i, "na": na[k], "nb": nb[k], "px": a[k], "disp": a[k, 0] - b[k, 0], "sharp": sharp[i]})
        if sharp:                                                            # drop the blurriest quarter: smeared features carry a vertical error of their own
            lim = 0.5 * float(np.median(list(sharp.values())))
            keep = []
            for o in out:
                (keep if o["sharp"] >= lim else skipped).append(o if o["sharp"] >= lim else (o["i"], "blurred"))
            out = keep
        return out, skipped

    # -- 2. solve ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def solve(self, data):
        """Fit (dRx, dRy, dRz, dir_y, dir_z, log s) to the pooled correspondences by minimising the Sampson distance in pixels. Returns (theta, std, rms_px, n)."""
        K1, D1, K2, D2, R0, T0 = self.p.arrays()
        T0 = T0.ravel()
        fm = 0.5 * (K1[0, 0] + K2[0, 0])
        xl = np.concatenate([np.c_[d["na"], np.ones(len(d["na"]))] for d in data])
        xr0 = np.concatenate([np.c_[d["nb"], np.ones(len(d["nb"]))] for d in data])
        t0 = T0 / np.linalg.norm(T0)

        def residual(th):
            R = _rot(th[:3]) @ R0
            t = t0 + np.array([0.0, np.radians(th[3]), np.radians(th[4])])
            t = t / np.linalg.norm(t)
            xr = xr0 * np.array([1 / np.exp(th[5]), 1 / np.exp(th[5]), 1.0])
            E = _skew(t) @ R
            Ex, Etx = xl @ E.T, xr @ E
            num = np.einsum("ij,ij->i", xr, Ex)
            den = np.sqrt(Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2) + 1e-12
            return num / den * fm

        th0 = np.zeros(6)
        sol = least_squares(residual, th0, loss="soft_l1", f_scale=0.5, x_scale=[np.radians(0.1)] * 3 + [0.1, 0.1, 0.001], diff_step=1e-6)
        r = sol.fun
        inl = np.abs(r) < 1.5
        J = sol.jac[inl]
        sigma2 = float(np.sum(r[inl] ** 2) / max(inl.sum() - 6, 1))
        try:
            cov = np.linalg.inv(J.T @ J) * sigma2
            std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            std = np.full(6, np.inf)
        return sol.x, std, float(np.sqrt(np.mean(r[inl] ** 2))), int(inl.sum()), residual

    # -- 3. the whole check ------------------------------------------------------------------------------------------------------------------------------------------------------------
    def run(self, frames, progress=None):
        frames = list(frames)
        data, skipped = self.collect(frames, progress)
        res = {"frames_asked": len(frames), "frames_used": len(data), "skipped": skipped, "checks": [], "proposed": {}, "std": {}, "unobservable": [], "verdict": "INSUFFICIENT DATA",
               "note": "Not estimated, by design: the baseline length (no known size in a room: use a tape-measured distance), the yaw R.ry_deg and the principal point x."}
        chk = res["checks"]
        npts = sum(len(d["na"]) for d in data)
        chk.append(("enough pairs", len(data) >= MIN_FRAMES, f"{len(data)} usable pairs of {len(frames)} (need {MIN_FRAMES})"))
        chk.append(("enough matches", npts >= MIN_POINTS, f"{npts} consistent matches (need {MIN_POINTS})"))
        if len(data) < MIN_FRAMES or npts < MIN_POINTS:
            return res
        w, h = self.p.size
        px = np.concatenate([d["px"] for d in data])
        cells = np.zeros((3, 4), int)
        np.add.at(cells, (np.clip((px[:, 1] / h * 3).astype(int), 0, 2), np.clip((px[:, 0] / w * 4).astype(int), 0, 3)), 1)
        filled = int((cells >= 20).sum())
        chk.append(("covers the picture", filled >= MIN_CELLS, f"{filled} of 12 picture regions hold 20+ matches (need {MIN_CELLS}); a calibration from one corner says nothing about the rest"))
        dstd = float(np.std(np.concatenate([d["disp"] for d in data])))
        res["coverage"] = cells.tolist()
        # like the OAK viewer's dataConfidence: 0..1, the weakest-link average of pairs, matches, picture coverage and depth spread
        res["data_confidence"] = float(np.mean([min(1, len(data) / 30), min(1, npts / 3000), filled / 12, min(1, dstd / (2 * MIN_DISP_STD))]))
        chk.append(("near and far both seen", dstd >= MIN_DISP_STD, f"disparity spread {dstd:.1f} px (need {MIN_DISP_STD}); only one distance cannot separate rotation from the baseline direction"))
        if not all(c[1] for c in chk):
            return res
        order = np.arange(len(data))
        train, test = [data[i] for i in order[::2]], [data[i] for i in order[1::2]]
        th, std, rms, n_in, _ = self.solve(data)
        chk.append(("model fits the matches", rms <= MAX_RESID_PX, f"remaining row error {rms:.2f} px RMS (need <= {MAX_RESID_PX}); more means the lens distortion is wrong, which rotation cannot fix"))
        a, sa, *_ = self.solve(train)
        b, sb, *_ = self.solve(test)
        names = ["R.rx_deg", "R.ry_deg", "R.rz_deg", "dir_y_deg", "dir_z_deg", "s"]
        scale = [180 / np.pi] * 3 + [1, 1, 1]
        est, diag = {}, {}
        for k, nm in enumerate(names):
            e, sd = th[k] * scale[k], std[k] * scale[k]
            half = abs(a[k] - b[k]) * scale[k]
            lim = STD_LIMIT.get(nm)
            res["std"][nm] = float(sd)
            if nm == "R.ry_deg":
                res["unobservable"].append((nm, "yaw shifts disparity, not rows: the epipolar geometry does not see it"))
                continue
            good = sd <= lim and half <= 2 * lim
            if nm in ("R.rx_deg", "R.rz_deg"):                                # pitch and roll: the only ones applied
                chk.append((f"{nm} well determined", good, f"estimate {e:+.3f} deg +- {sd:.3f}; even/odd pairs differ by {half:.3f} (limit +-{lim})"))
                if good:
                    est[nm] = e
                else:
                    res["unobservable"].append((nm, f"uncertainty +-{sd:.3f} / halves differ {half:.3f}: too weak to apply"))
            else:                                                             # baseline direction and right focal scale: shown, never applied (they trade against lens distortion and depth)
                diag[nm] = {"value": float(e), "std": float(sd), "stable": bool(good)}
                res["unobservable"].append((nm, f"estimate {e:+.3f} +- {sd:.3f}: reported only; it trades against lens distortion, so it is not applied"))
        res["diagnostic"] = diag
        p2 = self.p.copy()
        if est:
            R_new = _rot(np.array([est.get("R.rx_deg", 0.0), 0.0, est.get("R.rz_deg", 0.0)]) * np.pi / 180.0) @ self.p.arrays()[4]
            rv = cv2.Rodrigues(R_new)[0].ravel() * 180 / np.pi
            p2.values.update({"R.rx_deg": rv[0], "R.ry_deg": rv[1], "R.rz_deg": rv[2]})   # composing the rotations moves ry by a hair; the yaw itself was not estimated
            p2._rect = None
        res["estimates"], res["row_rms_px"] = {k: float(v) for k, v in est.items()}, rms
        # an independent judge: the real rectified pictures of held-out pairs, rows measured by their own features, with the current numbers and with the proposed ones
        cur, new = [], []
        for d in test[:8]:
            l, r, _ = self.get_pair(d["i"])
            for par, out in ((self.p, cur), (p2, new)):
                st = epipolar_stats(*par.rect().apply(l, r))
                if st:
                    out.append(st["row_err_px"])
        b0, b1 = (float(np.median(cur)) if cur else float("nan")), (float(np.median(new)) if new else float("nan"))
        res["row_err_current_px"], res["row_err_proposed_px"] = b0, b1
        if np.isfinite(b0):                                                   # like calibrationConfidence: 1 when held-out rows sit at the matching noise (~0.5 px), 0 at 2 px or worse
            res["calibration_confidence"] = float(np.clip(1 - (b0 - 0.5) / 1.5, 0, 1))
        helps = bool(est) and np.isfinite(b0) and b1 <= 0.9 * b0
        chk.append(("the change improves held-out pairs", helps or not est or b0 <= MAX_RESID_PX, f"median row error of 8 pairs not used for the fit: {b0:.2f} px now, {b1:.2f} px with the change" + ("" if helps else " (no gain: nothing to improve)")))
        big = max([abs(v) for v in est.values()] or [0.0])
        chk.append(("the correction is small", big <= 1.5, f"largest rotation change {big:.2f} deg (over 1.5 deg usually means a wrong layout or a moved camera, not a refinement)"))
        hard = [c for c in chk if "well determined" not in c[0]]
        if not all(c[1] for c in hard):
            res["verdict"] = "UNRELIABLE (not applied)"
        elif not est:
            res["verdict"] = "INSUFFICIENT DATA"
        elif not helps:
            res["verdict"] = f"CONFIRMED: the current calibration already fits (rows {b0:.2f} px, about the matching noise)"
        else:
            res["verdict"] = "RELIABLE" if len(est) == 2 else "PARTIAL"
        if res["verdict"] in ("RELIABLE", "PARTIAL"):
            res["params"] = p2
            res["proposed"] = {n: float(p2.values[n]) for n in p2.changed(self.p, tol=1e-9)}
        return res
