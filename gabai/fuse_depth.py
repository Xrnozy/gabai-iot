"""Fuse a noisy stereo disparity with a monocular metric depth model.

Stereo knows distances (from geometry) but is noisy and fails on blank or shiny areas. A monocular model gives clean,
sharp structure but its overall scale is only roughly right. So:
  1. turn the monocular depth into disparity (d = f*B/Z);
  2. fit a robust scale and offset to the stereo pixels we trust, then correct the slow remaining difference with a
     heavily blurred copy of the residual (low-frequency only, so the monocular detail stays);
  3. blend per pixel by inverse variance: where stereo agrees with the aligned monocular map it is trusted, where it is
     noisy the monocular detail wins, where stereo failed the monocular map fills in (flagged `from_mono`).
All in disparity units (pixels), because stereo noise is roughly constant there.
"""
import cv2
import numpy as np


def masked_blur(x, mask, sigma):
    """Gaussian blur using only pixels where mask is True (normalised convolution). Wide blurs are computed at a
    quarter of the resolution, which is indistinguishable for such a smooth result and about 10x faster."""
    if sigma >= 16:
        h, w = x.shape
        sm = (max(w // 4, 1), max(h // 4, 1))
        xs = cv2.resize(np.where(mask, x, 0).astype(np.float32), sm, interpolation=cv2.INTER_AREA)
        ms = cv2.resize(mask.astype(np.float32), sm, interpolation=cv2.INTER_AREA)
        num = cv2.GaussianBlur(xs, (0, 0), sigma / 4)
        den = cv2.GaussianBlur(ms, (0, 0), sigma / 4)
        return cv2.resize(num / np.maximum(den, 1e-6), (w, h), interpolation=cv2.INTER_LINEAR)
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(np.where(mask, x, 0).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def fit_affine(dm, ds, trusted, max_pixels=20000, iters=5, huber=1.5):
    """Robust ds ~ a*dm + b over trusted pixels (Huber-weighted least squares, solved with the 2x2 normal equations)."""
    if dm.ndim == 2:
        dm, ds, trusted = dm[::2, ::2], ds[::2, ::2], trusted[::2, ::2]   # a quarter of the pixels is plenty for 2 numbers
    dm2, ds2, tr2 = dm, ds, trusted
    idx = np.flatnonzero(tr2 & (ds2 > 1) & np.isfinite(ds2) & np.isfinite(dm2))
    if len(idx) < 50:
        return 1.0, 0.0
    if len(idx) > max_pixels:
        idx = idx[:: len(idx) // max_pixels]                                  # an even stride is as good as a random sample
    x, y = dm2.ravel()[idx].astype(np.float64), ds2.ravel()[idx].astype(np.float64)
    w = np.ones_like(x)
    a, b = 1.0, 0.0
    for _ in range(iters):
        sw, sx, sy = w.sum(), (w * x).sum(), (w * y).sum()
        sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
        den = sw * sxx - sx * sx
        if abs(den) < 1e-9:
            break
        a, b = (sw * sxy - sx * sy) / den, (sxx * sy - sx * sxy) / den
        w = 1.0 / np.maximum(1.0, np.abs(y - (a * x + b)) / huber)
    return float(a), float(b)


def low_correction(dm, ds, trusted, a, b, sigma=40.0):
    """Slowly varying part of (stereo - scaled monocular), from trusted pixels only."""
    r = ds - (a * dm + b)
    if not trusted.any():
        return np.zeros_like(r)
    tr_vals = np.abs(r[trusted])
    ok = trusted & np.isfinite(r) & (np.abs(r) < 4.0 * max(np.median(tr_vals[:: max(len(tr_vals) // 20000, 1)]) * 1.4826, 0.5))
    return masked_blur(r, ok, sigma) if ok.any() else np.zeros_like(r)


def align_mono(dm, ds, trusted, sigma=40.0, prior=None, low_prior=None):
    """Monocular disparity brought onto the stereo scale. Returns (aligned map, {'a','b','low'}). `prior` = (a, b) and
    `low_prior` = a correction map, to use instead of fitting this frame (the DepthFuser passes smoothed ones)."""
    a, b = fit_affine(dm, ds, trusted) if prior is None else prior
    low = low_correction(dm, ds, trusted, a, b, sigma) if low_prior is None else low_prior
    return (a * dm + b + low).astype(np.float32), {"a": a, "b": b, "low": low}


def fuse(ds, trusted, dm, sigma_m=0.5, rel_m=0.01, floor_s=0.3, sigma_local=8.0, prior=None, low_prior=None):
    """Returns (fused disparity, info). `ds` stereo disparity (px), `trusted` its validity mask, `dm` monocular
    disparity (px, any scale). info: a, b, weight_stereo (0..1 per pixel), from_mono (bool: no stereo support)."""
    ds = np.where(np.isfinite(ds), ds, 0).astype(np.float32)
    trusted = trusted & (ds > 0.5)
    aligned, info = align_mono(dm, ds, trusted, prior=prior, low_prior=low_prior)
    # Each map's own noise is read from its own high-frequency content (robust L1, so edges barely count). Comparing
    # stereo to the aligned monocular map instead would blame stereo for the monocular model's noise too.
    # Noise estimates are slowly varying, so they are computed at half resolution and upsampled (about 4x cheaper).
    h, wd = ds.shape
    ds_h = np.ascontiguousarray(ds[::2, ::2])      # decimate, do not average: averaging would hide the very noise we measure
    tr_h = np.ascontiguousarray(trusted[::2, ::2])
    half = (ds_h.shape[1], ds_h.shape[0])
    hf_s = np.abs(ds_h - cv2.medianBlur(ds_h, 5))
    if trusted.any():
        sig_s = np.maximum(1.2533 * masked_blur(hf_s, tr_h, sigma_local / 2), floor_s)
        sig_s = cv2.resize(sig_s, (wd, h), interpolation=cv2.INTER_LINEAR)
    else:
        sig_s = np.full_like(ds, 1e3)
    al_q = np.ascontiguousarray(aligned[::4, ::4])
    sig_m_seen = 1.2533 * float(np.median(np.abs(al_q - cv2.medianBlur(al_q, 5))))
    var_s = sig_s ** 2
    var_m = np.maximum(sigma_m + rel_m * np.abs(aligned), sig_m_seen) ** 2
    w = np.where(trusted, var_m / (var_m + var_s), 0.0).astype(np.float32)   # stereo weight
    fused = w * ds + (1 - w) * aligned
    info.update(weight_stereo=w, from_mono=~trusted, aligned=aligned)
    return fused.astype(np.float32), info


def warp_right_to_left(right_map, disp_left):
    """Sample a right-view map at x - d for every left-view pixel, so it describes the same scene points as the left view.
    Pixels whose match falls outside the right image, or that have no disparity, come back as NaN."""
    h, w = disp_left.shape
    xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    ys = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    mx = xs - disp_left.astype(np.float32)
    out = cv2.remap(right_map.astype(np.float32), mx, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    bad = ~np.isfinite(disp_left) | (mx < 0) | (mx > w - 1)
    out[bad] = np.nan
    return out


def combine_mono_pair(dm_left, dm_right_in_left, tol=3.0):
    """Average the two monocular estimates where they agree (noise falls by sqrt 2); where they disagree by more than tol
    pixels keep the left view's own and mark the pixel as disagreeing. Returns (combined disparity, agree mask)."""
    r = dm_right_in_left
    valid = np.isfinite(r)
    agree = valid & (np.abs(dm_left - np.where(valid, r, 0)) <= tol)
    return np.where(agree, 0.5 * (dm_left + np.where(valid, r, 0)), dm_left).astype(np.float32), agree


class DepthFuser:
    """fuse() with memory. In a dark room most stereo pixels are rejected, so the scale fit each frame rests on a few
    pixels and jumps around, which made the monocular map flicker by pixels. Here the scale and offset are averaged
    over time (alpha = how fast it follows), and the fit is only trusted when enough pixels support it. A sudden large
    change (a new scene, a model glitch) restarts the average instead of being smoothed into mush."""

    def __init__(self, alpha=0.25, min_pixels=300, jump=0.35):
        self.alpha, self.min_pixels, self.jump = alpha, min_pixels, jump
        self.ab = None
        self.low = None

    def _track(self, ds, good, dm):
        """Update the smoothed scale/offset and low-frequency correction from this frame's trusted pixels. Returns (prior, low, reset)."""
        reset = False
        if good.sum() >= self.min_pixels:
            a, b = fit_affine(dm, ds, good)
            if self.ab is None:
                self.ab = (a, b)
            else:
                pa, pb = self.ab
                if abs(a - pa) > self.jump * abs(pa):
                    reset, self.ab = True, (a, b)
                else:
                    self.ab = (pa + self.alpha * (a - pa), pb + self.alpha * (b - pb))
        prior = self.ab
        if prior is not None and good.sum() >= self.min_pixels:
            raw = low_correction(dm, ds, good, prior[0], prior[1])
            self.low = raw.copy() if (self.low is None or reset) else self.low + self.alpha * (raw - self.low)
        return prior, self.low, reset

    def fuse(self, ds, trusted, dm, **kw):
        good = trusted & np.isfinite(ds) & (ds > 1)
        prior, low, reset = self._track(ds, good, dm)
        fused, info = fuse(ds, trusted, dm, prior=prior, low_prior=low, **kw)
        info["reset"] = reset
        return fused, info

    def fill(self, ds, trusted, dm, max_resid_px=1.5, min_disp=1.5, min_corr=0.1):
        """Monocular disparity on the stereo scale for the pixels stereo did not measure, and nothing else. Returns (aligned disparity, fill mask, info).
        Every trusted stereo pixel stays exactly as it is: the monocular map never replaces or blends into a measurement. The fill is refused (empty mask)
        when too few trusted pixels support the scale fit, or when the aligned monocular map disagrees with those pixels by more than max_resid_px (median),
        which means the model misread this picture and its structure cannot be trusted at the holes either. It is also refused when the two maps are uncorrelated
        (the smooth correction alone can match a smooth stereo scene, so a monocular map with no structure would pass the residual test)."""
        ds = np.where(np.isfinite(ds), ds, 0).astype(np.float32)
        good = trusted & (ds > 1)
        prior, low, reset = self._track(ds, good, dm)
        none = np.zeros(ds.shape, bool)
        if prior is None or low is None or good.sum() < self.min_pixels:
            return ds, none, {"ok": False, "why": f"only {int(good.sum())} trusted stereo pixels to fit the scale (need {self.min_pixels})", "reset": reset}
        aligned, _ = align_mono(dm, ds, good, prior=prior, low_prior=low)
        resid = float(np.median(np.abs(aligned - ds)[good]))
        x, y = dm[good][::7].astype(np.float64), ds[good][::7].astype(np.float64)
        corr = float(np.corrcoef(x, y)[0, 1]) if y.std() > 3.0 and x.std() > 1e-6 else 1.0           # does the monocular map carry the scene's structure at all? A flat scene has none to test
        why = "" if resid <= max_resid_px and corr >= min_corr else (f"aligned monocular map is {resid:.2f} px from stereo (limit {max_resid_px})" if resid > max_resid_px else
                                                                      f"monocular map and stereo are uncorrelated (r = {corr:.2f}, need {min_corr})")
        info = {"ok": not why, "a": prior[0], "b": prior[1], "resid_px": resid, "corr": corr, "reset": reset, "why": why}
        if not info["ok"]:
            return aligned, none, info
        fill = ~good & np.isfinite(aligned) & (aligned >= min_disp)
        info["filled_px"] = int(fill.sum())
        return aligned, fill, info


class TemporalFilter:
    """Per-pixel gated temporal average of the stereo disparity. On a steady scene it averages roughly 1/alpha frames, so
    sensor noise drops by about sqrt of that; when a pixel's value jumps by more than the gate (something moved, the
    rig moved, a new obstacle) it takes the new value at once, so real changes are never smeared. Pixels the stereo
    rejects keep their last good value for `hold` frames (so flicker in the rejection mask does not blink holes) and
    then become unknown (NaN), never stale. Call reset() or pass alpha=1 while the rig turns fast."""

    def __init__(self, alpha=0.3, gate_px=1.5, gate_rel=0.015, hold=2, moving_frac=0.3):
        self.alpha, self.gate_px, self.gate_rel, self.hold = alpha, gate_px, gate_rel, hold
        self.moving_frac = moving_frac     # if this share of the overlapping pixels jumped, the whole view moved
        self.moving = False
        self.max_sigma = 1.0               # px: the noisiest dark scene measured so far flickers about 1.0 px
        self.reset()

    def reset(self):
        self.ema = None
        self.age = None
        self.sigma = None   # running estimate of the stereo noise, so the gate adapts to a noisy dark room

    def update(self, d, ok, alpha=None):
        a = self.alpha if alpha is None else alpha
        d = np.asarray(d, np.float32)
        valid = np.asarray(ok, bool) & np.isfinite(d) & (d > 0.5)
        if self.ema is None:
            self.ema = np.where(valid, d, np.nan).astype(np.float32)
            self.age = np.where(valid, 0, 99).astype(np.int16)
            return self.ema.copy()
        prev = self.ema
        known = np.isfinite(prev)
        both = known & valid
        if both.sum() > 200:
            # robust: moving parts barely count; capped, because when the whole view moves this would learn the motion as "noise"
            seen = min(1.4826 * float(np.median(np.abs(d - prev)[both])), self.max_sigma)
            self.sigma = seen if self.sigma is None else self.sigma + 0.2 * (seen - self.sigma)
        gate = max(self.gate_px, 3.0 * (self.sigma or 0.0)) + self.gate_rel * np.abs(np.where(known, prev, 0))
        close = both & (np.abs(d - prev) <= gate)
        n_both = int(both.sum())
        self.moving = n_both > 200 and 1.0 - close.sum() / n_both > self.moving_frac
        if self.moving:                    # the rig is moving or walking: averaging would only smear, take the new frame
            a = 1.0
        new = np.where(valid, np.where(close, prev + a * (d - prev), d), prev)
        self.age = np.where(valid, 0, self.age + 1).astype(np.int16)
        hold = 0 if (self.moving or a >= 1.0) else self.hold      # while moving, a held value is a trail behind the object
        new = np.where(self.age > hold, np.nan, new)
        self.ema = new.astype(np.float32)
        return self.ema.copy()


def mono_veto(ds, trusted, dm, tol_px=6.0, tol_rel=0.25, sigma=40.0):
    """Use the monocular depth only as a referee, never as a source of depth. The monocular map (disparity `dm`) is put on
    the stereo scale, and a trusted stereo pixel that disagrees with it by more than max(tol_px, tol_rel * disparity) is
    marked untrusted: that is a gross mismatch (repeating bars, a saturated near object), not noise. Agreeing pixels keep
    their pure stereo value, so the monocular map's own flicker never reaches the output.
    Returns (trusted after the veto, vetoed mask, info of the alignment)."""
    aligned, info = align_mono(dm, ds, trusted, sigma=sigma)
    bad = trusted & (np.abs(ds - aligned) > np.maximum(tol_px, tol_rel * np.abs(aligned)))
    return trusted & ~bad, bad, info
