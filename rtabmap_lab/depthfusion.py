"""Confidence-aware multi-source depth fusion: several stereo depth maps and two monocular ones in, ONE depth map and ONE confidence map out, with a record of what every pixel is.

Rules (nothing is averaged blindly):
 1. MEASURED depth comes from the stereo sources only. Each has its own noise, sigma_z = z^2 * sigma_px / (f * B), at least rel_floor * z. Two sources AGREE at a pixel when they differ by less than k * sqrt(sigma_a^2 + sigma_b^2).
    The largest group of agreeing sources wins the pixel (ties go to the earlier source in the list: put the best one first) and its depth is their inverse-variance weighted mean. Confidence:
    two agreeing sources 0.75, three 1.0, a lone source 0.5, each times a noise factor (near and precise pixels count more than far and noisy ones), halved next to depth edges (where flying pixels live).
 2. CONFLICTS (sources that all have a value but no two agree) are not averaged. The monocular depth, scaled to the pixels the stereo sources agree on, arbitrates: if exactly one stereo source is close to it,
    that one is taken with low confidence (0.4); otherwise the pixel is REJECTED.
 3. MONOCULAR depth is never measurement. Both pictures' maps are first fitted (robust scale + offset in disparity space) to the agreed stereo pixels; the right one is warped into the left view and fitted on its
    own; where the two agree they are averaged, where they do not the pixel is not usable. They only fill pixels that have no measured depth, and only where: the pixel is within `reach` px of measured pixels, the
    filled value matches the surrounding measured depth (within 25%), and the monocular map has no depth edge there. A filled pixel is class 3 ('estimated'), confidence at most 0.3, and never replaces a measurement.
 4. Clean-up: isolated pixels and tiny estimated blobs are removed.

Output classes: 0 none, 1 measured, sources agree, 2 measured by one source (or won an arbitration), 3 estimated from the monocular maps. `rejected` marks pixels that had evidence but too contradictory to use.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from gabai.fuse_depth import combine_mono_pair, fit_affine, warp_right_to_left

MONO_FB = 30.0                                       # f*B stand-in to turn monocular metres into disparity; the affine fit absorbs the real value (as in the main pipeline)


@dataclass
class StereoIn:
    z: np.ndarray                                    # metres, NaN where the source has no (or no trusted) depth
    sigma_px: float                                  # the source's disparity noise in pixels
    name: str = ""


@dataclass
class FusionParams:
    zmin: float = 0.3
    zmax: float = 4.0
    k: float = 3.0
    rel_floor: float = 0.02
    reach: int = 40
    local_tol: float = 0.25
    mono_tol: float = 0.10                           # a stereo source is "close" to the monocular depth within this share of the depth (conflict arbitration)
    pair_tol_px: float = 2.0
    min_ref_pixels: int = 300


@dataclass
class FusedDepth:
    z: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
    rejected: np.ndarray
    n_agree: np.ndarray
    mono: np.ndarray                                 # the combined, aligned monocular depth (NaN where none), for display
    info: dict = field(default_factory=dict)


def _edges(z, rel=0.06, floor=0.05):
    zz = np.where(np.isfinite(z), z, 0).astype(np.float32)
    big = np.where(np.isfinite(z), z, 99.0).astype(np.float32)
    k = np.ones((5, 5), np.uint8)
    return (cv2.dilate(zz, k) - cv2.erode(big, k)) > rel * np.nan_to_num(z, nan=1.0) + floor


def fuse(stereo, mono_left, mono_right, cal, p=None):
    p = p or FusionParams()
    fB = float(cal["f"] * cal["B"])
    S, (h, w) = len(stereo), stereo[0].z.shape
    Z = np.stack([np.where((s.z >= p.zmin) & (s.z <= p.zmax), s.z, np.nan) for s in stereo]).astype(np.float32)
    valid = np.isfinite(Z)
    sig = np.stack([np.maximum(p.rel_floor * Z[i], Z[i] ** 2 * stereo[i].sigma_px / fB) for i in range(S)]).astype(np.float32)
    nvalid = valid.sum(0)
    # -- 1. agreement between the stereo sources
    agree = np.zeros((S, S, h * w), bool)
    for i in range(S):
        for j in range(i + 1, S):
            both = (valid[i] & valid[j]).ravel()
            tol = p.k * np.sqrt(np.nan_to_num(sig[i]) ** 2 + np.nan_to_num(sig[j]) ** 2).ravel()
            a = both & (np.abs(Z[i] - Z[j]).ravel() < tol)
            agree[i, j] = agree[j, i] = a
    count = agree.sum(1)                                                                          # (S, pixels): how many others agree with source i
    prio = (S - np.arange(S))[:, None]
    score = np.where(valid.reshape(S, -1), count * 10 + prio, -1)
    best = score.argmax(0)
    member = agree.transpose(2, 0, 1)[np.arange(h * w), best] | (np.arange(S)[None, :] == best[:, None])        # (pixels, S)
    member = member.T.reshape(S, h, w) & valid
    n = member.sum(0)
    wgt = np.where(member, 1.0 / np.maximum(sig, 1e-4) ** 2, 0)
    zf = np.where(n > 0, (wgt * np.nan_to_num(Z)).sum(0) / np.maximum(wgt.sum(0), 1e-12), np.nan)
    sf = np.where(n > 0, 1.0 / np.sqrt(np.maximum(wgt.sum(0), 1e-12)), np.nan)
    nf = np.clip(0.04 / np.where(np.isfinite(sf), sf, 1.0), 0.4, 1.0)
    cls = np.zeros((h, w), np.uint8)
    conf = np.zeros((h, w), np.float32)
    agreed, single = n >= 2, (n == 1) & (nvalid == 1)
    cls[agreed], cls[single] = 1, 2
    conf[agreed] = (np.where(n >= 3, 1.0, 0.75) * nf)[agreed]
    conf[single] = (0.5 * nf)[single]
    conflict = (n == 1) & (nvalid >= 2)
    z = np.where(cls > 0, zf, np.nan).astype(np.float32)
    info = {"sources": [s.name for s in stereo], "valid": [float(valid[i].mean()) for i in range(S)]}
    # -- 2. monocular depth, scaled to the pixels the stereo sources agree on
    ref = cls == 1
    if ref.sum() < p.min_ref_pixels:
        ref = cls >= 1
    zm = np.full((h, w), np.nan, np.float32)
    mono_ok = mono_left is not None and ref.sum() >= p.min_ref_pixels
    mono_pair = False
    if mono_ok:
        ds = np.where(ref, fB / np.where(ref, zf, 1.0), np.nan).astype(np.float32)
        dm_l = MONO_FB / np.clip(mono_left, 0.2, 30.0)
        a, b = fit_affine(dm_l, ds, ref)
        al = (a * dm_l + b).astype(np.float32)
        info["mono_left_fit"] = (a, b)
        info["mono_left_resid_px"] = float(np.nanmedian(np.abs(al - ds)[ref]))
        comb, ok_m = al, np.ones((h, w), bool)
        if mono_right is not None:
            dm_r = MONO_FB / np.clip(mono_right, 0.2, 30.0)
            r_in_left = warp_right_to_left(dm_r, np.where(ref, ds, al))
            good = ref & np.isfinite(r_in_left)
            if good.sum() >= p.min_ref_pixels:
                ar, br = fit_affine(r_in_left, ds, good)
                r_al = (ar * r_in_left + br).astype(np.float32)
                comb, ok_m = combine_mono_pair(al, r_al, p.pair_tol_px)
                mono_pair = True
                info["mono_pair_agree"] = float(ok_m[np.isfinite(r_al)].mean()) if np.isfinite(r_al).any() else 0.0
        zm = np.where(comb > 1.5, fB / np.maximum(comb, 1e-3), np.nan).astype(np.float32)
        zm = np.where((zm >= p.zmin) & (zm <= p.zmax), zm, np.nan)
        mono_agree = ok_m if mono_pair else np.ones((h, w), bool)
    # -- 3. conflicts: the monocular depth arbitrates, or the pixel is rejected
    rejected = np.zeros((h, w), bool)
    if conflict.any():
        if mono_ok:
            close = valid & (np.abs(Z - zm[None]) < p.mono_tol * zm[None])
            nc = close.sum(0)
            pick = conflict & (nc == 1)
            idx = close.argmax(0)
            zp = np.take_along_axis(Z, idx[None], 0)[0]
            sp = np.take_along_axis(sig, idx[None], 0)[0]
            z = np.where(pick, zp, z)
            cls[pick] = 2
            conf[pick] = (0.4 * np.clip(0.04 / np.maximum(sp, 1e-4), 0.4, 1.0))[pick]
            rejected = conflict & ~pick
        else:
            rejected = conflict.copy()
    # -- 4. holes: estimated depth where the evidence allows it
    if mono_ok:
        measured = cls > 0
        near = cv2.dilate(measured.astype(np.uint8), np.ones((2 * p.reach + 1,) * 2, np.uint8)).astype(bool)
        m32 = measured.astype(np.float32)
        cnt = cv2.boxFilter(m32, -1, (31, 31), normalize=False)
        loc = cv2.boxFilter(np.where(measured, z, 0).astype(np.float32), -1, (31, 31), normalize=False) / np.maximum(cnt, 1)
        consistent = (cnt >= 20) & (np.abs(zm - loc) < p.local_tol * np.nan_to_num(loc, nan=1.0))
        fill = ~measured & near & np.isfinite(zm) & mono_agree & consistent & ~_edges(zm)
        z = np.where(fill, zm, z)
        cls[fill] = 3
        conf[fill] = (0.3 if mono_pair else 0.2)
        rejected &= ~fill
    # -- 5. clean-up
    live = cls > 0
    nb = cv2.boxFilter(live.astype(np.float32), -1, (3, 3), normalize=False) - live
    drop = live & (nb < np.where(cls == 1, 2, 3))
    est = (cls == 3).astype(np.uint8)
    if est.any():
        k, lab, st, _ = cv2.connectedComponentsWithStats(est, connectivity=8)
        small = np.isin(lab, [i for i in range(1, k) if st[i, cv2.CC_STAT_AREA] < 50])
        drop |= small & (cls == 3)
    z[drop], cls[drop], conf[drop] = np.nan, 0, 0
    edge = _edges(np.where(cls > 0, z, np.nan))
    conf = np.where(edge & (cls > 0), conf * 0.5, conf).astype(np.float32)                         # flying pixels live at depth edges: they count half
    info.update({f"share_{name}": float((cls == c).mean()) for c, name in ((1, "agreed"), (2, "single"), (3, "estimated"))}, share_rejected=float(rejected.mean()))
    return FusedDepth(np.where(cls > 0, z, np.nan).astype(np.float32), conf, cls, rejected, n.astype(np.uint8), zm, info)
