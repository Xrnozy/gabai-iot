"""Turning a solve into files (always in a NEW directory), validating it on frames the solve never saw, and gating updates.

Nothing here opens, modifies or deletes an existing calibration file. Output goes to data/selfcal/<run>/ ; the only file shared with the rest of the system is data/selfcal/active.json, written by
`activate` after an explicit --confirm.
"""
import json
import os
import time

import cv2
import numpy as np

from mapper.selfcal.core import K_of, D_of, NC

PASS_MEDIAN_ROW_PX = 0.40
PASS_P90_ROW_PX = 1.00
PASS_MIN_MATCHES = 150
MAX_FOCAL_REL_STD = 0.01


def to_npz(path, ba, size, baseline_mm=None):
    """Writes a calibration in the format the pipeline reads (gabai Rectifier / rtabmap_lab CalibParams). T is in metres only when a baseline was measured; otherwise it is in baseline UNITS and the file says so."""
    cL, cR, Rs, ts = ba["cL"], ba["cR"], ba["Rs"], ba["ts"]
    metric = baseline_mm is not None
    T = (ts * (baseline_mm / 1000.0 if metric else 1.0)).reshape(3, 1)
    K1, K2, D1, D2 = K_of(cL), K_of(cR), D_of(cL).reshape(1, 5), D_of(cR).reshape(1, 5)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, tuple(size), Rs, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    np.savez(path, K1=K1, D1=D1, K2=K2, D2=D2, R=Rs, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, size=np.array(size), baseline=np.array(float(np.linalg.norm(T))), focal=np.array(float(P1[0, 0])),
             cx=np.array(float(P1[0, 2])), cy=np.array(float(P1[1, 2])), rms_stereo=np.array(ba["inlier_rms_px"]), epipolar_err_px=np.array(ba["inlier_rms_px"]), n_views=np.array(ba["obs"].n_frames),
             rotate_left=np.array(0), rotate_right=np.array(0), metric=np.array(metric), source=np.array("selfcal (no calibration target)"))
    return path


def row_error(rec, frames, npz, n_features=1500):
    """Rectify held-out pairs with a calibration file and measure how far matched points are from sharing a row. Returns dict(median, p90, matches, positive_disparity)."""
    z = np.load(npz)
    size = tuple(int(v) for v in z["size"])
    m1 = cv2.initUndistortRectifyMap(z["K1"], z["D1"], z["R1"], z["P1"], size, cv2.CV_16SC2)
    m2 = cv2.initUndistortRectifyMap(z["K2"], z["D2"], z["R2"], z["P2"], size, cv2.CV_16SC2)
    sift = cv2.SIFT_create(nfeatures=n_features)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    dys, pos = [], []
    for i in frames:
        l, r = rec.raw(i)
        l, r = cv2.remap(l, *m1, cv2.INTER_LINEAR), cv2.remap(r, *m2, cv2.INTER_LINEAR)
        kl, dl = sift.detectAndCompute(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), None)
        kr, dr = sift.detectAndCompute(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), None)
        if dl is None or dr is None:
            continue
        for m in bf.match(dl, dr):
            a, b = np.array(kl[m.queryIdx].pt), np.array(kr[m.trainIdx].pt)
            if abs(a[1] - b[1]) < 4.0:                                          # the same physical point, not a repeated texture on another row
                dys.append(abs(a[1] - b[1]))
                pos.append(a[0] - b[0] > 0)
    if not dys:
        return {"median": float("nan"), "p90": float("nan"), "matches": 0, "positive_disparity": float("nan")}
    return {"median": float(np.median(dys)), "p90": float(np.percentile(dys, 90)), "matches": len(dys), "positive_disparity": float(np.mean(pos))}


def validate(rec, held_frames, npz, compare=None):
    """Independent validation on frames the solve did not use. Returns (passed, report dict)."""
    new = row_error(rec, held_frames, npz)
    rep = {"held_out_frames": list(map(int, held_frames)), "new": new}
    if compare:
        rep["existing_for_comparison"] = row_error(rec, held_frames, compare)
    reasons = []
    if new["matches"] < PASS_MIN_MATCHES:
        reasons.append(f"only {new['matches']} matches on held-out frames (need {PASS_MIN_MATCHES})")
    else:
        if new["median"] > PASS_MEDIAN_ROW_PX:
            reasons.append(f"median row error {new['median']:.2f} px > {PASS_MEDIAN_ROW_PX}")
        if new["p90"] > PASS_P90_ROW_PX:
            reasons.append(f"90th percentile row error {new['p90']:.2f} px > {PASS_P90_ROW_PX}")
        if new["positive_disparity"] < 0.95:
            reasons.append(f"only {new['positive_disparity'] * 100:.0f}% of matches have positive disparity (cameras swapped or rotated?)")
    rep["reasons_for_failure"] = reasons
    return not reasons, rep


def params_report(ba):
    names = ["f", "cx", "cy", "k1", "k2"]
    std = ba["std"]
    out = {"left": {n: {"value": float(ba["cL"][i]), "std": float(std[i])} for i, n in enumerate(names)},
           "right": {n: {"value": float(ba["cR"][i]), "std": float(std[NC + i])} for i, n in enumerate(names)},
           "stereo_rotation_deg": float(np.degrees(np.linalg.norm(cv2.Rodrigues(ba["Rs"])[0]))), "baseline_direction": ba["ts"].tolist(),
           "baseline_direction_std_deg": float(np.degrees(np.linalg.norm(std[2 * NC + 3:2 * NC + 5]))),
           "observations": int(len(ba["obs"].frame)), "frames": int(ba["obs"].n_frames), "points": int(ba["obs"].n_points), "reprojection_rms_px": ba["rms_px"], "reprojection_inlier_rms_px": ba["inlier_rms_px"]}
    return out


def gate_update(old, new, max_sigma_jump=4.0):
    """Online refinement rule: should a newer solve replace the current self-calibration? Returns (accept, reasons). Never accepts a solve that is less certain, whose parameters jumped by more than
    `max_sigma_jump` of the older (larger) uncertainty (a jump that size is either a different rig state or noise, and either way is not applied silently), or whose reprojection error is worse."""
    reasons = []
    if new["std"][0] / new["cL"][0] > MAX_FOCAL_REL_STD:
        reasons.append("new focal length is not determined to 1%")
    if new["inlier_rms_px"] > max(old["inlier_rms_px"] * 1.2, 0.6):
        reasons.append(f"reprojection error rose from {old['inlier_rms_px']:.2f} to {new['inlier_rms_px']:.2f} px")
    sig = np.maximum(old["std"][:2 * NC], new["std"][:2 * NC])
    jump = np.abs(np.r_[new["cL"], new["cR"]] - np.r_[old["cL"], old["cR"]]) / np.maximum(sig, 1e-9)
    if (jump > max_sigma_jump).any():
        reasons.append(f"parameters moved by up to {jump.max():.1f} sigma: reported, not applied")
    if np.sum(new["std"][:2 * NC] ** 2) > 1.0 * np.sum(old["std"][:2 * NC] ** 2) * 1.5:
        reasons.append("new estimate is less certain than the current one")
    return not reasons, reasons


def new_run_dir(root="data/selfcal"):
    d = os.path.join(root, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=False)                                              # never reuse a directory: exist_ok=False so nothing is overwritten
    return d


def write_json(path, obj):
    with open(path, "x") as f:                                                  # 'x': refuses to overwrite an existing file
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
