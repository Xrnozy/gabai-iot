"""Analyse a folder of checkerboard pairs (pair_NN_L.png / pair_NN_R.png) and write a NEW stereo calibration from them.

    py scripts/analyze_stereo_pairs.py --square-mm 19.4                       data/calib_pairs_15 -> data/stereo_calib_new.npz/.json + report
    py scripts/analyze_stereo_pairs.py --compare data/stereo_calib.npz        also test an existing calibration on the same pairs

It never reads or writes an existing calibration unless --compare names one (read only). What it does:
  1. finds the board in both pictures of every pair and measures how flat the board is (a paper sheet that bends is the commonest hidden error),
  2. tries three lens models (k1 k2 | k1 k2 k3 | rational) with leave-one-pair-out cross-validation and keeps the one that predicts the held-out pair best,
  3. calibrates with it, drops pairs that fit badly (and says why),
  4. validates: reprojection error, epipolar (row) error after rectification, depth of the board from stereo against its distance from PnP,
     baseline against the designed 42 mm and the printed square size that would make them agree.
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from gabai.tools.calibrate_stereo import RIG_BASELINE_MM, board_points, calibrate_from_corners, check_baseline, find_corners, save   # noqa: E402

MODELS = {"k1k2": dict(fix_k3=True, rational=False), "k1k2k3": dict(fix_k3=False, rational=False), "rational": dict(fix_k3=False, rational=True)}


def load_pairs(folder, pattern, upright=False):
    """upright: the layout of the other device order. Every picture is turned 180 deg and left and right trade places, which is what the same two
    cameras give when the sender reads --left /dev/video2 --right /dev/video0 (left turned 180 deg by the receiver, right as it is)."""
    rows = []
    for lf in sorted(glob.glob(os.path.join(folder, "pair_*_L.png"))):
        rf = lf[:-6] + "_R.png"
        l, r = cv2.imread(lf), cv2.imread(rf)
        if upright:
            l, r = cv2.rotate(r, cv2.ROTATE_180), cv2.rotate(l, cv2.ROTATE_180)
        gl, gr = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        rows.append({"name": os.path.basename(lf)[:7], "left": l, "right": r, "cl": find_corners(gl, pattern), "cr": find_corners(gr, pattern)})
    return rows


def flatness(corners, pattern):
    """RMS (px) of the detected corners from the best homography of the ideal grid: ~0.1-0.2 px for a flat board, more for a bent sheet or strong lens distortion."""
    grid = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2).astype(np.float32)
    H, _ = cv2.findHomography(grid, corners.reshape(-1, 2), 0)
    p = cv2.perspectiveTransform(grid[None], H)[0]
    return float(np.sqrt(((p - corners.reshape(-1, 2)) ** 2).sum(1).mean()))


def sharp(img, corners):
    x, y, w, h = cv2.boundingRect(corners.reshape(-1, 2).astype(np.float32))
    g = cv2.cvtColor(img[max(y, 0):y + h, max(x, 0):x + w], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def stereo_fold_error(c, obj, cl, cr):
    """Held-out pair: board pose from the LEFT view with the fitted intrinsics, projected into the right camera with the fitted R, T. Returns
    (left reprojection px, right prediction px): the second is the disparity error this calibration would give on a new view."""
    ok, rv, tv = cv2.solvePnP(obj, cl, c["K1"], c["D1"], flags=cv2.SOLVEPNP_ITERATIVE)
    pl, _ = cv2.projectPoints(obj, rv, tv, c["K1"], c["D1"])
    Rl = cv2.Rodrigues(rv)[0]
    rvr = cv2.Rodrigues(c["R"] @ Rl)[0]
    tvr = c["R"] @ tv + c["T"]
    pr, _ = cv2.projectPoints(obj, rvr, tvr, c["K2"], c["D2"])
    return float(np.linalg.norm(pl.reshape(-1, 2) - cl.reshape(-1, 2), axis=1).mean()), float(np.linalg.norm(pr.reshape(-1, 2) - cr.reshape(-1, 2), axis=1).mean())


def cross_validate(obj, CL, CR, size):
    out = {}
    np.seterr(all="ignore")
    for name, kw in MODELS.items():
        ml, mr, B = [], [], []
        for i in range(len(CL)):
            idx = [j for j in range(len(CL)) if j != i]
            try:
                c = calibrate_from_corners([obj] * len(idx), [CL[j] for j in idx], [CR[j] for j in idx], size, min_views=8, **kw)
            except Exception:
                continue
            a, b = stereo_fold_error(c, obj, CL[i], CR[i])
            ml.append(a), mr.append(b), B.append(c["baseline"] * 1000)
        out[name] = {"heldout_left_px": round(float(np.mean(ml)), 3), "heldout_right_prediction_px": round(float(np.mean(mr)), 3),
                     "heldout_right_prediction_px_worst": round(float(np.max(mr)), 3), "baseline_mm_mean": round(float(np.mean(B)), 2), "baseline_mm_std": round(float(np.std(B)), 2)}
    return out


def model_difference(obj, CL, CR, size, chosen, used_idx):
    """Where the lens models disagree: rectified position of a 16x9 grid of pixels under k1k2 and under k1k2k3, split by whether any board
    corner of that camera was seen near the pixel. A big difference where no corner was seen is extrapolation the data cannot check."""
    a = calibrate_from_corners([obj] * len(CL), CL, CR, size, min_views=10, **MODELS["k1k2"])
    b = calibrate_from_corners([obj] * len(CL), CL, CR, size, min_views=10, **MODELS["k1k2k3"])
    gx, gy = np.meshgrid(np.linspace(8, size[0] - 8, 16), np.linspace(8, size[1] - 8, 9))
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)[:, None]
    out = {}
    for cam, K, Dk, Rk, Pk, C in (("left", "K1", "D1", "R1", "P1", CL), ("right", "K2", "D2", "R2", "P2", CR)):
        ra = cv2.undistortPoints(pts, a[K], a[Dk], R=a[Rk], P=a[Pk]).reshape(-1, 2)
        rb = cv2.undistortPoints(pts, b[K], b[Dk], R=b[Rk], P=b[Pk]).reshape(-1, 2)
        dev = np.linalg.norm(ra - rb, axis=1)
        seen = np.concatenate([c_.reshape(-1, 2) for c_ in C])
        near = np.array([(np.linalg.norm(seen - p[0], axis=1) < 40).any() for p in pts])
        out[cam] = {"mean_px_where_seen": round(float(dev[near].mean()), 2), "max_px_where_seen": round(float(dev[near].max()), 2),
                    "mean_px_not_seen": round(float(dev[~near].mean()), 2) if (~near).any() else None, "max_px_not_seen": round(float(dev[~near].max()), 2) if (~near).any() else None,
                    "share_of_picture_never_seen_pct": round(float((~near).mean() * 100), 1)}
    out["baseline_mm_k1k2"], out["baseline_mm_k1k2k3"] = round(a["baseline"] * 1000, 2), round(b["baseline"] * 1000, 2)
    return out


def validate(c, obj, CL, CR, names):
    rows = []
    for i, (cl, cr, n) in enumerate(zip(CL, CR, names)):
        ua = cv2.undistortPoints(cl, c["K1"], c["D1"], R=c["R1"], P=c["P1"]).reshape(-1, 2)
        ub = cv2.undistortPoints(cr, c["K2"], c["D2"], R=c["R2"], P=c["P2"]).reshape(-1, 2)
        dy = np.abs(ua[:, 1] - ub[:, 1])
        d = ua[:, 0] - ub[:, 0]
        ok, rv, tv = cv2.solvePnP(obj, cl, c["K1"], c["D1"])
        Z_pnp = float(np.median((cv2.Rodrigues(rv)[0] @ obj.T + tv).T[:, 2]))
        Z_st = float(np.median(c["focal"] * c["baseline"] / d))
        pl, _ = cv2.projectPoints(obj, rv, tv, c["K1"], c["D1"])
        a, b = stereo_fold_error(c, obj, cl, cr)
        rows.append({"pair": n, "row_error_px": round(float(dy.mean()), 3), "row_error_max_px": round(float(dy.max()), 3), "disparity_px": round(float(np.median(d)), 2),
                     "board_distance_pnp_cm": round(Z_pnp * 100, 1), "board_distance_stereo_cm": round(Z_st * 100, 1), "depth_error_pct": round((Z_st / Z_pnp - 1) * 100, 2),
                     "left_reprojection_px": round(a, 3), "right_prediction_px": round(b, 3)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.path.join("data", "calib_pairs_15"))
    ap.add_argument("--square-mm", type=float, required=True, help="side of one printed square measured with a ruler")
    ap.add_argument("--cols", type=int, default=9), ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--out", default=os.path.join("data", "stereo_calib_new"))
    ap.add_argument("--model", choices=list(MODELS) + ["auto"], default="auto", help="lens model; auto = best leave-one-out result")
    ap.add_argument("--layout", choices=["capture", "upright"], default="capture",
                    help="capture: the pairs as they were saved (the sender's --left/--right order at capture time); upright: the same cameras read in the other order "
                         "(--left /dev/video2 --right /dev/video0: upright pictures)")
    ap.add_argument("--devices", default="", help="left,right camera devices of the layout, stored in the calibration so a live run can check the sender, e.g. /dev/video2,/dev/video0")
    a = ap.parse_args()
    pattern = (a.cols, a.rows)
    obj = board_points(a.cols, a.rows, a.square_mm / 1000)
    rot = tuple(json.load(open(os.path.join(a.images, "rotation.json"))))
    rows = load_pairs(a.images, pattern, a.layout == "upright")
    size = rows[0]["left"].shape[1::-1]
    rep = {"layout": a.layout, "devices": a.devices, "folder": a.images, "pictures": f"{size[0]}x{size[1]}", "rotation_used_when_capturing": list(rot), "square_mm": a.square_mm, "pairs": []}
    good = []
    for r in rows:
        info = {"pair": r["name"], "board_left": r["cl"] is not None, "board_right": r["cr"] is not None}
        if r["cl"] is not None and r["cr"] is not None:
            info.update(flat_left_px=round(flatness(r["cl"], pattern), 3), flat_right_px=round(flatness(r["cr"], pattern), 3),
                        sharp_left=round(sharp(r["left"], r["cl"]), 1), sharp_right=round(sharp(r["right"], r["cr"]), 1),
                        board_width_px=round(float(np.ptp(r["cl"].reshape(-1, 2)[:, 0])), 1))
            good.append(r)
        rep["pairs"].append(info)
    print(f"board found in both pictures: {len(good)}/{len(rows)} pairs")
    names = [r["name"] for r in good]
    CL, CR = [r["cl"] for r in good], [r["cr"] for r in good]
    rep["cross_validation"] = cross_validate(obj, CL, CR, size)
    for k, v in rep["cross_validation"].items():
        print(f"  model {k:9s} held-out: left {v['heldout_left_px']:.3f} px, right prediction {v['heldout_right_prediction_px']:.3f} px (worst {v['heldout_right_prediction_px_worst']:.2f}), baseline {v['baseline_mm_mean']:.2f} +- {v['baseline_mm_std']:.2f} mm")
    cv = {k: v["heldout_right_prediction_px"] for k, v in rep["cross_validation"].items() if np.isfinite(v["heldout_right_prediction_px"])}
    best = min(cv.values())
    model = a.model if a.model != "auto" else next(k for k in MODELS if k in cv and cv[k] <= 1.03 * best)   # simplest model within 3% of the best
    rep["model"] = model
    print(f"lens model used: {model} (the simplest whose held-out error is within 3% of the best; a model that fails or diverges is listed as inf)")
    c = calibrate_from_corners([obj] * len(good), CL, CR, size, min_views=10, **MODELS[model])
    used = [names[i] for i in c["used"]]
    rep["model_difference"] = model_difference(obj, CL, CR, size, c, used_idx=c["used"])
    print("rectification difference between k1k2 and k1k2k3:", rep["model_difference"])
    rep["pairs_used"], rep["pairs_dropped"] = used, [n for n in names if n not in used]
    meta = save(c, a.out, a.square_mm, rot)
    z = dict(np.load(a.out + ".npz"))
    z.update(lens_model=np.array(model), layout=np.array(a.layout), date=np.array(time.strftime("%Y-%m-%d %H:%M:%S")))
    if a.devices:
        z.update(left_dev=np.array(a.devices.split(",")[0]), right_dev=np.array(a.devices.split(",")[1]))
    np.savez(a.out + ".npz", **z)
    sub = [names.index(n) for n in used]
    rep["validation"] = validate(c, obj, [CL[i] for i in sub], [CR[i] for i in sub], used)
    ve = rep["validation"]
    rep["summary"] = {"rms_stereo_px": round(c["rms_stereo"], 3), "epipolar_err_px": round(c["epipolar_err_px"], 3), "baseline_mm": round(c["baseline"] * 1000, 2),
                      "baseline_check": check_baseline(c, a.square_mm), "implied_square_mm_for_42mm": round(a.square_mm * RIG_BASELINE_MM / (c["baseline"] * 1000), 2),
                      "focal_px": round(c["focal"], 1), "f_left_px": round(float(c["K1"][0, 0]), 1), "f_right_px": round(float(c["K2"][0, 0]), 1),
                      "median_depth_error_vs_pnp_pct": round(float(np.median([v["depth_error_pct"] for v in ve])), 2),
                      "worst_row_error_px": max(v["row_error_max_px"] for v in ve), "distortion_left": [round(float(x), 4) for x in c["D1"].ravel()],
                      "distortion_right": [round(float(x), 4) for x in c["D2"].ravel()], "verdict": meta["verdict"], "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(rep["summary"], indent=1))
    json.dump(rep, open(a.out + "_report.json", "w"), indent=1)
    print(f"saved {a.out}.npz  {a.out}.json  {a.out}_report.json")


if __name__ == "__main__":
    main()
