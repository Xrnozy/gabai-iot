"""Test two stereo calibrations on the SAME checkerboard pairs and compare depth maps and point clouds.

    py scripts/compare_calibrations.py                                   old data/stereo_calib.npz against new data/stereo_calib_new.npz on data/calib_pairs_15

Both calibrations are read only. The pairs are put in the layout of the sender order both files were made for (left /dev/video2 turned 180 deg,
right /dev/video0): every saved picture is turned 180 deg and left and right trade places. Per calibration and pair it measures
  row error     how far the same board corner sits at a different row in the rectified left and right picture (stereo matching wants < 0.5 px)
  image rows    the pipeline's own rectification check on the whole rectified pictures (features of the room, not only the board)
  coverage      share of pixels LAS2 keeps after the left-right check, and the cause of the rest (gabai/depthq.py)
  board depth   median depth of the board from stereo against the distance PnP gives from the board's known size, and how thick the board looks
                (RMS of its stereo points from a fitted plane; the board is nearly flat, so thickness is noise)
Writes data/bench/calibration_compare/{results.json, depth_compare.png, cloud_compare.png}.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from gabai import depthq as Q                                              # noqa: E402
from gabai.pipeline.run import default_depth                               # noqa: E402
from gabai.spike.autorect import vertical_residual                          # noqa: E402
from gabai.tools.calibrate_stereo import Rectifier, board_points, find_corners   # noqa: E402


def load(folder, upright=True):
    out = []
    for lf in sorted(glob.glob(os.path.join(folder, "pair_*_L.png"))):
        l, r = cv2.imread(lf), cv2.imread(lf[:-6] + "_R.png")
        out.append((os.path.basename(lf)[:7], cv2.rotate(r, cv2.ROTATE_180), cv2.rotate(l, cv2.ROTATE_180)) if upright else (os.path.basename(lf)[:7], l, r))
    return out


def corners_in(z, img, K, D, R, P, pattern):
    c = find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pattern)
    return None if c is None else (c, cv2.undistortPoints(c, K, D, R=R, P=P).reshape(-1, 2))


def plane_rms_mm(pts):
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    return float(s[-1] / np.sqrt(len(pts)) * 1000)


def board_stats(z, rect, left, right, d, ok, pattern, obj):
    f, B, cx, cy = float(z["focal"]), float(z["baseline"]), float(z["cx"]), float(z["cy"])
    cl = corners_in(z, left, z["K1"], z["D1"], z["R1"], z["P1"], pattern)
    cr = corners_in(z, right, z["K2"], z["D2"], z["R2"], z["P2"], pattern)
    if cl is None or cr is None:
        return None, None
    dy = np.abs(cl[1][:, 1] - cr[1][:, 1])
    _, rv, tv = cv2.solvePnP(obj, cl[0], z["K1"], z["D1"])
    z_pnp = float(np.median((cv2.Rodrigues(rv)[0] @ obj.T + tv).T[:, 2]))
    hull = cv2.convexHull(cl[1].astype(np.float32))
    m = np.zeros(d.shape, np.uint8)
    cv2.fillConvexPoly(m, hull.astype(np.int32), 1)
    m = cv2.erode(m, np.ones((9, 9), np.uint8)).astype(bool) & ok & (d > 1)
    out = {"row_error_px": float(dy.mean()), "board_pixels_with_depth": int(m.sum())}
    if m.sum() > 200:
        ys, xs = np.nonzero(m)
        Z = f * B / d[ys, xs]
        P3 = np.stack([(xs - cx) * Z / f, (ys - cy) * Z / f, Z], 1)
        out.update(board_distance_stereo_cm=float(np.median(Z) * 100), board_distance_pnp_cm=z_pnp * 100, depth_error_pct=float((np.median(Z) / z_pnp - 1) * 100),
                   board_thickness_rms_mm=plane_rms_mm(P3))
    return out, (m, f, B, cx, cy)


def cloud(d, ok, f, B, cx, cy):
    ys, xs = np.nonzero(ok & (d > 1))
    Z = f * B / d[ys, xs]
    return np.stack([(xs - cx) * Z / f, (ys - cy) * Z / f, Z], 1)[::3]


def top_view(P, span, size=(430, 400)):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    ok = (P[:, 2] > 0) & (P[:, 2] < span[3])
    P = P[ok]
    ux = ((P[:, 0] - span[0]) / (span[1] - span[0]) * (size[0] - 1)).astype(int)
    uy = ((span[3] - P[:, 2]) / (span[3] - span[2]) * (size[1] - 1)).astype(int)
    m = (ux >= 0) & (ux < size[0]) & (uy >= 0) & (uy < size[1])
    img[uy[m], ux[m]] = (80, 220, 80)
    return img


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    return cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.path.join("data", "calib_pairs_15"))
    ap.add_argument("--old", default=os.path.join("data", "stereo_calib.npz")), ap.add_argument("--new", default=os.path.join("data", "stereo_calib_new.npz"))
    ap.add_argument("--square-mm", type=float, default=19.4), ap.add_argument("--model", default="h")
    ap.add_argument("--show", default="pair_03,pair_12", help="pairs drawn in the pictures")
    ap.add_argument("--out", default=os.path.join("data", "bench", "calibration_compare"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pattern, obj = (9, 6), board_points(9, 6, a.square_mm / 1000)
    pairs_up, pairs_cap = load(a.images), load(a.images, upright=False)
    fn = default_depth(a.model, 100, 1.0)
    res, shots = {}, {}
    for tag, path, pairs in (("old", a.old, pairs_up), ("new", a.new, pairs_up), ("old_wrong_order", a.old, pairs_cap)):
        z, rect = np.load(path), Rectifier(path)
        rows, cov, causes, vres = [], [], [], []
        for name, l, r in pairs:
            rl, rr = rect.apply(l, r)
            d, ok, _ = fn(rl, rr)
            d = np.nan_to_num(d)
            cov.append(float(ok.mean()))
            causes.append(Q.reason_fractions(fn.reasons))
            vres.append(float(vertical_residual(cv2.resize(rl, None, fx=0.5, fy=0.5), cv2.resize(rr, None, fx=0.5, fy=0.5))))
            st, extra = board_stats(z, rect, l, r, d, ok, pattern, obj)
            rows.append({"pair": name, **{k: round(v, 3) for k, v in (st or {}).items()}})
            shots[(tag, name)] = (rl, d, ok, extra, float(ok.mean()))
        thick = [x["board_thickness_rms_mm"] for x in rows if "board_thickness_rms_mm" in x]
        derr = [abs(x["depth_error_pct"]) for x in rows if "depth_error_pct" in x]
        res[tag] = {"file": path, "baseline_mm": round(float(z["baseline"]) * 1000, 2),
                    "row_error_corners_px_mean": round(float(np.mean([x["row_error_px"] for x in rows if "row_error_px" in x])), 3),
                    "row_error_corners_px_worst": round(float(np.max([x["row_error_px"] for x in rows if "row_error_px" in x])), 3),
                    "image_rows_residual_px_median": round(float(np.median(vres)), 3), "valid_point_coverage_pct": round(float(np.mean(cov)) * 100, 2),
                    "pixels_dropped_by_cause_pct": {k: round(float(np.mean([c[k] for c in causes])) * 100, 2) for k in Q.LABELS},
                    "board_depth_error_vs_pnp_pct_median_abs": round(float(np.median(derr)), 2) if derr else None,
                    "board_thickness_rms_mm_median": round(float(np.median(thick)), 2) if thick else None, "per_pair": rows}
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    show = a.show.split(",")
    strips, clouds = [], []
    for name in show:
        cols = []
        for tag in ("old", "new", "old_wrong_order"):
            rl, d, ok, extra, c = shots[(tag, name)]
            top = float(np.percentile(d[ok], 98)) if ok.any() else 1.0
            v = cv2.applyColorMap(np.clip(d / top * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            v[~ok] = 0
            cols.append(label(v, f"{name} {tag}: {c * 100:.0f}% valid"))
        strips.append(np.hstack([label(shots[("new", name)][0].copy(), f"{name} rectified left (new)")] + cols))
        pts = {tag: cloud(*shots[(tag, name)][1:3], *shots[(tag, name)][3][1:]) for tag in ("old", "new", "old_wrong_order")}
        allp = np.concatenate(list(pts.values()))
        zmax = float(np.percentile(allp[:, 2], 99)) * 1.05
        span = (float(np.percentile(allp[:, 0], 1)), float(np.percentile(allp[:, 0], 99)), 0.0, zmax)
        clouds.append(np.hstack([label(top_view(pts[t], span), f"{name} {t}: top view, {len(pts[t])} points") for t in ("old", "new", "old_wrong_order")]))
    cv2.imwrite(os.path.join(a.out, "depth_compare.png"), np.vstack(strips))
    cv2.imwrite(os.path.join(a.out, "cloud_compare.png"), np.vstack(clouds))
    print(json.dumps({t: {k: v for k, v in res[t].items() if k != "per_pair"} for t in res}, indent=1))


if __name__ == "__main__":
    main()
