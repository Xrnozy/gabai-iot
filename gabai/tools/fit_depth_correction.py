"""Measure the stereo model's distance error on the calibration board pairs and store a correction (gabai/depthcal.py).

    py -m gabai.tools.fit_depth_correction                         fits LAS2 s, m and h on data/calib_pairs_15, prints the leave-one-pair-out result
    py -m gabai.tools.fit_depth_correction --models h --write      also writes data/calibration/depth_correction.json (entries of other models are kept)

The error is (estimated - true) / true distance over the board area, true from solvePnP on the known square size. 'leave one out' fits the curve on all pairs but one
and scores the pair left out, so the number is not a fit to the pictures it is judged on. An existing file is only changed with --write, and only the entries of the
models fitted now are replaced. Nothing else in data/calibration is touched.
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from gabai import depthcal
from gabai.tools.benchmark_depth import board_depth
from gabai.tools.calibrate_stereo import Rectifier


def collect(model, pairs_dir, calib, cols=9, rows=6):
    """Per pair with the board found: (true distance m, true disparity samples, predicted disparity samples), the model's left-right checked pixels only."""
    from gabai.spike.las2_runner import Las2Depth
    rect, z, las = Rectifier(calib), np.load(calib), Las2Depth(model)
    out = []
    for lf in sorted(glob.glob(os.path.join(pairs_dir, "*_L.png"))):
        L, R = cv2.imread(lf), cv2.imread(lf[:-6] + "_R.png")
        if L is None or R is None:
            continue
        l, r = rect.apply(L, R)
        truth = board_depth(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), z["P1"], cols, rows, float(z["square_mm"]) / 1000)
        if truth is None:
            continue
        m = np.isfinite(truth)
        disp, ok, _ = las.disparity_lr(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        k = m & ok & (disp > 0.5)
        out.append((float(np.median(truth[m])), rect.f * rect.B / truth[k], disp[k].astype(float)))
    return out, rect.f * rect.B


def board_error(corr, rows, fb):
    """Median error per pair (signed, fraction of the true distance)."""
    e = []
    for _, true_d, pred_d in rows:
        d = pred_d if corr is None else corr(pred_d)
        e.append(float(np.median((fb / d - fb / true_d) / (fb / true_d))))
    return np.array(e)


def leave_one_out(rows, fb, n_knots):
    e = []
    for i in range(len(rows)):
        rest = rows[:i] + rows[i + 1:]
        corr = depthcal.fit_knots(np.concatenate([r[2] for r in rest]), np.concatenate([r[1] for r in rest]), n_knots)
        e.append(board_error(corr, [rows[i]], fb)[0])
    return np.array(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-dir", default="data/calib_pairs_15")
    ap.add_argument("--calib", default="data/stereo_calib_new.npz")
    ap.add_argument("--models", default="s,m,h")
    ap.add_argument("--knots", type=int, default=3)
    ap.add_argument("--out", default=depthcal.DEFAULT_PATH)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    doc = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {"models": {}}
    for model in a.models.split(","):
        rows, fb = collect(model, a.pairs_dir, a.calib)
        if len(rows) < 6:
            print(f"LAS2-{model}: only {len(rows)} pairs with the board: not enough to fit")
            continue
        before, loo = board_error(None, rows, fb), leave_one_out(rows, fb, a.knots)
        corr = depthcal.fit_knots(np.concatenate([r[2] for r in rows]), np.concatenate([r[1] for r in rows]), a.knots)
        dist = [r[0] for r in rows]
        print(f"LAS2-{model}: {len(rows)} pairs, board at {min(dist):.2f}-{max(dist):.2f} m. median |error| {np.median(np.abs(before))*100:.2f}% raw -> {np.median(np.abs(loo))*100:.2f}% corrected "
              f"(leave one out); mean bias {np.mean(before)*100:+.2f}% -> {np.mean(loo)*100:+.2f}%; worst {np.max(np.abs(before))*100:.1f}% -> {np.max(np.abs(loo))*100:.1f}%")
        print(f"          knots (disparity px) {np.round(corr.knots, 1).tolist()} ratio {np.round(corr.ratio, 3).tolist()}  (valid between {fb/corr.knots[-1]:.2f} and {fb/corr.knots[0]:.2f} m, tapering outside)")
        doc["models"][model] = dict(corr.to_dict(), pairs=len(rows), true_distance_m=[round(min(dist), 3), round(max(dist), 3)], fitted=time.strftime("%Y-%m-%d"),
                                    median_abs_error_raw=round(float(np.median(np.abs(before))), 4), median_abs_error_loo=round(float(np.median(np.abs(loo))), 4))
    if a.write:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(doc, open(a.out, "w", encoding="utf-8"), indent=1)
        print("wrote", a.out)
    else:
        print("not written (add --write)")


if __name__ == "__main__":
    main()
