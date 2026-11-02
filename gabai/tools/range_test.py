"""Measure depth accuracy at distance, live from the Pi, with the calibrated rig.

Hold the checkerboard in view of both cameras at a distance you can check with a tape measure. The tool finds
the board in the left camera, works out its true distance with solvePnP (this depends on --square-mm being the
REAL square size of the board you hold), and compares it with the depth the model gives over the board.
Press SPACE to record a sample, q to finish. Prints an error table by distance and saves data/range_test.csv.

The calibration board squares are small (25 mm), which is hard to see beyond about 1.5 m. For 2 to 3 m use a
bigger board (for example 60-100 mm squares, printed on a flat rigid sheet) and pass its real size.

    py -m gabai.tools.range_test --square-mm 60 --model m
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

BINS = [(0.0, 0.8, "<0.8 m"), (0.8, 1.5, "0.8-1.5 m"), (1.5, 3.0, "1.5-3 m"), (3.0, 99.0, ">3 m")]


def bucket(z):
    for lo, hi, name in BINS:
        if lo <= z < hi:
            return name
    return BINS[-1][2]


def summarize(samples):
    """samples: (true_m, est_median_m, est_p10_m, est_p90_m). Returns {bucket: stats} for non-empty buckets."""
    out = {}
    for lo, hi, name in BINS:
        rows = [s for s in samples if lo <= s[0] < hi]
        if not rows:
            continue
        err = np.array([(e - t) / t * 100 for t, e, _, _ in rows])
        spread = np.array([(p90 - p10) / t * 100 for t, _, p10, p90 in rows])
        out[name] = dict(n=len(rows), median_err_pct=float(np.median(err)), worst_err_pct=float(err[np.argmax(np.abs(err))]),
                         median_spread_pct=float(np.median(spread)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--square-mm", type=float, required=True, help="REAL side of one square of the board you hold")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--model", choices=["s", "m", "h", "ffs"], default="m")
    ap.add_argument("--gray", action="store_true", help="feed grayscale to the model (default is colour)")
    ap.add_argument("--calib", default="data/stereo_calib_new.npz")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--out", default="data/range_test.csv")
    ap.add_argument("--auto-record", type=int, default=0, help="testing: record this many samples automatically, no window, then finish")
    a = ap.parse_args()

    from gabai.pc.receiver import LiveReceiver
    from gabai.tools.benchmark_depth import board_depth
    from gabai.tools.calibrate_stereo import Rectifier
    rect = Rectifier(a.calib)
    P1 = np.load(a.calib)["P1"]
    if a.model == "ffs":
        from gabai.spike.ffs_runner import FfsDepth
        model = FfsDepth()
    else:
        from gabai.spike.las2_runner import Las2Depth
        model = Las2Depth(a.model)
    rx = LiveReceiver(a.port, a.token, rotate_left=rect.rotation[0], rotate_right=rect.rotation[1])
    rx.start()
    print(f"waiting for the Pi on port {rx.port} ... SPACE records a sample, q finishes", flush=True)
    samples, seq, last, win = [], 0, None, "Range test (SPACE = record, q = finish)"
    g = (lambda x: cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)) if a.gray else (lambda x: x)
    while True:
        pair = rx.latest()
        if pair is not None and pair.seq != seq:
            seq = pair.seq
            l, r = rect.apply(g(pair.left), g(pair.right))
            truth = board_depth(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), P1, a.cols, a.rows, a.square_mm / 1000)
            view = l.copy()
            line = "board not found"
            last = None
            if truth is not None:
                disp, _ = model.disparity(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
                m = np.isfinite(truth)
                zt = float(np.median(truth[m]))
                zp = rect.f * rect.B / np.maximum(disp[m], 0.5)
                e = (zp - truth[m]) / truth[m]  # per-pixel relative error: a tilted board has a depth range, the error is what matters
                est = zt * (1 + float(np.median(e)))
                p10, p90 = zt * (1 + float(np.percentile(e, 10))), zt * (1 + float(np.percentile(e, 90)))
                last = (zt, est, p10, p90)
                view[m] = (0.6 * view[m] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
                line = f"true {zt:.2f} m   estimated {est:.2f} m   error {100 * (est - zt) / zt:+.1f}%   spread {100 * (p90 - p10) / zt:.1f}%"
            cv2.putText(view, line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, f"{len(samples)} samples recorded", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if not a.auto_record:
                cv2.imshow(win, view)
            elif last is not None and len(samples) < a.auto_record:
                samples.append(last)
                print(f"recorded: true {last[0]:.2f} m, estimated {last[1]:.2f} m ({100 * (last[1] - last[0]) / last[0]:+.1f}%)", flush=True)
        if a.auto_record and len(samples) >= a.auto_record:
            break
        k = cv2.waitKey(20) & 0xFF if not a.auto_record else 255
        if k in (ord("q"), 27):
            break
        if k == ord(" ") and last is not None:
            samples.append(last)
            print(f"recorded: true {last[0]:.2f} m, estimated {last[1]:.2f} m ({100 * (last[1] - last[0]) / last[0]:+.1f}%)", flush=True)
    rx.stop()
    cv2.destroyAllWindows()
    if not samples:
        return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_m", "est_median_m", "est_p10_m", "est_p90_m"])
        w.writerows(samples)
    print(f"\nmodel {a.model}; {len(samples)} samples saved to {a.out}")
    print(f"{'distance':12s} {'n':>3s} {'median error':>13s} {'worst error':>12s} {'error spread(p10-p90)':>16s}")
    for name, s in summarize(samples).items():
        print(f"{name:12s} {s['n']:3d} {s['median_err_pct']:+12.1f}% {s['worst_err_pct']:+11.1f}% {s['median_spread_pct']:15.1f}%")


if __name__ == "__main__":
    main()
