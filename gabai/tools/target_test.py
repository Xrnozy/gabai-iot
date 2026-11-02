"""Depth error on real targets (wall, chair, dark bag, thin pole, glass, low light), live from the Pi.

Put the target in the middle box, measure the real distance from the lens with a tape measure, then record it:

    py -m gabai.tools.target_test --label wall --true-m 1.0 --light lit
    py -m gabai.tools.target_test --label "thin pole" --true-m 2.0 --light lit --box 40

SPACE records one measurement (the median of the last 15 frames, left-right checked); q finishes. Each measurement is
appended to data/target_tests.csv with the share of the box that got any depth at all: a target the model cannot see
(glass, a black bag, a pole thinner than the box) shows up as a low share or as a wide spread, not as a good number.
Run it again for every target and distance; `make_report` turns the file into the accuracy graph.
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

COLS = ["time", "label", "light", "true_m", "est_m", "p10_m", "p90_m", "valid_pct", "model"]


def region_stats(disp, trusted, box, f, B):
    """Depth over a box (x0, y0, x1, y1) of the left image. Only trusted pixels count; valid_pct is their share of the box."""
    x0, y0, x1, y1 = box
    d, t = disp[y0:y1, x0:x1], trusted[y0:y1, x0:x1]
    ok = t & np.isfinite(d) & (d > 0.5)
    pct = 100.0 * float(ok.mean()) if ok.size else 0.0
    if not ok.any():
        return {"est_m": None, "p10_m": None, "p90_m": None, "valid_pct": pct}
    z = f * B / d[ok]
    return {"est_m": float(np.median(z)), "p10_m": float(np.percentile(z, 10)), "p90_m": float(np.percentile(z, 90)), "valid_pct": pct}


def summarize_targets(rows):
    """rows: dicts with label, true_m, est_m (None = no depth), valid_pct. Returns {label: stats}."""
    out = {}
    for label in sorted({r["label"] for r in rows}):
        rs = [r for r in rows if r["label"] == label]
        have = [r for r in rs if r["est_m"] is not None]
        err = np.array([(r["est_m"] - r["true_m"]) / r["true_m"] * 100 for r in have])
        out[label] = dict(n=len(rs), no_depth=len(rs) - len(have), valid_pct=float(np.mean([r["valid_pct"] for r in rs])),
                          median_err_pct=float(np.median(err)) if err.size else None,
                          worst_err_pct=float(err[np.argmax(np.abs(err))]) if err.size else None)
    return out


def load_targets(path):
    def num(v):
        return float(v) if v not in ("", None) else None
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(label=r["label"], light=r.get("light", ""), true_m=float(r["true_m"]), est_m=num(r["est_m"]),
                             p10_m=num(r.get("p10_m")), p90_m=num(r.get("p90_m")), valid_pct=float(r["valid_pct"])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--true-m", type=float, required=True, help="tape-measured distance from the lens to the target")
    ap.add_argument("--light", default="lit", help="lit | dim | dark (free text, goes into the file)")
    ap.add_argument("--box", type=int, default=80, help="side of the measuring box in pixels (use a small one for a thin pole)")
    ap.add_argument("--model", choices=["s", "m", "h"], default="m")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--calib", default="data/stereo_calib_new.npz")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--out", default="data/target_tests.csv")
    ap.add_argument("--auto-record", type=int, default=0, help="testing: record this many measurements without a window")
    a = ap.parse_args()

    from gabai.pc.receiver import LiveReceiver
    from gabai.spike.las2_runner import Las2Depth
    from gabai.tools.calibrate_stereo import Rectifier
    rect = Rectifier(a.calib)
    las = Las2Depth(a.model)
    lr_pass = getattr(las, "disparity_lr_batched", las.disparity_lr)
    rx = LiveReceiver(a.port, a.token, rotate_left=rect.rotation[0], rotate_right=rect.rotation[1])
    rx.start()
    print(f"waiting for the Pi on port {rx.port} ... '{a.label}' at {a.true_m} m; SPACE records, q finishes", flush=True)
    h, w = rect.size[1], rect.size[0]
    half = a.box // 2
    box = (w // 2 - half, h // 2 - half, w // 2 + half, h // 2 + half)
    recent, seq, saved, win = [], 0, 0, f"Target test: {a.label} at {a.true_m} m (SPACE = record, q = finish)"
    new = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    f_out = open(a.out, "a", newline="")
    wr = csv.writer(f_out)
    if new:
        wr.writerow(COLS)
    while True:
        pair = rx.latest()
        if pair is not None and pair.seq != seq:
            seq = pair.seq
            l, r = rect.apply(pair.left, pair.right)
            disp, trusted, _ = lr_pass(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
            recent = (recent + [region_stats(disp, trusted, box, rect.f, rect.B)])[-a.frames:]
            view = l.copy()
            cv2.rectangle(view, box[:2], box[2:], (0, 255, 255), 2)
            ests = [s["est_m"] for s in recent if s["est_m"] is not None]
            line = (f"est {np.median(ests):.2f} m (true {a.true_m:.2f}, {100 * (np.median(ests) - a.true_m) / a.true_m:+.1f}%)  "
                    f"depth on {np.mean([s['valid_pct'] for s in recent]):.0f}% of the box") if ests else "no depth in the box"
            cv2.putText(view, line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, f"{saved} recorded", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if not a.auto_record:
                cv2.imshow(win, view)
        enough = len(recent) >= a.frames
        k = 255 if a.auto_record else cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27) or (a.auto_record and saved >= a.auto_record):
            break
        if enough and (k == ord(" ") or (a.auto_record and saved < a.auto_record)):
            ests = [s for s in recent if s["est_m"] is not None]
            row = [time.strftime("%Y-%m-%d %H:%M:%S"), a.label, a.light, a.true_m,
                   "" if not ests else round(float(np.median([s["est_m"] for s in ests])), 4),
                   "" if not ests else round(float(np.median([s["p10_m"] for s in ests])), 4),
                   "" if not ests else round(float(np.median([s["p90_m"] for s in ests])), 4),
                   round(float(np.mean([s["valid_pct"] for s in recent])), 1), a.model]
            wr.writerow(row)
            f_out.flush()
            saved += 1
            recent = []
            print("recorded:", row[1:], flush=True)
    f_out.close()
    rx.stop()
    cv2.destroyAllWindows()
    print(f"{saved} measurement(s) appended to {a.out}")


if __name__ == "__main__":
    main()
