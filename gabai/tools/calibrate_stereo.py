"""One-time stereo calibration from a checkerboard, live from the Pi or from saved pairs.

Live (Pi must be sending, see docs/live-link-guide.md):
    py -m gabai.tools.calibrate_stereo --square-mm 25
Offline, from a folder of NNN_L.png / NNN_R.png pairs saved by an earlier run:
    py -m gabai.tools.calibrate_stereo --images data/calib_pairs --square-mm 25

Hold the board (9x6 inner corners, the 10x7-square pattern in data/checkerboard_9x6.png) in view of BOTH
cameras. A pair is captured automatically when the board is steady and in a new position. Cover the whole
image, near and far, tilted both ways. Result is saved to data/stereo_calib.npz (+ .json) and reused by
the viewer. Redo it if you move a camera or change resolution.
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np


def board_points(cols, rows, square_m):
    p = np.zeros((cols * rows, 3), np.float32)
    p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return p


def pose_is_new(corners, accepted, min_shift):
    """True if the board moved at least `min_shift` px (mean corner displacement) from every accepted pose."""
    c = np.asarray(corners).reshape(-1, 2)
    return all(np.linalg.norm(c - np.asarray(a).reshape(-1, 2), axis=1).mean() >= min_shift for a in accepted)


def _per_view_errors(objpts, pts, size, flags):
    r = cv2.calibrateCameraExtended(objpts, pts, size, None, None, flags=flags)
    return r[1], r[2], np.asarray(r[7]).ravel()


def calibrate_from_corners(objpts, lpts, rpts, size, rational=False, min_views=10, fix_k3=False):
    if len(objpts) < min_views:
        raise ValueError(f"only {len(objpts)} board views; need at least {min_views} (aim for 25)")
    flags = (cv2.CALIB_ZERO_TANGENT_DIST | (cv2.CALIB_RATIONAL_MODEL if rational else 0)
             | (cv2.CALIB_FIX_K3 if fix_k3 else 0))
    idx = list(range(len(objpts)))
    for _ in range(3):  # drop views whose corners fit badly, then refit
        o = [objpts[i] for i in idx]
        K1, D1, e1 = _per_view_errors(o, [lpts[i] for i in idx], size, flags)
        K2, D2, e2 = _per_view_errors(o, [rpts[i] for i in idx], size, flags)
        err = np.maximum(e1, e2)
        limit = max(2.5 * float(np.median(err)), 0.8)
        keep = [i for i, e in zip(idx, err) if e <= limit]
        if len(keep) == len(idx) or len(keep) < min_views:
            break
        idx = keep
    if len(idx) < min_views:
        raise ValueError(f"only {len(idx)} good views left after removing bad detections; capture more")
    o = [objpts[i] for i in idx]
    lp, rp = [lpts[i] for i in idx], [rpts[i] for i in idx]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    # pass 1 fixes each camera's own intrinsics; pass 2 refines everything jointly, which pins the baseline
    # down far better (synthetic test: worst baseline error 0.3% vs 7% with pass 1 alone)
    rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        o, lp, rp, K1, D1, K2, D2, size, flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
    rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        o, lp, rp, K1, D1, K2, D2, size, R, T,
        flags=(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | (cv2.CALIB_RATIONAL_MODEL if rational else 0)
               | (cv2.CALIB_FIX_K3 if fix_k3 else 0)),
        criteria=crit)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    ey = []
    for a, b in zip(lp, rp):
        ua = cv2.undistortPoints(a, K1, D1, R=R1, P=P1).reshape(-1, 2)
        ub = cv2.undistortPoints(b, K2, D2, R=R2, P=P2).reshape(-1, 2)
        ey.append(np.abs(ua[:, 1] - ub[:, 1]).mean())
    return dict(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
                size=np.array(size), baseline=float(np.linalg.norm(T)), focal=float(P1[0, 0]),
                cx=float(P1[0, 2]), cy=float(P1[1, 2]), rms_stereo=float(rms),
                epipolar_err_px=float(np.mean(ey)), used=idx, n_views=len(idx),
                swapped=bool(T.ravel()[0] > 0))  # right camera ended up on the LEFT: disparities would be negative


class Rectifier:
    """Loads data/stereo_calib.npz and rectifies live frames. f, B, cy are metric and trustworthy."""

    def __init__(self, path):
        z = np.load(path)
        self.size = tuple(int(v) for v in z["size"])
        self.f, self.B, self.cy = float(z["focal"]), float(z["baseline"]), float(z["cy"])
        self.cx = float(z["cx"]) if "cx" in z else self.size[0] / 2.0
        self.rotation = (int(z["rotate_left"]), int(z["rotate_right"])) if "rotate_left" in z else (0, 0)
        self.m1 = cv2.initUndistortRectifyMap(z["K1"], z["D1"], z["R1"], z["P1"], self.size, cv2.CV_16SC2)
        self.m2 = cv2.initUndistortRectifyMap(z["K2"], z["D2"], z["R2"], z["P2"], self.size, cv2.CV_16SC2)

    def apply(self, left, right):
        return cv2.remap(left, *self.m1, cv2.INTER_LINEAR), cv2.remap(right, *self.m2, cv2.INTER_LINEAR)


RIG_BASELINE_MM = 42.0   # lens-centre spacing in the Fusion 360 model of the printed case
BASELINE_TOL = 0.03


def check_baseline(c, square_mm, nominal_mm=RIG_BASELINE_MM, tol=BASELINE_TOL):
    """Compare the measured baseline with the spacing the case was designed for. The calibrated baseline scales with the square size typed in,
    so a wrong --square-mm shows up as a baseline off by the same factor: the square that would make it match is named."""
    mm = float(c["baseline"]) * 1000.0
    if abs(mm - nominal_mm) <= tol * nominal_mm:
        return f"baseline {mm:.2f} mm agrees with the {nominal_mm:g} mm lens spacing of the case ({(mm - nominal_mm) / nominal_mm * 100:+.2f}%)"
    msg = f"baseline {mm:.2f} mm DOES NOT AGREE with the {nominal_mm:g} mm lens spacing of the case ({(mm - nominal_mm) / nominal_mm * 100:+.1f}%)."
    if square_mm and square_mm > 0:
        msg += f" The baseline scales with --square-mm ({square_mm:g} mm was used): a square of {square_mm * nominal_mm / mm:.2f} mm would make it {nominal_mm:g} mm. Measure the printed square again."
    return msg


def verdict(c):
    if c["rms_stereo"] < 0.5 and c["epipolar_err_px"] < 0.5:
        return "GOOD"
    if c["rms_stereo"] < 1.0 and c["epipolar_err_px"] < 1.0:
        return "OK (usable; more varied views would help)"
    return "POOR (recapture: flatter board, better light, more varied angles)"


def save(c, out, square_mm, rotation=(0, 0)):
    meta = {k: c[k] for k in ("baseline", "focal", "cx", "cy", "rms_stereo", "epipolar_err_px", "n_views", "swapped")}
    meta.update(size=[int(v) for v in c["size"]], square_mm=square_mm, rotate_left=rotation[0], rotate_right=rotation[1], verdict=verdict(c),
                date=time.strftime("%Y-%m-%d %H:%M:%S"))
    np.savez(out + ".npz", **{k: v for k, v in c.items() if k not in ("used", "swapped")}, square_mm=square_mm,
             rotate_left=rotation[0], rotate_right=rotation[1])
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def find_corners(gray, pattern):
    ok, c = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    return (c.astype(np.float32) if ok else None)


def report(c, meta):
    print(f"\ncalibration from {c['n_views']} views")
    print(f"  baseline {c['baseline']*100:.2f} cm   focal {c['focal']:.1f} px "
          f"(horizontal FOV about {np.degrees(2*np.arctan(c['size'][0]/2/c['focal'])):.0f} deg after rectification)")
    print(f"  stereo reprojection error {c['rms_stereo']:.3f} px   epipolar (row) error {c['epipolar_err_px']:.3f} px")
    print(f"  verdict: {meta['verdict']}")
    print("  " + check_baseline(c, meta.get("square_mm") or 0.0))
    if c["swapped"]:
        print("  WARNING: the cameras are SWAPPED (the 'right' camera is physically on the left). "
              "Depth will be badly wrong.")
        print("           Swap them on the Pi (--left <other> --right <other>) and recalibrate, "
              "or swap the saved pairs.")


def run_offline(a):
    pattern = (a.cols, a.rows)
    obj = board_points(a.cols, a.rows, a.square_mm / 1000)
    O, L, R, size = [], [], [], None
    for lf in sorted(glob.glob(os.path.join(a.images, "*_L.png"))):
        rf = lf[:-6] + "_R.png"
        if not os.path.exists(rf):
            continue
        gl, gr = (cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in (lf, rf))
        size = gl.shape[::-1]
        cl, cr = find_corners(gl, pattern), find_corners(gr, pattern)
        if cl is not None and cr is not None:
            O.append(obj), L.append(cl), R.append(cr)
    print(f"board found in {len(O)} pairs")
    rot = (0, 0)
    rj = os.path.join(a.images, "rotation.json")
    if os.path.exists(rj):  # the saved pairs were already rotated by the live run; remember how
        rot = tuple(json.load(open(rj)))
    c = calibrate_from_corners(O, L, R, size, a.rational, fix_k3=a.fix_k3)
    report(c, save(c, a.out, a.square_mm, rot))


def run_live(a):
    from gabai.pc.receiver import LiveReceiver
    pattern = (a.cols, a.rows)
    obj = board_points(a.cols, a.rows, a.square_mm / 1000)
    os.makedirs(a.pairs_dir, exist_ok=True)
    for f in glob.glob(os.path.join(a.pairs_dir, "*.png")):
        os.remove(f)  # a new calibration starts from fresh pairs
    rx = LiveReceiver(a.port, a.token, rotate_left=a.rotate_left, rotate_right=a.rotate_right)
    json.dump([a.rotate_left, a.rotate_right], open(os.path.join(a.pairs_dir, "rotation.json"), "w"))
    rx.start()
    print(f"waiting for the Pi on port {rx.port} ... start: python3 -m gabai.pi.sender --host <PC_IP>", flush=True)
    O, L, R, saved, centers = [], [], [], [], []
    seq, prev, stable, size, msg = 0, None, 0, None, "show the board to both cameras"
    win = "Stereo calibration (c = calibrate now, q = quit)"
    result = None
    while True:
        pair = rx.latest()
        canvas = np.zeros((480, 1280, 3), np.uint8)
        if pair is not None and pair.seq != seq and result is None:
            seq = pair.seq
            gl = cv2.cvtColor(pair.left, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(pair.right, cv2.COLOR_BGR2GRAY)
            size = gl.shape[::-1]
            cl, cr = find_corners(gl, pattern), find_corners(gr, pattern)
            vl, vr = pair.left.copy(), pair.right.copy()
            if cl is not None:
                cv2.drawChessboardCorners(vl, pattern, cl, True)
            if cr is not None:
                cv2.drawChessboardCorners(vr, pattern, cr, True)
            canvas = np.hstack([vl, vr])
            if cl is None or cr is None:
                msg, stable, prev = ("board not seen by the " + ("right" if cl is not None else
                                     "left" if cr is not None else "left or right") + " camera"), 0, None
            else:
                moved = np.abs(cl - prev).mean() if prev is not None else 99
                prev = cl
                stable = stable + 1 if moved < 1.5 else 0
                if not pose_is_new(cl, saved, a.min_shift):
                    msg = "move the board to a NEW position/angle/distance"
                elif stable < 3:
                    msg = "hold still..."
                else:
                    n = len(O)
                    cv2.imwrite(os.path.join(a.pairs_dir, f"{n:03d}_L.png"), pair.left)
                    cv2.imwrite(os.path.join(a.pairs_dir, f"{n:03d}_R.png"), pair.right)
                    O.append(obj), L.append(cl), R.append(cr), saved.append(cl)
                    centers.append(tuple(cl.reshape(-1, 2).mean(0).astype(int)))
                    msg, stable = f"CAPTURED pair {len(O)}/{a.pairs}", 0
                    print(msg, flush=True)
            for cx, cy in centers:
                cv2.circle(canvas, (cx, cy), 6, (0, 255, 0), -1)
            if len(O) >= a.pairs:
                key = ord("c")
            else:
                key = None
        elif pair is not None and result is None:
            canvas = np.hstack([pair.left, pair.right])
            key = None
        else:
            key = None
        if result is None:
            cv2.putText(canvas, f"{len(O)}/{a.pairs} pairs   {msg}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
            if not a.no_window:
                cv2.imshow(win, canvas)
        k = (cv2.waitKey(30) & 0xFF if not a.no_window else (time.sleep(0.03) or 255)) if key is None else key
        if k in (ord("q"), 27):
            break
        if k == ord("c") and result is None:
            try:
                c = calibrate_from_corners(O, L, R, size, a.rational, fix_k3=a.fix_k3)
                meta = save(c, a.out, a.square_mm, (a.rotate_left, a.rotate_right))
                report(c, meta)
                result = meta
                shown = np.zeros((300, 900, 3), np.uint8)
                for i, t in enumerate([f"Saved {a.out}.npz",
                                       f"baseline {c['baseline']*100:.2f} cm   focal {c['focal']:.0f} px",
                                       f"reprojection {c['rms_stereo']:.2f} px   row error {c['epipolar_err_px']:.2f} px",
                                       f"verdict: {meta['verdict']}", "press q to close"]):
                    cv2.putText(shown, t, (20, 50 + 55 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if not a.no_window:
                    cv2.imshow(win, shown)
                else:
                    break
            except ValueError as e:
                msg = str(e)
                print(msg, flush=True)
    rx.stop()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--square-mm", type=float, required=True, help="side of one printed/displayed square, measured with a ruler")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--pairs", type=int, default=25, help="board views to capture before calibrating")
    ap.add_argument("--min-shift", type=float, default=40, help="px the board must move between captured pairs")
    ap.add_argument("--rational", action="store_true", help="stronger lens model for very wide lenses (needs more views)")
    ap.add_argument("--no-window", action="store_true", help="testing: no GUI, exit after calibrating")
    ap.add_argument("--rotate-left", type=int, default=0, choices=[0, 90, 180, 270], help="turn the left image (e.g. 180 if that camera is mounted upside down)")
    ap.add_argument("--rotate-right", type=int, default=0, choices=[0, 90, 180, 270], help="turn the right image")
    ap.add_argument("--fix-k3", action="store_true", help="simpler lens model (k1,k2 only); avoids wild edge distortion when the fit is shaky")
    ap.add_argument("--images", help="offline: folder of NNN_L.png / NNN_R.png pairs")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--pairs-dir", default="data/calib_pairs")
    ap.add_argument("--out", default="data/stereo_calib", help="output path without extension")
    a = ap.parse_args()
    run_offline(a) if a.images else run_live(a)


if __name__ == "__main__":
    main()
