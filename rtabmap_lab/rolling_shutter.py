"""Opposite rolling shutters: the left camera is mounted upside down, so it reads its rows bottom-to-top while the right one reads top-to-bottom.

The two pictures of a pair are therefore NOT taken at one instant per row: the right row v is exposed dt(v) = a + b * (v / h - 0.5) after the left row v. Measured on walk_20261102_230115
(220 pairs, SIFT matches of rectified pairs against the recorded gyro): a = +1.05 ms (the pair offset the Pi reports, 1.1 ms), b = +51 ms (correlation 0.89). While the rig turns, every row of
the right picture is seen from a slightly rotated camera:
    vertical:   y_R - y_L = f * w_x * dt(v)            -> rows no longer line up, the matcher fails or matches the wrong thing
    horizontal: x_L - x_R = d_true + f * w_y * dt(v)   -> disparity is biased, + or - depending on the row and the turn: at 24 deg/s yaw up to 6 px at the top and bottom of the picture
(2 m is 14 px of disparity with this rig). That is why walls bend, why every matcher has the same "noise", and why depth maps of nodes taken while turning disagree.

correct(right, w_cam, f, cx, cy) resamples the right rectified picture at the left picture's time for every row (first-order rotational image motion, w_cam = angular rate in the optical frame of the
pictures RTAB-Map gets, rad/s). It cannot fix translation during the readout (small at walking speed) and it is only as good as the gyro (0.9 deg agreement with the odometry on that walk).
The real fix is mechanical: mount both cameras the same way up (or flip the left sensor in hardware, if its driver offers vertical_flip/horizontal_flip, and drop the software 180 deg turn).

  py -m rtabmap_lab.rolling_shutter data/recordings/<folder> --calib data/stereo_calib_charuco_45.npz [--save data/rolling_shutter.json]
measures a and b on a recording (and, with --save, writes them for live.py / the lab), and shows the row error before and after the correction.
"""
import argparse
import json
import os

import cv2
import numpy as np

DEFAULT_PATH = "data/rolling_shutter.json"


class ShutterCorrection:
    def __init__(self, a_ms, b_ms):
        """a, b in the CALIBRATED picture frame (cameras as calibrated, upright)."""
        self.a, self.b = a_ms / 1000.0, b_ms / 1000.0
        self._grid = None

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        if not path or not os.path.exists(path):
            return None
        d = json.load(open(path))
        return cls(d["a_ms"], d["b_ms"])

    def dt(self, h, turned=False, a_ms=None):
        """Right-minus-left exposure time of every row (s). a_ms: this pair's measured offset (the Pi's dt_ms, signed, in the frame of these pictures), else the stored one.
        turned: the rig is upside down and the pictures were turned 180 deg and traded (flip): the stored offset changes sign, the slope does not."""
        v = (np.arange(h) + 0.5) / h - 0.5
        a = a_ms / 1000.0 if a_ms is not None else (-self.a if turned else self.a)
        return a + self.b * v

    def correct(self, right, w_cam, f, cx, cy, turned=False, a_ms=None):
        """The right picture as it would look at the left picture's time, row by row. w_cam: (3,) rad/s, optical frame of these pictures."""
        if w_cam is None:
            return right
        h, w = right.shape[:2]
        if self._grid is None or self._grid[0].shape != (h, w):
            v, u = np.mgrid[0:h, 0:w].astype(np.float32)
            self._grid = (u, v)
        u, v = self._grid
        x, y = u - cx, v - cy
        wx, wy, wz = (float(c) for c in w_cam)
        dt = self.dt(h, turned, a_ms).astype(np.float32)[:, None]
        mu = (-f * wy + wz * y) * dt                                       # image motion of a fixed point while the camera turns (first order), times the row's time offset
        mv = (f * wx - wz * x) * dt
        return cv2.remap(right, u + mu, v + mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def gyro_rate(imu, t, delay=0.0, window=0.02):
    """Mean gyro (rad/s, IMU axes) within +-window of t, or None. imu rows: t, gyro deg/s (3), ..."""
    s = np.abs(imu[:, 0] + delay - t) < window
    return np.radians(imu[s, 1:4].mean(0)) if s.any() else None


def row_stats(left, right):
    """(row of each match in the left picture, y_R - y_L, x_L - x_R) for SIFT matches of a rectified pair, or None."""
    sift, bf = cv2.SIFT_create(3000), cv2.BFMatcher()
    gl, gr = (cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if p.ndim == 3 else p for p in (left, right))
    ka, da = sift.detectAndCompute(gl, None)
    kb, db = sift.detectAndCompute(gr, None)
    if da is None or db is None or len(ka) < 50 or len(kb) < 2:
        return None
    m = [a for a, b in bf.knnMatch(da, db, 2) if a.distance < 0.75 * b.distance]
    if len(m) < 60:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    dy, dx = pb[:, 1] - pa[:, 1], pa[:, 0] - pb[:, 0]
    ok = (np.abs(dy) < 8) & (dx > -2) & (dx < 120)
    return (pa[ok, 1], dy[ok], dx[ok]) if ok.sum() >= 50 else None


def measure(rec, rect, corr=None, step=3):
    """Fits dy = f * w_x * (a + b (v/h - 0.5)) over the pairs of a recording (rotation turned into the calibrated frame). Returns a dict, and the row error with and without `corr`."""
    turned = rec.flip                                                         # a layout swap gives back the calibrated cameras: only a flip turns the frame
    F = np.diag([-1.0, -1.0, 1.0]) if turned else np.eye(3)
    X, Y, res = [], [], []
    for i in range(0, rec.n, step):
        wi = gyro_rate(rec.imu, rec.t[i], rec.imu_delay)
        if wi is None:
            continue
        l, r = rec.prepare(i, rect, shutter=corr)
        h = l.shape[0]
        f = rect.f * rec.k
        wc = F @ rec.R_ci @ wi
        s = row_stats(l, r)
        if s is None:
            continue
        v, dy, _ = s
        rv = v / h - 0.5
        X.append(np.stack([f * wc[0] * np.ones_like(rv), f * wc[0] * rv], 1))
        Y.append(dy)
        res.append(np.median(np.abs(dy - np.median(dy))))
    X, Y = np.concatenate(X), np.concatenate(Y)
    A = np.column_stack([X, np.ones(len(X))])
    coef = np.linalg.lstsq(A, Y, rcond=None)[0]
    pred = A @ coef
    sign = -1.0 if turned else 1.0                                           # back to the calibrated frame
    return {"a_ms": float(coef[0] * 1000 * sign), "b_ms": float(coef[1] * 1000), "pairs": len(res), "matches": int(len(Y)),
            "row_spread_px": float(np.median(res)), "explained": float(1 - np.var(Y - pred) / np.var(Y))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--save", help=f"write a and b here (live.py and the lab read {DEFAULT_PATH})")
    ap.add_argument("--step", type=int, default=3)
    a = ap.parse_args(argv)
    from rtabmap_lab.calib_lab import params as P
    from rtabmap_lab.calib_lab.recording import Recording
    rect = P.CalibParams.from_npz(a.calib).rect()
    rec = Recording(a.folder).decide(rect)
    if rec.imu is None or rec.R_ci is None:
        raise SystemExit("this recording has no imu.csv (or no camera-IMU rotation file): the readout cannot be measured")
    rec.shutter = None
    m = measure(rec, rect, step=a.step)
    print(f"right-minus-left exposure time: {m['a_ms']:+.2f} ms at the middle row, changing by {m['b_ms']:+.1f} ms from the top row to the bottom row "
          f"({m['pairs']} pairs, {m['matches']} matches, gyro explains {m['explained'] * 100:.0f}% of the row mismatch)")
    print("  opposite rolling shutters" if abs(m["b_ms"]) > 10 else "  no row-dependent offset worth correcting (global shutter or same readout direction)")
    after = measure(rec, rect, ShutterCorrection(m["a_ms"], m["b_ms"]), step=a.step)
    print(f"after the correction the gyro explains {after['explained'] * 100:.0f}% of what is left (slope {after['b_ms']:+.1f} ms); row spread per pair {m['row_spread_px']:.2f} -> {after['row_spread_px']:.2f} px")
    if a.save:
        json.dump({**m, "recording": rec.folder, "calib": a.calib, "after": after, "note": "a, b in the calibrated (upright) frame; a is replaced by each pair's measured offset when known; see rtabmap_lab/rolling_shutter.py"}, open(a.save, "w"), indent=1)
        print("saved", a.save)


if __name__ == "__main__":
    main()
