"""Compare the orientation filter before and after the 2026-10-30 change on REAL sensor recordings.

py scripts/evaluate_orientation.py                              all EuRoC-format SLAM recordings under data/slam (gyro + accelerometer, no compass)
py scripts/evaluate_orientation.py --euroc data/slam/live_20261026_181526
py scripts/evaluate_orientation.py --sensors data/sensors/still_xxx.npz     a gabai.tools.record_sensors recording (gyro + accelerometer + compass)

'before' is the original gradient-descent Madgwick filter, kept here only as the yardstick; 'after' is gabai.orient.Madgwick.
There is no ground truth for a hand-held recording, so the metrics are the ones that need none:
  correction  the angular rate (deg/s) the filter adds to the gyro. Rest or turn, a good filter adds little when it is not needed.
  tilt error  angle between the filter's 'up' and the accelerometer's, taken only where the accelerometer length is within 3% of 1 g (so it is gravity alone).
              A filter that has drifted or lags shows it here; a filter that ignores the accelerometer fails it.
  wobble      rms angle between the filter's 'up' and its own 0.5 s average while the device is still: the jitter you see on a resting board.
  heading     (recordings with a compass) rms of the heading's sample-to-sample step while the device is still.
Classes by the 0.25 s mean of |gyro|: still < 3, slow 3 to 30, fast > 30 deg/s."""
import argparse
import glob
import math
import os

import numpy as np

from gabai.orient import GyroBias, HeadingTracker, Madgwick, q_from_accel, q_mul, q_norm, q_to_R


class GradientMadgwick:
    """The filter that ran until 2026-10-29, verbatim, with the compass term left out (the EuRoC recordings have none)."""

    def __init__(self, beta=0.05):
        self.beta, self.q = beta, np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro_rad, accel, dt):
        q1, q2, q3, q4 = self.q
        a = np.asarray(accel, float)
        n = np.linalg.norm(a)
        qdot = 0.5 * q_mul(self.q, np.array([0.0, *gyro_rad]))
        if n > 1e-6:
            ax, ay, az = a / n
            f = np.array([2 * (q2 * q4 - q1 * q3) - ax, 2 * (q1 * q2 + q3 * q4) - ay, 2 * (0.5 - q2 * q2 - q3 * q3) - az])
            J = np.array([[-2 * q3, 2 * q4, -2 * q1, 2 * q2], [2 * q2, 2 * q1, 2 * q4, 2 * q3], [0.0, -4 * q2, -4 * q3, 0.0]])
            step = J.T @ f
            sn = np.linalg.norm(step)
            if sn > 0:
                qdot = qdot - self.beta * step / sn
        self.q = q_norm(self.q + qdot * dt)
        return self.q


def load_euroc(folder):
    d = np.loadtxt(os.path.join(folder, "mav0", "imu0", "data.csv"), delimiter=",", skiprows=1)
    return d[:, 0] * 1e-9, np.degrees(d[:, 1:4]), d[:, 4:7] / 9.80665, None


def load_sensors(path):
    d = np.load(path)
    mag = None
    if "mag" in d.files and len(d["mag"]):
        from gabai.mag import apply_cal, load_mag_cfg
        cfg = load_mag_cfg()
        m = d["mag"]
        mag = (m[:, 0] - cfg["delay_s"], (cfg["R_im"] @ apply_cal(cfg, m[:, 1:4]).T).T)
    return d["imu"][:, 0], d["imu"][:, 1:4], d["imu"][:, 4:7], mag


def body_rate(q0, q1, dt):
    c = q_mul(np.array([q0[0], -q0[1], -q0[2], -q0[3]]), q1)
    if c[0] < 0:
        c = -c
    s = np.linalg.norm(c[1:])
    return c[1:] / s * 2 * math.atan2(s, c[0]) / dt if s > 1e-12 else np.zeros(3)


def run(filt, t, g, a, mag=None, use_bias=True):
    """Feeds every sample through filt (and GyroBias, as gabai.fusion does). Returns per-sample: correction (dps), the filter's 'up' in the body frame, heading (deg)."""
    bias = GyroBias() if use_bias else None
    tracker = HeadingTracker()                                                   # heading of the body axis that is most horizontal at the start (a level board has no horizontal z)
    filt.q = q_from_accel(a[0])
    n = len(t)
    corr, up, head = np.zeros((n, 3)), np.zeros((n, 3)), np.full(n, np.nan)
    j = 0
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if not 0 < dt < 0.2:
            up[i] = up[i - 1]
            continue
        gyro = bias.feed(t[i], g[i], a[i]) if bias else g[i]
        q0 = filt.q.copy()
        m = None
        if mag is not None:
            while j + 1 < len(mag[0]) and mag[0][j + 1] <= t[i]:
                j += 1
            m = mag[1][j] if mag[0][j] <= t[i] and t[i] - mag[0][j] < 0.1 else None
        if m is not None and hasattr(filt, "update_marg"):
            filt.update_marg(np.radians(gyro), a[i], m, dt)
        else:
            filt.update(np.radians(gyro), a[i], dt)
        corr[i] = np.degrees(body_rate(q0, filt.q, dt)) - gyro
        up[i] = q_to_R(filt.q).T @ [0, 0, 1.0]
        head[i] = tracker.update(filt.q)[1]                                      # continuous heading in degrees
    return corr, up, head


def evaluate(name, t, g, a, mag):
    k = max(3, int(0.25 / np.median(np.diff(t))))
    spin = np.convolve(np.linalg.norm(g, axis=1), np.ones(k) / k, mode="same")
    an = np.linalg.norm(a, axis=1)
    classes = {"still": spin < 3, "slow": (spin >= 3) & (spin < 30), "fast": spin >= 30}
    u_acc = a / an[:, None]
    rows = {}
    for label, factory in (("before", lambda: GradientMadgwick(0.05)), ("after", lambda: Madgwick(0.05))):
        corr, up, head = run(factory(), t, g, a, mag if label == "after" else None)
        tilt = np.degrees(np.arccos(np.clip((up * u_acc).sum(1), -1, 1)))
        w = int(0.5 / np.median(np.diff(t)))
        ma = np.array([np.convolve(up[:, i], np.ones(w) / w, mode="same") for i in range(3)]).T
        ma /= np.maximum(np.linalg.norm(ma, axis=1, keepdims=True), 1e-9)
        wob = np.degrees(np.arccos(np.clip((up * ma).sum(1), -1, 1)))
        r = {}
        for c, m in classes.items():
            if m.sum() < 100:
                continue
            trusted = m & (np.abs(an - 1.0) < 0.03)
            r[c] = dict(n=int(m.sum()), correction_rms=float(np.sqrt((np.linalg.norm(corr[m], axis=1) ** 2).mean())),
                        tilt_med=float(np.median(tilt[trusted])) if trusted.sum() > 20 else float("nan"),
                        tilt_p95=float(np.percentile(tilt[trusted], 95)) if trusted.sum() > 20 else float("nan"),
                        wobble_rms=float(np.sqrt((wob[m] ** 2).mean())))
        if mag is not None and label == "after":
            st = classes["still"] & np.isfinite(head)
            step = np.abs(np.diff(head[st]))
            r["heading_step_rms_still"] = float(np.sqrt((step ** 2).mean())) if len(step) else float("nan")
        rows[label] = r
    print(f"\n== {name}: {len(t)} samples, {t[-1]-t[0]:.1f} s, {len(t)/(t[-1]-t[0]):.0f} Hz, compass {'yes' if mag is not None else 'no'}")
    print(f"   {'':6s} {'class':6s} {'n':>6s} {'correction dps':>15s} {'tilt err med/p95 deg':>22s} {'wobble deg':>11s}")
    for label in ("before", "after"):
        for c in ("still", "slow", "fast"):
            if c in rows[label]:
                x = rows[label][c]
                print(f"   {label:6s} {c:6s} {x['n']:6d} {x['correction_rms']:15.2f} {x['tilt_med']:10.2f} /{x['tilt_p95']:8.2f} {x['wobble_rms']:11.4f}")
    if "heading_step_rms_still" in rows["after"]:
        print(f"   heading step rms while still: {rows['after']['heading_step_rms_still']:.4f} deg per sample")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", nargs="*")
    ap.add_argument("--sensors", nargs="*")
    a = ap.parse_args()
    srcs = []
    if a.euroc is None and a.sensors is None:
        a.euroc = [os.path.dirname(os.path.dirname(os.path.dirname(p))) for p in sorted(glob.glob("data/slam/*/mav0/imu0/data.csv")) if "_mono" not in p and "_realigned" not in p and os.path.getsize(p) > 200_000]
    srcs += [(os.path.basename(p), load_euroc(p)) for p in a.euroc or []]
    srcs += [(os.path.basename(p), load_sensors(p)) for p in a.sensors or []]
    for name, (t, g, acc, mag) in srcs:
        evaluate(name, t, g, acc, mag)


if __name__ == "__main__":
    main()
