"""Calibrate the accelerometer (bias and scale per axis) by holding the rig still in the six face-up positions.

    py -m gabai.tools.calibrate_accel                  # live: start the sender on the Pi first
    py -m gabai.tools.calibrate_accel --from data/slam/rec_<time> ...   # fit from recordings that already hold several still poses

A still accelerometer must read exactly 9.807 m/s^2 in every direction. Yours reads about 4 percent low (9.42), and Basalt
integrates that error into metres of drift, so its result becomes unstable. Live: put the rig down with each of its six sides
facing the ground in turn (the cameras up, down, left, right, forward, back). Hold each one dead still for two seconds. The
tool says which sides it still needs. Add a few tilted holds in between if you can; more directions give a better fit.

It writes data/accel_calib.json. record_basalt and live_basalt then pass those numbers to Basalt.
"""
import argparse
import json
import os
import time

import numpy as np

from gabai.slam.accel_cal import (G, StillCapture, cluster_orientations, faces_covered, fit_bias_scale, still_windows,
                                  to_basalt_params)

MIN_ORIENTATIONS = 6


def report_and_save(means, out, force=False):
    cov = faces_covered(means)
    missing = [k for k, v in cov.items() if not v]
    fit = fit_bias_scale(means)
    print(f"\n{len(means)} orientations; faces covered: " + " ".join(f"{k}{'+' if v else '-'}" for k, v in cov.items()))
    print(f"bias  (m/s^2): {np.round(fit['bias'], 3)}  +- {np.round(fit['std'][:3], 3)}")
    print(f"scale (error): {np.round(fit['scale'], 4)}  +- {np.round(fit['std'][3:], 4)}")
    print(f"gravity error before: {fit['rms_before']:.3f} m/s^2 rms, after: {fit['rms_after']:.3f} m/s^2 rms")
    ok = fit["observable"] and len(means) >= MIN_ORIENTATIONS and not missing
    if not ok and not force:
        why = []
        if missing:
            why.append("missing faces: " + " ".join(missing))
        if not fit["observable"] or len(means) < MIN_ORIENTATIONS:
            why.append("the orientations do not pin down bias and scale separately")
        print("NOT SAVED (" + "; ".join(why) + "). Add holds in those directions, or use --force to save anyway.")
        return False
    json.dump({"bias": fit["bias"].tolist(), "scale": fit["scale"].tolist(), "basalt_params": to_basalt_params(fit["bias"], fit["scale"]),
               "rms_before": fit["rms_before"], "rms_after": fit["rms_after"], "n_orientations": len(means),
               "faces_missing": missing, "date": time.strftime("%Y-%m-%d %H:%M:%S")}, open(out, "w"), indent=2)
    print(f"saved {out}")
    return True


def from_recordings(folders):
    means = []
    for d in folders:
        imu = np.loadtxt(f"{d}/mav0/imu0/data.csv", delimiter=",", comments="#")
        means += list(still_windows(imu[:, 0] * 1e-9, imu[:, 4:7], imu[:, 1:4]))
    return cluster_orientations(np.array(means)) if means else np.empty((0, 3))


def live(a):
    from gabai.pc.receiver import LiveReceiver
    rx = LiveReceiver(a.port, a.token)
    rx.start()
    cap, means = StillCapture(min_s=a.hold_s), []
    print(f"waiting for the Pi on port {rx.port} ... then put each side of the rig on the table in turn, hold dead still for {a.hold_s:.0f} s", flush=True)
    try:
        while True:
            samples = rx.drain_imu()
            if not samples:
                time.sleep(0.01)
            for t, gx, gy, gz, ax, ay, az in samples:
                m = cap.feed(t, np.array([ax, ay, az]) * G, np.radians([gx, gy, gz]))
                if m is not None:
                    means = list(cluster_orientations(np.vstack(means + [m]))) if means else [m]
                    cov = faces_covered(np.array(means))
                    need = [k for k, v in cov.items() if not v]
                    print(f"hold {len(means)} captured: reading {np.round(m, 2)} (|a| = {np.linalg.norm(m):.3f}); still needed: {' '.join(need) or 'none'}", flush=True)
                    if not need and len(means) >= MIN_ORIENTATIONS:
                        raise StopIteration
    except (StopIteration, KeyboardInterrupt):
        pass
    finally:
        rx.stop()
    return np.array(means)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="folders", nargs="+", help="recording folders to fit from instead of the live stream")
    ap.add_argument("--out", default="data/accel_calib.json")
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--force", action="store_true", help="save even if the data cannot separate bias from scale")
    a = ap.parse_args()
    means = from_recordings(a.folders) if a.folders else live(a)
    if len(means) < 3:
        raise SystemExit(f"only {len(means)} still orientation(s) found: not enough to calibrate")
    report_and_save(np.asarray(means), a.out, a.force)


if __name__ == "__main__":
    main()
