"""Synchronise the camera, the BMI160 (gyro) and the HMC5883L (compass) in one recording.

All three are stamped on the Pi's clock when read, but each one's stamp is late by a different, unknown amount (the
camera by its exposure and USB transfer, the compass by its internal averaging and polling). The gyro is the
reference. From ONE recording of you rotating the rig this tool measures, with no board and no light:

  * the compass calibration (hard/soft iron) from the sphere the readings trace,
  * the camera delay and which gyro axis is which camera axis (motion in the pictures vs the gyro),
  * the compass delay and which compass axis is which gyro axis (the field's rate of change vs the gyro).

Run with the Pi streaming (sender must report IMU found and compass found), a textured scene in front of the cameras:
    py -m gabai.tools.sync_sensors
Then rotate the rig by hand for 30 s: tumble it through EVERY direction (up, down, upside down, left, right),
quickly and smoothly. Results: data/imu_sync.json (camera), data/mag_cal.json and data/mag_sync.json (compass).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from gabai.mag import apply_cal, calibrate_ellipsoid
from gabai.mag_sync import estimate_mag_delay
from gabai.tools.sync_imu_camera import estimate_axes, estimate_delay, estimate_delay_xy, visual_rates


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--calib", default="data/stereo_calib_new.npz")
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()

    from gabai.pc.receiver import LiveReceiver
    rot, fx = (0, 0), 740.0
    if os.path.exists(a.calib):
        z = np.load(a.calib)
        rot = (int(z["rotate_left"]), int(z["rotate_right"])) if "rotate_left" in z else (0, 0)
        fx = float(z["K1"][0, 0])
    rx = LiveReceiver(a.port, a.token, rotate_left=rot[0], rotate_right=rot[1])
    rx.start()
    print(f"waiting for the Pi on port {rx.port} ...", flush=True)
    frames, times, imu, mag, seq, t_start = [], [], [], [], 0, None
    while True:
        pair = rx.latest()
        imu += rx.drain_imu()
        mag += rx.drain_mag()
        if pair is not None and pair.seq != seq:
            seq = pair.seq
            if t_start is None:
                t_start = time.monotonic()
                print(f"recording {a.seconds:.0f} s: TUMBLE THE RIG through every direction (also upside down), quickly and smoothly", flush=True)
            frames.append(cv2.resize(cv2.cvtColor(pair.left, cv2.COLOR_BGR2GRAY), (320, 240), interpolation=cv2.INTER_AREA))
            times.append(pair.t_capture)
        if t_start is not None and time.monotonic() - t_start > a.seconds:
            break
        time.sleep(0.005)
    imu += rx.drain_imu()
    mag += rx.drain_mag()
    rx.stop()
    print(f"collected {len(frames)} frames, {len(imu)} gyro samples, {len(mag)} compass samples", flush=True)
    if len(frames) < 60 or len(imu) < 800:
        raise SystemExit("not enough camera or gyro data; is the sender reporting 'IMU found'?")

    os.makedirs(a.data_dir, exist_ok=True)
    arr = np.array(imu)
    t_imu, gyro = arr[:, 0], np.radians(arr[:, 1:4]).T
    speed = np.linalg.norm(gyro, axis=0)
    fast = np.degrees(np.percentile(speed, 90))
    if fast < 30:
        raise SystemExit(f"the rig moved too gently (fast gyro speed {fast:.0f} deg/s); rotate harder, aim for 60+ deg/s")
    rows = []

    # ---- camera vs gyro
    tv, wx, wy = (np.array(v) for v in zip(*visual_rates(frames, times, fx * 320 / 640)))
    cam_delay, cam_q = estimate_delay(tv, np.hypot(wx, wy), t_imu, speed)
    R_ci, C = estimate_axes(tv, wx, wy, t_imu, gyro, cam_delay)
    cam_delay, cam_q = estimate_delay_xy(tv, wx, wy, t_imu, gyro, R_ci)     # second pass, with signs and the axis map
    R_ci, C = estimate_axes(tv, wx, wy, t_imu, gyro, cam_delay)
    json.dump({"delay_s": cam_delay, "R_ci": R_ci.tolist(), "quality": cam_q, "date": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(a.data_dir, "imu_sync.json"), "w"), indent=2)
    rows.append(("camera (left image)", cam_delay, cam_q, 0.5, "correlation of picture rotation and gyro, per axis"))

    # ---- compass
    if len(mag) < 300:
        print("\nNo compass data (fewer than 300 readings): the sender must print 'compass found'. Compass skipped.")
    else:
        m = np.array(mag)
        t_mag, raw = m[:, 0], m[:, 1:4]
        cal = calibrate_ellipsoid(raw)
        print(f"\ncompass calibration: offset {np.round(cal['offset'], 1)}, scale {np.round(cal['scale'], 3)}, "
              f"quality {cal['quality']:.3f}, directions covered {cal['coverage'] * 100:.0f}%, method {cal['method']}, field after calibration {cal['radius_counts'] * 0.092:.0f} uT")
        if cal["coverage"] < 0.75 or cal["quality"] > 0.08:
            print("  WARNING: the compass was not turned through enough directions; its sync result will be unreliable. "
                  "Rerun and tumble the rig through every direction.")
        fixed = apply_cal(cal, raw)
        json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cal.items()},
                  open(os.path.join(a.data_dir, "mag_cal.json"), "w"), indent=2)
        mag_delay, R_im, score = estimate_mag_delay(t_mag, fixed, t_imu, gyro.T)
        json.dump({"delay_s": mag_delay, "R_im": R_im.tolist(), "score": score, "date": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(a.data_dir, "mag_sync.json"), "w"), indent=2)
        rows.append(("compass", mag_delay, score, 0.8, "share of the field's rate of change explained by the gyro"))
        print("compass axes -> IMU axes (imu = R @ compass):\n", R_im.astype(int))

    print("\n" + "=" * 78)
    print(f"{'sensor':22s} {'delay vs gyro':>14s} {'match':>7s}   verdict")
    print(f"{'gyro (reference)':22s} {'0 ms':>14s} {'':>7s}   reference")
    for name, d, q, need, what in rows:
        print(f"{name:22s} {d * 1000:+11.0f} ms {q:7.2f}   {'GOOD' if q >= need else 'WEAK'}  ({what})")
    print("=" * 78)
    print("Delay > 0 means the sensor's time stamp is later than the motion it reports. The viewer now looks each sensor up "
          "at (frame time - delay), so camera, gyro and compass describe the same instant.")
    print(f"Saved in {a.data_dir}/: imu_sync.json, mag_cal.json, mag_sync.json")


if __name__ == "__main__":
    main()
