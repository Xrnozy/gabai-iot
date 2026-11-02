"""Line up the IMU with the camera: measure the delay between them and which gyro axis is which camera axis.

Method: the camera's own motion is measured from the pictures (phase correlation between consecutive frames gives
the image shift, and shift = focal * angular velocity * dt), then compared with the gyro over time. The lag with the
best match is the delay; the gyro axes that move with the picture's horizontal and vertical shift are the camera's
pitch and yaw axes. No light, no board.

Run with the Pi streaming (sender with the IMU), then ROTATE THE RIG by hand for about 25 s: sweep it left-right and
up-down, quickly and smoothly, with a textured scene in view (shelves, posters, not a blank wall).

    py -m gabai.tools.sync_imu_camera
Writes data/imu_sync.json, which the viewer uses to time-match the orientation to each camera frame.

Camera frame convention (OpenCV): x right, y down, z forward. A small turn with angular velocity w moves a point on the
optical axis by dx = -f * w_y * dt, dy = +f * w_x * dt pixels.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np


def resample(t_src, y_src, t_new):
    return np.interp(t_new, t_src, y_src)


def visual_rates(frames, times, f_px):
    """Camera angular velocity about its x and y axes from consecutive frames: list of (t_mid, wx, wy) in rad/s."""
    out = []
    win = cv2.createHanningWindow((frames[0].shape[1], frames[0].shape[0]), cv2.CV_32F)
    prev = np.float32(frames[0])
    for k in range(1, len(frames)):
        cur = np.float32(frames[k])
        (dx, dy), _ = cv2.phaseCorrelate(prev, cur, win)
        dt = times[k] - times[k - 1]
        if dt > 0:
            out.append(((times[k] + times[k - 1]) / 2, dy / (f_px * dt), -dx / (f_px * dt)))
        prev = cur
    return out


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def estimate_delay(t_vis, vis_signal, t_imu, imu_signal, lags=None):
    """Delay (s) such that vis(t) matches imu(t - delay), i.e. the frame stamp is later than the motion it shows.
    Returns (delay, correlation at that delay)."""
    lags = np.arange(-0.25, 0.4001, 0.002) if lags is None else lags
    best = (0.0, -2.0)
    for lag in lags:
        c = _corr(vis_signal, resample(t_imu, imu_signal, t_vis - lag))
        if c > best[1]:
            best = (float(lag), c)
    return best


def estimate_delay_xy(t_vis, wx_vis, wy_vis, t_imu, gyro_xyz, R_ci, lags=None):
    """Delay refined with signs: once the axis map is known, the picture's x and y rotation rates must follow the
    matching gyro axes exactly, which pins the lag down better than comparing overall speeds."""
    lags = np.arange(-0.25, 0.4001, 0.002) if lags is None else lags
    cam = R_ci @ np.asarray(gyro_xyz)
    best = (0.0, -2.0)
    for lag in lags:
        c = 0.5 * (_corr(wx_vis, resample(t_imu, cam[0], t_vis - lag)) + _corr(wy_vis, resample(t_imu, cam[1], t_vis - lag)))
        if c > best[1]:
            best = (float(lag), c)
    return best


def estimate_axes(t_vis, wx_vis, wy_vis, t_imu, gyro_xyz, delay):
    """R_ci (3x3 signed permutation, cam = R_ci @ imu) from how each gyro axis correlates with the visual x/y rates."""
    g = np.stack([resample(t_imu, gyro_xyz[i], t_vis - delay) for i in range(3)])
    C = np.array([[_corr(g[i], v) for v in (wx_vis, wy_vis)] for i in range(3)])
    R = np.zeros((3, 3))
    ix = int(np.argmax(np.abs(C[:, 0])))
    R[0, ix] = np.sign(C[ix, 0]) or 1.0
    order = [i for i in np.argsort(-np.abs(C[:, 1])) if i != ix]
    iy = int(order[0])
    R[1, iy] = np.sign(C[iy, 1]) or 1.0
    R[2] = np.cross(R[0], R[1])
    return R, C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=25)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--calib", default="data/stereo_calib_new.npz")
    ap.add_argument("--out", default="data/imu_sync.json")
    a = ap.parse_args()

    from gabai.pc.receiver import LiveReceiver
    rot, fx = (0, 0), 740.0
    if os.path.exists(a.calib):
        z = np.load(a.calib)
        rot = (int(z["rotate_left"]), int(z["rotate_right"])) if "rotate_left" in z else (0, 0)
        fx = float(z["K1"][0, 0])
    rect = None
    if os.path.exists(a.calib):
        from gabai.tools.calibrate_stereo import Rectifier
        from gabai.pipeline.sources import layout_needs_swap
        rect = Rectifier(a.calib)
    swap, n_seen = None if rect is not None else False, 0
    rx = LiveReceiver(a.port, a.token, rotate_left=rot[0], rotate_right=rot[1])
    rx.start()
    print(f"waiting for the Pi on port {rx.port} (sender must have the IMU working) ...", flush=True)
    frames, times, imu, seq, t_start = [], [], [], 0, None
    while True:
        pair = rx.latest()
        imu += rx.drain_imu()
        if pair is not None and pair.seq != seq:
            seq = pair.seq
            if swap is None:                                       # the live pipeline's camera-layout decision: the axes found here must be those of the live pictures
                swap = layout_needs_swap(rect, pair.left, pair.right, n_seen)[0]
                n_seen += 1
                if swap is None:
                    continue
                print("camera layout: " + ("SWAPPED (both pictures turned 180 deg and traded)" if swap else "as calibrated"), flush=True)
            if t_start is None:
                t_start = time.monotonic()
                print(f"recording {a.seconds:.0f} s: ROTATE THE RIG by hand, left-right and up-down, quickly and smoothly", flush=True)
            small = cv2.resize(cv2.cvtColor(cv2.rotate(pair.right, cv2.ROTATE_180) if swap else pair.left, cv2.COLOR_BGR2GRAY), (320, 240), interpolation=cv2.INTER_AREA)
            frames.append(small)
            times.append(pair.t_capture)
        if t_start is not None and time.monotonic() - t_start > a.seconds:
            break
        time.sleep(0.005)
    imu += rx.drain_imu()
    rx.stop()
    if len(frames) < 60 or len(imu) < 500:
        raise SystemExit(f"not enough data (frames {len(frames)}, IMU samples {len(imu)}); is the IMU streaming?")

    arr = np.array(imu)
    t_imu, gyro = arr[:, 0], np.radians(arr[:, 1:4]).T
    rates = visual_rates(frames, times, fx * 320 / 640)
    tv, wx, wy = (np.array(v) for v in zip(*rates))
    speed_imu = np.linalg.norm(gyro, axis=0)
    motion = np.degrees(np.percentile(speed_imu, 90))
    print(f"{len(frames)} frames, {len(imu)} IMU samples; typical fast gyro speed {motion:.0f} deg/s")
    if motion < 20:
        raise SystemExit("the rig barely moved; rotate it harder (aim for 60 deg/s or more) and rerun")
    vis_speed = np.hypot(wx, wy)
    delay, q = estimate_delay(tv, vis_speed, t_imu, speed_imu)       # first pass, signs unknown
    R_ci, C = estimate_axes(tv, wx, wy, t_imu, gyro, delay)
    delay, q = estimate_delay_xy(tv, wx, wy, t_imu, gyro, R_ci)       # second pass, with the axis map
    R_ci, C = estimate_axes(tv, wx, wy, t_imu, gyro, delay)
    print(f"camera-IMU delay: {delay * 1000:+.0f} ms (the frame time stamp is {'later' if delay > 0 else 'earlier'} than the motion it shows)")
    print(f"match quality (correlation of visual and gyro speed): {q:.2f}   {'GOOD' if q > 0.7 else 'WEAK: move more, use a textured scene'}")
    print("gyro axis vs camera axis correlation (rows gyro x,y,z; columns camera pitch x, yaw y):")
    print(np.round(C, 2))
    print("axis map cam = R_ci @ imu:\n", R_ci.astype(int))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"delay_s": delay, "R_ci": R_ci.tolist(), "quality": q, "date": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
