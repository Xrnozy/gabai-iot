"""Record a sequence from the Pi in the folder layout Basalt replays (EuRoC format) with a matching calibration file.

    py -m gabai.tools.record_basalt --seconds 60

Start this INSTEAD of the viewer (they share port 5600), then start the sender on the Pi. Hold the rig still for the
first 3 seconds (Basalt finds gravity from the IMU), then walk slowly with the cameras looking at textured things at
a few metres: no blank walls, no fast turns. The recording is written to data/slam/rec_<time>/ together with
calib.json (the camera model and camera-IMU rotation), recording_info.json and run_basalt.cmd (the command to replay it).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from gabai.slam.euroc import EurocWriter, basalt_calib, load_accel_calib, load_imu_extrinsics

BASALT_DIR = r"D:\slam\basalt"
VCPKG_BIN = r"D:\slam\vcpkg\installed\x64-windows\bin"
CONFIG = r"data\euroc\euroc_config.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", help="output folder (default data/slam/rec_<time>)")
    ap.add_argument("--calib", default="data/stereo_calib_new.npz")
    ap.add_argument("--imu-sync", default="data/imu_sync.json", help="camera-IMU rotation and delay from sync_imu_camera")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    a = ap.parse_args()

    from gabai.pc.receiver import LiveReceiver
    from gabai.tools.calibrate_stereo import Rectifier
    from gabai.pipeline.sources import layout_needs_swap
    rect = Rectifier(a.calib)
    cx = float(np.load(a.calib)["cx"])
    ext = load_imu_extrinsics(a.imu_sync)
    out = a.out or os.path.join("data", "slam", time.strftime("rec_%Y%m%d_%H%M%S"))
    w = EurocWriter(out, cam_delay_s=ext["delay_s"])
    if not ext["calibrated"]:
        print(f"WARNING: {a.imu_sync} not found. The camera-IMU rotation is the identity and the delay a guess "
              f"({ext['delay_s'] * 1000:.0f} ms). Basalt will not track on this recording until you run sync_imu_camera and re-convert.", flush=True)

    if load_accel_calib() is None:
        print("WARNING: data/accel_calib.json not found: the accelerometer is uncalibrated and Basalt will be unreliable. Run gabai.tools.calibrate_accel.", flush=True)
    rx = LiveReceiver(a.port, a.token, rotate_left=rect.rotation[0], rotate_right=rect.rotation[1])
    rx.start()
    print(f"waiting for the Pi on port {rx.port} ... (recording starts at the first frame; hold still for 3 s)", flush=True)
    seq, t_start, t_report, skipped, last_t, max_gap = 0, None, 0.0, 0, None, 0.0
    swap, n_seen = None, 0
    try:
        while True:
            pair = rx.latest()
            w.add_imu(rx.drain_imu())
            if pair is not None and pair.seq != seq:
                skipped += max(0, pair.seq - seq - 1) if seq else 0
                seq = pair.seq
                now = time.monotonic()
                if t_start is None:
                    t_start = now
                    print("recording ...", flush=True)
                if last_t is not None:
                    max_gap = max(max_gap, now - last_t)
                last_t = now
                if swap is None:                                           # the same camera-layout decision as the live pipeline, so the pictures (and the camera-IMU rotation fitted on them) are the live ones
                    swap = layout_needs_swap(rect, pair.left, pair.right, n_seen)[0]
                    n_seen += 1
                    if swap is None:
                        continue
                    print("camera layout: " + ("SWAPPED (both pictures turned 180 deg and traded)" if swap else "as calibrated"), flush=True)
                left, right = (cv2.rotate(pair.right, cv2.ROTATE_180), cv2.rotate(pair.left, cv2.ROTATE_180)) if swap else (pair.left, pair.right)
                l, r = rect.apply(left, right)
                g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                w.add_frame(pair.t_capture, g(l), g(r))
            else:
                time.sleep(0.003)
            if t_start is not None:
                el = time.monotonic() - t_start
                if el - t_report >= 5:
                    t_report = el
                    print(f"{el:5.0f} s: {w.n_frames} frames ({w.n_frames / el:.1f} /s), {w.n_imu} IMU samples ({w.n_imu / el:.0f} /s)", flush=True)
                if el >= a.seconds:
                    break
    except KeyboardInterrupt:
        print("stopped early")
    finally:
        w.close()
        rx.stop()
    if w.n_frames == 0:
        raise SystemExit("no frames received: is the sender running?")
    dur = time.monotonic() - t_start
    json.dump(basalt_calib(rect.f, cx, rect.cy, rect.size, rect.B, ext["R_ic"], accel_params=load_accel_calib()), open(os.path.join(out, "calib.json"), "w"), indent=2)
    info = dict(seconds=round(dur, 1), frames=w.n_frames, imu_samples=w.n_imu, frame_rate=round(w.n_frames / dur, 2),
                imu_rate=round(w.n_imu / dur, 1), skipped_pairs=skipped, longest_gap_s=round(max_gap, 2),
                imu_extrinsics_calibrated=ext["calibrated"], cam_delay_ms=round(ext["delay_s"] * 1000, 1))
    json.dump(info, open(os.path.join(out, "recording_info.json"), "w"), indent=2)
    absout = os.path.abspath(out)
    cmd = (f'set PATH={VCPKG_BIN};%PATH%\ncd /d {BASALT_DIR}\n.\\build\\basalt_vio.exe --show-gui 1 --dataset-path "{absout}" --dataset-type euroc '
           f'--cam-calib "{os.path.join(absout, "calib.json")}" --config-path {CONFIG} '
           f'--save-trajectory tum --save-trajectory-fn "{os.path.join(absout, "trajectory.txt")}"\n')
    open(os.path.join(out, "run_basalt.cmd"), "w").write(cmd)
    print(json.dumps(info, indent=2))
    print(f"\nrecorded to {absout}\nreplay with: {os.path.join(absout, 'run_basalt.cmd')}")
    if info["longest_gap_s"] > 0.3 or info["imu_rate"] < 50:
        print("NOTE: gaps in the stream or a low IMU rate will hurt tracking.")


if __name__ == "__main__":
    main()
