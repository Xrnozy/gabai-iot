"""Run Basalt live on the Pi stream and watch it (step S6 of docs/slam-plan.md).

    py -m gabai.tools.live_basalt                                   # live from the Pi
    py -m gabai.tools.live_basalt --replay data/slam/rec_<time>     # same code, a recorded sequence at real time

Start this INSTEAD of the depth viewer (they share port 5600), then start the sender on the Pi. Hold the rig completely
still for the first 3 seconds (Basalt finds gravity from the IMU), then move slowly.

Two windows: Basalt's own view (camera tracks, map points, 3D path) and ours (left camera plus a top-down path, status
and a warning when no pose arrives). A live run is also saved as a normal recording (data/slam/live_<time>/), so it can be
replayed and checked later; trajectory.txt is written at the end (TUM format: time x y z qx qy qz qw).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from gabai.slam.euroc import DEG, G, EurocWriter, basalt_calib, load_accel_calib, load_imu_extrinsics
from gabai.diag import log as dlog
from gabai.slam.run_diag import SlamRunDiag
from gabai.slam.still import StillGate

log = dlog.get("slam")

BASALT_CONFIG = "D:/slam/basalt/data/euroc/euroc_config.json"
CFG_DIR = r"D:\slam\cfg"      # Basalt reads its settings from a command-line string that splits on spaces, so this path must have none
NO_POSE_WARN_S = 1.0


def safe_toml_path(out):
    """Where to put basalt.toml: Basalt parses '--config=<path>' as a command line, so a path with a space (like 'MY PC') breaks it."""
    os.makedirs(CFG_DIR, exist_ok=True)
    return os.path.join(CFG_DIR, os.path.basename(os.path.normpath(out)).replace(" ", "_") + ".toml")


def write_toml(path, calib_path, show_ui, threads=0, deterministic=False):
    open(path, "w").write(
        f'show-gui={int(show_ui)}\ncam-calib="{calib_path}"\nconfig-path="{BASALT_CONFIG}"\n'
        f"print-queue=0\nuse-double=0\ndeterministic={int(deterministic)}\nnum-threads={threads}\n")


def replay_events(folder, speed=1.0):
    """IMU samples and frames of a recorded EuRoC folder in time order, paced to wall-clock time."""
    imu = np.loadtxt(f"{folder}/mav0/imu0/data.csv", delimiter=",", comments="#")
    cam = [l.strip().split(",") for l in open(f"{folder}/mav0/cam0/data.csv") if not l.startswith("#")]
    ev = [(int(r[0]), 0, r) for r in imu] + [(int(t), 1, name) for t, name in cam]
    ev.sort(key=lambda e: (e[0], e[1]))
    t0, wall0 = ev[0][0], time.monotonic()
    for t_ns, kind, payload in ev:
        wait = wall0 + (t_ns - t0) * 1e-9 / speed - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        if kind == 0:
            yield ("imu", t_ns, payload[4:7], payload[1:4])
        else:
            l = cv2.imread(f"{folder}/mav0/cam0/data/{payload}", 0)
            r = cv2.imread(f"{folder}/mav0/cam1/data/{payload}", 0)
            yield ("frame", t_ns, l, r)


def live_events(rx, rect, delay_s):
    """Same events from the Pi receiver. IMU samples are drained before the newest frame so they cover its stamp."""
    seq = 0
    while True:
        for t, gx, gy, gz, ax, ay, az in rx.drain_imu():
            yield ("imu", int(round(t * 1e9)), (ax * G, ay * G, az * G), (gx * DEG, gy * DEG, gz * DEG))
        pair = rx.latest()
        if pair is not None and pair.seq != seq:
            seq = pair.seq
            l, r = rect.apply(pair.left, pair.right)
            g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            yield ("frame", int(round((pair.t_capture - delay_s) * 1e9)), g(l), g(r))
        else:
            time.sleep(0.003)
            yield ("idle", 0, None, None)


class PathView:
    """Top-down trail of the poses (x right, y up, metres) with the camera's heading."""

    def __init__(self, size=480):
        self.size, self.pts, self.fwd = size, [], None

    def add(self, p, fwd_xy):
        self.pts.append((p[0], p[1]))
        self.fwd = fwd_xy

    def draw(self):
        img = np.full((self.size, self.size, 3), 24, np.uint8)
        if not self.pts:
            return img
        a = np.array(self.pts)
        lo, hi = a.min(0), a.max(0)
        c, span = (lo + hi) / 2, max(float((hi - lo).max()), 2.0) * 1.2
        px = lambda q: (int(self.size / 2 + (q[0] - c[0]) / span * self.size), int(self.size / 2 - (q[1] - c[1]) / span * self.size))
        cv2.polylines(img, [np.array([px(q) for q in a], np.int32)], False, (200, 160, 60), 2, cv2.LINE_AA)
        cv2.circle(img, px(a[0]), 6, (80, 200, 80), -1)
        cur = px(a[-1])
        cv2.circle(img, cur, 6, (60, 60, 240), -1)
        if self.fwd is not None and np.linalg.norm(self.fwd) > 1e-6:
            f = self.fwd / np.linalg.norm(self.fwd)
            cv2.arrowedLine(img, cur, (int(cur[0] + 28 * f[0]), int(cur[1] - 28 * f[1])), (60, 60, 240), 2, tipLength=0.4)
        scale_m = 1.0
        cv2.line(img, (12, self.size - 14), (12 + int(scale_m / span * self.size), self.size - 14), (230, 230, 230), 2)
        cv2.putText(img, "1 m", (14, self.size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        cv2.putText(img, "start", (px(a[0])[0] + 8, px(a[0])[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 200, 80), 1)
        return img


def forward_xy(q, R_ic):
    from scipy.spatial.transform import Rotation as R
    return (R.from_quat(q).as_matrix() @ R_ic @ np.array([0.0, 0.0, 1.0]))[:2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", help="a recorded folder (data/slam/rec_*) instead of the live Pi stream")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = real time)")
    ap.add_argument("--calib", default="data/stereo_calib_new.npz")
    ap.add_argument("--imu-sync", default="data/imu_sync.json")
    ap.add_argument("--accel-calib", default="data/accel_calib.json")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--out", help="output folder (default data/slam/live_<time>)")
    ap.add_argument("--no-basalt-ui", action="store_true", help="do not open Basalt's own window")
    ap.add_argument("--no-window", action="store_true", help="no windows at all (testing)")
    ap.add_argument("--threads", type=int, default=0, help="Basalt threads (0 = all)")
    ap.add_argument("--no-still-gate", action="store_true", help="start Basalt at once even if the rig is moving (results then vary from run to run)")
    ap.add_argument("--still-s", type=float, default=1.0, help="seconds of stillness required before Basalt starts")
    ap.add_argument("--no-save", action="store_true", help="do not save the live run as a recording")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = until you quit)")
    ap.add_argument("--log-level", default=None, help="diagnostic log level: DEBUG (one line per pose), INFO or WARNING (default INFO)")
    ap.add_argument("--log-dir", default=None, help="where the diagnostic log goes (default logs/)")
    a = ap.parse_args()
    log_run = dlog.configure(log_dir=a.log_dir, run_name="slam", level=a.log_level)
    print(f"diagnostic log: {log_run}", flush=True)
    from gabai.diag.env import calibration_summary, environment_info
    log.info("slam.start", args={k: v for k, v in vars(a).items() if k != "token"}, environment=environment_info(), calibration=calibration_summary(),
             note="frame id (fid) = Pi monotonic capture time in ms, the same number the Pi and the depth viewer log")

    from gabai.slam.vit import BasaltTracker
    out = os.path.abspath(a.out or os.path.join("data", "slam", time.strftime("live_%Y%m%d_%H%M%S")))
    os.makedirs(out, exist_ok=True)
    ext = load_imu_extrinsics(a.imu_sync)
    if not ext["calibrated"]:
        print(f"WARNING: {a.imu_sync} not found: the camera-IMU rotation is a guess and Basalt will diverge.", flush=True)
        log.error("slam.imu_sync_missing", path=a.imu_sync, hint="run gabai.tools.sync_imu_camera, then gabai.tools.check_extrinsics --write on a recording")

    accel = load_accel_calib(a.accel_calib)
    if accel is None:
        print(f"WARNING: {a.accel_calib} not found. The accelerometer is uncalibrated (it reads about 4 percent low), so Basalt's result is "
              "unstable. Run: py -m gabai.tools.calibrate_accel", flush=True)
        log.error("slam.accel_calib_missing", path=a.accel_calib, hint="run gabai.tools.calibrate_accel: an uncalibrated accelerometer makes Basalt drift metres")
    from gabai.tools.calibrate_stereo import Rectifier
    rect = Rectifier(a.calib)
    calib_path = os.path.join(out, "calib.json")
    json.dump(basalt_calib(rect.f, float(np.load(a.calib)["cx"]), rect.cy, rect.size, rect.B, ext["R_ic"], accel_params=accel),
              open(calib_path, "w"), indent=2)
    rx = writer = None
    if a.replay:
        events = replay_events(a.replay, a.speed)
    else:
        from gabai.pc.receiver import LiveReceiver
        rx = LiveReceiver(a.port, a.token, rotate_left=rect.rotation[0], rotate_right=rect.rotation[1])
        rx.start()
        events = live_events(rx, rect, ext["delay_s"])
        if not a.no_save:
            writer = EurocWriter(out, cam_delay_s=0.0)      # stamps below already have the delay removed
        print(f"waiting for the Pi on port {rx.port} ... hold the rig still until the HOLD STILL bar reaches 100%", flush=True)

    toml = safe_toml_path(out)
    write_toml(toml, calib_path.replace("\\", "/"), show_ui=not (a.no_basalt_ui or a.no_window), threads=a.threads)
    tracker = BasaltTracker(toml, show_ui=not (a.no_basalt_ui or a.no_window))
    from gabai.slam.vit import BASALT_DLL
    log.info("slam.tracker_ready", config=toml, timing_extension=tracker.enable_timing(), basalt_dll=BASALT_DLL,
             dll_built=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(BASALT_DLL))))
    sd = SlamRunDiag(out, f_px=rect.f, cam_delay_s=ext["delay_s"])
    t_started, gate_logged = time.monotonic(), False
    gate = None if a.no_still_gate else StillGate(a.still_s)
    view, traj, pushed, pop_lat = PathView(), [], {}, []
    n_frames = n_imu = 0
    t_run, last_pose_wall, last_img, last_log = None, None, None, time.monotonic()
    last_draw = 0.0
    imu_buf = []
    try:
        for kind, t_ns, x, y in events:
            now = time.monotonic()
            if kind == "imu":
                for ts, acc, gyr in (gate.feed(t_ns, x, y) if gate is not None else [(t_ns, x, y)]):
                    if tracker.push_imu(ts, acc, gyr):
                        n_imu += 1
                        sd.note_imu(ts * 1e-9, gyr, acc)
                        if writer is not None:
                            imu_buf.append((ts * 1e-9, gyr[0] / DEG, gyr[1] / DEG, gyr[2] / DEG, acc[0] / G, acc[1] / G, acc[2] / G))
            elif kind == "frame":
                if n_imu < 20 or (gate is not None and not gate.ready):
                    last_img = x
                    continue                               # Basalt needs IMU data, and a rig at rest, before the first picture
                if gate is not None and gate.ready and not gate_logged:
                    gate_logged = True
                    log.info("slam.gate_open", waited_s=round(now - t_started, 2), still_s=a.still_s, imu_samples_handed_over=n_imu,
                             note="Basalt starts from this still window; its first orientation comes from the accelerometer here")
                if tracker.push_frame(t_ns, x, y):
                    sd.note_image(x, fid=sd.fid_of(t_ns))
                    sd.note_frame_pushed(sd.fid_of(t_ns))
                    n_frames += 1
                    pushed[t_ns] = now
                    last_img = x
                    t_run = t_run or now
                    if writer is not None:
                        writer.add_imu(imu_buf)
                        imu_buf = []
                        writer.add_frame(t_ns * 1e-9, x, y)
            while True:
                p = tracker.pop_pose()
                if p is None:
                    break
                last_pose_wall = time.monotonic()
                pw = pushed.pop(p["t_ns"], None)
                if pw is not None:
                    pop_lat.append(last_pose_wall - pw)
                sd.note_pose(p, pw, last_pose_wall)
                traj.append((p["t_ns"] * 1e-9, *p["p"], *p["q"]))
                view.add(p["p"], forward_xy(p["q"], ext["R_ic"]))
            sd.tick(now, gate=gate, n_frames=n_frames, n_imu=n_imu, started_wall=t_started)
            if a.seconds and t_run and now - t_run > a.seconds:
                break
            if not a.no_window and kind != "imu" and now - last_draw >= 0.04:
                last_draw = now
                img = cv2.cvtColor(last_img, cv2.COLOR_GRAY2BGR) if last_img is not None else np.zeros((480, 640, 3), np.uint8)
                lost = t_run is not None and (last_pose_wall is None or now - last_pose_wall > NO_POSE_WARN_S)
                lines = [f"frames {n_frames}  imu {n_imu}  poses {len(traj)}"]
                if gate is not None and not gate.ready:
                    lines.append(f"HOLD STILL to start Basalt: {gate.progress * 100:.0f}%")
                if traj:
                    lines.append("pos %.2f %.2f %.2f m" % tuple(traj[-1][1:4]))
                if pop_lat:
                    lines.append("pose delay %.0f ms" % (1000 * np.median(pop_lat[-30:])))
                if lost:
                    lines.append("NO POSE: tracking lost")
                for i, ln in enumerate(lines):
                    cv2.putText(img, ln, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if "NO POSE" in ln else (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Gabai SLAM (q quits)", np.hstack([cv2.resize(img, (640, 480)), view.draw()]))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            if now - last_log >= 5 and t_run:
                last_log = now
                print(f"{now - t_run:5.0f} s: {n_frames} frames, {n_imu} IMU, {len(traj)} poses"
                      + (f", pose delay {1000 * np.median(pop_lat[-30:]):.0f} ms" if pop_lat else ""), flush=True)
    except KeyboardInterrupt:
        print("stopped")
    except Exception:
        log.error("slam.crash", exc=True, frames=n_frames, imu=n_imu, poses=len(traj))
        raise
    finally:
        tracker.close()
        if writer is not None:
            writer.add_imu(imu_buf)
            writer.close()
        cv2.destroyAllWindows()
        if rx is not None:
            rx.stop()
    summary = sd.finish()
    print(f"\nverdict: {summary['verdict']}")
    for f_ in summary["findings"]:
        print(f"  {f_['level'].upper():9s} {f_['code']}: {f_['message']}")
    print(f"details: {os.path.join(out, 'slam_summary.json')}   log: {log_run}", flush=True)
    dlog.shutdown()
    if traj:
        np.savetxt(os.path.join(out, "trajectory.txt"), np.array(traj), fmt="%.9f")
        p = np.array(traj)[:, 1:4]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        print(f"\n{len(traj)} poses of {n_frames} frames; path {step.sum():.2f} m; start to end {np.linalg.norm(p[-1] - p[0]):.2f} m; "
              f"box {np.round(p.max(0) - p.min(0), 2)} m"
              + (f"; pose delay median {1000 * np.median(pop_lat):.0f} ms, p95 {1000 * np.percentile(pop_lat, 95):.0f} ms" if pop_lat else ""))
        print("trajectory saved to", os.path.join(out, "trajectory.txt"))
    else:
        print("no poses were produced")


if __name__ == "__main__":
    main()
