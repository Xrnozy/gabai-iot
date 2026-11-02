"""Record synchronized left/right videos from the Pi's two cameras at 640x360, to be replayed through the depth pipeline as test input.

    Pi:  python3 -m gabai.pi.sender --host <PC_IP> --left /dev/video2 --right /dev/video0 --width 640 --height 360 --fps 15 --capture-fps 30
    PC:  py -m gabai.tools.record_pairs --name still            (hold the rig still, 20 s)
         py -m gabai.tools.record_pairs --name normal           (walk or turn the rig at an ordinary pace)
         py -m gabai.tools.record_pairs --name rapid            (fast turns and sweeps)
    Stop the viewer first (they share port 5600). Recording stops after --seconds (default 30) or when you press q in the window / Ctrl-C.

Each recording is a folder  data/recordings/<name>_<date_time>/  holding
    left.avi  right.avi   one frame per pair, in step (frame n of the left video belongs to frame n of the right one), 640x360, FFV1 (lossless: the replay sees exactly
                          what the PC received; about 8 MB/s, a 30 s recording is about 250 MB; --codec mjpg is 5x smaller and loses a little detail)
    pairs.csv             per pair: capture time on the Pi clock, receive time, the signed left/right capture offset the Pi measured (dt_ms, right minus left),
                          the camera exposure / gain / picture brightness the Pi reported, and the receiver's sequence number (a gap = a pair the PC missed)
    imu.csv               the IMU samples the Pi sent (t on the Pi clock, gyro deg/s, accel g): gabai.tools.fake_pi plays them back with the pairs
    meta.json             devices the Pi read, calibration file and its numbers, the turns applied, counts, Pi power flags
The pictures are stored exactly as the live pipeline receives them (turned by the calibration's rotation, NOT yet rectified), so the replay rectifies, corrects the
offset and estimates depth with the same code as the live run, and the same recording can be replayed with different settings (before / after a change).
Replay:   py -m gabai.tools.rerun_pipeline --source replay --folder data/recordings/<folder> --mode stereo_only --model h
Compare:  py scripts/evaluate_recording.py data/recordings/<folder>
"""
import argparse
import csv
import json
import os
import queue
import threading
import time

import cv2
import numpy as np

from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.sources import calibration_banner, check_devices
from gabai.tools.calibrate_stereo import Rectifier


class PairRecorder:
    """Writes pairs to a folder from a background thread, so a slow disk never holds up the receiver. add() never blocks: when the queue is full the pair is dropped and counted."""

    def __init__(self, folder, size, calib, fps=15.0, codec="ffv1", queue_len=200):
        self.folder, self.size, self.calib, self.codec = folder, size, calib, codec
        os.makedirs(folder, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec.upper())
        self.vl = cv2.VideoWriter(os.path.join(folder, "left.avi"), fourcc, fps, size)
        self.vr = cv2.VideoWriter(os.path.join(folder, "right.avi"), fourcc, fps, size)
        for v in (self.vl, self.vr):
            if not v.isOpened():
                raise SystemExit(f"cannot write the video files: this OpenCV has no {codec.upper()} writer (try --codec mjpg)")
        self.csv_file = open(os.path.join(folder, "pairs.csv"), "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["index", "t_capture", "t_recv", "dt_ms", "exposure_ms", "gain", "picture_mean", "seq", "throttled", "rtt_ms", "temp_c", "load1", "cam_fps_left", "cam_fps_right"])
        self.q = queue.Queue(maxsize=queue_len)
        self.n, self.dropped, self.seen_seq, self.missed = 0, 0, None, 0
        self.first_t = self.last_t = None
        self.dts, self.t0 = [], time.time()
        self.imu = []
        self.th = threading.Thread(target=self._work, daemon=True)
        self.th.start()

    def add(self, left, right, t_capture, t_recv, dt_ms, seq, health):
        if self.seen_seq is not None and seq > self.seen_seq + 1:
            self.missed += seq - self.seen_seq - 1                       # pairs the receiver overwrote before we took them
        self.seen_seq = seq
        try:
            self.q.put_nowait((left, right, t_capture, t_recv, dt_ms, seq, health))
        except queue.Full:
            self.dropped += 1
            return False
        return True

    def add_imu(self, rows):
        self.imu.extend(rows)

    def _work(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            left, right, tc, tr, dt, seq, h = item
            self.vl.write(left)
            self.vr.write(right)
            blank = lambda k, f=lambda v: v: "" if h.get(k) is None else f(h[k])
            self.csv.writerow([self.n, f"{tc:.4f}", f"{tr:.4f}", "" if dt is None else f"{dt:.2f}", blank("exposure_ms"), blank("gain"), blank("picture_mean"), seq,
                               blank("throttled", lambda v: int(str(v), 16) if isinstance(v, str) else int(v)), blank("rtt_s", lambda v: round(v * 1000, 1)), blank("temp_c"), blank("load1"),
                               blank("cam_fps_left"), blank("cam_fps_right")])
            self.n += 1
            self.first_t = tc if self.first_t is None else self.first_t
            self.last_t = tc
            if dt is not None:
                self.dts.append(abs(dt))

    def close(self, hello=None, health=None, rotation=(0, 0)):
        self.q.put(None)
        self.th.join()
        self.vl.release(), self.vr.release(), self.csv_file.close()
        if self.imu:
            np.savetxt(os.path.join(self.folder, "imu.csv"), np.array(self.imu, float), delimiter=",", header="t,gx_dps,gy_dps,gz_dps,ax_g,ay_g,az_g", comments="", fmt="%.6f")
        z = np.load(self.calib)
        dur = (self.last_t - self.first_t) if self.n > 1 else 0.0
        meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs_recorded": self.n, "pairs_missed_by_the_receiver": self.missed, "pairs_dropped_by_the_recorder": self.dropped,
                "duration_s": round(dur, 2), "pairs_per_second": round((self.n - 1) / dur, 2) if dur > 0 else None, "imu_samples": len(self.imu), "picture_size": list(self.size), "video_codec": self.codec,
                "left_right_offset_ms": None if not self.dts else {"median": round(float(np.median(self.dts)), 2), "max": round(float(np.max(self.dts)), 2)},
                "pictures_turned_deg_left_right": list(rotation), "sender_hello": hello or {}, "pi_health_at_end": health or {},
                "calibration": {"file": self.calib, "baseline_mm": round(float(z["baseline"]) * 1000, 2), "focal_px": round(float(z["focal"]), 1),
                                "left_dev": str(z["left_dev"]) if "left_dev" in z.files else None, "right_dev": str(z["right_dev"]) if "right_dev" in z.files else None},
                "note": "pictures are stored as the live pipeline receives them (turned, not rectified); replay with: py -m gabai.tools.rerun_pipeline --source replay --folder <this folder>"}
        json.dump(meta, open(os.path.join(self.folder, "meta.json"), "w"), indent=1)
        return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="rec", help="what is being recorded, e.g. still / normal / rapid: becomes part of the folder name")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default=os.path.join("data", "recordings"))
    ap.add_argument("--calib", default=PipelineConfig().source.calib)
    ap.add_argument("--port", type=int, default=5600), ap.add_argument("--token", default="gabai")
    ap.add_argument("--codec", default="ffv1", choices=["ffv1", "mjpg"], help="ffv1 = lossless (default), mjpg = about 5x smaller, slightly lossy")
    ap.add_argument("--countdown", type=int, default=3, help="seconds between the first pair arriving and the recording starting")
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args()
    from gabai.pc.receiver import LiveReceiver
    rect = Rectifier(a.calib)
    for line in calibration_banner(a.calib):
        print(line, flush=True)
    rx = LiveReceiver(a.port, a.token, rotate_left=rect.rotation[0], rotate_right=rect.rotation[1])
    rx.start()
    print(f"waiting for the Pi on port {rx.port} ...  python3 -m gabai.pi.sender --host <PC_IP> --left /dev/video2 --right /dev/video0 --width 640 --height 360", flush=True)
    folder = os.path.join(a.out, f"{a.name}_{time.strftime('%Y%m%d_%H%M%S')}")
    rec = seq = None
    win, started, t_first = "recording (q = stop)", None, None
    try:
        while True:
            imu_rows = rx.drain_imu()                                   # every iteration, so the receiver's queue never overflows
            if rec is not None:
                rec.add_imu(imu_rows)
            p = rx.latest()
            if p is not None and (seq is None or p.seq != seq):
                seq = p.seq
                if t_first is None:
                    t_first = time.time()
                    ok, text = check_devices(a.calib, rx.stats().get("hello") or {})
                    print(("OK: " if ok else "ERROR: ") + text, flush=True)
                    if not ok:
                        raise SystemExit(text)
                    if (p.left.shape[1], p.left.shape[0]) != tuple(rect.size):
                        raise SystemExit(f"the pictures are {p.left.shape[1]}x{p.left.shape[0]} but the calibration is {rect.size[0]}x{rect.size[1]}: start the sender with --width {rect.size[0]} --height {rect.size[1]}")
                    if p.dt_ms is None:
                        print("WARNING: the sender does not send the left/right offset (old sender): the offset correction cannot be tested with this recording", flush=True)
                    print(f"pairs are arriving; recording starts in {a.countdown} s, for {a.seconds:g} s. Move the rig the way this recording is named ('{a.name}').", flush=True)
                if started is None and time.time() - t_first >= a.countdown:
                    rec = PairRecorder(folder, tuple(rect.size), a.calib, codec=a.codec)
                    started = time.time()
                    print(f"RECORDING to {folder}", flush=True)
                if rec is not None:
                    rec.add(p.left, p.right, p.t_capture, p.t_recv, p.dt_ms, p.seq, rx.stats().get("health") or {})
                if not a.no_window:
                    img = np.hstack([p.left, p.right]).copy()
                    msg = "waiting..." if rec is None else f"REC {rec.n} pairs  {time.time() - started:.0f}/{a.seconds:g} s"
                    cv2.putText(img, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if rec else (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(win, img)
            k = cv2.waitKey(1) & 0xFF if not a.no_window else (time.sleep(0.002) or 255)
            if k in (ord("q"), 27) or (started is not None and time.time() - started >= a.seconds):
                break
    except KeyboardInterrupt:
        pass
    finally:
        h = rx.stats()
        rx.stop()
        cv2.destroyAllWindows()
    if rec is None:
        raise SystemExit("nothing was recorded: no pairs arrived")
    meta = rec.close(h.get("hello"), h.get("health"), rect.rotation)
    print(f"\nsaved {meta['pairs_recorded']} pairs in {folder}: {meta['pairs_per_second']} pairs/s, offset between the cameras median {meta['left_right_offset_ms']}, "
          f"missed by the receiver {meta['pairs_missed_by_the_receiver']}, dropped by the recorder {meta['pairs_dropped_by_the_recorder']}")
    print(f"replay:   py -m gabai.tools.rerun_pipeline --source replay --folder {folder} --mode stereo_only --model h\ncompare:  py scripts/evaluate_recording.py {folder}")


if __name__ == "__main__":
    main()
