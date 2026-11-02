"""The whole live path with a fake Pi: stereo pairs, IMU and compass over the real protocol, the real run() loop, a stand-in pose provider that follows a known motion.
Checks that the measured reference is computed for the SAME frame as each pose, that the camera delay is applied to the stamps, and that Rerun gets the timeline first."""
import json
import socket
import tempfile
import threading
import time

import cv2
import numpy as np

import gabai.pipeline.run as run_mod
from gabai.diag import log as dlog
from gabai.net import protocol as P
from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.poses import PoseResult
from gabai.refframe import LiveReference
from gabai.slam.euroc import load_live_extrinsics
from tests.test_pipeline_run import ListViz
from tests.test_refframe import CAL, R_DEV_IMU, R_IMU_MAG, Rz, sensors, world_to_cam

N, BETA = 14, 37.0


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class MovingProvider:
    """Stands in for Basalt: returns the true pose of every frame it is given (turned by BETA about z, as a tracker's arbitrary yaw would be)."""
    name, uses_imu, world = "fake_pose", False, "test world"

    def __init__(self):
        self.k = 0

    def feed_imu(self, samples):
        pass

    def push(self, t_ns, gl, gr, disp=None, trusted=None):
        k = int(round((float(gl.mean()) - 10.0) / 8.0))                                  # the sender paints the frame number into the picture: frames may be skipped
        R = Rz(BETA) @ world_to_cam(10.0 + 4.0 * k, 6.0)
        return [PoseResult(int(t_ns), R, np.array([0.1 * k, 0.0, 1.0]), True, self.name, {})]

    def correct(self, R, t):
        pass

    def close(self):
        pass


def test_the_live_loop_computes_the_measured_reference_for_each_posed_frame():
    dlog.configure(log_dir=tempfile.mkdtemp(), run_name="live_ref", level="WARNING")
    cfg = PipelineConfig()
    cfg.pipeline.mode, cfg.source.kind, cfg.source.port, cfg.source.max_frames = "stereo_only", "live", _port(), N
    cfg.perf.drop_when_busy, cfg.perf.prefetch, cfg.perf.low_priority, cfg.perf.async_viz, cfg.perf.staged = True, False, False, False, False
    cfg.hands.enabled = cfg.detector.enabled = False
    d = tempfile.mkdtemp()
    cal = {"offset": np.zeros(3).tolist(), "scale": [1, 1, 1], "W": np.eye(3).tolist(), "radius_counts": 470.0}
    json.dump(cal, open(d + "/mag_cal.json", "w"))
    json.dump({"R_im": R_IMU_MAG.tolist(), "delay_s": 0.0}, open(d + "/mag_sync.json", "w"))
    json.dump({"R_dev_imu": R_DEV_IMU.tolist(), "date": "2026-10-29 14:21:57"}, open(d + "/device_frame.json", "w"))
    cfg.source.mag_cal, cfg.source.mag_sync, cfg.source.device_frame = d + "/mag_cal.json", d + "/mag_sync.json", d + "/device_frame.json"
    cfg.source.imu_sync = d + "/no_imu_sync.json"                                       # absent: the gravity map is used, and says so
    cfg.source.accel_calib = d + "/no_accel.json"                                        # the simulated accelerometer is perfect: no bias to remove
    ext = load_live_extrinsics(cfg.source.imu_sync, cfg.source.device_frame)
    assert np.allclose(ext["R_ic"], R_DEV_IMU.T)

    seen = []
    orig = LiveReference.update

    def spy(self, imu, mag, R_wc):
        out = orig(self, imu, mag, R_wc)
        seen.append((len(imu), len(mag), out))
        return out

    orig_make = run_mod.make_provider
    LiveReference.update = spy
    run_mod.make_provider = lambda c, cal_: MovingProvider()
    from gabai.tools.calibrate_stereo import Rectifier
    w, h = Rectifier(cfg.source.calib).size
    img_of = lambda k: cv2.imencode(".jpg", np.full((h, w, 3), 8 * k + 10, np.uint8))[1].tobytes()
    delay = ext["delay_s"]

    def pi():
        s = None
        for _ in range(200):
            try:
                s = socket.create_connection(("127.0.0.1", cfg.source.port), timeout=1)
                break
            except OSError:
                time.sleep(0.05)
        s.sendall(P.pack(P.HELLO, time.monotonic(), json.dumps({"token": cfg.source.token, "name": "fake-pi"}).encode()))
        t0 = time.monotonic()
        for k in range(N + 4):
            R = world_to_cam(10.0 + 4.0 * k, 6.0)
            tc = t0 + 0.2 * k
            samples, mags = [], []
            for i in range(8):                                                           # IMU and compass samples for the 0.2 s before this frame's stamp (after the camera delay)
                ts = tc - delay - 0.19 + i * 0.025
                imu, mg = sensors(R, t=ts, n=1)
                a = imu[0][2] / 9.80665
                samples.append([ts, 0.0, 0.0, 0.0, *a])
                mags.append([mg[0][0], *mg[0][1:]])
            s.sendall(P.pack_json(P.IMU, tc, {"gyro_dps": [0, 0, 0], "peak_dps": 0.0, "samples": samples, "mag": mags}))
            s.sendall(P.pack(P.FRAME, tc, P.pack_frame(img_of(k), img_of(k))))
            time.sleep(0.2)
        time.sleep(0.5)
        s.close()

    threading.Thread(target=pi, daemon=True).start()
    ListViz.instances.clear()
    depth = lambda l, r: (np.full(l.shape[:2], 20.0, np.float32), np.ones(l.shape[:2], bool), 1.0)
    try:
        run_mod.run(cfg, depth_fn=depth, viz_factory=lambda c, s_, sv, vw: ListViz(c, s_, sv, vw), quiet=True)
    finally:
        LiveReference.update = orig
        run_mod.make_provider = orig_make

    good = [o for _, _, o in seen if o["tilt_error_deg"] is not None]
    assert len(seen) >= N - 3 and len(good) >= N - 4, (len(seen), len(good))
    assert all(ni >= 3 and nm >= 3 for ni, nm, _ in seen)                                # every frame carried its IMU and compass samples
    assert max(o["tilt_error_deg"] for o in good) < 0.5, [(round(o["tilt_error_deg"], 1), o["accel_g"]) for o in good]
    offs = [o["heading_offset_deg"] for o in good if o["heading_offset_deg"] is not None]
    assert len(offs) >= 5 and np.allclose(offs, -BETA, atol=0.5)                          # the pose's yaw offset to magnetic north is the same on every frame
    paths = ListViz.instances[-1].paths
    assert {"ref/tilt_error_deg", "ref/heading_offset_deg", "world/reference/north", "sync/frame_gap_ms"} <= paths
