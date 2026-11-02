"""Does every stage of the live pipeline use the calibration file? Runs the real pipeline for a few frames on a fake Pi and prints, per stage, the numbers it
was given next to the numbers in the calibration file.

    py scripts/check_calibration_use.py                          data/stereo_calib_new.npz (the pipeline default)
    py scripts/check_calibration_use.py --calib data/other.npz

Checks: the file is self-consistent (baseline from T and from the rectified projection matrices agree), the rectifier and the live frames carry its focal length,
principal point and baseline, the pose provider (visual odometry), the map, the Rerun views and the Basalt calibration file all receive them, the camera-order check
reads the file's devices, and no tool in gabai/ still points at another stereo calibration by default.
"""
import argparse
import glob
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from gabai.net import protocol as P                         # noqa: E402
from gabai.pipeline import run as R                          # noqa: E402
from gabai.pipeline import sources                           # noqa: E402
from gabai.pipeline.config import PipelineConfig             # noqa: E402
from gabai.tools.calibrate_stereo import Rectifier           # noqa: E402

rows = []


def check(stage, what, want, got, tol=1e-6):
    ok = bool(np.allclose(np.asarray(want, float), np.asarray(got, float), rtol=tol, atol=tol))
    rows.append((stage, what, want, got, ok))
    return ok


def fake_pi(port, token, left_dev, right_dev, size, n=60):
    pics = sorted(glob.glob(os.path.join("data", "calib_pairs_15", "pair_*_L.png")))
    imgs = []
    for lf in pics[:6]:
        l, r = cv2.imread(lf), cv2.imread(lf[:-6] + "_R.png")
        # what the two cameras themselves deliver for the layout of stereo_calib_new.npz: /dev/video2 (left) is the saved right picture, /dev/video0 (right) the saved left one turned 180 deg;
        # the receiver then turns the left picture 180 deg
        imgs.append((cv2.imencode(".jpg", cv2.resize(r, size))[1].tobytes(), cv2.imencode(".jpg", cv2.resize(cv2.rotate(l, cv2.ROTATE_180), size))[1].tobytes()))

    def run():
        s = None
        for _ in range(200):
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except OSError:
                time.sleep(0.05)
        s.sendall(P.pack(P.HELLO, time.monotonic(), json.dumps({"token": token, "name": "fake-pi", "width": size[0], "height": size[1], "fps": 15, "left": left_dev, "right": right_dev}).encode()))
        try:
            for i in range(n):
                lj, rj = imgs[i % len(imgs)]
                s.sendall(P.pack(P.FRAME, time.monotonic(), P.pack_frame(lj, rj, 4.0)))
                time.sleep(0.08)
        except OSError:                                    # the pipeline took its 8 frames and closed the link: the fake Pi is done
            pass
        finally:
            s.close()
    threading.Thread(target=run, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default=PipelineConfig().source.calib)
    a = ap.parse_args()
    z = np.load(a.calib)
    f, B, cx, cy = float(z["focal"]), float(z["baseline"]), float(z["cx"]), float(z["cy"])
    size = tuple(int(v) for v in z["size"])
    print(f"calibration file: {a.calib}  ({f:.1f} px, {B * 1000:.2f} mm, cx {cx:.1f}, cy {cy:.1f}, {size[0]}x{size[1]}, made {z['date'] if 'date' in z.files else 'date not recorded'})")
    # 1. the file itself
    check("file", "baseline from T == baseline from P2/P1", np.linalg.norm(z["T"]), -float(z["P2"][0, 3]) / float(z["P1"][0, 0]), 1e-4)
    check("file", "focal / cx / cy == P1", [f, cx, cy], [z["P1"][0, 0], z["P1"][0, 2], z["P1"][1, 2]])
    check("file", "both rectified cameras share one focal length and one row", [z["P1"][0, 0], z["P1"][1, 2]], [z["P2"][0, 0], z["P2"][1, 2]])
    # 2. pipeline run
    cfg = PipelineConfig.load("config/pipeline.toml", {"pipeline.mode": "stereo_only", "source.kind": "live", "source.port": 0, "source.max_frames": 8, "source.calib": a.calib,
                                                      "source.folder": tempfile.mkdtemp(), "hands.enabled": False, "detector.enabled": False, "viz.viewer": False, "perf.drop_when_busy": False,
                                                      "perf.prefetch": False})
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    cfg.source.port = s.getsockname()[1]
    s.close()
    seen = {}
    from gabai import kfvo
    from gabai.tools import slam_cloud_view as SC
    k0, c0, mp = kfvo.KeyframeVO.__init__, SC.SlamCloud.__init__, R.make_provider

    def kf_init(self, fx, fy, cx_, cy_, *aa, **kk):
        seen["odometry"] = [fx, cx_, cy_]
        k0(self, fx, fy, cx_, cy_, *aa, **kk)

    def sb(self, B_):
        seen["odometry_B"] = B_
        seen["odometry_fy"] = self.fy
    kfvo.KeyframeVO.__init__ = kf_init
    sb0 = kfvo.KeyframeVO.set_baseline

    def set_baseline(self, B_):
        sb(self, B_)
        sb0(self, B_)
    kfvo.KeyframeVO.set_baseline = set_baseline

    def c_init(self, f_, B_, cx_, cy_, *aa, **kk):
        seen["map"] = [f_, B_, cx_, cy_]
        c0(self, f_, B_, cx_, cy_, *aa, **kk)
    SC.SlamCloud.__init__ = c_init

    class Spy:
        from gabai.pipeline.viz import Viz as _V

        def __init__(self, cal, size_, save=None, viewer=False):
            seen["views"] = [cal["f"], cal["B"], cal["cx"], cal["cy"]]
            seen["views_source"] = cal["source"]
            self.cal, self.n = cal, 0

        def __getattr__(self, name):
            return lambda *aa, **kk: None
    depth_calls = []

    def depth(l, r):
        depth_calls.append(l.shape)
        d = np.full(l.shape[:2], f * B / 1.5, np.float32)
        return d, np.ones(l.shape[:2], bool), 1.0
    devs = (str(z["left_dev"]), str(z["right_dev"])) if "left_dev" in z.files else ("/dev/video2", "/dev/video0")
    fake_pi(cfg.source.port, cfg.source.token, devs[0], devs[1], size)
    print("running the pipeline on a fake Pi ...", flush=True)
    summary = R.run(cfg, depth_fn=depth, viz_factory=Spy, log_dir=tempfile.mkdtemp(), quiet=True)
    kfvo.KeyframeVO.__init__, kfvo.KeyframeVO.set_baseline, SC.SlamCloud.__init__ = k0, sb0, c0
    check("rectifier", "focal / cx / cy / baseline", [f, cx, cy, B], [Rectifier(a.calib).f, Rectifier(a.calib).cx, Rectifier(a.calib).cy, Rectifier(a.calib).B])
    check("depth model", "picture size it receives == calibration size", [size[1], size[0]], list(depth_calls[0][:2]) if depth_calls else [0, 0])
    check("odometry", "fx / cx / cy", [f, cx, cy], seen.get("odometry", [0, 0, 0]))
    check("odometry", "fy == fx, baseline", [f, B], [seen.get("odometry_fy", 0), seen.get("odometry_B", 0)])
    check("map (3D points)", "focal / baseline / cx / cy", [f, B, cx, cy], seen.get("map", [0] * 4))
    check("rerun views (depth in metres)", "focal / baseline / cx / cy", [f, B, cx, cy], seen.get("views", [0] * 4))
    rows.append(("rerun views", "source line in the status panel", a.calib, seen.get("views_source", ""), a.calib in seen.get("views_source", "")))
    rows.append(("run", "frames processed, poses accepted, map frames", "8, >0, >0", f"{summary['frames']}, {summary['poses_ok']}, {summary['map_frames']}", summary["frames"] == 8 and summary["poses_ok"] > 0 and summary["map_frames"] > 0))
    # 3. Basalt (stereo_imu) calibration written from the same file
    cfg2 = PipelineConfig.load("config/pipeline.toml", {"pipeline.mode": "stereo_imu", "source.kind": "live", "source.calib": a.calib, "source.folder": tempfile.mkdtemp()})
    rect, delay, k = sources.prepare_live(cfg2)
    c = json.load(open(os.path.join(cfg2.source.folder, "calib.json")))["value0"]
    ki = c["intrinsics"][0]["intrinsics"]
    check("Basalt calibration", "fx / cx / cy (scaled by the live width factor)", [f * k, cx * k, cy * k], [ki["fx"], ki["cx"], ki["cy"]])
    check("Basalt calibration", "baseline between the two cameras", B, np.linalg.norm(np.array([c["T_imu_cam"][1][x] for x in ("px", "py", "pz")]) - np.array([c["T_imu_cam"][0][x] for x in ("px", "py", "pz")])), 1e-6)
    check("Basalt calibration", "picture size", [int(round(size[0] * k)), int(round(size[1] * k))], c["resolution"][0])
    # 4. camera order and the other tools
    ok, text = sources.check_devices(a.calib, {"left": devs[0], "right": devs[1]})
    rows.append(("camera order", "the sender's devices against the file's", f"left {devs[0]}, right {devs[1]}", text[:70], ok))
    stale = []
    for p in glob.glob("gabai/**/*.py", recursive=True):
        for i, line in enumerate(open(p, encoding="utf-8", errors="ignore"), 1):
            if re.search(r'default\s*=\s*["\']data[/\\]stereo_calib\.npz["\']', line) or re.search(r'calib:\s*str\s*=\s*["\']data/stereo_calib\.npz', line):
                stale.append(f"{p}:{i}")
    rows.append(("other tools", "defaults that still name data/stereo_calib.npz", "none", ", ".join(stale) or "none", not stale))
    w = max(len(r[0]) for r in rows)
    bad = 0
    for stage, what, want, got, ok in rows:
        fmt = lambda v: "[" + ", ".join(f"{x:.4g}" for x in np.ravel(v)) + "]" if not isinstance(v, str) else v
        print(f"{'OK  ' if ok else 'FAIL'} {stage:<{w}}  {what}\n       file {fmt(want)}\n       used {fmt(got)}")
        bad += not ok
    print(f"\n{len(rows) - bad}/{len(rows)} checks pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
