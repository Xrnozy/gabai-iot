"""Runs on the Pi. Records both webcams with the SAME pairing the sender uses, for sync measurement and calibration shots.

Modes:
  sync   : writes frames.csv (capture times + mean brightness of an ROI in each camera). Use with a flashing light and
           `py -m gabai.spike.analyze_sync` on the PC to measure how far apart the two cameras really expose.
  pairs  : saves PNG stereo pairs every --interval seconds.
Needs: sudo apt install python3-opencv v4l-utils
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

from gabai.pi.cams import SyncedCameras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sync", "pairs"])
    ap.add_argument("--left", default="/dev/video2")
    ap.add_argument("--right", default="/dev/video0")
    ap.add_argument("--capture-fps", type=int, default=30, help="camera rate; the pairing error is about half a frame period")
    ap.add_argument("--fps", type=int, default=15, help="pairs recorded per second")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--exposure", default="300", help="'auto' or exposure_time_absolute in 0.1 ms units; use the SAME fixed value for a fair sync test")
    ap.add_argument("--gain", type=int, default=None)
    ap.add_argument("--roi", default="0,0,64,64", help="x,y,w,h of the region containing the flashing light")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--out", default="session")
    a = ap.parse_args()

    cams = SyncedCameras(a.left, a.right, a.width, a.height, a.capture_fps, a.exposure, a.gain, None, 90, a.fps)
    os.makedirs(a.out, exist_ok=True)
    x, y, w, h = map(int, a.roi.split(","))
    dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    t_end = time.monotonic() + a.duration
    last_save, n = 0.0, 0
    grids_t, grids_l, grids_r = [], [], []
    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["idx", "t_left", "t_right", "roi_left", "roi_right"])
        while time.monotonic() < t_end:
            t, lj, rj = cams.read()
            fl, fr = dec(lj), dec(rj)
            gl, gr = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if a.mode == "sync":  # coarse 6x8 brightness grid per camera so the flashing light can be found automatically
                grids_t.append(t)
                grids_l.append(cv2.resize(gl, (8, 6), interpolation=cv2.INTER_AREA))
                grids_r.append(cv2.resize(gr, (8, 6), interpolation=cv2.INTER_AREA))
            # t_left/t_right are the pair's capture times; both written so the analysis keeps working
            wr.writerow([n, f"{t:.6f}", f"{t:.6f}", f"{gl[y:y+h, x:x+w].mean():.2f}", f"{gr[y:y+h, x:x+w].mean():.2f}"])
            if a.mode == "pairs" and t - last_save >= a.interval:
                k = int(t * 1000)
                cv2.imwrite(os.path.join(a.out, f"L_{k}.png"), fl)
                cv2.imwrite(os.path.join(a.out, f"R_{k}.png"), fr)
                last_save = t
                print("saved pair", k, flush=True)
            n += 1
    cams.close()
    if grids_t:
        np.savez(os.path.join(a.out, "grid.npz"), t=np.array(grids_t), left=np.array(grids_l), right=np.array(grids_r))
    s = cams.stats()
    print(f"pairs={n} effective_pairs_per_s={n / a.duration:.2f}  camera fps L/R={s['cam_fps_left']}/{s['cam_fps_right']}  "
          f"pairing offset median={s['pair_dt_ms']} ms worst={s['pair_dt_max_ms']} ms")


if __name__ == "__main__":
    main()
