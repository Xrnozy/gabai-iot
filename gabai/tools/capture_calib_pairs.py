"""Capture exactly 15 left/right checkerboard pairs from the Pi at 640x360 and (optionally) calibrate from them.

1. Pi:  python3 -m gabai.pi.sender --host <PC_IP> --width 640 --height 360 --fps 10 --capture-fps 15
2. PC:  py -m gabai.tools.capture_calib_pairs --square-mm 19.4
        (measure one printed square with a ruler and type that number; the baseline scales with it)

In the window: SPACE = save this pair, U = undo the last pair, Q = quit. The tool saves a pair only when the board is found in BOTH
pictures, is steady, the two cameras were captured less than --max-offset-ms apart, and the board is somewhere new. Follow the pose
printed on screen (the same list is in docs/calibration-guide.md). The ENTIRE board must stay inside both pictures.

Files (in data/calib_pairs_15, an older folder of that name is kept as ..._prev_<time>):
    pair_01_L.png  pair_01_R.png ... pair_15_L.png  pair_15_R.png    lossless, already turned by --rotate-left/--rotate-right
    capture_log.json   per pair: time, capture offset between the cameras, board position and size
    rotation.json      the turns used (calibrate_stereo --images reads it)
After the 15th pair the calibration runs into data/stereo_calib_15.npz (your current data/stereo_calib.npz is NOT touched) and prints the
baseline next to the 42 mm of the printed case. Calibrate again later with:
    py -m gabai.tools.calibrate_stereo --images data/calib_pairs_15 --square-mm 19.4 --out data/stereo_calib_15
"""
import argparse
import json
import os
import shutil
import time
from types import SimpleNamespace

import cv2
import numpy as np

from gabai.tools.calibrate_stereo import RIG_BASELINE_MM, find_corners, pose_is_new, run_offline

SIZE = (640, 360)
POSES = [
    "centred, board flat to the cameras, medium distance (about half the picture wide)",
    "centred, flat, CLOSE (about 80% of the picture wide)",
    "centred, flat, FAR (about 25% of the picture wide)",
    "flat, pushed to the LEFT edge of the picture",
    "flat, pushed to the RIGHT edge of the picture",
    "flat, pushed to the TOP edge",
    "flat, pushed to the BOTTOM edge",
    "turned about 30 deg around the vertical axis: left side nearer the cameras",
    "turned about 30 deg around the vertical axis: right side nearer the cameras",
    "tilted about 30 deg: top edge leaning AWAY from the cameras",
    "tilted about 30 deg: bottom edge leaning AWAY from the cameras",
    "rotated about 30 deg in its own plane (like turning a steering wheel), centred",
    "top-left corner of the picture, tilted",
    "bottom-right corner of the picture, tilted",
    "close and strongly tilted (about 45 deg), any side",
]


class Capturer:
    """The decision part, without any window: feed it pairs, it says whether the current pair can be saved and saves it on request."""

    def __init__(self, out, pairs=15, pattern=(9, 6), min_shift=30.0, max_offset_ms=10.0, still_px=1.5, still_pairs=2, rotation=(180, 0), size=SIZE):
        self.out, self.n_goal, self.pattern, self.min_shift = out, pairs, pattern, min_shift
        self.max_offset, self.still_px, self.still_pairs, self.rotation, self.size = max_offset_ms, still_px, still_pairs, rotation, size
        self.saved, self.log, self.prev, self.steady, self.cur = [], [], None, 0, None
        self.status, self.ok = "show the board to both cameras", False
        os.makedirs(out, exist_ok=True)
        json.dump(list(rotation), open(os.path.join(out, "rotation.json"), "w"))

    @property
    def n(self):
        return len(self.saved)

    def consider(self, left, right, dt_ms=None):
        """Look at one pair. Returns (left corners or None, right corners or None); sets self.ok / self.status."""
        self.ok, self.cur = False, None
        if (left.shape[1], left.shape[0]) != self.size or (right.shape[1], right.shape[0]) != self.size:
            self.status = f"pictures are {left.shape[1]}x{left.shape[0]}, need {self.size[0]}x{self.size[1]}: start the sender with --width 640 --height 360"
            return None, None
        cl = find_corners(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), self.pattern)
        cr = find_corners(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), self.pattern)
        if cl is None or cr is None:
            self.status = "board not found in the " + ("both pictures" if cl is None and cr is None else "right picture" if cl is not None else "left picture") + ": keep ALL of it in view"
            self.prev, self.steady = None, 0
            return cl, cr
        moved = float(np.abs(cl - self.prev).mean()) if self.prev is not None else 99.0
        self.prev = cl
        self.steady = self.steady + 1 if moved < self.still_px else 0
        if dt_ms is not None and abs(dt_ms) > self.max_offset:
            self.status = f"the two cameras were captured {abs(dt_ms):.0f} ms apart (limit {self.max_offset:g}): hold still, a better pair is coming"
        elif self.steady < self.still_pairs:
            self.status = "hold the board still..."
        elif not pose_is_new(cl, self.saved, self.min_shift):
            self.status = "too close to a pair you already saved: move the board to a NEW position, distance or angle"
        elif self.n >= self.n_goal:
            self.status = "all pairs captured"
        else:
            self.status, self.ok = "READY: press SPACE to save this pair", True
        self.cur = (cl, cr, dt_ms)
        return cl, cr

    def save(self, left, right):
        """Save the pair last passed to consider() if it was ok. Returns the pair number or None."""
        if not self.ok or self.cur is None or self.n >= self.n_goal:
            return None
        cl, cr, dt = self.cur
        k = self.n + 1
        cv2.imwrite(os.path.join(self.out, f"pair_{k:02d}_L.png"), left)
        cv2.imwrite(os.path.join(self.out, f"pair_{k:02d}_R.png"), right)
        c = cl.reshape(-1, 2)
        self.saved.append(cl)
        self.log.append({"pair": k, "time": time.strftime("%H:%M:%S"), "camera_offset_ms": None if dt is None else round(float(dt), 2),
                         "board_centre_px": [round(float(v), 1) for v in c.mean(0)], "board_width_px": round(float(np.ptp(c[:, 0])), 1),
                         "board_height_px": round(float(np.ptp(c[:, 1])), 1), "pose_asked": POSES[k - 1] if k <= len(POSES) else ""})
        json.dump(self.log, open(os.path.join(self.out, "capture_log.json"), "w"), indent=1)
        self.ok, self.steady = False, 0
        return k

    def undo(self):
        if not self.saved:
            return None
        k = self.n
        for s in "LR":
            p = os.path.join(self.out, f"pair_{k:02d}_{s}.png")
            os.path.exists(p) and os.remove(p)
        self.saved.pop(), self.log.pop()
        json.dump(self.log, open(os.path.join(self.out, "capture_log.json"), "w"), indent=1)
        return k


def fresh_folder(path):
    """An existing folder of pairs is moved aside, never overwritten."""
    if os.path.isdir(path) and os.listdir(path):
        aside = f"{path}_prev_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(path, aside)
        print(f"the older pairs in {path} were moved to {aside}", flush=True)


def draw(cap, left, right, cl, cr):
    vl, vr = left.copy(), right.copy()
    for v, c in ((vl, cl), (vr, cr)):
        if c is not None:
            cv2.drawChessboardCorners(v, cap.pattern, c, True)
    img = np.hstack([vl, vr])
    for c in cap.log:                                                         # where the board already was
        cv2.circle(img, tuple(int(x) for x in c["board_centre_px"]), 5, (0, 200, 0), -1)
    panel = np.zeros((130, img.shape[1], 3), np.uint8)
    nxt = POSES[cap.n] if cap.n < len(POSES) else "done"
    col = (0, 220, 0) if cap.ok else (0, 200, 255)
    cv2.putText(panel, f"{cap.n}/{cap.n_goal} pairs saved", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, cap.status, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    cv2.putText(panel, f"next pose {cap.n + 1}: {nxt}", (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "SPACE save   U undo   Q quit", (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    return np.vstack([img, panel])


def calibrate(a, out):
    print(f"\ncalibrating from {out} (square {a.square_mm:g} mm) ...", flush=True)
    ns = SimpleNamespace(images=out, cols=a.cols, rows=a.rows, square_mm=a.square_mm, rational=False, fix_k3=False, out=a.calib_out)
    run_offline(ns)
    print(f"\nreference: the case was designed with {RIG_BASELINE_MM:g} mm between the lens centres. Result saved to {a.calib_out}.npz; "
          "data/stereo_calib.npz was not changed.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--square-mm", type=float, default=19.4, help="side of one printed square, measured with a ruler (default: the value of the current calibration)")
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--out", default=os.path.join("data", "calib_pairs_15"))
    ap.add_argument("--calib-out", default=os.path.join("data", "stereo_calib_15"), help="where the calibration is written after the last pair (no extension)")
    ap.add_argument("--no-calibrate", action="store_true", help="only capture")
    ap.add_argument("--rotate-left", type=int, default=180, choices=[0, 90, 180, 270], help="the turn the live pipeline uses (data/stereo_calib.json: 180 / 0)")
    ap.add_argument("--rotate-right", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--min-shift", type=float, default=30.0, help="px the board must move from every saved pair")
    ap.add_argument("--max-offset-ms", type=float, default=10.0, help="refuse pairs whose cameras were captured further apart than this")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    a = ap.parse_args()
    from gabai.pc.receiver import LiveReceiver
    fresh_folder(a.out)
    cap = Capturer(a.out, a.pairs, (a.cols, a.rows), a.min_shift, a.max_offset_ms, rotation=(a.rotate_left, a.rotate_right))
    rx = LiveReceiver(a.port, a.token, rotate_left=a.rotate_left, rotate_right=a.rotate_right)
    rx.start()
    print(f"waiting for the Pi on port {rx.port} ...  python3 -m gabai.pi.sender --host <PC_IP> --width 640 --height 360 --fps 10 --capture-fps 15", flush=True)
    win, seq, pair, cl, cr = "Stereo calibration pairs", 0, None, None, None
    while cap.n < cap.n_goal:
        p = rx.latest()
        if p is not None and p.seq != seq:
            seq, pair = p.seq, p
            cl, cr = cap.consider(p.left, p.right, p.dt_ms)
        if pair is None:
            frame = np.zeros((130, 640, 3), np.uint8)
            cv2.putText(frame, "waiting for the Pi ...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            frame = draw(cap, pair.left, pair.right, cl, cr)
        cv2.imshow(win, frame)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            print(f"stopped with {cap.n}/{cap.n_goal} pairs in {a.out}")
            rx.stop()
            return
        if k == ord("u") and (n := cap.undo()):
            print(f"removed pair {n}; {cap.n}/{cap.n_goal}", flush=True)
        if k == ord(" ") and pair is not None:
            n = cap.save(pair.left, pair.right)
            if n:
                print(f"CAPTURED pair {n}/{cap.n_goal}  (camera offset {cap.log[-1]['camera_offset_ms']} ms)", flush=True)
            else:
                print("not saved: " + cap.status, flush=True)
    rx.stop()
    cv2.destroyAllWindows()
    print(f"\n{cap.n_goal}/{cap.n_goal} pairs saved in {a.out}", flush=True)
    if not a.no_calibrate:
        calibrate(a, a.out)


if __name__ == "__main__":
    main()
