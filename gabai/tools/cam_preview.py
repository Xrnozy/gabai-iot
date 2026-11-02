"""Show the two Pi cameras side by side, nothing else (no depth model, no IMU).

PC:  py -m gabai.tools.cam_preview [--rotate-left 180]
Pi:  python3 -m gabai.pi.sender --host <PC_IP> --left /dev/video2 --right /dev/video0 --no-imu --no-mag

Press q or Esc to quit. Left is on the left. If the pictures are swapped, swap --left and --right on the Pi.
"""
import argparse
import time

import cv2
import numpy as np

from gabai.pc.receiver import LiveReceiver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--rotate-left", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--rotate-right", type=int, default=0, choices=[0, 90, 180, 270])
    a = ap.parse_args()
    rx = LiveReceiver(a.port, a.token, rotate_left=a.rotate_left, rotate_right=a.rotate_right)
    rx.start()
    print(f"listening on port {rx.port} (token '{a.token}'). Start the sender on the Pi. q or Esc in the window quits.")
    last_seq, shown = -1, 0
    blank = np.zeros((480, 1280, 3), np.uint8)
    cv2.putText(blank, "waiting for the Pi ...", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imshow("cameras: left | right", blank)
    try:
        while True:
            f = rx.latest()
            if f is not None and f.seq != last_seq:
                last_seq, shown = f.seq, shown + 1
                img = np.hstack([f.left, f.right])
                s = rx.stats()
                cv2.putText(img, f"left | right   pair {f.seq}   {s['fps']:.1f} fps", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("cameras: left | right", img)
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                break
    finally:
        rx.stop()


if __name__ == "__main__":
    main()
