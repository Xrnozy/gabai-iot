"""A fake Pi for testing the sync tools without hardware.

Simulates a rig being tumbled by hand: pictures of a textured scene seen from the rotating camera, a gyro and an
accelerometer, and a compass, all on one clock but with chosen time-stamp delays, axis mappings and hard/soft-iron
distortion. Run:  py tests/fake_pi_sensors.py PORT [seconds]
"""
import socket
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from gabai.net import protocol as P

CAM_DELAY = 0.07                       # camera frames are stamped this much after the motion they show
MAG_DELAY = 0.04                       # compass readings likewise
A = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float)      # imu_vector = A @ camera_vector
M = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)       # compass_vector = M @ imu_vector
B_WORLD = np.array([18.0, 0.0, -42.0]) * (450.0 / 45.8)       # field in counts (about 41 uT)
MAG_OFFSET = np.array([100.0, -60.0, 30.0])
MAG_SCALE = np.array([1.0, 0.9, 1.1])
F = 600.0


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def main():
    port = int(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
    rng = np.random.default_rng(5)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (2400, 3200), dtype=np.uint8), (0, 0), 3.0)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)
    K = np.array([[F, 0, 320], [0, F, 240], [0, 0, 1]])
    Ktex = np.array([[F, 0, 1600], [0, F, 1200], [0, 0, 1]])
    sock = socket.create_connection(("127.0.0.1", port))
    sock.sendall(P.pack_json(P.HELLO, time.monotonic(), {"token": "gabai"}))

    rate = 200.0
    n_hist = int(seconds * rate) + 10
    hist_t = np.arange(n_hist) / rate
    smooth = lambda: np.convolve(rng.normal(0, 1, n_hist), np.ones(160) / 160, mode="same")
    om = np.stack([smooth() for _ in range(3)], axis=1)
    om = 1.4 * om / om.std(axis=0)                                 # rad/s about the camera axes (sizeable rotations)
    R = np.eye(3)
    Rs = []
    for k in range(n_hist):
        Rs.append(R.copy())
        R = R @ (np.eye(3) + skew(om[k]) / rate)
        u, _, vt = np.linalg.svd(R)
        R = u @ vt
    Rs = np.array(Rs)

    t0 = time.monotonic()
    last = 0.0
    k_sent = 0
    while True:
        now = time.monotonic() - t0
        if now > seconds:
            break
        # state at times (now - delay) is what the sensors report
        def idx(t):
            return int(np.clip(round(t * rate), 0, n_hist - 1))
        samples, mags = [], []
        for k in range(k_sent, idx(now) + 1):
            tk = k / rate
            w_imu = A @ om[k]
            R_wi = Rs[k] @ A.T
            acc = R_wi.T @ np.array([0, 0, 1.0])
            samples.append([round(t0 + tk, 5), *np.degrees(w_imu), *acc])
            if k % 4 == 0:                                          # compass at 50 Hz, stamped late
                j = idx(tk - MAG_DELAY)
                m_imu = (Rs[j] @ A.T).T @ B_WORLD
                m_mag = MAG_OFFSET + MAG_SCALE * (M @ m_imu) + rng.normal(0, 1.5, 3)
                mags.append([round(t0 + tk, 5), *m_mag])
        k_sent = idx(now) + 1
        Rc = Rs[idx(now - CAM_DELAY)]
        H = Ktex @ Rc @ np.linalg.inv(K)
        img = cv2.warpPerspective(tex, H, (640, 480), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))[1].tobytes()
        try:
            sock.sendall(P.pack(P.FRAME, t0 + now, P.pack_frame(jpg, jpg)))
            sock.sendall(P.pack_json(P.IMU, t0 + now, {"samples": samples, "mag": mags, "peak_dps": 0.0}))
        except OSError:
            break
        time.sleep(max(0.0, 1 / 15 - (time.monotonic() - t0 - now)))


if __name__ == "__main__":
    main()
