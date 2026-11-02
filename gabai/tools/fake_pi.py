"""Pretend to be the Pi: play a recording into the live receiver, with the stutter of a starved Pi, so `--source live` can be tested again and again without the rig.

    PC window 1:  py -m gabai.tools.rerun_pipeline --source live --mode stereo_imu --model h
    PC window 2:  py -m gabai.tools.fake_pi --folder data/recordings/<folder> --stutter log

The folder is one written by gabai.tools.record_pairs (left.avi, right.avi, pairs.csv, meta.json, and imu.csv when it was recorded with this version). The pictures are sent
the way the Pi sends them (turned back to what the camera saw, JPEG), so the PC rectifies and turns them with the same code as a live run.

--stutter  log       the link gaps of the 01:19-01:23 run with the Pi in under-voltage: 90 stalls from 0.3 s to 3.8 s (list below), repeated when the recording is longer
           none      steady 15 pairs/s
           recorded  the arrival times of the recording itself (a recording made on a starved Pi keeps its own stutter)
A stall means what it means on the Pi: the camera keeps running, so the pairs captured during the stall are never sent; the IMU samples of the stall arrive with the next pair.
--imu      file (default when imu.csv exists) | synthetic (rig lying still, gravity only: exercises the link and stall handling, says NOTHING about tracking quality) | none
--imu-hz   thin the IMU to about this rate (the starved Pi measured 61 to 92 Hz in pi.imu_rate_low)
"""
import argparse
import csv
import json
import os
import socket
import time

import cv2
import numpy as np

from gabai.net import protocol as P

# (gap_ms, pairs received so far) of every `link.gap` line of that run, in order; the first row is the pipeline.stutter of 4054 ms at the start.
LOG_STUTTER = [(4054, 3), (344, 15), (547, 24), (359, 30), (766, 43), (359, 50), (453, 58), (797, 71), (344, 83), (469, 92), (343, 98), (375, 107), (406, 115), (875, 123), (437, 135),
               (625, 142), (672, 152), (359, 160), (438, 170), (594, 176), (438, 181), (484, 191), (375, 202), (907, 212), (641, 222), (968, 229), (344, 237), (797, 245), (344, 250),
               (312, 259), (469, 267), (688, 274), (500, 280), (640, 288), (406, 297), (562, 303), (343, 311), (406, 317), (1094, 323), (750, 325), (1235, 330), (797, 335), (1657, 339),
               (2093, 347), (1110, 355), (453, 361), (1656, 365), (1579, 368), (1078, 373), (813, 378), (891, 385), (1813, 387), (2156, 388), (2187, 389), (2515, 391), (422, 397),
               (1625, 401), (609, 408), (562, 414), (578, 424), (781, 431), (3094, 432), (1094, 435), (937, 441), (1203, 447), (438, 454), (2453, 455), (344, 466), (1859, 472),
               (1407, 478), (2234, 479), (1391, 482), (2016, 483), (2250, 484), (344, 490), (2203, 491), (2062, 493), (2656, 495), (1735, 498), (1640, 503), (2172, 504), (343, 509),
               (1141, 513), (1125, 516), (1922, 519), (1062, 525), (1531, 527), (640, 533), (3797, 540), (1562, 543), (2375, 547), (1329, 550)]
FPS = 15.0
# The log delivered 550 pairs in 243.6 s. The 90 listed stalls explain only part of that time; the rest is slower-than-15-per-second pace between them (pi_send_fps 0.4 to 5.8), so the
# pace between stalls in the 'log' profile is what makes the total match, not 1/15 s.
LOG_PACE_S = (243.6 - sum(g for g, _ in LOG_STUTTER) / 1000.0) / (LOG_STUTTER[-1][1] - len(LOG_STUTTER))


def stutter_times(profile, fps, length_s):
    """Seconds (from the first pair) at which the Pi manages to send a pair: normal pace between stalls, and each stall as long as the log says. Repeats the profile to fill length_s."""
    out, s, i = [], 0.0, 0
    while s < length_s:
        j = lambda k: k if k < len(profile) else 1 + (k - 1) % (len(profile) - 1)      # the 4 s start-up stall (first row) happens once; the rest repeats
        gap_ms, cum = profile[j(i)]
        nxt = profile[j(i + 1)][1]
        s += gap_ms / 1000.0                                               # the stall, ending with the late pair
        out.append(s)
        for _ in range(max((nxt - cum - 1) if nxt > cum else 8, 0)):       # then pairs at the normal pace until the next stall
            s += 1.0 / fps
            out.append(s)
        i += 1
    return [t for t in out if t < length_s]


def read_pairs(folder):
    rows = list(csv.DictReader(open(os.path.join(folder, "pairs.csv"))))
    return np.array([float(r["t_capture"]) for r in rows]), np.array([float(r["t_recv"]) for r in rows]), [None if r["dt_ms"] == "" else float(r["dt_ms"]) for r in rows]


def read_imu(folder, mode, tc, R_ic_path):
    """(n,7) rows t, gyro deg/s (3), accel g (3) on the Pi clock, or None."""
    p = os.path.join(folder, "imu.csv")
    if mode == "file" or (mode == "auto" and os.path.exists(p)):
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found: record again with the current gabai.tools.record_pairs (it now saves the IMU), or use --imu synthetic / --imu none")
        return np.loadtxt(p, delimiter=",", skiprows=1)
    if mode == "synthetic":
        from gabai.slam.euroc import load_imu_extrinsics
        R_ic = load_imu_extrinsics(R_ic_path)["R_ic"]
        up = R_ic @ np.array([0.0, -1.0, 0.0])                            # the accelerometer reads +up; the camera's up is -y
        t = np.arange(tc[0] - 1.0, tc[-1] + 1.0, 0.01)
        rng = np.random.default_rng(0)
        return np.c_[t, rng.normal(0, 0.1, (len(t), 3)), np.tile(up, (len(t), 1)) + rng.normal(0, 0.003, (len(t), 3))]
    return None


def unturn(img, deg):
    """The receiver turned the picture by `deg` (clockwise); give it back what the camera saw."""
    code = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}.get(int(deg) % 360)
    return img if code is None else cv2.rotate(img, code)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--host", default="127.0.0.1"), ap.add_argument("--port", type=int, default=5600), ap.add_argument("--token", default="gabai")
    ap.add_argument("--stutter", choices=["log", "none", "recorded"], default="log")
    ap.add_argument("--imu", choices=["auto", "file", "synthetic", "none"], default="auto")
    ap.add_argument("--imu-hz", type=float, default=0.0, help="thin the IMU to about this rate (0 = as recorded)")
    ap.add_argument("--imu-sync", default="data/imu_sync.json", help="only for --imu synthetic")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = the whole recording)")
    ap.add_argument("--jpeg", type=int, default=90)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.folder, "meta.json")))
    tc, tr, dts = read_pairs(a.folder)
    rel = tc - tc[0]
    length = float(rel[-1]) if not a.seconds else min(a.seconds, float(rel[-1]))
    imu = read_imu(a.folder, a.imu, tc, a.imu_sync)
    if imu is not None and a.imu_hz > 0:
        rate = (len(imu) - 1) / (imu[-1, 0] - imu[0, 0])
        imu = imu[::max(int(round(rate / a.imu_hz)), 1)]
    if a.stutter == "log":
        times = stutter_times(LOG_STUTTER, 1.0 / LOG_PACE_S, length)
    elif a.stutter == "recorded":
        times = list(tr[tr - tr[0] < length] - tr[0])
    else:
        times = list(np.arange(0.0, length, 1.0 / FPS))
    print(f"{a.folder}: {len(tc)} pairs over {rel[-1]:.1f} s, IMU {'none' if imu is None else f'{len(imu)} samples'}; stutter '{a.stutter}': {len(times)} pairs will be sent in {length:.0f} s "
          f"({len(times) / max(length, 1e-6):.1f} per second)", flush=True)

    cap_l, cap_r = cv2.VideoCapture(os.path.join(a.folder, "left.avi")), cv2.VideoCapture(os.path.join(a.folder, "right.avi"))
    pos, last = [0], [None]                                              # the videos are read once, in order; pairs that a stall skips are read and dropped

    def pair(i):
        while pos[0] <= i:
            okl, l = cap_l.read()
            okr, r = cap_r.read()
            if not (okl and okr):
                return None
            last[0], pos[0] = (l, r), pos[0] + 1
        return last[0]                                                    # the same recorded pair again when the pace is faster than the recording

    turned = meta.get("pictures_turned_deg_left_right") or [0, 0]
    hello = dict(meta.get("sender_hello") or {}, token=a.token)
    hello.setdefault("name", "fake_pi")
    health = dict(meta.get("pi_health_at_end") or {})
    health.setdefault("throttled", "0x50005" if a.stutter != "none" else "0x0")
    sock = socket.create_connection((a.host, a.port), timeout=10)
    sock.sendall(P.pack_json(P.HELLO, time.monotonic(), hello))
    t_wall0, last_imu_t, sent, last_health = time.monotonic(), tc[0] - 1.0, 0, -9.0
    try:
        for s in times:
            i = int(np.searchsorted(rel, s))
            if i >= len(rel):
                break
            lag = t_wall0 + s - time.monotonic()
            if lag > 0:
                time.sleep(lag)
            if s - last_health >= 1.0:
                sock.sendall(P.pack_json(P.HEALTH, tc[i], dict(health, fps=round(sent / max(s, 1e-6), 1), clock_offset_s=0.0)))
                last_health = s
            if imu is not None:
                k = (imu[:, 0] > last_imu_t) & (imu[:, 0] <= tc[i])
                rows = imu[k]
                last_imu_t = tc[i]
                if len(rows):
                    sock.sendall(P.pack_json(P.IMU, tc[i], {"gyro_dps": rows[-1, 1:4].tolist(), "peak_dps": float(np.abs(rows[:, 1:4]).max()), "accel_g": rows[-1, 4:7].tolist(),
                                                             "samples": rows.tolist(), "mag": []}))
            got = pair(i)
            if got is None:
                break
            l, r = unturn(got[0], turned[0]), unturn(got[1], turned[1])
            lj = cv2.imencode(".jpg", l, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg])[1].tobytes()
            rj = cv2.imencode(".jpg", r, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg])[1].tobytes()
            sock.sendall(P.pack(P.FRAME, tc[i], P.pack_frame(lj, rj, dts[i])))
            sent += 1
    except (BrokenPipeError, ConnectionError, OSError) as e:
        print(f"the PC closed the connection: {e}", flush=True)
    finally:
        sock.close()
    print(f"sent {sent} pairs", flush=True)


if __name__ == "__main__":
    main()
