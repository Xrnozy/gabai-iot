"""Check the HMC5883L compass live: raw readings, field strength, heading, and a calibration scatter plot.

Needs the Pi sender running with the compass found ("compass found (bus 3, 0x1E)"). Then:
    py -m gabai.tools.compass_check

What to do, in order (the window also tells you the status):
 1. HOLD STILL. The three numbers should be steady and the field strength should sit between 25 and 65 microtesla.
 2. TURN THE BOARD FLAT, one slow full circle. The heading should sweep about 360 degrees and the field strength
    should stay about the same.
 3. TUMBLE IT in every direction for 20 s (like drawing a figure of eight in the air, including upside down). The
    three scatter plots should fill out into circles. Press c: it computes the correction and saves data/mag_cal.json.
    Quality under 0.03 is good, under 0.06 usable.
 4. HOLD A PHONE or magnet near it: the field strength should jump. That proves the sensor reacts to real fields.
Keys: c calibrate and save, r reset, s save a screenshot, q quit.
"""
import argparse
import json
import os
import time
from collections import deque

import cv2
import numpy as np

from gabai.mag import EARTH_RANGE_UT, apply_cal, calibrate_ellipsoid as calibrate, field_ut, heading_deg, sweep_deg


def draw_scatter(points, size, centre=(0.0, 0.0), extent=300.0, title="", color=(255, 160, 60), ring=None):
    """Plot 2D points (n, 2) around `centre` with a cross at the origin and an optional ring of radius `ring`."""
    img = np.full((size, size, 3), 30, np.uint8)
    s = (size / 2 - 8) / extent
    cx = cy = size // 2
    cv2.line(img, (0, cy), (size, cy), (70, 70, 70), 1)
    cv2.line(img, (cx, 0), (cx, size), (70, 70, 70), 1)
    if ring:
        cv2.circle(img, (cx, cy), max(int(ring * s), 1), (90, 160, 90), 1, cv2.LINE_AA)
    p = np.asarray(points, float)
    if len(p):
        xs = np.clip(((p[:, 0] - centre[0]) * s + cx).astype(int), 0, size - 1)
        ys = np.clip((cy - (p[:, 1] - centre[1]) * s).astype(int), 0, size - 1)
        img[ys, xs] = color
        for x, y in zip(xs[-1:], ys[-1:]):
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
    cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (110, 110, 110), 1)
    return img


def draw_dial(heading, size=240, label=""):
    img = np.full((size, size, 3), 30, np.uint8)
    c = size // 2
    r = c - 14
    cv2.circle(img, (c, c), r, (110, 110, 110), 1, cv2.LINE_AA)
    for name, ang in (("0", 0), ("90", 90), ("180", 180), ("270", 270)):
        a = np.radians(ang)
        cv2.putText(img, name, (int(c + (r - 14) * np.sin(a)) - 8, int(c - (r - 14) * np.cos(a)) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    if heading is not None:
        a = np.radians(heading)   # like a compass card: 0 at the top, 90 at the right, 180 at the bottom, 270 at the left
        cv2.arrowedLine(img, (c, c), (int(c + (r - 8) * np.sin(a)), int(c - (r - 8) * np.cos(a))), (0, 220, 255), 3, cv2.LINE_AA, tipLength=0.2)
    cv2.putText(img, label, (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--cal", default="data/mag_cal.json")
    ap.add_argument("--no-window", action="store_true", help="testing: no GUI")
    ap.add_argument("--calibrate-after", type=float, default=0, help="testing: calibrate and exit after N seconds")
    a = ap.parse_args()

    from gabai.pc.receiver import LiveReceiver
    rx = LiveReceiver(a.port, a.token)
    rx.start()
    print(f"waiting for the Pi on port {rx.port}; its sender must print 'compass found' ...", flush=True)
    pts = []                       # raw (x, y, z) since the last reset
    recent = deque(maxlen=60)      # (t_pc, |B| in microtesla) for rate and steadiness
    heads, cal, last_v = [], None, None
    if os.path.exists(a.cal):
        d = json.load(open(a.cal))
        cal = {"offset": np.array(d["offset"]), "scale": np.array(d["scale"]), "quality": d["quality"], "coverage": d["coverage"], **({"W": np.array(d["W"])} if d.get("W") is not None else {})}
    t0 = time.time()
    win = "Compass check (c calibrate, r reset, q quit)"
    msg = ""
    while True:
        for (_, x, y, z) in rx.drain_mag():
            v = np.array([x, y, z], float)
            last_v = v
            pts.append(v)
            recent.append((time.time(), field_ut(v)))
            c = v if cal is None else apply_cal(cal, v)
            heads.append(heading_deg(c[0], c[1]))
        if a.calibrate_after and time.time() - t0 > a.calibrate_after:
            break
        if not a.no_window:
            P = np.array(pts[-2500:]) if pts else np.zeros((0, 3))
            ref = cal["offset"] if cal is not None else (np.median(P, axis=0) if len(P) else np.zeros(3))
            ext = max(float(np.abs(P - ref).max()) * 1.15, 120.0) if len(P) else 300.0
            ring = None
            if cal is not None and len(P):
                ring = float(np.mean(np.linalg.norm(P - cal["offset"], axis=1)))
            plots = [draw_scatter(P[:, [i, j]] if len(P) else P, 290, (ref[i], ref[j]), ext, f"{n} (raw counts)", ring=ring)
                     for (i, j, n) in ((0, 1, "X-Y"), (0, 2, "X-Z"), (1, 2, "Y-Z"))]
            top = np.hstack(plots)
            hd = heads[-1] if heads else None
            dial = draw_dial(hd, 240, f"heading {hd:.0f} deg" if hd is not None else "no data")
            info = np.full((240, top.shape[1] - 240, 3), 30, np.uint8)
            lines = []
            if last_v is None:
                lines.append("no compass data yet: sender must print 'compass found'")
            else:
                ut = field_ut(last_v)
                ok = EARTH_RANGE_UT[0] <= ut <= EARTH_RANGE_UT[1]
                rate = (len(recent) - 1) / (recent[-1][0] - recent[0][0]) if len(recent) > 5 and recent[-1][0] > recent[0][0] else 0.0
                steady = np.std([u for _, u in recent]) if len(recent) > 5 else 0.0
                lines += [f"raw counts  x {last_v[0]:+5.0f}  y {last_v[1]:+5.0f}  z {last_v[2]:+5.0f}",
                          f"field strength {ut:5.1f} uT  " + ("(plausible Earth field 25-65)" if ok else "(OUTSIDE 25-65: metal or magnet nearby, or bad gain)"),
                          f"readings {rate:4.1f}/s   short-term wobble {steady:4.2f} uT   overflows {rx.stats().get('mag_overflows', 0)}",
                          f"heading sweep since reset: {sweep_deg(heads[-3000:]):.0f} deg   points {len(pts)}"]
                if len(pts) > 200:
                    cc = calibrate(np.array(pts[-4000:]))
                    lines.append(f"if calibrated now: quality {cc['quality']:.3f} ({'GOOD' if cc['quality'] < 0.03 else 'usable' if cc['quality'] < 0.06 else 'needs more rotation'}), "
                                 f"directions covered {cc['coverage'] * 100:.0f}%")
                if cal is not None:
                    lines.append(f"saved calibration in use: quality {cal['quality']:.3f}")
                if msg:
                    lines.append(msg)
            for i, s in enumerate(lines):
                cv2.putText(info, s, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            frame = np.vstack([top, np.hstack([dial, info])])
            cv2.imshow(win, frame)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("r"):
                pts, heads, msg = [], [], "reset"
            if k == ord("s"):
                cv2.imwrite(f"data/compass_{int(time.time())}.png", frame)
                msg = "screenshot saved in data/"
            if k == ord("c") and len(pts) > 200:
                cal = calibrate(np.array(pts))
                os.makedirs(os.path.dirname(a.cal) or ".", exist_ok=True)
                json.dump({k2: (v.tolist() if hasattr(v, "tolist") else v) for k2, v in cal.items()}, open(a.cal, "w"), indent=2)
                msg = f"calibrated and saved: quality {cal['quality']:.3f}, coverage {cal['coverage'] * 100:.0f}%"
                heads = []
        else:
            time.sleep(0.03)
    if a.calibrate_after and len(pts) > 200:
        cal = calibrate(np.array(pts))
        os.makedirs(os.path.dirname(a.cal) or ".", exist_ok=True)
        json.dump({k2: (v.tolist() if hasattr(v, "tolist") else v) for k2, v in cal.items()}, open(a.cal, "w"), indent=2)
        print(f"{len(pts)} readings; offset {np.round(cal['offset'], 1)}, scale {np.round(cal['scale'], 3)}, "
              f"quality {cal['quality']:.3f}, coverage {cal['coverage'] * 100:.0f}%", flush=True)
    rx.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
