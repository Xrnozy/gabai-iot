"""Find the exposure and gain for the room your cameras are in, on the Pi, in about 40 seconds.

    python3 -m gabai.pi.exposure_probe --left /dev/video2 --right /dev/video0

Hold the rig STILL, pointed at the kind of scene you work in, with the lights you normally have. For every exposure x gain in a small grid the probe sets
BOTH cameras to the same values, waits for the picture to settle, and measures each camera's brightness, clipped pixels and noise. It also tries the cameras'
own auto exposure for comparison, checks that the exposure really changes the picture (a control a camera ignores is the usual reason pictures stay dark), and
prints the power state of the Pi. Then it prints the shortest exposure that gives a bright enough, matching picture, and the sender command that uses it.
Short exposure = less smear when the rig moves; gain = more noise; the table shows both so you can see the price.
"""
import argparse
import subprocess
import time

import cv2
import numpy as np

from gabai.pi.cams import control_limits, open_cam, v4l2

EXPOSURES_MS = (4, 8, 12, 16, 24, 33)
GAIN_FRACTIONS = (0.0, 0.5, 1.0)


def frame_stats(grays):
    """grays: consecutive greyscale frames of a still scene. median brightness, share of clipped pixels, temporal noise (grey levels, from frame differences)."""
    g = np.stack([x.astype(np.float32) for x in grays])
    mean = float(np.median(g))
    mid = (g[0] > 20) & (g[0] < 235)
    noise = float(np.std((g[1:] - g[:-1])[:, mid]) / np.sqrt(2)) if len(g) > 1 and mid.any() else float("nan")
    return {"median": mean, "clipped_pct": float((g >= 250).mean() * 100), "noise": noise}


def responds(rows, factor=1.25):
    """Does a longer exposure make a brighter picture? True/False/None (not enough data). Compares each exposure with the next longer one at the same gain, picture not clipped."""
    ratios = []
    for gain in sorted({r["gain"] for r in rows if r["mode"] == "manual"}):
        rs = sorted([r for r in rows if r["mode"] == "manual" and r["gain"] == gain], key=lambda r: r["exposure_ms"])
        for a, b in zip(rs, rs[1:]):
            if b["exposure_ms"] >= 1.5 * a["exposure_ms"] and a["left"]["clipped_pct"] < 1 and b["left"]["clipped_pct"] < 1 and a["left"]["median"] > 5:
                ratios.append(b["left"]["median"] / a["left"]["median"])
    return None if not ratios else bool(np.median(ratios) >= factor)


def recommend(rows, cap_ms=33.0, lo=70.0, hi=150.0, max_clip=2.0, max_lr_ratio=1.15):
    """The manual row with the shortest exposure (then the lowest gain) whose BOTH pictures are bright enough, not clipped and match each other.
    Returns (row or None, reason). Without such a row, the brightest unclipped row under the exposure cap, with a reason saying why it is not enough."""
    manual = [r for r in rows if r["mode"] == "manual" and r["exposure_ms"] <= cap_ms]
    ok = [r for r in manual if lo <= min(r["left"]["median"], r["right"]["median"]) and max(r["left"]["median"], r["right"]["median"]) <= hi
          and max(r["left"]["clipped_pct"], r["right"]["clipped_pct"]) <= max_clip
          and max(r["left"]["median"], r["right"]["median"]) / max(min(r["left"]["median"], r["right"]["median"]), 1.0) <= max_lr_ratio]
    if ok:
        return min(ok, key=lambda r: (r["exposure_ms"], r["gain"])), "shortest exposure that gives a bright, unclipped, matching picture"
    if not manual:
        return None, "no manual setting was measured"
    best = max(manual, key=lambda r: min(r["left"]["median"], r["right"]["median"]) - 10 * max(r["left"]["clipped_pct"], r["right"]["clipped_pct"]))
    return best, f"nothing in the grid reaches brightness {lo:g}-{hi:g} with matching, unclipped pictures within {cap_ms:g} ms: this is the best of them, add light"


def power_state():
    out = {}
    for key, cmd in (("throttled", ["vcgencmd", "get_throttled"]), ("arm_clock", ["vcgencmd", "measure_clock", "arm"]), ("temp", ["vcgencmd", "measure_temp"])):
        try:
            out[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out[key] = "unavailable"
    return out


def grab_gray(cap, n):
    out = []
    for i in range(n + 4):                                    # the first frames after a control change are still the old picture
        ok = cap.grab()
        ok, fr = cap.retrieve() if ok else (False, None)
        if ok and i >= 4:
            img = cv2.imdecode(fr.reshape(-1), cv2.IMREAD_GRAYSCALE) if fr.ndim == 2 or fr.shape[0] == 1 else cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            out.append(img)
    return out


def measure(caps, n=6):
    left, right = grab_gray(caps["L"], n), grab_gray(caps["R"], n)
    return frame_stats(left), frame_stats(right)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="/dev/video2"), ap.add_argument("--right", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640), ap.add_argument("--height", type=int, default=360), ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-exposure-ms", type=float, default=16.0, help="the longest exposure you accept (smear of a moving scene = turn rate x focal length x this)")
    ap.add_argument("--brightness-low", type=float, default=70.0, help="the darkest picture median you accept")
    a = ap.parse_args()
    print("power:", power_state(), flush=True)
    lim = {d: control_limits(d) for d in (a.left, a.right)}
    e_lim = [lim[d].get("exposure_time_absolute") for d in (a.left, a.right)]
    g_lim = [lim[d].get("gain") for d in (a.left, a.right)]
    if None in e_lim:
        raise SystemExit("a camera has no exposure_time_absolute control (v4l2-ctl --list-ctrls): only auto exposure is possible")
    print(f"limits: exposure_time_absolute {e_lim[0]} (0.1 ms units), gain {g_lim[0]}", flush=True)
    caps = {"L": open_cam(a.left, a.width, a.height, a.fps), "R": open_cam(a.right, a.width, a.height, a.fps)}
    devs = {"L": a.left, "R": a.right}
    for d in devs.values():
        v4l2(d, white_balance_automatic=0, focus_automatic_continuous=0, exposure_dynamic_framerate=0)
    rows = []
    gains = sorted({int(round(g_lim[0][0] + f * (g_lim[0][1] - g_lim[0][0]))) for f in GAIN_FRACTIONS}) if g_lim[0] else [0]
    for g in gains:
        for e_ms in EXPOSURES_MS:
            e = int(e_ms * 10)
            if not (e_lim[0][0] <= e <= e_lim[0][1]):
                continue
            for d in devs.values():
                v4l2(d, auto_exposure=1, exposure_time_absolute=e, **({"gain": g} if g_lim[0] else {}))
            time.sleep(0.7)
            sl, sr = measure(caps)
            rows.append({"mode": "manual", "exposure_ms": e_ms, "gain": g, "left": sl, "right": sr})
            print(f"  manual {e_ms:>4g} ms gain {g:>3}: left {sl['median']:5.1f} right {sr['median']:5.1f}  clipped {max(sl['clipped_pct'], sr['clipped_pct']):4.1f}%  noise {sl['noise']:4.1f}", flush=True)
    for d in devs.values():
        v4l2(d, auto_exposure=3)
    time.sleep(2.5)
    sl, sr = measure(caps)
    got = subprocess.run(["v4l2-ctl", "-d", a.left, "--get-ctrl=exposure_time_absolute,gain"], capture_output=True, text=True).stdout.split()
    print(f"  camera AUTO: left {sl['median']:5.1f} right {sr['median']:5.1f}  clipped {max(sl['clipped_pct'], sr['clipped_pct']):4.1f}%  noise {sl['noise']:4.1f}  (it reports {' '.join(got)})", flush=True)
    rows.append({"mode": "auto", "exposure_ms": None, "gain": None, "left": sl, "right": sr})
    for c in caps.values():
        c.release()
    r = responds(rows)
    print("\nDoes a longer exposure make a brighter picture?", {True: "yes", False: "NO: the camera ignores exposure_time_absolute while streaming (or the picture is limited by something else)", None: "not enough data"}[r])
    best, why = recommend(rows, a.max_exposure_ms, a.brightness_low)
    if best is None:
        print("no recommendation:", why)
        return
    print(f"\nRECOMMENDED: exposure {best['exposure_ms']:g} ms, gain {best['gain']} ({why})")
    print(f"  left {best['left']['median']:.0f} right {best['right']['median']:.0f}, noise {best['left']['noise']:.1f}, smear at 60 deg/s about {618 * 1.047 * best['exposure_ms'] / 1000:.0f} px")
    print(f"  python3 -m gabai.pi.sender --host <PC_IP> --left {a.left} --right {a.right} --width {a.width} --height {a.height} --fps 15 --capture-fps 30 "
          f"--exposure {int(best['exposure_ms'] * 10)} --gain {best['gain']}")
    if rows[-1]["mode"] == "auto":
        print(f"  for comparison the camera's own auto exposure gives left {rows[-1]['left']['median']:.0f} / right {rows[-1]['right']['median']:.0f} (its exposure is not under your control)")


if __name__ == "__main__":
    main()
