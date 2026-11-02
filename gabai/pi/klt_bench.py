"""Measures what --klt costs on this Pi: runs the real sender twice with the same arguments, once without and once with --klt, and prints both side by side.

  python3 -m gabai.pi.klt_bench --seconds 60 -- --host 192.168.1.11 --left /dev/video2 --right /dev/video0 --width 640 --height 360 --fps 15 --capture-fps 30 --exposure auto
  add  --klt-args "--klt-stereo --klt-fb-px 1"  to measure a variant

The PC must be listening (python -m gabai.pc.receiver, or a viewer) for the whole run. Per second it reads /proc for the sender's CPU time and memory, and
vcgencmd for temperature and throttling; frame rate, send times, pairing offset, bandwidth and tracking time come from the sender's own pi.stats log lines.
A Pi has no power sensor: CPU seconds, temperature and the under-voltage flags are the available stand-ins, and a USB power meter is the real answer.
"""
import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time

from gabai.pi.sender import read_pi_slow

TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
KEYS = ["sent_fps", "mbit_s", "send_ms_p95", "loop_gap_max_ms", "cam_fps_left", "cam_fps_right", "pair_dt_ms", "pair_dt_max_ms",
        "klt_ms_p50", "klt_ms_max", "klt_tracked", "klt_new", "klt_lost", "klt_kbit_s"]


def cpu_s(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / TICK                              # utime + stime


def mem_mb(pid):
    out = {}
    for line in open(f"/proc/{pid}/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            out[line.split(":")[0]] = int(line.split()[1]) / 1024
    return out.get("VmRSS", 0.0), out.get("VmHWM", 0.0)


def run_phase(label, sender_args, seconds, warmup):
    logdir = tempfile.mkdtemp(prefix=f"kltbench_{label}_")
    p = subprocess.Popen([sys.executable, "-m", "gabai.pi.sender", *sender_args, "--log-dir", logdir, "--stats-every", "2"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[{label}] sender pid {p.pid}: waiting {warmup:g} s for cameras, phase alignment and exposure, then measuring {seconds:g} s", flush=True)
    time.sleep(warmup)
    c0, t0, samples, temps, throttled = cpu_s(p.pid), time.monotonic(), [], [], set()
    while time.monotonic() - t0 < seconds and p.poll() is None:
        time.sleep(1.0)
        samples.append(mem_mb(p.pid))
        s = read_pi_slow()
        temps.append(s["temp_c"])
        throttled.add(s["throttled"])
    wall, cpu = time.monotonic() - t0, cpu_s(p.pid) - c0
    p.terminate()
    p.wait(timeout=10)
    stats = [json.loads(l) for f in glob.glob(os.path.join(logdir, "**", "gabai.jsonl"), recursive=True) for l in open(f) if '"pi.stats"' in l]
    stats = [s for s in stats if s.get("window_s", 0) >= 1.5][1:]       # the first window includes start-up
    row = {"cpu_pct_of_one_core": 100 * cpu / wall, "cpu_seconds": cpu, "rss_mb": samples[-1][0] if samples else None, "peak_rss_mb": max((m[1] for m in samples), default=None),
           "temp_c_max": max((t for t in temps if t is not None), default=None), "throttled_flags": sorted(x for x in throttled if x), "stat_windows": len(stats)}
    for k in KEYS:
        v = [s[k] for s in stats if s.get(k) is not None]
        row[k] = (sum(v) / len(v)) if v else None
    row["klt_ms_max"] = max((s["klt_ms_max"] for s in stats if s.get("klt_ms_max") is not None), default=None)
    row["frames_missed"] = sum(s.get("grab_fail_left", 0) + s.get("grab_fail_right", 0) for s in stats[-1:])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0, help="measured time per phase")
    ap.add_argument("--warmup", type=float, default=25.0, help="seconds to let the sender start and lock exposure before measuring")
    ap.add_argument("--cooldown", type=float, default=20.0, help="pause between the phases so the second one does not start hot")
    ap.add_argument("--klt-args", default="--klt", help="what the KLT phase adds to the sender arguments")
    ap.add_argument("sender", nargs=argparse.REMAINDER, help="after --: the sender arguments (without --klt)")
    a = ap.parse_args()
    base = [x for x in a.sender if x != "--"]
    if not base:
        sys.exit("give the sender arguments after --, e.g. -- --host 192.168.1.11 --width 640 --height 360")
    off = run_phase("off", base, a.seconds, a.warmup)
    time.sleep(a.cooldown)
    on = run_phase("klt", base + shlex.split(a.klt_args), a.seconds, a.warmup)
    print(f"\n{'':24}{'KLT off':>12}{'KLT on':>12}{'difference':>12}")
    for k in ["cpu_pct_of_one_core", "cpu_seconds", "rss_mb", "peak_rss_mb", "temp_c_max", *KEYS, "frames_missed"]:
        x, y = off.get(k), on.get(k)
        f = lambda v: "-" if v is None else f"{v:.2f}"
        print(f"{k:24}{f(x):>12}{f(y):>12}{('-' if x is None or y is None else f'{y - x:+.2f}'):>12}")
    print(f"throttled flags: off {off['throttled_flags']}  on {on['throttled_flags']}   (0x0 = none; see docs/pi-power-guide.md)")
    print("klt_kbit_s is the extra network traffic of the points; mbit_s is the video. Power: no sensor on the Pi, read it from a USB meter and compare the two phases.")


if __name__ == "__main__":
    main()
