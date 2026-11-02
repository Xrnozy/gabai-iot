"""Collect one zip you can send to an AI coding agent (or keep) when something looks wrong.

    py -m gabai.tools.collect_diag                  the newest 3 runs, calibration, summaries, a self-check, a report
    py -m gabai.tools.collect_diag --with-images    also the saved diagnostic pictures (depth maps, grid, left frames)
    py -m gabai.tools.collect_diag --runs 6 --max-mb 100

It contains: the log folders of the newest runs (structured logs, the last frame traces, operator marks), the calibration files, the CSV run
logs, SLAM recording summaries (trajectory, IMU samples and timestamps, but NOT the camera pictures), the environment, a fresh
self-check, a diagnostic report of the newest run, and MANIFEST.json (every file with its size and SHA-256, and what was left out and why).
It never contains camera footage unless you ask (--with-images). Tokens and passwords are replaced by <redacted> in every text file.
Copy the Pi's log folder (~/gabai_logs/pi_...) next to the zip if the problem involves the Pi.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import re
import time
import zipfile

PICTURE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".mp4", ".avi")
TEXT_EXT = (".json", ".jsonl", ".csv", ".txt", ".md", ".log", ".toml")
README = """Gabai diagnostic package
========================
Made {when} by gabai.tools.collect_diag.

What this is: logs, calibration and summaries from one or more runs, collected so a problem can be traced to its source.
What to do with it: give the whole zip to the AI coding agent together with a sentence on what looked wrong and when (the time, and what
the rig was doing). Start the agent at diag_report.txt, then logs/<run>/gabai.jsonl (one JSON object per line; `ev` is the event, `fid`
the frame id that ties the PC log to the Pi log), then selfcheck.json and the calibration files.
What is NOT here: camera footage ({images}), model weights, anything outside the project folder. Tokens are redacted.
See MANIFEST.json for every file with its SHA-256 and the list of files that were left out and why.
"""


def redact(text):
    text = re.sub(r'("(?:token|password|secret|api_key)"\s*:\s*)"[^"]*"', r'\1"<redacted>"', text, flags=re.I)
    text = re.sub(r"\b(token|password|secret|api_key)=\S+", r"\1=<redacted>", text, flags=re.I)
    return text


def _newest(paths, n):
    return sorted(paths, key=os.path.getmtime, reverse=True)[:n]


def collect(root=".", out=None, runs=3, with_images=False, max_mb=50.0, run_selfcheck=True, log_dir=None):
    root = os.path.abspath(root)
    log_dir = log_dir or os.path.join(root, "logs")
    out = out or os.path.join(root, "data", "diag", time.strftime("diag_%Y%m%d_%H%M%S.zip"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cap = max_mb * 1e6
    used = 0
    manifest, left_out, queued = [], [], []

    def want(path, arc, tail_lines=None):
        queued.append((path, arc, tail_lines))

    # logs of the newest runs
    run_dirs = _newest([d for d in glob.glob(os.path.join(log_dir, "*")) if os.path.exists(os.path.join(d, "run.json"))], runs)
    for d in run_dirs:
        for dp, _, fs in os.walk(d):
            for f in fs:
                p = os.path.join(dp, f)
                arc = os.path.join("logs", os.path.relpath(p, log_dir)).replace("\\", "/")
                if f.lower().endswith(PICTURE_EXT) or f.endswith(".npz"):
                    if not with_images:
                        left_out.append({"file": arc, "why": "pictures excluded (use --with-images)"})
                        continue
                want(p, arc)
    for name in ("stereo_calib_new.json", "stereo_calib_new.npz", "stereo_calib.json", "stereo_calib.npz", "imu_sync.json", "accel_calib.json", "mag_cal.json", "mag_sync.json"):
        p = os.path.join(root, "data", name)
        if os.path.exists(p):
            want(p, f"data/{name}")
    for p in _newest(glob.glob(os.path.join(root, "data", "runs", "*.csv")), 6):
        want(p, "data/runs/" + os.path.basename(p))
    p = os.path.join(root, "data", "telemetry_log.csv")
    if os.path.exists(p):
        want(p, "data/telemetry_log.csv", tail_lines=2000)
    for d in _newest([x for x in glob.glob(os.path.join(root, "data", "slam", "*")) if os.path.isdir(x)], 2):
        base = os.path.basename(d)
        for rel in ("recording_info.json", "slam_summary.json", "trajectory.txt", "calib.json", "mav0/imu0/data.csv", "mav0/cam0/data.csv"):
            q = os.path.join(d, rel)
            if os.path.exists(q):
                want(q, f"data/slam/{base}/{rel}")
        pics = len(glob.glob(os.path.join(d, "mav0", "cam*", "data", "*")))
        if pics:
            left_out.append({"file": f"data/slam/{base}/mav0/cam*/data/ ({pics} pictures)", "why": "pictures excluded (use --with-images)" if not with_images else "too large: not included"})
    for p in _newest(glob.glob(os.path.join(root, "data", "reports", "*", "report.md")), 1):
        want(p, "data/reports_latest/report.md")

    generated = {}
    from gabai.diag.env import calibration_summary, environment_info
    generated["environment.json"] = json.dumps({"environment": environment_info(), "calibration": calibration_summary(root)}, indent=2, default=str)
    if run_selfcheck:
        try:
            from gabai.tools import selfcheck
            from dataclasses import asdict
            res = selfcheck.run(root, log_dir=log_dir)
            generated["selfcheck.json"] = json.dumps({"results": [asdict(r) for r in res], "summary": selfcheck.summarize(res)}, indent=2)
        except Exception as e:
            generated["selfcheck.json"] = json.dumps({"error": f"{type(e).__name__}: {e}"})
    if run_dirs:
        try:
            from gabai.tools.diag_report import analyze_events, format_report, load_events
            generated["diag_report.txt"] = format_report(analyze_events(load_events(run_dirs[0])))
        except Exception as e:
            generated["diag_report.txt"] = f"report failed: {type(e).__name__}: {e}"

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        def add_bytes(arc, data, src=None):
            nonlocal used
            if used + len(data) > cap:
                left_out.append({"file": arc, "why": f"size cap {max_mb} MB reached"})
                return
            z.writestr(arc, data)
            used += len(data)
            manifest.append({"file": arc, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "source": src})

        for arc, text in generated.items():
            add_bytes(arc, redact(text).encode("utf-8"))
        for p, arc, tail in queued:
            if arc.lower().endswith(TEXT_EXT):
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                if tail and len(lines) > tail + 1:
                    lines = lines[:1] + lines[-tail:]
                data = redact("".join(lines)).encode("utf-8")
            else:
                data = open(p, "rb").read()
            add_bytes(arc, data, os.path.relpath(p, root))
        z.writestr("README.txt", README.format(when=time.strftime("%Y-%m-%d %H:%M"), images="included" if with_images else "excluded"))
        z.writestr("MANIFEST.json", json.dumps({"made": time.strftime("%Y-%m-%dT%H:%M:%S"), "root": root, "files": manifest, "left_out": left_out, "total_bytes": used}, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--with-images", action="store_true")
    ap.add_argument("--max-mb", type=float, default=50.0)
    ap.add_argument("--out")
    ap.add_argument("--root", default=".")
    ap.add_argument("--log-dir", default=os.environ.get("GABAI_LOG_DIR"))
    ap.add_argument("--no-selfcheck", action="store_true")
    a = ap.parse_args()
    z = collect(a.root, a.out, a.runs, a.with_images, a.max_mb, not a.no_selfcheck, a.log_dir)
    m = json.loads(zipfile.ZipFile(z).read("MANIFEST.json"))
    print(f"wrote {z}\n  {len(m['files'])} files, {m['total_bytes'] / 1e6:.1f} MB; left out: {len(m['left_out'])} (see MANIFEST.json)")
    print("Send this zip to the AI coding agent with a sentence on what looked wrong and when. If the Pi was involved, also copy ~/gabai_logs from the Pi.")


if __name__ == "__main__":
    main()
