"""Checkerboard-free stereo self-calibration.

  py -m mapper.selfcal calibrate data/recordings/<rec> [more recordings] [--baseline-mm 47.0] [--compare data/stereo_calib_charuco_45.npz]
  py -m mapper.selfcal status
  py -m mapper.selfcal activate data/selfcal/<run> --confirm

`calibrate` writes ONLY into a new data/selfcal/run_<time>/ (calib.npz, report.json, log.txt, frames.json). It never touches an existing calibration file. If the pictures do not determine the
calibration it writes a refusal report and no calib.npz: nothing is guessed. `activate` records the run as the active self-calibration in data/selfcal/active.json after validation passed and
--confirm was given; until then every consumer keeps the existing calibration. To use a run explicitly:  py -m mapper.run_offline ... --calib data/selfcal/<run>/calib.npz   (or --calib active).

What to capture (no target): in a textured, well-lit room, hold the rig STILL for about a second at 20-30 different orientations (pan, tilt and roll by 15-30 degrees between holds), with things at 0.7-4 m
in view. A walk is not usable: the two cameras read their rows in opposite directions, so any motion becomes a row error, and the solve needs the rig to be still at each view.
Metric scale: pictures cannot give the baseline. Pass --baseline-mm from a caliper measurement of the lens centres; without it the result is saved as NOT metric and cannot be activated.
"""
import argparse
import json
import os
import sys

import numpy as np

from mapper.selfcal import output
from mapper.selfcal.core import Insufficient, calibrate
from mapper.selfcal.matching import build_obs, select_frames

ACTIVE = "data/selfcal/active.json"


def cmd_calibrate(a):
    from rtabmap_lab.calib_lab.recording import Recording
    out = output.new_run_dir(a.root)
    logf = open(os.path.join(out, "log.txt"), "x", encoding="utf-8")

    def log(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"run directory {out}; recordings {a.recordings}; no existing calibration is read or written")
    recs, fit, held = [], [], []
    for folder in a.recordings:
        rec = Recording(folder)
        frames, _ = select_frames(rec, n=a.frames, max_dt_ms=a.max_dt_ms, max_flow_px=a.max_flow_px, min_gap_s=a.min_gap_s, log=log)
        recs.append((rec, frames))
    if len(recs) != 1:
        log("one recording at a time for now (frames of different sessions share one rig only if it did not move between them)")
        write_refusal(out, "one recording at a time")
        return 2
    rec, frames = recs[0]
    held = frames[3::4]
    fit = [f for f in frames if f not in held]
    log(f"{len(fit)} frames for the solve, {len(held)} held out for validation")
    if len(fit) < 10:
        return refuse(out, log, f"only {len(fit)} usable still, sharp, synchronised frames (need 10+). Capture protocol: see `py -m mapper.selfcal --help`")
    obs = build_obs(rec, fit, log=log)
    if obs is None:
        return refuse(out, log, "no feature tracks")
    try:
        ba = calibrate(obs, log=log)
    except Insufficient as e:
        return refuse(out, log, str(e))
    if ba["inlier_rms_px"] > 1.0:
        return refuse(out, log, f"reprojection error {ba['inlier_rms_px']:.2f} px after the solve: the model does not explain the views (moving objects, wrong matches, or the rig flexed)")
    rep = {"recording": rec.folder, "params": output.params_report(ba), "metric": a.baseline_mm is not None, "baseline_mm": a.baseline_mm}
    if ba["ts"][0] > 0:
        rep["warning"] = "the right picture appears to come from the left camera: cameras swapped or a picture rotated 180 degrees. Not activatable until the layout is fixed."
    npz = output.to_npz(os.path.join(out, "calib.npz"), ba, rec.size, a.baseline_mm)
    ok, val = output.validate(rec, held, npz, compare=a.compare)
    rep["validation"] = val
    rep["validated"] = bool(ok)
    rep["activatable"] = bool(ok and a.baseline_mm is not None and "warning" not in rep)
    log("validation on held-out frames: " + json.dumps(val["new"]))
    if a.compare:
        log("existing calibration on the same held-out frames (comparison only): " + json.dumps(val["existing_for_comparison"]))
    for r in val["reasons_for_failure"]:
        log("FAILED: " + r)
    output.write_json(os.path.join(out, "report.json"), rep)
    output.write_json(os.path.join(out, "frames.json"), {"fit": fit, "held_out": held})
    log(f"validated: {ok}; activatable: {rep['activatable']}" + ("" if a.baseline_mm else " (no --baseline-mm: not metric)"))
    log(f"nothing is active yet. To adopt:  py -m mapper.selfcal activate {out} --confirm")
    return 0


def refuse(out, log, why):
    log("REFUSED: " + why)
    write_refusal(out, why)
    return 2


def write_refusal(out, why):
    output.write_json(os.path.join(out, "report.json"), {"refused": True, "reason": why, "validated": False, "activatable": False})


def cmd_check(a):
    """Quick look at a capture before the long solve: how many separate still holds, and how different their views are."""
    import cv2
    from rtabmap_lab.calib_lab.recording import Recording
    from mapper.selfcal.matching import frame_stats
    rec = Recording(a.recording)
    prev, stats = None, []
    for i in range(rec.n):
        st, prev = frame_stats(rec, i, prev)
        stats.append(st)
    good = [st["i"] for st in stats if st["flow"] < a.max_flow_px and st["dt"] is not None and abs(st["dt"]) <= a.max_dt_ms and st["sharp"] >= 120 and st["bright"] >= 40]
    holds, cur = [], []
    for i in good:                                                              # consecutive good frames = one hold
        if cur and i != cur[-1] + 1:
            holds.append(cur); cur = []
        cur.append(i)
    if cur:
        holds.append(cur)
    holds = [h for h in holds if len(h) >= a.min_hold_frames]
    reps, kept = [], []
    for h in holds:                                                             # one representative per hold; a hold counts if its view differs from every earlier one
        g = cv2.cvtColor(rec.raw(h[len(h) // 2])[0], cv2.COLOR_BGR2GRAY)
        new = True
        for r in reps:
            p0 = cv2.goodFeaturesToTrack(r, 300, 0.01, 8)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(r, g, p0, None, winSize=(21, 21), maxLevel=4)
            ok = st[:, 0] == 1
            if ok.sum() >= 15 and np.median(np.linalg.norm((p1 - p0)[ok, 0], axis=1)) < a.min_view_shift_px:
                new = False
                break
        if new:
            reps.append(g); kept.append(h)
    print(f"{rec.n} pairs: {len(good)} still, sharp, synchronised; {len(holds)} holds of {a.min_hold_frames}+ frames; {len(kept)} with clearly different views (shift >= {a.min_view_shift_px:g} px)")
    need = 15
    print(("ENOUGH: run calibrate" if len(kept) >= need else f"NOT ENOUGH: need {need}+ different views, have {len(kept)}. Hold still at {need - len(kept)}+ more orientations (tilt, pan and roll by 15-30 degrees between holds).")
          + "  (a capture with fewer than 10 views is refused by calibrate)")
    return 0 if len(kept) >= need else 1


def cmd_status(a):
    if not os.path.exists(ACTIVE):
        print("no self-calibration is active: every consumer uses its existing calibration")
        return 0
    print(open(ACTIVE).read())
    return 0


def cmd_activate(a):
    rep = json.load(open(os.path.join(a.run, "report.json")))
    if not rep.get("activatable"):
        print("REFUSED: this run is not activatable:", rep.get("reason") or rep.get("validation", {}).get("reasons_for_failure") or "no metric baseline / layout warning / not validated")
        return 2
    if not a.confirm:
        print("this would make", a.run, "the active self-calibration (the existing calibration files stay as they are). Re-run with --confirm to do it.")
        return 1
    os.makedirs(os.path.dirname(ACTIVE), exist_ok=True)
    prev = open(ACTIVE).read() if os.path.exists(ACTIVE) else None
    if prev:
        os.replace(ACTIVE, ACTIVE + time_suffix())                                # the previous pointer is kept, so reverting is a rename
    with open(ACTIVE, "w") as f:
        json.dump({"calib": os.path.abspath(os.path.join(a.run, "calib.npz")), "run": os.path.abspath(a.run), "baseline_mm": rep["baseline_mm"]}, f, indent=1)
    print("active self-calibration:", a.run, "(previous pointer kept)" if prev else "")
    return 0


def time_suffix():
    import time
    return time.strftime(".prev_%Y%m%d_%H%M%S")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("recordings", nargs="+")
    c.add_argument("--baseline-mm", type=float, help="measured lens-centre spacing; without it the result is not metric")
    c.add_argument("--compare", help="an existing calibration .npz, evaluated read-only on the same held-out frames for comparison")
    c.add_argument("--frames", type=int, default=40)
    c.add_argument("--max-flow-px", type=float, default=0.5)
    c.add_argument("--max-dt-ms", type=float, default=3.0)
    c.add_argument("--min-gap-s", type=float, default=0.4)
    c.add_argument("--root", default="data/selfcal")
    k = sub.add_parser("check", help="how many still holds with clearly different views a recording has (run this before calibrate)")
    k.add_argument("recording")
    k.add_argument("--max-flow-px", type=float, default=0.5)
    k.add_argument("--max-dt-ms", type=float, default=3.0)
    k.add_argument("--min-hold-frames", type=int, default=6)
    k.add_argument("--min-view-shift-px", type=float, default=40.0)
    sub.add_parser("status")
    s = sub.add_parser("activate")
    s.add_argument("run")
    s.add_argument("--confirm", action="store_true")
    a = ap.parse_args(argv)
    return {"calibrate": cmd_calibrate, "status": cmd_status, "activate": cmd_activate, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
