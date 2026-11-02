"""Stereo RGB-D and visual-inertial pipeline in Rerun. See docs/rerun-pipeline.md.

    py -m gabai.tools.rerun_pipeline                                       config/pipeline.toml as it is
    py -m gabai.tools.rerun_pipeline --mode stereo_only --left data/left.mp4 --right data/right.mp4 --max-frames 0
    py -m gabai.tools.rerun_pipeline --mode stereo_imu --source recording --folder data/slam/live_20261026_181526
    py -m gabai.tools.rerun_pipeline --source live --mode stereo_imu     live from the Pi (start this, then the sender on the Pi)
    py -m gabai.tools.rerun_pipeline --check                               only check that the chosen mode can run, and say why not

Command-line options override the config file. --save writes a .rrd you can open later with `rerun file.rrd`.
"""
import argparse
import json

from gabai.pipeline.config import PipelineConfig, validate

DEFAULT_CONFIG = "config/pipeline.toml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--mode", choices=["stereo_only", "stereo_imu"])
    ap.add_argument("--source", choices=["video", "recording", "live", "replay"], help="live = the Pi's two cameras and IMU, drawn in Rerun as they arrive; replay = a folder from gabai.tools.record_pairs (--folder)")
    ap.add_argument("--rotate-left", type=int, choices=[0, 90, 180, 270], help="live: turn the left picture (default: the rotation stored in the stereo calibration)")
    ap.add_argument("--rotate-right", type=int, choices=[0, 90, 180, 270], help="live: turn the right picture")
    ap.add_argument("--scale", type=float, help="enlarge the pictures by this factor (1.0 to 2.0) before the depth model: fewer large errors at thin structures, slower (see docs/rerun-pipeline.md)")
    ap.add_argument("--flip180", action="store_true", help="live: turn both pictures 180 deg (use it when the view is upside down); the pictures trade places so the calibration stays valid")
    ap.add_argument("--no-flip180", action="store_true", help="live: do not turn the pictures (the config file turns them by default)")
    ap.add_argument("--mono", choices=["dav2"], help="compensate stereo noise with this monocular metric depth model (see docs/rerun-pipeline.md)")
    ap.add_argument("--mono-mode", choices=["blend", "check"], help="blend (default): average with trusted stereo; check: only veto gross disagreements")
    ap.add_argument("--port", type=int, help="live: port the Pi sender connects to")
    ap.add_argument("--left"), ap.add_argument("--right"), ap.add_argument("--folder")
    ap.add_argument("--start", type=float), ap.add_argument("--skip", type=int), ap.add_argument("--max-frames", type=int, help="0 = the whole sequence")
    ap.add_argument("--model", choices=["s", "m", "h"])
    ap.add_argument("--still-gate", choices=["auto", "on", "off"])
    ap.add_argument("--save"), ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--no-hands", action="store_true"), ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--klt", action="store_true", help="live + stereo_imu: fuse the feature tracks the Pi sends (start the sender with --klt) into the pose, KLT favoured (see docs/pi-klt.md)")
    ap.add_argument("--klt-weight", type=float, help="with --klt: share of the KLT pose while it agrees with the PC tracker (0.5 equal, 1 KLT only; default 0.75)")
    ap.add_argument("--filter", action="store_true", help="stereo_imu: the Kalman filter on the pose (IMU prediction, pictures as measurement, the Pi's own motion from sender --klt-motion); replaces --klt's direct fusion")
    a = ap.parse_args()
    ov = {}
    if a.filter:
        ov["filter.enabled"] = True
    if a.klt:
        ov["klt.enabled"] = True
    if a.klt_weight is not None:
        ov["klt.weight"] = a.klt_weight
    for k, v in (("pipeline.mode", a.mode), ("source.kind", a.source), ("source.left", a.left), ("source.right", a.right), ("source.folder", a.folder), ("source.start", a.start),
                 ("source.skip", a.skip), ("source.max_frames", a.max_frames), ("stereo.model", a.model), ("imu.still_gate", a.still_gate), ("viz.save", a.save)):
        if v is not None:
            ov[k] = v
    for k, v in (("source.rotate_left", a.rotate_left), ("source.rotate_right", a.rotate_right)):
        if v is not None:
            ov[k] = v
    for k, v in (("stereo.mono", a.mono), ("stereo.mono_mode", a.mono_mode)):
        if v is not None:
            ov[k] = v
    if a.scale is not None:
        ov["stereo.scale"] = a.scale
    if a.flip180:
        ov["source.flip180"] = True
    if a.no_flip180:
        ov["source.flip180"] = False
    if a.port is not None:
        ov["source.port"] = a.port
    if a.source == "replay":                               # a recording from gabai.tools.record_pairs: every pair, no frame skipping
        ov.setdefault("source.max_frames", 0)
    if a.source == "live":                                 # a live stream: every frame, newest first, no read-ahead
        ov.setdefault("source.max_frames", 0)
        ov["perf.drop_when_busy"], ov["perf.prefetch"] = True, False
    if a.no_viewer:
        ov["viz.viewer"] = False
    if a.no_hands:
        ov["hands.enabled"] = False
    if a.no_detect:
        ov["detector.enabled"] = False
    cfg = PipelineConfig.load(a.config, ov)
    errs, warns = validate(cfg)
    print(f"mode: {cfg.pipeline.mode}   source: {cfg.source.kind}")
    for w in warns:
        print("WARNING:", w)
    for e in errs:
        print("ERROR:", e)
    if a.check or errs:
        raise SystemExit(1 if errs else 0)
    from gabai.pipeline.run import run
    from gabai.pipeline import shutdown
    shutdown.install()                      # Ctrl-C ends the run even if it is stuck in a native call; see gabai/pipeline/shutdown.py
    s = run(cfg)
    print(json.dumps({k: v for k, v in s.items() if k not in ("warnings",)}, indent=2, default=str))
    if cfg.viz.save:
        print(f"saved {cfg.viz.save}: open with  rerun {cfg.viz.save}")


if __name__ == "__main__":
    main()
