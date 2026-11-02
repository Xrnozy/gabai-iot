"""How much does the visual tracker lose when frames arrive late or far apart (a stalled Pi link, a slow PC)? Replays a recording at its full rate and again with every k-th
frame only, then compares the poses at the frames both runs have. The full-rate run is the reference, so this is only meaningful on a recording the full-rate run tracks well
(data/recordings/normal_*); the error is what the gaps cost, not the absolute accuracy.

py scripts/evaluate_lag.py data/recordings/normal_20261029_235458 [--skip 1 2 3 5 8] [--set tracking.x=y ...]

Both runs start from the same first frame, so their poses share the origin; no fitting is done. Per skip: poses held, position error rms / at the end, rotation error rms / at the end.
"""
import argparse
import json
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation as R

from evaluate_still import Log, parse
from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.run import run


def go(folder, skip, model, extra):
    ov = {"pipeline.mode": "stereo_only", "source.kind": "replay", "source.folder": folder, "source.replay_skip": skip, "source.max_frames": 0, "stereo.model": model, "stereo.mono": "",
          "viz.viewer": False, "hands.enabled": False, "detector.enabled": False, "perf.async_viz": False, "perf.staged": False, "perf.low_priority": False, "tracking.gyro_prior": False}
    ov.update(extra)
    s = run(PipelineConfig.load("config/pipeline.toml", ov), viz_factory=Log, log_dir=tempfile.mkdtemp(), quiet=True)
    return s, {round(t, 4): (Rm, tw) for t, Rm, tw in Log.last.poses}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="h")
    ap.add_argument("--skip", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    extra = parse(a.set)
    ref = None
    for k in a.skip:
        s, poses = go(a.folder, k, a.model, extra)
        if ref is None:
            ref = poses
            print(f"skip {k}: reference, {len(poses)} poses, held {s['poses_held']}, inconsistent {s['frames_skipped_inconsistent']}", flush=True)
            continue
        common = sorted(set(ref) & set(poses))
        e = np.array([np.linalg.norm(poses[t][1] - ref[t][1]) for t in common]) * 100
        r = np.array([(R.from_matrix(poses[t][0]).inv() * R.from_matrix(ref[t][0])).magnitude() for t in common]) * 180 / np.pi
        print(f"skip {k}: {len(poses)} poses ({len(common)} shared), held {s['poses_held']}, inconsistent {s['frames_skipped_inconsistent']} | position error rms {np.sqrt((e ** 2).mean()):.1f} cm, end {e[-1]:.1f} cm"
              f" | rotation error rms {np.sqrt((r ** 2).mean()):.2f} deg, end {r[-1]:.2f} deg", flush=True)


if __name__ == "__main__":
    main()
