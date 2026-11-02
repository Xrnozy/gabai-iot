"""Compare ways of tracking the camera on one IMU recording (data/slam/rec_*, live_*): what the map looks like and how well each pose agrees with the depth.

py scripts/evaluate_tracking.py data/slam/rec_20261026_155458 [--model s] [--frames 0]
    [--variants "pipeline.mode=stereo_only tracking.gyro_prior=false" "pipeline.mode=stereo_imu"]

Per variant, replayed through the real pipeline (mono off, so only the tracking differs):
  held / inconsistent   frames whose pose was held / kept out of the map because pose and depth disagreed
  align                 median distance between the map and each frame's own depth (cm), smaller = the new frame lands on what is already mapped
  thickness / isolated  median distance of a map point from the plane through its neighbours (cm) and share of points with no neighbour: how chaotic the cloud is
  vs Basalt             when data/slam/<rec>/trajectory.txt exists: position error of the path after the best rigid fit (a reference, NOT ground truth;
                        meaningless for the stereo_imu variant, which IS Basalt)
"""
import argparse
import os
import tempfile

import numpy as np

from evaluate_cloud import metrics, parse
from evaluate_motion import PoseViz, versus_reference
from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.run import run
from gabai.relocalize import load_map

DEFAULT = ["pipeline.mode=stereo_only tracking.gyro_prior=false", "pipeline.mode=stereo_only tracking.gyro_prior=true", "pipeline.mode=stereo_imu"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="s")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=DEFAULT)
    a = ap.parse_args()
    ref = os.path.join(a.folder, "trajectory.txt")
    for v in a.variants:
        out = os.path.join(tempfile.mkdtemp(), "m.npz")
        ov = {"source.kind": "recording", "source.folder": a.folder, "source.skip": 1, "source.max_frames": a.frames, "stereo.model": a.model, "stereo.mono": "", "viz.viewer": False,
              "hands.enabled": False, "detector.enabled": False, "perf.async_viz": False, "perf.staged": False, "perf.low_priority": False, "map.save": out}
        ov.update(parse(v))
        cfg = PipelineConfig.load("config/pipeline.toml", ov)
        s = run(cfg, viz_factory=PoseViz, log_dir=tempfile.mkdtemp(), quiet=True)
        al = s.get("map_alignment_median_cm")
        line = f"[{v}] frames {s['frames']} held {s['poses_held']} inconsistent {s['frames_skipped_inconsistent']} resets {s['map_resets']} align {al['median'] if al else None} cm"
        if os.path.exists(out):
            vm, _ = load_map(out)
            p = vm.points(min_obs=3, core=cfg.map.core, faces_only=True)
            if len(p) > 50:
                m = metrics(p, cfg.map.voxel)
                line += f" | points {len(p)} thickness {m['thick_med']:.2f} cm (p90 {m['thick_p90']:.2f}) isolated {m['isolated']:.1f}%"
        print(line, flush=True)
        r = versus_reference(PoseViz.last.poses, ref) if os.path.exists(ref) and "stereo_imu" not in v else None
        if r:
            print(f"    vs Basalt over {r['n']} frames, path {r['path_m']:.1f} m: rms {r['rms_cm']:.1f} cm max {r['max_cm']:.1f} cm end {r['end_pct']:.1f}% of the path", flush=True)


if __name__ == "__main__":
    main()
