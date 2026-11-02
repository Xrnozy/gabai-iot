"""Run the Rerun pipeline on the same video segment with different settings and print one line per run: speed, stage times and map consistency.

    py -m gabai.tools.pipeline_bench --start 30 --frames 150 [--hands]

Each variant runs in this process one after another, with the viewer off. The numbers are for THIS PC, this video and the assumed (not calibrated)
camera geometry of plain video; map_alignment is how far the map lies from the depth of the frame that was just placed (a consistency measure, not the
distance to the real world).
"""
import argparse
import json

from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.run import run

VARIANTS = {
    "original (frame VO, no map refine)": {"tracking.vo": "frame", "map.refine_pose": False, "map.weighted": False},
    "keyframe VO": {"tracking.vo": "keyframe", "map.refine_pose": False},
    "keyframe VO + weighted map": {"tracking.vo": "keyframe", "map.refine_pose": False, "map.weighted": True},
    "keyframe VO + weighted map + map refine": {"tracking.vo": "keyframe", "map.refine_pose": True, "map.weighted": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.toml")
    ap.add_argument("--start", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--hands", action="store_true", help="leave hand and object analysis on")
    ap.add_argument("--only", help="run only variants whose name contains this text")
    ap.add_argument("--out", help="write all summaries to this json file")
    a = ap.parse_args()
    rows = {}
    for name, ov in VARIANTS.items():
        if a.only and a.only not in name:
            continue
        o = {"source.start": a.start, "source.max_frames": a.frames, "source.skip": a.skip, "viz.viewer": False, "viz.save": "", **ov}
        if not a.hands:
            o.update({"hands.enabled": False, "detector.enabled": False})
        s = run(PipelineConfig.load(a.config, o), quiet=True)
        sm = {k: s["stage_ms"].get(k, {}).get("median_ms") for k in ("depth", "pose", "map", "alignment")}
        al = s["map_alignment_median_cm"] or {}
        rows[name] = s
        print(f"{name:42s} {s['fps_steady']:5.1f} fps  frame {s['frame_ms_median']:5.1f} ms (p95 {s['frame_ms_p95']:5.1f})  pose {sm['pose']:5.1f}  map {sm['map']:5.1f}  "
              f"voxels {s['map_voxels']:6d}  align median {al.get('median')} cm  last {al.get('last')} cm  held {s['poses_held']}  path {s['path_m']} m  refine {s.get('map_refine')}", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
