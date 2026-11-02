"""Run the pipeline on a stretch of video and check that the map does not run away. Exit code 1 if it does.

    py -m gabai.tools.map_growth_check --start 20 --frames 450            # about 30 s
    py -m gabai.tools.map_growth_check --start 20 --frames 700 --hands    # the stretch where the 2026-10-28 runaway showed

Prints the voxel count every 25 map frames, how many frames were skipped for disagreeing with the map, map resets, and the map alignment. FAIL when:
the map passes max_voxels, grows by more than 50 percent and 2000 voxels within 25 frames, or is reset more than once. Plain video has no calibration, so
this checks stability of the mapping, not accuracy.
"""
import argparse
import sys

from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.run import run


def verdict(s, max_voxels):
    tr = s["map_voxels_every_25_frames"]
    bad = []
    if s["map_voxels_peak"] > max_voxels:
        bad.append(f"the map passed the cap ({s['map_voxels_peak']} > {max_voxels})")
    for a, b in zip(tr, tr[1:]):
        if b - a > 2000 and b > 1.5 * a:
            bad.append(f"the map grew from {a} to {b} voxels in 25 frames")
            break
    if s["map_resets"] > 1:
        bad.append(f"the map was reset {s['map_resets']} times")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.toml")
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--frames", type=int, default=450)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--hands", action="store_true")
    a = ap.parse_args()
    o = {"source.start": a.start, "source.max_frames": a.frames, "source.skip": a.skip, "viz.viewer": False, "viz.save": ""}
    if not a.hands:
        o.update({"hands.enabled": False, "detector.enabled": False})
    cfg = PipelineConfig.load(a.config, o)
    s = run(cfg, quiet=True)
    al = s["map_alignment_median_cm"] or {}
    print(f"frames {s['frames']}  fps {s['fps_steady']}  voxels final {s['map_voxels']} peak {s['map_voxels_peak']} evicted {s['map_voxels_evicted']}  path {s['path_m']} m")
    print("voxels every 25 map frames:", s["map_voxels_every_25_frames"])
    print(f"skipped (map disagreed) {s['frames_skipped_inconsistent']}  resets {s['map_resets']}  pose jumps {s['pose_jumps']}  alignment median {al.get('median')} cm last {al.get('last')} cm")
    print("refiner:", s["map_refine"])
    bad = verdict(s, cfg.map.max_voxels or 10 ** 9)
    print("FAIL: " + "; ".join(bad) if bad else "OK: the map stayed bounded")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
