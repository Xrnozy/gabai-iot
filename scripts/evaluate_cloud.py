"""Compare the fused point cloud made by two pipeline settings on the same recording.

py scripts/evaluate_cloud.py data/recordings/normal_20261029_235458 --model h
py scripts/evaluate_cloud.py <folder> --a "map.voxel=0.05 map.stride=3 map.window_m=30 stereo.mono= stereo.depth_correction=" --b "map.voxel=0.03 map.stride=2"

Each setting replays the recording (gabai.tools.record_pairs folder) through the real pipeline, saves the fused map, and reports what can be measured without ground truth:
  points      solid voxels (seen at least 3 times) in the shown map
  thickness   median / p90 distance of a point from the best plane through its 12 nearest neighbours: how much noise lies on a surface (smaller = cleaner)
  isolated    share of points with no other point within 3 voxels: scatter that is not part of any surface
  spacing     median distance to the nearest neighbour (smaller = finer)
It also writes <out>_a.png and <out>_b.png, top and front views coloured by height, to look at. Thickness and spacing say how clean and fine the cloud is; they cannot say
the cloud is placed correctly. For distance accuracy use gabai.tools.fit_depth_correction.
"""
import argparse
import json
import os
import tempfile

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.run import run
from gabai.relocalize import load_map
from tests.test_pipeline_run import ListViz

OLD = "map.voxel=0.05 map.stride=3 map.window_m=30 map.carve_m=0 stereo.mono= stereo.depth_correction= tracking.loop_closure=false"
NEW = "map.voxel=0.03 map.stride=2 stereo.mono=dav2 stereo.mono_mode=fill"


def parse(kv):
    out = {}
    for item in kv.split():
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def replay(folder, model, overrides, tag):
    ov = {"pipeline.mode": "stereo_only", "source.kind": "replay", "source.folder": folder, "source.max_frames": 0, "stereo.model": model, "viz.viewer": False,
          "hands.enabled": False, "detector.enabled": False, "perf.async_viz": False, "perf.staged": False, "perf.low_priority": False, "map.save": os.path.join(tempfile.mkdtemp(), tag + ".npz")}
    ov.update(parse(overrides))
    cfg = PipelineConfig.load("config/pipeline.toml", ov)
    s = run(cfg, viz_factory=ListViz, log_dir=tempfile.mkdtemp(), quiet=True)
    vm, _ = load_map(cfg.map.save)
    return vm.points(min_obs=3, core=cfg.map.core, faces_only=True), s, cfg


def metrics(p, voxel):
    t = cKDTree(p)
    d, idx = t.query(p, k=13)
    nn = d[:, 1]
    sel = np.random.default_rng(0).choice(len(p), min(len(p), 20000), replace=False)
    thick = []
    for i in sel:
        q = p[idx[i]] - p[idx[i]].mean(0)
        thick.append(np.sqrt(np.linalg.eigvalsh(q.T @ q / len(q))[0]))
    thick = np.array(thick)
    return dict(points=len(p), thick_med=float(np.median(thick)) * 100, thick_p90=float(np.percentile(thick, 90)) * 100,
                isolated=float((nn > 3 * voxel).mean()) * 100, spacing_cm=float(np.median(nn)) * 100)


def picture(p, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    z = p[:, 2]
    ax[0].scatter(p[:, 0], p[:, 1], c=z, s=0.3, cmap="turbo")
    ax[0].set_aspect("equal"); ax[0].set_title(title + ": top view (x, y)")
    ax[1].scatter(p[:, 0], p[:, 2], c=p[:, 1], s=0.3, cmap="viridis")
    ax[1].set_aspect("equal"); ax[1].set_title("front view (x, z)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="h")
    ap.add_argument("--a", default=OLD)
    ap.add_argument("--b", default=NEW)
    ap.add_argument("--out", default="data/reports/cloud_compare")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for tag, ov in (("a", a.a), ("b", a.b)):
        p, s, cfg = replay(a.folder, a.model, ov, tag)
        m = metrics(p, cfg.map.voxel)
        print(f"[{tag}] {ov}\n    frames {s['frames']} map_frames {s['map_frames']} resets {s['map_resets']}  points {m['points']}  thickness {m['thick_med']:.2f} cm (p90 {m['thick_p90']:.2f})  "
              f"isolated {m['isolated']:.1f}%  spacing {m['spacing_cm']:.2f} cm  depth {s['depth_ms_median']} ms  frame {s['frame_ms_median']} ms")
        picture(p, f"{a.out}_{tag}.png", f"[{tag}]")


if __name__ == "__main__":
    main()
