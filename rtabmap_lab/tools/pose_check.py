"""Are the camera poses good enough to fuse depth? Each node's depth is projected into the other nodes that look at the same place, and the depths are compared where the two surfaces meet.
If the poses were exact, the only difference would be depth noise; extra difference is pose error (or a wrong baseline / focal length). It is reported against the noise to expect from the depth itself.

  py -m rtabmap_lab.tools.pose_check nodes.npz          (a file written by  live.py --dump-nodes)
  py -m rtabmap_lab.tools.pose_check map.db             (a saved map; the depth is RTAB-Map's own then)
"""
import argparse
import os

import numpy as np

from rtabmap_lab.tsdf import OPTICAL


SIGMA_PX = 0.5


def consistency(nodes, cal, max_pairs=300, stride=8, max_dist=0.5):
    """nodes: {id: (depth_mm, pose 4x4 base)}. Returns dict with the median relative depth difference of pairs of overlapping nodes, the depth noise expected, and per-pair values."""
    ids = sorted(nodes)
    f, cx, cy = cal["f"], cal["cx"], cal["cy"]
    fB = cal["f"] * cal["B"]
    pts = {}
    for i in ids:
        d, T = nodes[i]
        v, u = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
        z = d[v, u].astype(np.float32) / 1000.0
        ok = z > 0
        P = np.stack([(u[ok] - cx) * z[ok] / f, (v[ok] - cy) * z[ok] / f, z[ok]], 1)
        Tw = T @ OPTICAL
        pts[i] = (P @ Tw[:3, :3].T + Tw[:3, 3], Tw)
    pairs = [(a, b) for k, a in enumerate(ids) for b in ids[k + 1:] if np.linalg.norm(pts[a][1][:3, 3] - pts[b][1][:3, 3]) < max_dist]
    rng = np.random.default_rng(0)
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    rel, noise, per = [], [], []
    for a, b in pairs:
        db_, Tb = nodes[b][0], pts[b][1]
        Pc = (pts[a][0] - Tb[:3, 3]) @ Tb[:3, :3]                          # node a's points in node b's camera
        z = Pc[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            u, v = np.rint(f * Pc[:, 0] / z + cx).astype(int), np.rint(f * Pc[:, 1] / z + cy).astype(int)
        h, w = db_.shape
        ok = (z > 0.3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        zb = np.where(ok, db_[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)], 0).astype(np.float32) / 1000.0
        ok &= zb > 0
        if ok.sum() < 300:
            continue
        dz = np.abs(z[ok] - zb[ok])
        near = dz < 0.15 * z[ok]                                           # the same surface, not a different object behind or in front
        if near.sum() < 200:
            continue
        r = np.median(dz[near] / z[ok][near])
        rel.append(r)
        noise.append(0.6745 * np.median(np.sqrt(2) * z[ok][near] * SIGMA_PX / fB))              # median |difference| of two pictures, each with SIGMA_PX of disparity noise (0.5 px measured, tools/stereo_stages.py), relative to the depth
        per.append((a, b, float(r)))
    return {"pairs": len(rel), "median_rel": float(np.median(rel)) if rel else np.nan, "p90_rel": float(np.percentile(rel, 90)) if rel else np.nan,
            "expected_rel_from_depth_noise": float(np.median(noise)) if noise else np.nan, "per_pair": per}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--calib", help="the calibration the nodes were made with (default: source.calib of config/pipeline.toml, which is the WRONG one for a run made with another file)")
    a = ap.parse_args(argv)
    from gabai.pipeline.config import PipelineConfig
    from gabai.tools.calibrate_stereo import Rectifier
    rect = Rectifier(a.calib or PipelineConfig.load("config/pipeline.toml").source.calib)
    cal = {"f": rect.f, "B": rect.B, "cx": rect.cx, "cy": rect.cy}
    if a.path.endswith(".npz"):
        z = np.load(a.path)
        nodes = {int(i): (z["depth"][k], z["poses"][k]) for k, i in enumerate(z["ids"])}
    else:
        from rtabmap_lab.tools.surface_lab import load_nodes
        nodes = {i: (d, T) for i, (d, b, T, r) in load_nodes(a.path, cal, rect.size).items()}
    r = consistency(nodes, cal)
    print(f"{len(nodes)} nodes, {r['pairs']} overlapping pairs checked")
    print(f"depth difference between overlapping nodes: median {r['median_rel'] * 100:.1f}% of the depth, 90th percentile {r['p90_rel'] * 100:.1f}%")
    print(f"expected from depth noise alone: about {r['expected_rel_from_depth_noise'] * 100:.1f}%  ->  the poses (and calibration) add {max(r['median_rel'] - r['expected_rel_from_depth_noise'], 0) * 100:.1f} points")


if __name__ == "__main__":
    main()
