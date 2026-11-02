"""Rebuilds the surface of a SAVED map with different settings and measures how flat the walls come out: no camera, no Pi, a few seconds per variant.

  py -m rtabmap_lab.tools.surface_lab path/to/map.db [--orientation as-is|flip] [--out DIR]

The bridge loads the database and sends every node's depth, colour and optimised pose (the same messages a live run gets); each variant fuses them its own way, and flatness.py measures the result.
"""
import argparse
import os
import time
from types import SimpleNamespace

import numpy as np

from rtabmap_lab import flatness
from rtabmap_lab.bridge_client import Bridge, LAB


def load_nodes(db, cal, size, port=5751, max_depth_std=0.08, max_range=4.0):
    """{node id: (depth_mm, bgr, optimised base pose, right gray)} from a saved map."""
    b = Bridge(db=db, port=port, localize=True, config=os.path.join(LAB, "config", "rtabmap_stereo.ini"))
    nodes, poses = {}, {}
    try:
        b.init(cal["f"], cal["f"], cal["cx"], cal["cy"], cal["B"], size[0], size[1], 4, 0.3, max_range, 0, 15, max_depth_std)
        end = time.time() + 120
        last = time.time()
        while time.time() < end and (time.time() - last < 3.0 or not b.ready):
            for m in b.poll():
                last = time.time()
                if m["type"] == "depth":
                    nodes[m["id"]] = (m["depth_mm"], m["bgr"], m["right"])
                elif m["type"] == "graph":
                    poses.update({i: T for i, T in m["poses"].items() if T is not None})
            time.sleep(0.05)
    finally:
        b.close(save=False)
    return {i: (nodes[i][0], nodes[i][1], poses[i], nodes[i][2]) for i in nodes if i in poses}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", help="a saved map (.db), or with --npz a file written by live.py --dump-nodes")
    ap.add_argument("--npz", action="store_true")
    ap.add_argument("--orientation", default="as-is", choices=["as-is", "flip"])
    ap.add_argument("--out", help="write each variant's mesh / points here as .ply")
    ap.add_argument("--gpu", action="store_true", help="compare the nvblox-style GPU TSDF (rtabmap_lab/gpu_tsdf.py) and its parts with the Open3D baseline")
    ap.add_argument("--calib", help="stereo calibration the nodes were made with (default: source.calib of config/pipeline.toml); a dumped run is only right with its own calibration")
    ap.add_argument("--only", help="comma-separated variant names")
    ap.add_argument("--fusion", type=int, default=0, nargs="?", const=1, help="compare the depth sources and the confidence-aware fusion (loads FFS, LAS2 and Depth Anything; same pictures and poses for every row)")
    a = ap.parse_args(argv)
    from gabai.pipeline.config import PipelineConfig
    from gabai.tools.calibrate_stereo import Rectifier
    cfg = PipelineConfig.load("config/pipeline.toml")
    rect = Rectifier(a.calib or cfg.source.calib)
    w, h = rect.size
    cal = {"f": rect.f, "B": rect.B, "cx": rect.cx, "cy": rect.cy}
    if a.orientation == "flip":
        cal["cx"], cal["cy"] = (w - 1) - cal["cx"], (h - 1) - cal["cy"]
    if a.npz:
        z = np.load(a.db)
        nodes = {int(i): (z["depth"][k], z["bgr"][k], z["poses"][k], None) for k, i in enumerate(z["ids"])}
    else:
        nodes = load_nodes(a.db, cal, (w, h))
    print(f"{len(nodes)} nodes loaded from {a.db}\n", flush=True)
    from rtabmap_lab.variants import VARIANTS, VARIANTS_FUSION, VARIANTS_FUSION2, VARIANTS_GPU
    for name, build in (VARIANTS_GPU if a.gpu else VARIANTS_FUSION2 if a.fusion == 2 else VARIANTS_FUSION if a.fusion else VARIANTS).items():
        if a.only and name not in a.only.split(";"):
            continue
        t0 = time.time()
        pts, save = build(nodes, cal, (w, h))
        r = flatness.report(pts)
        print(flatness.line(name, r) + f"   [{time.time() - t0:.0f} s]", flush=True)
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            save(os.path.join(a.out, "".join(ch if ch.isalnum() else "_" for ch in name) + ".ply"))


if __name__ == "__main__":
    main()
