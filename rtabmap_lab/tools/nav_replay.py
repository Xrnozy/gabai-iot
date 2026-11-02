"""Run the egocentric navigation map (rtabmap_lab/navmap.py) over the nodes of a run (live.py --dump-nodes) and report its speed and what it found.

  py -m rtabmap_lab.tools.nav_replay nodes.npz --calib data/stereo_calib_new.npz --out nav.png
"""
import argparse
import time

import cv2
import numpy as np

from rtabmap_lab.navmap import NavMap
from rtabmap_lab.tsdf import OPTICAL


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--calib", help="the stereo calibration the nodes were made with (default: source.calib of config/pipeline.toml)")
    ap.add_argument("--out", help="write the final top-down picture here (.png)")
    ap.add_argument("--cell", type=float, default=0.05)
    a = ap.parse_args(argv)
    from gabai.pipeline.config import PipelineConfig
    from gabai.tools.calibrate_stereo import Rectifier
    rect = Rectifier(a.calib or PipelineConfig.load("config/pipeline.toml").source.calib)
    z = dict(np.load(a.npz))                                          # read every array once: indexing the npz file decompresses the whole array each time
    nm = NavMap(cell_m=a.cell)
    ms, res = [], None
    for k, i in enumerate(z["ids"]):
        T = z["poses"][k] @ OPTICAL
        t0 = time.perf_counter()
        res = nm.update(k * 0.5, z["depth"][k].astype(np.float32) / 1000.0, rect.f, rect.cx, rect.cy, T)
        ms.append((time.perf_counter() - t0) * 1000.0)
    print(f"{len(ms)} nodes: {np.median(ms):.1f} ms median, {np.max(ms):.1f} ms worst per update (grid {nm.grid.n}x{nm.grid.n} cells of {a.cell * 100:g} cm)")
    print(f"floor found in {nm.stats['floor_ok']} pictures, missing in {nm.stats['floor_missing']}")
    if res and res.get("ready"):
        st = res["state"]
        print(f"cells: free {(st == 1).sum()}, occupied {(st == 2).sum()}, unknown {(st == 0).sum()}; path {len(res['path_cells'])} cells, unseen part {res['unseen_path_m']:.2f} m, "
              f"nearest obstacle ahead {res['nearest_obstacle_ahead_m']}, steer {res['steer_deg']}")
        if a.out:
            cv2.imwrite(a.out, nm.image(res))
            print("picture:", a.out)


if __name__ == "__main__":
    main()
