"""Numbers instead of a picture: runs the bridge on a recording and reports whether the odometry stayed sane and how clean the depth was.

  py -m rtabmap_lab.tools.diagnose data/recordings/vio_20261101_014854 [--skip 3] [--frames 200] [--orientation flip] [--max-range 3] [--max-depth-std 0.08]

--skip N uses every Nth pair only, which is what a busy bridge does live (it skips pairs it has no time for): bigger steps between frames are the harder case.
"""
import argparse
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np

from rtabmap_lab.bridge_client import EXE, Bridge, LAB


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--orientation", default="as-is", choices=["as-is", "flip"])
    ap.add_argument("--max-range", type=float, default=6.0)
    ap.add_argument("--max-depth-std", type=float, default=0.0)
    ap.add_argument("--port", type=int, default=5741)
    ap.add_argument("--dump", help="save the left picture next to RTAB-Map's disparity of the 6th disparity message to this .png")
    ap.add_argument("--arg", nargs=2, action="append", default=[], metavar=("Group/Name", "value"), help="an RTAB-Map parameter override, to compare settings, e.g. --arg Vis/MinInliers 12")
    a = ap.parse_args(argv)
    from rtabmap_lab.live import frames
    cfg, source = frames(SimpleNamespace(source="replay", folder=a.folder, port=5600, pipeline_config="config/pipeline.toml"))
    b = Bridge(db="", port=a.port, config=os.path.join(LAB, "config", "rtabmap_stereo.ini"), extra=[x for k, v in a.arg for x in ("--" + k.lstrip("-"), v)])
    st, nodes, depths, disps, lefts = [], [], [], [], []
    t0 = time.time()
    try:
        for i, fr in enumerate(source):
            if i % a.skip:
                continue
            left, right = fr.left, fr.right
            h, w = left.shape[:2]
            c = dict(fr.cal)
            if a.orientation == "flip":
                left, right = cv2.rotate(right, cv2.ROTATE_180), cv2.rotate(left, cv2.ROTATE_180)
                c["cx"], c["cy"] = (w - 1) - c["cx"], (h - 1) - c["cy"]
            if not st and not b.sent:
                b.init(c["f"], c["f"], c["cx"], c["cy"], c["B"], w, h, 4, 0.3, a.max_range, 3, 15, a.max_depth_std)
            b.wait_idle()
            b.send_frame(fr.idx, fr.t, left, cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
            b.wait_idle()
            for m in b.poll():
                if m["type"] == "disparity":
                    disps.append(m["image"])
                    lefts.append(left.copy())
                {"status": st, "node": nodes, "depth": depths}.get(m["type"], []).append(m)
            if b.sent >= a.frames:
                break
    finally:
        b.close(save=False)
    P = np.array([(s["map_correction"] @ s["odom"])[:3, 3] for s in st if s["odom"] is not None])
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    dt = np.diff([s["stamp"] for s in st if s["odom"] is not None])
    print(f"{len(st)} frames in {time.time() - t0:.0f} s, lost {sum(s['lost'] for s in st)}, restarts {st[-1]['restarts']}, nodes {st[-1]['nodes']}, loop closures {st[-1]['loops']}")
    print(f"travelled (odometry says) {st[-1]['distance']:.2f} m; path extent {np.round(P.max(0) - P.min(0), 2)} m; largest position step {step.max():.2f} m; speed p50/p95/max {np.percentile(step / np.maximum(dt, 1e-3), [50, 95, 100]).round(2)} m/s")
    if a.dump and len(disps) > 5:
        d = disps[5]
        col = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
        col[d == 0] = (40, 40, 40)
        cv2.imwrite(a.dump, np.hstack([cv2.resize(lefts[5], (d.shape[1], d.shape[0])), col]))
    ms = lambda k: np.array([x[k] for x in st])
    tot = ms("odom_ms") + ms("rtabmap_ms") + ms("disparity_ms")
    print(f"per frame ms (mean / p95 / max): odometry {ms('odom_ms').mean():.0f}/{np.percentile(ms('odom_ms'), 95):.0f}/{ms('odom_ms').max():.0f}, map update {ms('rtabmap_ms').mean():.0f}/{np.percentile(ms('rtabmap_ms'), 95):.0f}/{ms('rtabmap_ms').max():.0f}, "
          f"disparity {ms('disparity_ms').mean():.0f}/{np.percentile(ms('disparity_ms'), 95):.0f}/{ms('disparity_ms').max():.0f}; total {tot.mean():.0f}/{np.percentile(tot, 95):.0f}/{tot.max():.0f} -> a pair every {tot.mean():.0f} ms is the most this loop can take")
    if depths:
        d = np.concatenate([x["depth_mm"][x["depth_mm"] > 0] for x in depths]) / 1000.0
        cover = np.mean([(x["depth_mm"] > 0).mean() for x in depths])
        print(f"depth: {cover * 100:.1f}% of pixels valid, median {np.median(d):.2f} m, p90 {np.percentile(d, 90):.2f} m, share beyond 3 m {np.mean(d > 3) * 100:.0f}%")


if __name__ == "__main__":
    main()
