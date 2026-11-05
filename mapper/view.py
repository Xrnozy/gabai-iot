"""Shows a mapper run in Rerun: the semi-dense cloud, the camera path, and the camera at each node.

  py -m mapper.view data/mapper/walk [--point-size 0.004] [--min-conf 0.0] [--min-views 2] [--save out.rrd]

Point size is a viewer setting (metres, a Rerun radius): the stored cloud has no size. Colours: RGB from the pictures; --by-conf colours by depth confidence instead, --by-views by how many nodes saw the point.
"""
import argparse
import os

import numpy as np
import rerun as rr


def read_ply(path):
    with open(path, "rb") as f:
        n = int(next(l for l in iter(f.readline, b"") if l.startswith(b"element vertex")).split()[2])
        f.seek(0)
        while f.readline().strip() != b"end_header":
            pass
        rec = np.frombuffer(f.read(), np.dtype([("p", "<f4", 3), ("c", "u1", 3), ("q", "<f4"), ("v", "u1")]), n)
    return rec["p"], rec["c"], rec["q"], rec["v"]


def colormap(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
    return np.stack([255 * t, 255 * (1 - np.abs(2 * t - 1)), 255 * (1 - t)], 1).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--point-size", type=float, default=0.004, help="point radius in metres (viewer only)")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--by-conf", action="store_true")
    ap.add_argument("--by-views", action="store_true")
    ap.add_argument("--save", help="write a .rrd file instead of opening the viewer")
    a = ap.parse_args(argv)
    P, C, q, v = read_ply(os.path.join(a.folder, "cloud.ply"))
    keep = (q >= a.min_conf) & (v >= a.min_views)
    P, C, q, v = P[keep], C[keep], q[keep], v[keep]
    if a.by_conf:
        C = colormap(q, 0.5, 1.0)
    elif a.by_views:
        C = colormap(v.astype(float), 2, 6)
    tr = np.loadtxt(os.path.join(a.folder, "trajectory.txt"))
    rr.init("mapper", spawn=not a.save)
    if a.save:
        rr.save(a.save)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)       # RTAB-Map's world is z up; the optical camera poses below carry the rest
    rr.log("world/cloud", rr.Points3D(P, colors=C, radii=a.point_size), static=True)
    rr.log("world/path", rr.LineStrips3D([tr[:, 1:4]], colors=[(255, 200, 0)], radii=0.01), static=True)
    for k, row in enumerate(tr):
        rr.set_time("node", sequence=k)
        rr.log("world/camera", rr.Transform3D(translation=row[1:4], rotation=rr.Quaternion(xyzw=row[4:8])))
    print(f"{len(P)} points (confidence >= {a.min_conf}, views >= {a.min_views}), {len(tr)} camera poses, point radius {a.point_size} m")


if __name__ == "__main__":
    main()
