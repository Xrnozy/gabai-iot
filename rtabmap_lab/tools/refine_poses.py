"""Moves each node's pose so its depth lies on the surface of the nodes before it (TSDF registration, gpu_tsdf.GpuVolume.refine) and writes a new nodes file. Then measure it:

  py -m rtabmap_lab.tools.refine_poses nodes.npz refined.npz --calib data/stereo_calib_charuco_45.npz
  py -m rtabmap_lab.tools.pose_check nodes.npz          the depth disagreement between overlapping nodes before ...
  py -m rtabmap_lab.tools.pose_check refined.npz        ... and after: the only number here that is a direct measure of the poses, not of how a mesh looks

nodes.npz is what live.py --dump-nodes writes. The nodes are visited in order, as a live run would see them.
"""
import argparse
import collections

import numpy as np

from rtabmap_lab.gpu_tsdf import GpuVolume
from rtabmap_lab.tsdf import OPTICAL


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nodes")
    ap.add_argument("out")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--sigma-px", type=float, default=0.5)
    ap.add_argument("--max-t", type=float, default=0.06, help="largest move of one node, metres")
    ap.add_argument("--max-deg", type=float, default=3.0)
    ap.add_argument("--keep-worse", action="store_true", help="accept a move even when the robust cost does not drop (diagnosis)")
    a = ap.parse_args(argv)
    from gabai.tools.calibrate_stereo import Rectifier
    r = Rectifier(a.calib)
    w, h = r.size
    z = np.load(a.nodes, allow_pickle=True)
    ids, depth, bgr, poses = z["ids"], z["depth"], z["bgr"], np.array(z["poses"], float)
    vol = GpuVolume(0.02, 0.08, 4.0, r.f, r.f, r.cx, r.cy, (w, h), r.f * r.B, sigma_px=a.sigma_px, refine=True)
    inv = np.linalg.inv(OPTICAL)
    why, moved = collections.Counter(), []
    new = poses.copy()
    for k in range(len(ids)):
        Tc = poses[k] @ OPTICAL
        d = depth[k].astype(np.float32) / 1000.0
        if vol.n_in:
            n0 = len(vol.corrections)
            T2 = vol.refine(d, Tc, max_t=a.max_t, max_deg=a.max_deg)
            c = vol.corrections[n0]
            if c is None:
                why["too few points on the volume"] += 1
            elif c[4]:
                why["moved"] += 1
                moved.append(c[:2])
                Tc = T2
            elif c[0] / 1000.0 > a.max_t or c[1] > a.max_deg:
                why["move too large, refused"] += 1
            else:
                why["cost did not drop, refused"] += 1
        vol.integrate(d, np.ascontiguousarray(bgr[k][:, :, ::-1]), Tc, None)
        new[k] = Tc @ inv
    np.savez(a.out, ids=ids, depth=depth, bgr=bgr, poses=new, cls=z["cls"] if "cls" in z else np.zeros(len(ids), np.uint8))
    print(f"{len(ids)} nodes: " + ", ".join(f"{v} {k}" for k, v in why.items()))
    if moved:
        m = np.array(moved)
        print(f"moves: median {np.median(m[:, 0]):.1f} mm / {np.median(m[:, 1]):.2f} deg, 90th percentile {np.percentile(m[:, 0], 90):.1f} mm / {np.percentile(m[:, 1], 90):.2f} deg")
    print("written", a.out)


if __name__ == "__main__":
    main()
