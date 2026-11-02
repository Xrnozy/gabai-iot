"""How much of a mesh is debris? Reads .ply meshes (surface_lab --out) and reports area, number of separate pieces, and the share of the area in small pieces (floating rubble)
and in the biggest piece. Also the share of triangles that are strongly stretched or have a grazing normal against the view axis is not computed here: it needs the camera.

  py -m rtabmap_lab.tools.mesh_quality out_folder_or_files...
"""
import glob
import os
import sys

import numpy as np
import open3d as o3d


def quality(mesh, small_m2=0.05):
    mesh.remove_degenerate_triangles()
    if not len(mesh.triangles):
        return dict(area=0.0, pieces=0, small_share=0.0, largest_share=0.0, triangles=0)
    cl, ntri, area = mesh.cluster_connected_triangles()
    area = np.asarray(area)
    tot = float(area.sum())
    return dict(area=tot, pieces=len(area), small_share=float(area[area < small_m2].sum() / tot), largest_share=float(area.max() / tot), triangles=len(mesh.triangles),
                small_pieces=int((area < small_m2).sum()))


def main(argv=None):
    paths = []
    for a in (argv or sys.argv[1:]):
        paths += sorted(glob.glob(os.path.join(a, "*.ply"))) if os.path.isdir(a) else [a]
    print(f"{'mesh':44s} {'area m2':>8s} {'pieces':>7s} {'<5cm2 pcs':>9s} {'debris area':>11s} {'largest':>8s}")
    for p in paths:
        m = o3d.io.read_triangle_mesh(p)
        q = quality(m)
        print(f"{os.path.basename(p)[:44]:44s} {q['area']:8.2f} {q['pieces']:7d} {q.get('small_pieces', 0):9d} {q['small_share'] * 100:10.1f}% {q['largest_share'] * 100:7.1f}%")


if __name__ == "__main__":
    main()
