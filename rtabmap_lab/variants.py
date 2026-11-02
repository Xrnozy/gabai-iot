"""The surface builders the surface lab compares. Each takes {node id: (depth_mm, bgr, optimised base pose)}, the calibration and the picture size and returns (points to measure, save(path))."""
import numpy as np
import open3d as o3d

from rtabmap_lab.gabai_depth import points_from_depth
from rtabmap_lab.surfel import SurfelFusion
from rtabmap_lab.tsdf import OPTICAL, TsdfMap


def raw_points(nodes, cal, size, stride=4):
    """Every node's points placed with its pose, no fusion: what the 'stereo points' view shows."""
    P, C = [], []
    for i, (d, bgr, T, *_) in nodes.items():
        p, c = points_from_depth(d, bgr, cal["f"], cal["f"], cal["cx"], cal["cy"], stride)
        Tw = T @ OPTICAL
        P.append(p @ Tw[:3, :3].T + Tw[:3, 3])
        C.append(c[:, ::-1])
    P, C = np.concatenate(P), np.concatenate(C)

    def save(path):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        pc.colors = o3d.utility.Vector3dVector(C / 255.0)
        o3d.io.write_point_cloud(path, pc)
    return P, save


def tsdf(voxel, trunc, max_range=4.0):
    def build(nodes, cal, size):
        t = TsdfMap((cal["f"], cal["f"], cal["cx"], cal["cy"]), size, voxel=voxel, trunc=trunc, max_range=max_range, threaded=False)
        for i, (d, bgr, T, *_) in nodes.items():
            t.add_node(i, d, bgr)
            t.set_poses({i: T})
        mesh = t.flush()
        v = np.asarray(mesh.vertices)
        t.close()
        return v, lambda path: o3d.io.write_triangle_mesh(path, mesh)
    return build


def tsdf_gated(voxel, trunc, max_range=4.0, min_views=2):
    """Surfels decide which depth pixels are believed (seen from at least `min_views` pictures, agreeing in depth and normal); only those go into the TSDF."""
    def build(nodes, cal, size):
        f = SurfelFusion((cal["f"], cal["f"], cal["cx"], cal["cy"]), cal["f"] * cal["B"], threaded=False, min_views=min_views, max_range=max_range)
        for i, (d, bgr, T, *_) in nodes.items():
            f.add_node(i, d, bgr)
            f.set_poses({i: T})
        f.flush()
        t = TsdfMap((cal["f"], cal["f"], cal["cx"], cal["cy"]), size, voxel=voxel, trunc=trunc, max_range=max_range, threaded=False)
        for i, (d, bgr, T, *_) in nodes.items():
            t.add_node(i, f.map.gate_depth(d, T @ OPTICAL), bgr)
            t.set_poses({i: T})
        mesh = t.flush()
        v = np.asarray(mesh.vertices)
        t.close()
        f.close()
        return v, lambda path: o3d.io.write_triangle_mesh(path, mesh)
    return build


def gpu_tsdf(voxel=0.02, trunc=0.08, max_range=4.0, min_cluster_area=0.04, **opts):
    """nvblox-style GPU TSDF (rtabmap_lab/gpu_tsdf.py). opts go to GpuVolume: sigma_px, k_trunc, max_weight, min_weight, power, edge_rel, min_cos, free_space, max_std."""
    def build(nodes, cal, size):
        from rtabmap_lab.gpu_tsdf import GpuTsdfMap
        t = GpuTsdfMap((cal["f"], cal["f"], cal["cx"], cal["cy"]), size, voxel=voxel, trunc=trunc, max_range=max_range, fb=cal["f"] * cal["B"], volume_opts=opts,
                       min_cluster_area=min_cluster_area, threaded=False)
        for i, (d, bgr, T, *_) in nodes.items():
            t.add_node(i, d, bgr)
            t.set_poses({i: T})
        mesh = t.flush()
        v = np.asarray(mesh.vertices)
        t.close()
        return v, lambda path: o3d.io.write_triangle_mesh(path, mesh)
    return build


_SRC = {}


def per_node_sources(nodes, cal):
    """Runs FFS, LAS2, monocular x2 and the fusion on every node's pair ONCE (the saved pictures, the same poses for every variant) and keeps the results: {node: (parts, fused)}."""
    key = id(nodes)
    if key not in _SRC:
        from rtabmap_lab.fused_source import FusedSource
        src = FusedSource()
        out = {}
        for i, (d, bgr, T, right) in nodes.items():
            rtab = d
            out[i] = src.process(bgr, np.repeat(right[:, :, None], 3, 2), cal, rtab)[::-1]
        _SRC[key] = out
        _SRC["cal"] = cal
    return _SRC[key]


def refused(names, **params):
    """Fuse again from the cached source maps with another source set / settings: no model runs. Returns a pick function."""
    from rtabmap_lab.depthfusion import FusionParams, StereoIn, fuse
    from rtabmap_lab.fused_source import SIGMA_PX

    def pick(parts, fused):
        f = fuse([StereoIn(parts[n], SIGMA_PX[n], n) for n in names], parts.get("mono_left"), parts.get("mono_right"), _SRC["cal"], FusionParams(**params))
        return f.z, f.cls
    return pick


def _mm(z):
    return (np.nan_to_num(z, nan=0.0) * 1000.0).astype(np.uint16)


def tsdf_from(pick, voxel=0.02, trunc=0.08, max_range=4.0):
    """TSDF of the depth chosen by pick(parts, fused) -> (depth in m, class map or None)."""
    def build(nodes, cal, size):
        srcs = per_node_sources(nodes, cal)
        t = TsdfMap((cal["f"], cal["f"], cal["cx"], cal["cy"]), size, voxel=voxel, trunc=trunc, max_range=max_range, threaded=False)
        for i, (d, bgr, T, *_) in nodes.items():
            z, cls = pick(*srcs[i])
            t.add_node(i, _mm(z), bgr, cls)
            t.set_poses({i: T})
        mesh = t.flush()
        v = np.asarray(mesh.vertices)
        t.close()
        return v, lambda path: o3d.io.write_triangle_mesh(path, mesh)
    return build


def surfel_from(pick, min_views=2, max_range=4.0):
    def build(nodes, cal, size):
        srcs = per_node_sources(nodes, cal)
        f = SurfelFusion((cal["f"], cal["f"], cal["cx"], cal["cy"]), cal["f"] * cal["B"], threaded=False, min_views=min_views, max_range=max_range)
        for i, (d, bgr, T, *_) in nodes.items():
            z, conf = pick(*srcs[i])
            f.add_node(i, _mm(z), bgr, conf)
            f.set_poses({i: T})
        P, N, C = f.flush()
        f.close()
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
        pc.colors = o3d.utility.Vector3dVector(np.clip(C[:, ::-1] / 255.0, 0, 1))
        return P, lambda path: o3d.io.write_point_cloud(path, pc)
    return build


pick_rtab = lambda parts, fused: (parts["rtabmap"], None)
pick_las2 = lambda parts, fused: (parts["las2"], None)
pick_ffs = lambda parts, fused: (parts["ffs"], None)
pick_fused = lambda parts, fused: (fused.z, None)
pick_fused_w = lambda parts, fused: (fused.z, fused.cls)
conf_fused = lambda parts, fused: (fused.z, fused.conf)
conf_none = lambda parts, fused: (parts["ffs"], None)


def surfel(min_views=2, poisson=False, stride=2, max_range=4.0):
    def build(nodes, cal, size):
        f = SurfelFusion((cal["f"], cal["f"], cal["cx"], cal["cy"]), cal["f"] * cal["B"], threaded=False, min_views=min_views, stride=stride, max_range=max_range)
        for i, (d, bgr, T, *_) in nodes.items():
            f.add_node(i, d, bgr)
            f.set_poses({i: T})
        P, N, C = f.flush()
        f.close()
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
        pc.normals = o3d.utility.Vector3dVector(N.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(np.clip(C[:, ::-1] / 255.0, 0, 1))
        if not poisson:
            return P, lambda path: o3d.io.write_point_cloud(path, pc)
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8)
        mesh.remove_vertices_by_mask(np.asarray(dens) < np.quantile(dens, 0.08))
        return np.asarray(mesh.vertices), lambda path: o3d.io.write_triangle_mesh(path, mesh)
    return build


VARIANTS_FUSION = {
    "tsdf RTAB-Map depth (baseline)": tsdf_from(pick_rtab),
    "tsdf LAS2": tsdf_from(pick_las2),
    "tsdf Fast-FoundationStereo": tsdf_from(pick_ffs),
    "tsdf fused, unweighted": tsdf_from(pick_fused),
    "tsdf fused, confidence-weighted": tsdf_from(pick_fused_w),
    "tsdf fused weighted, 3cm/12cm": tsdf_from(pick_fused_w, 0.03, 0.12),
    "surfels (>=2) RTAB-Map depth": surfel_from(conf_none_rtab := (lambda parts, fused: (parts["rtabmap"], None))),
    "surfels (>=2) Fast-FoundationStereo": surfel_from(conf_none),
    "surfels (>=2) fused + confidence": surfel_from(conf_fused),
}

VARIANTS_FUSION2 = {
    "fused [ffs, las2, rtabmap] (default)": tsdf_from(pick_fused_w),
    "fused [ffs, las2]": tsdf_from(refused(["ffs", "las2"])),
    "fused [ffs] + mono fill": tsdf_from(refused(["ffs"])),
    "fused [ffs, las2], k=2": tsdf_from(refused(["ffs", "las2"], k=2.0)),
    "fused [ffs, rtabmap]": tsdf_from(refused(["ffs", "rtabmap"])),
    "ffs alone, no mono": tsdf_from(pick_ffs),
}

VARIANTS_GPU = {
    "open3d tsdf 2cm / 8cm (baseline)": tsdf(0.02, 0.08),
    "gpu tsdf 2cm (nvblox-style, defaults)": gpu_tsdf(),
    "gpu, no free-space carving": gpu_tsdf(free_space=False),
    "gpu, no cluster clean-up": gpu_tsdf(min_cluster_area=0.0),
    "gpu, min weight 0.2 (nearly ungated)": gpu_tsdf(min_weight=0.2),
    "gpu, min weight 2": gpu_tsdf(min_weight=2.0),
    "gpu, weight 1/z^4 (inverse variance)": gpu_tsdf(power=4.0),
    "gpu, constant weight": gpu_tsdf(power=0.0),
    "gpu, fixed trunc 8cm (k=0)": gpu_tsdf(k_trunc=0.0),
    "gpu, no edge/grazing gate": gpu_tsdf(edge_rel=99.0, min_cos=0.0),
    "gpu, max weight 20": gpu_tsdf(max_weight=20.0),
    "gpu, min weight 0.1": gpu_tsdf(min_weight=0.1),
    "gpu, min weight 0.1, no grazing drop": gpu_tsdf(min_weight=0.1, min_cos=0.0),
    "gpu, min weight 0.05, no grazing drop, noise 0.25": gpu_tsdf(min_weight=0.05, min_cos=0.0, max_std=0.25),
    "gpu, min weight 0.1, no gates at all": gpu_tsdf(min_weight=0.1, min_cos=0.0, edge_rel=99.0, max_std=0.25),
    "gpuA base (min w 0.1, no grazing drop)": gpu_tsdf(min_weight=0.1, min_cos=0.0),
    "gpuA, no fade behind surface": gpu_tsdf(min_weight=0.1, min_cos=0.0, fade=False),
    "gpuA, constant weight": gpu_tsdf(min_weight=0.1, min_cos=0.0, power=0.0),
    "gpuA, constant weight, no fade": gpu_tsdf(min_weight=0.1, min_cos=0.0, power=0.0, fade=False),
    "gpuA, no fade, max weight 1000": gpu_tsdf(min_weight=0.1, min_cos=0.0, fade=False, max_weight=1000.0),
    "gpuA, no fade, no carving": gpu_tsdf(min_weight=0.1, min_cos=0.0, fade=False, free_space=False),
    "gpuA, no fade, trunc 4cm": gpu_tsdf(0.02, 0.04, min_weight=0.1, min_cos=0.0, fade=False, k_trunc=0.0),
    "gpuA, no fade, voxel 3cm/12cm": gpu_tsdf(0.03, 0.12, min_weight=0.1, min_cos=0.0, fade=False),
    "gpu, noise limit 0.15 m": gpu_tsdf(max_std=0.15),
    "gpu, noise limit 0.25 m": gpu_tsdf(max_std=0.25),
    "gpu, noise limit 0.25 m, min weight 0.5": gpu_tsdf(max_std=0.25, min_weight=0.5),
    "gpu, noise limit 0.25 m, 1/z^4, min weight 0.1": gpu_tsdf(max_std=0.25, power=4.0, min_weight=0.1),
}

VARIANTS = {
    "raw stereo points": raw_points,
    "tsdf 2cm / 8cm (current)": tsdf(0.02, 0.08),
    "tsdf 3cm / 12cm": tsdf(0.03, 0.12),
    "tsdf 4cm / 16cm": tsdf(0.04, 0.16),
    "tsdf 3cm / 12cm, range 2.5 m": tsdf(0.03, 0.12, 2.5),
    "surfels, seen by >= 1 (no gate)": surfel(1),
    "surfels, seen by >= 2": surfel(2),
    "surfels, seen by >= 3": surfel(3),
    "surfels >= 2, Poisson mesh": surfel(2, poisson=True),
}
