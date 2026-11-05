"""Compares surface reconstruction methods on IDENTICAL input (the same nodes: depth, picture, RTAB-Map pose) and scores each against pictures it did not see.

  py -m rtabmap_lab.tools.recon_compare data/recordings/<folder> --calib data/stereo_calib_charuco_45.npz [--clip 0 860] --out data/recon/<name> --view
  py -m rtabmap_lab.tools.recon_compare nodes.npz [--calib file.npz] [--out folder] [--only tsdf,planes,...] [--view]

The input is a recording (RTAB-Map runs on it once, exactly as the calibration lab runs it, gyro included; the result is kept as <out>/nodes.npz) or such a nodes file
(also written by `live.py --dump-nodes`; one without its own calibration needs --calib). --view opens a browser viewer of every method on the same input.
Every 5th node is HELD OUT: it is not fused, and each method's surface is rendered into its view and compared with what that picture measured, in disparity (where the noise is the same everywhere,
about 0.5 px). A method that only looks smooth but moved surfaces away from the measurements scores worse here; a method that keeps the noise scores worse too, because the noise of the fused
pictures is not the noise of the held-out one. 'walls' is the same score on the pixels the held-out picture's own plane detector put on a large plane.
"""
import argparse
import json
import os
import time

import numpy as np
import open3d as o3d

from rtabmap_lab import flatness, planar_recon
from rtabmap_lab.tsdf import OPTICAL


def from_recording(folder, calib, clip, out):
    """Runs RTAB-Map on a clip of a recording (the calibration lab's runner: same pictures, same settings, gyro) and saves every node's depth, picture and pose."""
    from rtabmap_lab.calib_lab import params as P
    from rtabmap_lab.calib_lab.recording import Recording
    from rtabmap_lab.calib_lab.runner import run_headless
    base = P.CalibParams.from_npz(calib)
    rec = Recording(folder).decide(base.rect())
    r = run_headless(rec, base, clip or (0, rec.n), port=5761)
    t = r.tsdf
    ids = [i for i in sorted(t.frames) if i in t.poses]
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "nodes.npz")
    np.savez_compressed(path, ids=np.array(ids), depth=np.stack([t.frames[i][0] for i in ids]), bgr=np.stack([t.frames[i][1] for i in ids]), poses=np.stack([t.poses[i] for i in ids]),
                        cal=json.dumps({"f": r.cal["f"], "B": r.cal["B"], "cx": r.cal["cx"], "cy": r.cal["cy"], "size": list(r.cal["size"])}),
                        source=json.dumps({"recording": rec.folder, "calib": os.path.abspath(calib), "clip": list(r.clip), "odometry": r.metrics.get("odometry")}))
    print(f"RTAB-Map run: {json.dumps(r.metrics.get('odometry'))}\n{len(ids)} nodes saved to {path}", flush=True)
    return path


def load(path, calib=None):
    z = np.load(path, allow_pickle=False)
    if "cal" in z.files:
        cal = json.loads(str(z["cal"]))
    else:
        from gabai.tools.calibrate_stereo import Rectifier
        r = Rectifier(calib)
        cal = {"f": r.f, "B": r.B, "cx": r.cx, "cy": r.cy, "size": list(r.size)}
    depth, bgr, poses = z["depth"], z["bgr"], z["poses"]                          # once: indexing z[...] decompresses the whole array every time
    nodes = {int(i): (depth[k], bgr[k], poses[k], None) for k, i in enumerate(z["ids"])}
    return nodes, cal, tuple(cal["size"])


# ---- the methods: each returns ("mesh", TriangleMesh) or ("points", (P, colours), spacing m) -------------------------------------------------------------------------------------------------
def m_points(nodes, cal, size):
    """Every node's depth as points placed with its pose, no fusion: the input itself, noise included."""
    from rtabmap_lab.gabai_depth import points_from_depth
    P, C = [], []
    for d, bgr, T, *_ in nodes.values():
        p, c = points_from_depth(d, bgr, cal["f"], cal["f"], cal["cx"], cal["cy"], 4)
        Tw = T @ OPTICAL
        P.append(p @ Tw[:3, :3].T + Tw[:3, 3])
        C.append(c[:, ::-1] / 255.0)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(P)))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate(C))
    return "points", pc.voxel_down_sample(0.01), 0.01


def m_tsdf(nodes, cal, size):
    from rtabmap_lab.gpu_tsdf import GpuTsdfMap
    t = GpuTsdfMap((cal["f"], cal["f"], cal["cx"], cal["cy"]), size, voxel=0.02, trunc=0.08, max_range=4.0, fb=cal["f"] * cal["B"], threaded=False)
    for i, (d, bgr, T, *_) in nodes.items():
        t.add_node(i, d, bgr)
        t.set_poses({i: T})
    m = t.flush()
    t.close()
    return "mesh", m


def m_surfel(nodes, cal, size):
    from rtabmap_lab.surfel import SurfelFusion
    f = SurfelFusion((cal["f"], cal["f"], cal["cx"], cal["cy"]), cal["f"] * cal["B"], threaded=False, min_views=2, stride=2, max_range=4.0)
    for i, (d, bgr, T, *_) in nodes.items():
        f.add_node(i, d, bgr)
        f.set_poses({i: T})
    P, N, C = f.flush()
    f.close()
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(np.clip(C[:, ::-1] / 255.0, 0, 1))
    return "points", pc, 0.01


def m_occupancy(nodes, cal, size, vox=0.03, min_views=3):
    """Occupied voxels: a 3 cm cell is surface when at least 3 pictures put a point in it (no free-space reasoning beyond that)."""
    from rtabmap_lab.variants import raw_points
    keys, cols = {}, {}
    for i, nd in nodes.items():
        P, _ = raw_points({i: nd}, cal, size, stride=2)
        k = np.unique(np.floor(P / vox).astype(np.int64), axis=0)
        for t in map(tuple, k):
            keys[t] = keys.get(t, 0) + 1
    K = np.array([k for k, c in keys.items() if c >= min_views], np.float64)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((K + 0.5) * vox))
    return "points", pc, vox


def m_poisson(nodes, cal, size):
    """Screened Poisson surface on the 1 cm raw cloud (normals from neighbours, oriented to the cameras); low-density parts cut."""
    _, pc, _ = m_points(nodes, cal, size)
    pc = pc.voxel_down_sample(0.015)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    cams = np.array([(T @ OPTICAL)[:3, 3] for _, _, T, *_ in nodes.values()])
    pc.orient_normals_towards_camera_location(cams.mean(0))
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9)
    dens = np.asarray(dens)
    m.remove_vertices_by_mask(dens < np.quantile(dens, 0.1))
    return "mesh", m


def m_tsdf_frameplanes(nodes, cal, size):
    """The current GPU TSDF, fed depth whose flat parts were replaced by each picture's own fitted planes (planar_recon.denoise)."""
    fB = cal["f"] * cal["B"]
    den = {i: (planar_recon.denoise(d, cal["f"], cal["cx"], cal["cy"], fB)[0], bgr, T, *r) for i, (d, bgr, T, *r) in nodes.items()}
    return m_tsdf(den, cal, size)


def _planar(**kw):
    def build(nodes, cal, size):
        r = planar_recon.reconstruct(nodes, cal, size, **kw)
        build.result = r
        return "mesh", (r.combined() if kw.get("residual", True) else r.mesh)
    return build


METHODS = {"points": m_points, "tsdf": m_tsdf, "tsdf+frameplanes": m_tsdf_frameplanes, "surfel": m_surfel, "occupancy": m_occupancy, "poisson": m_poisson,
           "planes": _planar(residual=False), "hybrid": _planar(), "hybrid+snap": _planar(snap_deg=4.0)}


# ---- scoring ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def render(kind, geom, spacing, T_wc, cal, size):
    """Depth (m, 0 = nothing) of the reconstruction seen from a camera."""
    w, h = size
    f, cx, cy = cal["f"], cal["cx"], cal["cy"]
    if kind == "mesh":
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(geom))
        K = o3d.core.Tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]], o3d.core.float64)
        rays = sc.create_rays_pinhole(K, o3d.core.Tensor(np.linalg.inv(T_wc)), w, h)
        t = sc.cast_rays(rays)["t_hit"].numpy()
        d = np.asarray(rays.numpy()[..., 3:])
        z = t * (d @ (T_wc[:3, :3] @ np.array([0, 0, 1.0])))                  # ray length -> depth along the optical axis
        return np.where(np.isfinite(z), z, 0.0)
    P = np.asarray(geom.points)
    Tc = np.linalg.inv(T_wc)
    X = P @ Tc[:3, :3].T + Tc[:3, 3]
    X = X[X[:, 2] > 0.1]
    u, v = f * X[:, 0] / X[:, 2] + cx, f * X[:, 1] / X[:, 2] + cy
    zb = np.full((h, w), np.inf)
    rad = np.ceil(f * spacing / X[:, 2] / 2).astype(int)                         # a splat as wide as the point spacing, so the surface is closed
    for r in np.unique(np.clip(rad, 0, 6)):
        s = np.clip(rad, 0, 6) == r
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                uu, vv = np.round(u[s]).astype(int) + du, np.round(v[s]).astype(int) + dv
                ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                np.minimum.at(zb, (vv[ok], uu[ok]), X[s][ok, 2])
    return np.where(np.isfinite(zb), zb, 0.0)


def score(kind, geom, spacing, held, cal, size):
    fB = cal["f"] * cal["B"]
    errs, wall, cov = [], [], []
    for d_mm, T in held:
        T_wc = T @ OPTICAL
        z = render(kind, geom, spacing, T_wc, cal, size)
        m = (d_mm > 0) & (d_mm <= 4000)
        both = m & (z > 0)
        cov.append(both.sum() / max(m.sum(), 1))
        e = np.abs(fB / (d_mm[both] / 1000.0) - fB / z[both])
        errs.append(e)
        pl = np.zeros_like(m)
        for _, _, piece, *_ in planar_recon.frame_planes(d_mm, cal["f"], cal["cx"], cal["cy"], fB, max_planes=4):
            pl |= np.kron(piece, np.ones((4, 4), bool))[: m.shape[0], : m.shape[1]]
        wall.append(np.abs(fB / (d_mm[both & pl] / 1000.0) - fB / z[both & pl]))
    e, wl = np.concatenate(errs), np.concatenate(wall)
    return {"coverage": float(np.mean(cov)), "err_px": float(np.median(e)), "err_p90_px": float(np.percentile(e, 90)), "walls_err_px": float(np.median(wl)) if len(wl) else float("nan")}


def geometry_stats(kind, geom):
    if kind == "mesh":
        m = geom
        V = np.asarray(m.vertices)
        tri_cl, n_tri, _ = m.cluster_connected_triangles()
        n_tri = np.asarray(n_tri)
        return {"vertices": len(V), "triangles": len(m.triangles), "pieces_50+": int((n_tri >= 50).sum()), "specks_<50": int((n_tri < 50).sum()),
                "MB": round((V.nbytes + np.asarray(m.triangles).nbytes) / 1e6, 1), **({"roughness_mm": round(flatness.roughness(V)["median_mm"], 1), "spiky": round(flatness.roughness(V)["spiky"], 2)} if len(V) > 500 else {})}
    P = np.asarray(geom.points)
    return {"points": len(P), "MB": round(P.nbytes / 1e6, 1), **({"roughness_mm": round(flatness.roughness(P)["median_mm"], 1), "spiky": round(flatness.roughness(P)["spiky"], 2)} if len(P) > 500 else {})}


def room_report(r):
    """Angles between the large planes and distances between parallel ones: what a tape measure would check."""
    P = sorted(r.planes, key=lambda p: -p.S0)[:8]
    lines = []
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            c = abs(P[a].n @ P[b].n)
            ang = np.degrees(np.arccos(min(1.0, c)))
            if ang > 60:
                lines.append(f"  planes {a}-{b}: {ang:.1f} deg apart (90 for a corner)")
            elif ang < 15:
                s = np.sign(P[a].n @ P[b].n)
                lines.append(f"  planes {a}-{b}: parallel within {ang:.1f} deg, {abs(P[a].d - s * P[b].d):.3f} m apart")
    return lines


# ---- the viewer ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def export_view(path, kind, geom):
    """One method for the viewer: u32 vertex count, u32 triangle count, float32 xyz, uint8 rgb (padded to 4 bytes), uint32 triangles (none for points)."""
    if kind == "mesh":
        V, T = np.asarray(geom.vertices, np.float32), np.asarray(geom.triangles, np.uint32)
        C = np.asarray(geom.vertex_colors) if geom.has_vertex_colors() else np.full((len(V), 3), 0.6)
    else:
        V, T = np.asarray(geom.points, np.float32), np.zeros((0, 3), np.uint32)
        C = np.asarray(geom.colors) if geom.has_colors() else np.full((len(V), 3), 0.6)
    col = (np.clip(C, 0, 1) * 255).astype(np.uint8).tobytes()
    col += b"\0" * (-len(col) % 4)
    with open(path, "wb") as fh:
        fh.write(np.array([len(V), len(T)], np.uint32).tobytes() + V.tobytes() + col + T.tobytes())


def serve(out, port):
    """Static viewer over the folder (three.js copied from the calibration lab, works offline)."""
    import functools
    import http.server
    import shutil
    import threading
    import webbrowser
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("three.min.js", "OrbitControls.js"):
        shutil.copy(os.path.join(here, "..", "calib_lab", "static", f), out)
    shutil.copy(os.path.join(here, "recon_view.html"), os.path.join(out, "index.html"))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(http.server.SimpleHTTPRequestHandler, directory=out))
    url = f"http://127.0.0.1:{port}/"
    print(f"viewer: {url}   (Ctrl+C to stop)", flush=True)
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="a recording folder, or a nodes .npz")
    ap.add_argument("--calib")
    ap.add_argument("--clip", type=int, nargs=2)
    ap.add_argument("--out")
    ap.add_argument("--only")
    ap.add_argument("--hold", type=int, default=5, help="every Nth node is held out for scoring")
    ap.add_argument("--view", action="store_true", help="open the comparison viewer when done (needs --out)")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args(argv)
    if a.view and not a.out:
        ap.error("--view needs --out")
    if os.path.isdir(a.input):
        if not (a.calib and a.out):
            ap.error("a recording needs --calib and --out")
        a.input = from_recording(a.input, a.calib, a.clip, a.out)
    nodes, cal, size = load(a.input, a.calib)
    ids = sorted(nodes)
    held_ids = set(ids[a.hold // 2:: a.hold])
    fuse = {i: nodes[i] for i in ids if i not in held_ids}
    held = [(nodes[i][0], nodes[i][2]) for i in sorted(held_ids)]
    print(f"{len(fuse)} nodes fused, {len(held)} held out for scoring\n", flush=True)
    rows, extra = {}, {}
    for name, build in METHODS.items():
        if a.only and name not in a.only.split(","):
            continue
        t0 = time.perf_counter()
        kind, geom, *sp = build(fuse, cal, size)
        dt = time.perf_counter() - t0
        s = score(kind, geom, sp[0] if sp else 0.0, held, cal, size)
        g = geometry_stats(kind, geom)
        rows[name] = {"seconds": round(dt, 1), **{k: round(v, 3) for k, v in s.items()}, **g}
        print(f"{name:12s} {json.dumps(rows[name])}", flush=True)
        r = getattr(build, "result", None)
        if r is not None:
            print(f"   {len(r.planes)} planes; timings {json.dumps({k: round(v, 2) for k, v in r.timings.items()})}; snapped {[(k, round(d, 1)) for k, d in r.snapped]}")
            print("\n".join(room_report(r)))
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            fn = name.replace("+", "_")
            (o3d.io.write_triangle_mesh if kind == "mesh" else o3d.io.write_point_cloud)(os.path.join(a.out, fn + ".ply"), geom)
            export_view(os.path.join(a.out, fn + ".bin"), kind, geom)
            rows[name].update(file=fn + ".bin", kind=kind)
            if r is not None and "planes" not in extra:
                extra["planes"] = [{"n": p.n.round(4).tolist(), "d": round(p.d, 4), "views": len(p.frames)} for p in r.planes]
                extra["boundaries"] = [b.round(4).tolist() for b in r.boundaries]
                extra["room"] = room_report(r)
    if a.out:
        extra["cameras"] = [{"id": i, "T": (nodes[i][2] @ OPTICAL).round(5).tolist(), "held_out": i in held_ids} for i in ids]
        with open(os.path.join(a.out, "scores.json"), "w") as fh:
            json.dump({"methods": rows, **extra}, fh)
        if a.view:
            serve(a.out, a.port)


if __name__ == "__main__":
    main()
