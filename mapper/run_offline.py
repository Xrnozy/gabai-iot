"""M3 offline run: RTAB-Map (odometry, loop closure, graph) + retinify (default), LAS2 or TensorRT FoundationStereo depth with confidence -> persistent semi-dense cloud.

  py -m mapper.run_offline data/recordings/walk_20261102_230115 --calib data/stereo_calib_charuco_45.npz --out data/mapper/walk [--clip 0 860] [--shutter] [--gyro] [--compare]

Per RTAB-Map node: our depth + confidence for that pair (not RTAB-Map's own SGBM), stored once in the node's camera frame. Every graph update re-poses the nodes; the cloud is the nodes' points
under the newest poses. Outputs in --out: cloud.ply (x y z rgb confidence views), trajectory.txt (TUM: time x y z qx qy qz qw, the optimised node poses), nodes.npz (our depth in mm, picture, pose per node:
the input format of rtabmap_lab.tools.recon_compare). --compare scores the cloud and the TSDF / surfel / raw-point methods on those same nodes against every 5th node held out.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from mapper.cloud import OPTICAL, PointMap
from mapper.depth_trt import StereoTrt, paths
NEW_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "mapper_bridge.exe")
from rtabmap_lab import live as livemod
from rtabmap_lab.bridge_client import Bridge
from mapper.align import camera_up_prior
from gabai.relay.results import SlamGuide
from mapper.dynamic import PersonTracker, static_conf

FUSION_CONF_MIN = 0.5
from mapper.navigate import Navigator
from rtabmap_lab.calib_lab import params as P
from rtabmap_lab.calib_lab.recording import Recording


def quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()                       # x y z w


class Prof:
    """Where the main loop's time goes (--profile): seconds per named span, printed with the progress lines."""
    t, n = {}, {}

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *a):
        Prof.t[self.name] = Prof.t.get(self.name, 0.0) + time.perf_counter() - self.t0
        Prof.n[self.name] = Prof.n.get(self.name, 0) + 1

    @staticmethod
    def report(total):
        rows = sorted(Prof.t.items(), key=lambda kv: -kv[1])
        return "  profile (s, % of loop, calls, ms/call): " + "; ".join(f"{k} {v:.1f} {100 * v / total:.0f}% {Prof.n[k]} {1000 * v / Prof.n[k]:.0f}" for k, v in rows if v > 0.02 * total)


def free_port(port):
    """The first free local TCP port from `port` up: a bridge left over from an earlier run (or a crashed one) must not make the new one fail with 'could not connect to the bridge'."""
    import socket
    for p in range(port, port + 50):
        with socket.socket() as sk:
            try:
                sk.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return port


def run(rec, rect, clip, depth_model, shutter, conf_min=0.5, port=5710, min_views=2, view=None, tsdf=None, gyro=False, reset_countdown=3, min_inliers=0, min_sharp=0.0, fusion=None, segmenter=None, profile=False, sync_depth=False, relay=None, lean=True):
    ns = livemod.add_args(__import__("argparse").ArgumentParser()).parse_args([])
    if free_port(port) != port:
        print(f"bridge port {port} is taken (a bridge of an earlier run is still alive?): using {free_port(port)}", flush=True)
    port = free_port(port)
    cal = rect.camera(flip=rec.flip, k=rec.k)
    w, h = cal["size"]
    pm = PointMap((cal["f"], cal["f"], cal["cx"], cal["cy"]), min_views=min_views)
    # RTAB-Map's own recovery (Odom/ResetCountdown): after N lost pairs it restarts the odometry with a LARGE covariance, which makes RTAB-Map open a new map session; a loop closure joins the sessions
    # when a place is seen again. The lab bridge's manual restart instead gave the graph a confident link across the blind gap, which is a made-up pose.
    if reset_countdown:
        b = Bridge(db="", port=port, config=ns.config, exe=NEW_EXE, extra=("--Odom/ResetCountdown", str(reset_countdown)) + (("--lean", "1") if lean else ()))
    else:
        b = Bridge(db="", port=port, config=ns.config)
    __import__("atexit").register(lambda: b.proc.poll() is None and b.proc.kill())                 # never leave the bridge behind when this process dies
    b.init(cal["f"], cal["f"], cal["cx"], cal["cy"], cal["B"], w, h, ns.cloud_stride, ns.min_range, ns.max_range, ns.disparity_every, 10 ** 6 if reset_countdown else ns.lost_reset, ns.max_depth_std)
    nodes, poses, node_frame, ms, st = {}, {}, {}, [], {"lost": 0, "loops": 0, "restarts": 0}
    navigator, navst = Navigator(cal["f"], cal["cx"], cal["cy"]), {"ms": [], "cues": {}}
    guide = SlamGuide() if relay is not None else None
    tracker = PersonTracker(cal["f"], cal["cx"], cal["cy"])
    depth_pool = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(max_workers=1)     # one GPU job at a time, in pair order
    t_arr = {}
    dfut, dlat = {}, []                                                                                                    # pair index -> future of (depth result, ms from the pair's arrival to the depth being ready)

    def depth_job(idx, left, right, t_in):
        r = depth_model(np.ascontiguousarray(left[:, :, ::-1]), np.ascontiguousarray(right[:, :, ::-1]))
        if view is not None and idx % 2 == 0:                                       # live depth picture straight from the pair, not from the node RTAB-Map makes of it later
            view.depth(r, idx, frame=idx)
        return r, (time.perf_counter() - t_in) * 1e3
    extra = {}                                                     # node -> (confidence float16, class grid) for the semantic surfel map

    cur = {"left": None}
    bms = []                                                        # the bridge's own timings per pair: odometry, RTAB-Map map update, disparity
    inl = {}                                                        # frame -> odometry inliers: the odometry's own quality for that pose
    st_skip = {"quality": 0}

    def handle(m):
        t = m["type"]
        if view is not None:
            with Prof("view.message"):
                view.message(m, cur["left"])
        if t == "status":
            inl[m["id"]] = m["inliers"]
            bms.append((m["odom_ms"], m["rtabmap_ms"], m["disparity_ms"]))
            st["lost"] += int(m["lost"]); st["loops"], st["restarts"] = m["loops"], m["restarts"]
        elif t == "graph":
            with Prof("set_poses"):
                poses.update(m["poses"])
                pm.set_poses(m["poses"])
                if tsdf is not None:
                    tsdf.set_poses(m["poses"])
                if fusion is not None:
                    fusion.set_poses(m["poses"])
        elif t == "depth" and m["frame_id"] >= 0:
            fid = m["frame_id"]
            try:
                left, right = rec.prepare(fid, rect, shutter="default" if shutter else None)
            except KeyError:                                                     # live: the pair is older than the ring of kept pairs
                return
            # a pose the odometry itself is unsure of (few inliers) or a blurred picture (Laplacian variance) does not become map geometry; the node stays in the graph
            if inl.get(fid, 10 ** 9) < min_inliers or cv2.Laplacian(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < min_sharp:
                st_skip["quality"] += 1
                return
            fut = dfut.pop(fid, None) or depth_pool.submit(depth_job, fid, left, right, t_arr.get(fid, time.perf_counter()))
            with Prof("depth wait"):                                                # the depth of this pair has been computing since it arrived, alongside RTAB-Map
                r, delay_ms = fut.result()
            ms.append(r.ms)
            dlat.append(delay_ms)
            T_ = poses.get(m["id"], poses[max(poses)] if poses else None)
            sg, dets, people = None, [], []
            if segmenter is not None:                                            # one YOLO run: class grid for the semantic map AND the people, who are tracked in the video and kept out of the static maps
                try:
                    with Prof("segment"):
                        sg, dets = segmenter.analyse(left)
                except Exception as e:
                    print(f"segmentation of node {m['id']} failed: {type(e).__name__}: {e}", flush=True)
                if T_ is not None:
                    people = tracker.update(dets, r.depth, r.conf, T_, float(rec.t[fid]))
                    if view is not None:
                        view.people(people)
            conf_s = static_conf(r.conf, dets)                                   # the same confidence with every person pixel at 0: only this goes into the cloud, TSDF and surfels
            if T_ is not None:                                                   # walking guidance from this node's depth and pose (mapper/navigate.py), a few ms
                if len(nodes) % 10 == 0 or "up" not in navst:
                    navst["up"] = camera_up_prior(poses)
                with Prof("navigate"):
                    res = navigator.update(r.depth, r.conf, T_, navst["up"])     # unmasked: someone standing in the way NOW is an obstacle now
                    res["person_hazards"] = tracker.hazards(people)
                if relay is not None:                                            # the phone's Navigation screen speaks this (gabai/relay/results.py)
                    relay.publish(guide.result(fid, time.time(), res, T_))
                navst["ms"].append(res["ms"]); navst["cues"][res["cue"]] = navst["cues"].get(res["cue"], 0) + 1
                if view is not None:
                    with Prof("view.navigation"):
                        view.navigation(res, T_, navst["up"])
            with Prof("pm.add_node"):
                pm.add_node(m["id"], r.depth, conf_s, r.sigma_z, left, conf_min=conf_min)
            gated = np.where(conf_s >= conf_min, r.depth, 0.0)
            nodes[m["id"]] = (np.clip(gated * 1000, 0, 65535).astype(np.uint16), left.copy())
            node_frame[m["id"]] = fid
            if tsdf is not None:
                with Prof("tsdf.add_node"):
                    tsdf.add_node(m["id"], nodes[m["id"]][0], left)
            if fusion is not None and (segmenter is None or sg is not None):
                conf16 = np.where(conf_s >= FUSION_CONF_MIN, conf_s, 0).astype(np.float16)       # the surfel map is the CPU-heavy one and only needs surfaces for the labels: it keeps the strict gate however low --conf-min goes
                extra[m["id"]] = (conf16, sg)
                fusion.add_node(m["id"], nodes[m["id"]][0], left, conf16, sg)

    last = None
    t_start = time.perf_counter()
    for idx in (rec.indices() if hasattr(rec, "indices") else range(*clip)):
        if (idx - clip[0]) % 50 == 0:
            print(f"  pair {idx - clip[0]}" + ("" if hasattr(rec, "indices") else f"/{clip[1] - clip[0]}") + f", {len(nodes)} nodes, {time.perf_counter() - t_start:.0f} s", flush=True)
            if profile and idx > clip[0]:
                print(Prof.report(time.perf_counter() - t_start), flush=True)
        with Prof("prepare"):
            left, right = rec.prepare(idx, rect, shutter="default" if shutter else None)
        cur["left"] = left
        t_arr[idx] = time.perf_counter()
        if not sync_depth:
            dfut[idx] = depth_pool.submit(depth_job, idx, left, right, t_arr[idx])          # depth starts NOW, in parallel with RTAB-Map, instead of after the node message comes back
        for k in [k for k in t_arr if k < idx - 40]:
            del t_arr[k]
            dfut.pop(k, None)
        if view is not None:
            with Prof("view.pair"):
                view.at(idx, float(rec.t[idx]))
                view.pair(left, right)
        with Prof("bridge wait"):
            b.wait_idle()
        msgs = b.poll()
        g = rec.gyro_rotvec(last, idx) if (gyro and last is not None) else None
        with Prof("send"):
            if b.send_frame(idx, float(rec.t[idx] - rec.t[0]), left, cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), *(() if g is None else (g,))):
                last = idx
        with Prof("handle (all)"):                              # the bridge is now busy with this pair: the depth, navigation and map work of the PREVIOUS pair's results runs alongside it, not after it
            for m in msgs:
                handle(m)
        if view is not None:
            with Prof("view.cloud"):
                view.cloud(pm)
    b.wait_idle()
    if not nodes:
        print("NO NODES: the bridge sent " + str(b.sent) + " frames, skipped " + str(b.skipped) + ", alive=" + str(b.alive) + "; its last lines:\n  " + "\n  ".join(b.log[-15:]))
    for m in b.poll():
        handle(m)
    depth_pool.shutdown(wait=True)
    if view is not None:
        view.cloud(pm, force=True)
    print("  replay done; building the maps and scores (about a minute)...", flush=True)
    b.close(save=False)
    st["skipped_quality"] = st_skip["quality"]
    st["nav"] = navst
    st["bridge_ms"] = bms
    st["depth_delay_ms"] = dlat
    return pm, nodes, poses, node_frame, cal, ms, st, extra


def write_trajectory(path, rec, poses, node_frame):
    with open(path, "w") as f:
        for i in sorted(poses):
            T = poses[i] @ OPTICAL
            q = quat(T[:3, :3])
            f.write(f"{rec.t[node_frame[i]] - rec.t[0]:.4f} {T[0, 3]:.5f} {T[1, 3]:.5f} {T[2, 3]:.5f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def compare(nodes, poses, node_frame, cal, pm, hold=5, extra=None, semantic_kw=None, flatten=False):
    import open3d as o3d
    from rtabmap_lab.tools import recon_compare as rc
    ids = sorted(i for i in nodes if i in poses)
    held_ids = set(ids[hold // 2:: hold])
    size = tuple(cal["size"])
    fuse = {i: (nodes[i][0], nodes[i][1], poses[i], None) for i in ids if i not in held_ids}
    held = [(nodes[i][0], poses[i]) for i in sorted(held_ids)]
    print(f"{len(fuse)} nodes fused, {len(held)} held out", flush=True)
    rows = {}
    pm2 = PointMap(pm.k, pm.min_views)
    pm2.nodes = {i: pm.nodes[i] for i in fuse if i in pm.nodes}
    pm2.set_poses({i: poses[i] for i in pm2.nodes})
    Pw, C, cf, views = pm2.cloud()
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pw.astype(np.float64)))
    methods = {"semi-dense cloud (ours)": ("points", pc, 0.01)}
    for name, fn in (("raw points, 1 cm voxel", rc.m_points), ("surfel (>=2 views)", rc.m_surfel), ("tsdf 2 cm", rc.m_tsdf)):
        t0 = time.perf_counter()
        methods[name] = (*fn(fuse, cal, size), time.perf_counter() - t0)
    if extra:
        from mapper.semantics import K
        from mapper.surfel_fuse import SemanticSurfelFusion
        t0 = time.perf_counter()
        f2 = SemanticSurfelFusion((cal["f"], cal["f"], cal["cx"], cal["cy"]), cal["f"] * cal["B"], K, threaded=False, stride=2, max_range=2.5, **(semantic_kw or {}))
        for i in fuse:
            f2.add_node(i, nodes[i][0], nodes[i][1], extra[i][0], extra[i][1])
        f2.set_poses({i: poses[i] for i in fuse})
        m2 = f2.flush()
        Ps, _, _ = m2.points()
        ps = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Ps.astype(np.float64)))
        methods["semantic surfels (v2)"] = ("points", ps, 0.01, time.perf_counter() - t0)
        if flatten:
            t0 = time.perf_counter()
            Ff = flat_surfels(m2)[0]
            methods["semantic surfels + flatten"] = ("points", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Ff.astype(np.float64))), 0.01, time.perf_counter() - t0)
    for name, (kind, geom, *sp) in methods.items():
        s = rc.score(kind, geom, sp[0] if kind == "points" else 0.0, held, cal, size)
        g = rc.geometry_stats(kind, geom)
        rows[name] = {**{k: round(v, 3) for k, v in s.items()}, **g}
        print(f"{name:26s} {json.dumps(rows[name])}", flush=True)
    return rows


def save_surfels_ply(path, P, bgr, labels, prob):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "property char label\nproperty float prob\nend_header\n").encode())
        rec = np.empty(len(P), np.dtype([("p", "<f4", 3), ("c", "u1", 3), ("l", "i1"), ("q", "<f4")]))
        rec["p"], rec["c"], rec["l"], rec["q"] = P, np.clip(bgr[:, ::-1], 0, 255).astype(np.uint8), labels, prob
        f.write(rec.tobytes())


def flat_surfels(m):
    """Flattened stable surfels of a semantic map: (points, normals, colours bgr, labels, prob, planes)."""
    from mapper.flatten import flatten
    st = m.stable()
    P, N, Cb = m.P[st], m.N[st], m.C[st]
    labels, prob = m.labels()
    F, keep, planes = flatten(P, N, m.W[st], labels)
    return F[keep], N[keep], Cb[keep], labels[keep], prob[keep], planes


def report_semantics(fusion, out, flatten=False, ctx=None):
    """Final surfel map: class counts, objects (grouped surfels -> persistent ids by gabai.objects3d.ObjectMap) and, for every object, how flat and how big it came out."""
    from gabai.objects3d import ObjectMap
    from mapper.semantics import CLASSES, instances, plane_fit
    m = fusion.flush()
    if fusion.stats.get("error"):
        print("SURFEL WORKER ERROR (the live view of the surfels stopped updating):", fusion.stats["error"])
    P, N, Cb = m.points()
    labels, prob = m.labels()
    save_surfels_ply(os.path.join(out, "surfels.ply"), P, Cb, labels, prob)
    known = labels >= 0
    counts = {CLASSES[k]: int((labels == k).sum()) for k in range(len(CLASSES)) if (labels == k).any()}
    print(f"semantic surfels: {len(m)} surfels, {len(P)} seen by >= {m.min_views} nodes, {known.mean() * 100:.0f}% labelled; per class {counts}; map rebuilt {fusion.stats['rebuilds']} times")
    if flatten:
        F, Nf, Cf, lf, pf, planes = flat_surfels(m)
        save_surfels_ply(os.path.join(out, "surfels_flat.ply"), F, Cf, lf, pf)
        print(f"flattened: {len(F)} of {len(P)} surfels kept ({len(P) - len(F)} stacked-layer surfels dropped), {len(planes)} planes")
        for pl in planes:
            print(f"  plane {pl['label']:12s} {pl['members']:6d} surfels on it, {pl['dropped']:5d} stacked copies dropped, scatter {pl['rms_before_mm']:.1f} -> {pl['rms_after_mm']:.1f} mm, extent {pl['extent'][0]:.2f} x {pl['extent'][1]:.2f} m")
    if ctx is not None:                                                         # vision-only world alignment (mapper/align.py), judged by the accelerometer without using it
        from mapper.align import alignment, apply, up_error_vs_accel
        rec, poses, node_frame = ctx
        st = m.stable()
        A, info = alignment(poses, P, N, m.W[st])
        if A is None:
            print(f"world alignment refused: {info.get('reason')}; the map stays in RTAB-Map's frame (first camera = level)")
        else:
            import json as _json
            _json.dump({"A": A.tolist(), **{k: v for k, v in info.items()}}, open(os.path.join(out, "alignment.json"), "w"), indent=1)
            e0, e1 = up_error_vs_accel(rec, poses, node_frame), up_error_vs_accel(rec, poses, node_frame, A)
            print(f"world alignment: up from the camera-up prior (planes: {info['refinement']})")
            if len(e0):
                print(f"  map up against the accelerometer's gravity (judge only, {len(e0)} still nodes): median {np.median(e0):.1f} -> {np.median(e1):.1f} deg, 90th {np.percentile(e0, 90):.1f} -> {np.percentile(e1, 90):.1f} deg")
            Pa = apply(A, P)
            save_surfels_ply(os.path.join(out, "surfels_aligned.ply"), Pa, Cb, labels, prob)
            if flatten:
                save_surfels_ply(os.path.join(out, "surfels_flat_aligned.ply"), apply(A, F), Cf, lf, pf)
            write_trajectory(os.path.join(out, "trajectory_aligned.txt"), rec, {i: apply(A, T) for i, T in poses.items() if i in node_frame}, node_frame)
            print("  aligned files: surfels_aligned.ply" + (", surfels_flat_aligned.ply" if flatten else "") + ", trajectory_aligned.txt, alignment.json")
            report_semantics.A = A
            from mapper import walkable as wk
            g = wk.build(Pa)
            doors = wk.doorways(g)
            st_ = g["stats"]
            print(f"walkable floor: {st_['walkable_m2']:.1f} m2 at z {st_['floor_z']:.2f} m, {st_['measured_floor_m2']:.1f} m2 of it measured floor and {st_['inferred_m2']:.1f} m2 inferred (no floor points, nothing standing there), {st_['obstacle_m2']:.1f} m2 of obstacles; {len(doors)} doorways")
            Tl = A @ poses[max(poses)]
            gd = wk.guide(g, Tl[:2, 3], Tl[:2, 0], doors)
            for d in doors:
                rel = d["center"] - Tl[:2, 3]
                print(f"  doorway {d['width']:.2f} m wide at ({d['center'][0]:.1f}, {d['center'][1]:.1f}), {np.hypot(*rel):.1f} m from the last pose")
            print("  guide from the last pose: " + (f"to the {gd['target']}, {gd['dist']:.1f} m, bearing {gd['bearing_deg']:+.0f} deg" if gd else "none (the last pose is not on safe floor)"))
            r_, c_ = np.nonzero(g["walkable"])
            x_, y_ = wk.cell_xy(g, r_, c_)
            kind = np.where(g["inferred"][r_, c_], 2, np.where(g["safe"][r_, c_], 0, 1))
            pal = np.array([[40, 200, 90], [30, 110, 60], [150, 220, 80]], np.uint8)[kind]
            save_surfels_ply(os.path.join(out, "walkable.ply"), np.c_[x_, y_, np.full(len(x_), g["floor_z"])], pal[:, ::-1].astype(np.float32), np.full(len(x_), -1), np.zeros(len(x_), np.float32))
            print("  wrote walkable.ply (green safe, dark green walkable near obstacles, light green inferred floor)")
            if flatten:
                elev = lambda v: np.degrees(np.arcsin(np.clip(abs(v[2]), 0, 1)))
                for pl in planes:
                    if pl["label"] == "screen":
                        print(f"  screen plane ({pl['members']} surfels, {pl['extent'][0]:.2f} x {pl['extent'][1]:.2f} m): face normal elevation {elev(pl['normal']):.0f} deg in RTAB-Map's frame -> {elev(A[:3, :3] @ pl['normal']):.0f} deg aligned (0 = vertical face, 90 = facing up)")
    objs = ObjectMap(min_looks=1)
    dets = instances(P, labels, prob)
    objs.update(0.0, [{k: v for k, v in d.items() if k != "members"} for d in dets], lambda p: False)
    if ctx is not None and poses:                                                # what the user would hear at the end of the walk (mapper/egocentric.py -> gabai/net/lexicon.py)
        from gabai.net.lexicon import say
        from mapper.egocentric import Egocentric
        last = poses[max(poses)]
        msg = Egocentric().update(0.0, last, [{"id": i, **{k: v for k, v in d.items() if k != "members"}} for i, d in enumerate(dets)])
        print("egocentric at the last pose: " + (" | ".join(say(p) for p in msg.get("d", [])) or "nothing in range") + f"  ({len(str(msg).replace(' ', ''))} bytes: {str(msg).replace(' ', '')})")
    for d, T in zip(dets, objs.objs):
        n, rms, ext = plane_fit(P[d["members"]])
        print(f"  object {T['id']}: {d['label']:12s} {len(d['members']):6d} surfels, size {d['dims'][0]:.2f} x {d['dims'][1]:.2f} x {d['dims'][2]:.2f} m at {np.round(d['p_w'], 2)}; plane fit rms {rms * 1000:.1f} mm, in-plane extent {ext[0]:.2f} x {ext[1]:.2f} m")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="a recording folder; omit with --pi")
    ap.add_argument("--pi", action="store_true", help="live from the Raspberry Pi instead of a recording (python3 -m gabai.pi.sender --host <this PC> ...); stop with Ctrl-C and the maps are written")
    ap.add_argument("--port", type=int, default=5600, help="--pi: the port the Pi sender connects to (the sender default)")
    ap.add_argument("--bridge-port", type=int, default=5710, help="local port of the RTAB-Map bridge; change it to run two at once")
    ap.add_argument("--rotate-left", type=int, default=-1, help="--pi: turn the left picture by this many degrees (0 / 90 / 180 / 270). Default -1 = as the calibration was made, 180 for the current one")
    ap.add_argument("--rotate-right", type=int, default=-1, help="--pi: same for the right picture (default: as the calibration was made, 0)")
    ap.add_argument("--history", action="store_true", help="--live / --save-rrd: keep every update of the map entities in the timeline (scrub back through the map). Off by default: the viewer keeps only the latest cloud, surfels, mesh and floor, which is what keeps it fast; the file also stays small")
    ap.add_argument("--sync-depth", action="store_true", help="the old order: compute a pair's depth only after RTAB-Map has made a node of it (for comparing the delay)")
    ap.add_argument("--rtab-disparity", action="store_true", help="let the RTAB-Map bridge also compute its own block-matching disparity, node clouds and dense depth for every node (the RTAB-Map own disparity panel). Off by default: the mapper uses its own depth, and that work was the slowest part of the bridge")
    ap.add_argument("--profile", action="store_true", help="print where the main loop spends its time every 50 pairs")
    ap.add_argument("--seconds", type=float, default=0, help="--pi: stop by itself after this many seconds (0 = until Ctrl-C)")
    ap.add_argument("--relay", default="http://127.0.0.1:8090", help="--pi: send the walking guidance to this caregiver relay so the phone's Navigation screen speaks it ('' = off)")
    ap.add_argument("--flip", action="store_true", help="--pi: the rig is mounted upside down (turn the pictures 180 deg, swap left and right)")
    ap.add_argument("--calib", default="data/stereo_calib_charuco_45.npz", help="a calibration .npz, or 'active' for the confirmed self-calibration (mapper.selfcal)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", type=int, nargs=2, default=(0, 10 ** 9))
    ap.add_argument("--shutter", action="store_true", help="opt in: rolling-shutter correction of the right picture from the gyro (data/rolling_shutter.json). Off by default: RTAB-Map and the pictures only")
    ap.add_argument("--gyro", action="store_true", help="opt in: the gyro turns the pose when odometry restarts. Off by default")
    ap.add_argument("--depth", choices=["retinify", "las2", "ffs"], default="retinify", help="depth source: retinify (default), LAS2, or TensorRT Fast-FoundationStereo; all give depth + the same confidence map")
    ap.add_argument("--retinify-mode", choices=["fast", "balanced", "accurate"], default="balanced", help="retinify matching size: fast 640x320, balanced 640x480, accurate 1280x720")
    ap.add_argument("--las2-size", choices=["s", "m", "h"], default="s", help="LAS2 model size: s is fastest, h most accurate")
    ap.add_argument("--iters", type=int, default=8, help="ffs only")
    ap.add_argument("--max-disp", type=int, default=192)
    ap.add_argument("--conf-min", type=float, default=0.35, help="pixels below this depth confidence stay out of the maps (0.5 was strict enough to drop fine detail)")
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--reset-countdown", type=int, default=3, help="RTAB-Map Odom/ResetCountdown: restart odometry (new map session) after this many lost pairs; 0 = the lab bridge's manual restart after 15")
    ap.add_argument("--min-inliers", type=int, default=0, help="a node whose odometry had fewer inliers adds no geometry")
    ap.add_argument("--min-sharp", type=float, default=0.0, help="a node whose left picture has a smaller Laplacian variance adds no geometry (the walk: tracked 558, lost 73)")
    ap.add_argument("--semantic", action="store_true", help="also build the semantic surfel map (SemanticFusion-style: surfel update + Bayesian class fusion from YOLO11n-seg) and report objects")
    ap.add_argument("--flatten", action="store_true", help="with --semantic: flatten planes of the surfel map and drop stacked layers (mapper/flatten.py); adds surfels_flat.ply and a scored row")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--live", action="store_true", help="open the Rerun viewer and stream the run into it (RTAB-Map, depth, confidence, cloud, TSDF)")
    ap.add_argument("--save-rrd", help="stream the same into a .rrd file instead of (or as well as) the viewer")
    ap.add_argument("--point-size", type=float, default=0.004, help="point radius in metres (viewer only)")
    ap.add_argument("--no-tsdf", action="store_true", help="with --live: skip the TSDF surface")
    a = ap.parse_args(argv)
    if a.calib == "active":                                                    # the self-calibration adopted with `py -m mapper.selfcal activate --confirm`; otherwise the default file below stays in use
        if not os.path.exists("data/selfcal/active.json"):
            raise SystemExit("no self-calibration is active (py -m mapper.selfcal status); give --calib a file")
        a.calib = json.load(open("data/selfcal/active.json"))["calib"]
        print(f"calibration: self-calibration {a.calib}", flush=True)
    if a.pi:
        from mapper.pi_source import PiRecording
        rec = PiRecording(a.port, a.calib, a.flip, a.seconds, (a.rotate_left, a.rotate_right))
        print(f"waiting for the Pi on port {a.port} ...", flush=True)
        rec.wait_first()
        rect, clip = rec.rect, (0, 10 ** 9)
    else:
        if not a.folder:
            raise SystemExit("give a recording folder, or --pi for the live Raspberry Pi")
        base = P.CalibParams.from_npz(a.calib)
        rect = base.rect()
        rec = Recording(a.folder).decide(rect, shutter_file="data/rolling_shutter.json" if a.shutter else None)
        clip = (a.clip[0], min(a.clip[1], rec.n))
    cal = rect.camera(flip=rec.flip, k=rec.k)
    if a.depth == "retinify":
        from mapper.depth_retinify import RetinifyStereo
        model = RetinifyStereo(cal["f"], cal["B"], a.retinify_mode)
    elif a.depth == "las2":
        from mapper.depth_las2 import Las2Stereo
        model = Las2Stereo(cal["f"], cal["B"], a.las2_size)
    else:
        model = StereoTrt(cal["f"], cal["B"], paths(a.iters, a.max_disp)[1])
    view = tsdf = None
    if a.live or a.save_rrd:
        from mapper.live_view import LiveView
        view = LiveView(cal, tuple(cal["size"]), a.point_size, save=a.save_rrd, viewer=a.live, flatten=a.flatten, history=a.history)
        if not a.no_tsdf:
            from rtabmap_lab.gpu_tsdf import GpuTsdfMap
            tsdf = GpuTsdfMap((cal["f"], cal["f"], cal["cx"], cal["cy"]), tuple(cal["size"]), voxel=0.02, trunc=0.08, max_range=2.5, fb=cal["f"] * cal["B"], on_mesh=view.mesh_callback())
    fusion = segmenter = None
    if a.semantic:
        from mapper.semantics import K, Segmenter
        from mapper.surfel_fuse import SemanticSurfelFusion
        segmenter = Segmenter(stride=2)
        fusion = SemanticSurfelFusion((cal["f"], cal["f"], cal["cx"], cal["cy"]), cal["f"] * cal["B"], K, threaded=view is not None, stride=2, max_range=2.5,
                                      on_update=view.surfels if view is not None else None)
    relay = None
    if a.pi and a.relay:
        from gabai.relay.publish import RelayPublisher
        relay = RelayPublisher(a.relay)
        print(f"walking guidance goes to the phone through the relay at {a.relay}", flush=True)
    t0 = time.perf_counter()
    pm, nodes, poses, node_frame, cal, ms, st, extra = run(rec, rect, clip, model, a.shutter, a.conf_min, min_views=a.min_views, port=a.bridge_port, view=view, tsdf=tsdf, gyro=a.gyro, reset_countdown=a.reset_countdown, min_inliers=a.min_inliers, min_sharp=a.min_sharp, fusion=fusion, segmenter=segmenter, profile=a.profile, sync_depth=a.sync_depth, relay=relay, lean=not a.rtab_disparity)
    if tsdf is not None:
        tsdf.close()
    os.makedirs(a.out, exist_ok=True)
    Pw, C, cf, views = pm.cloud()
    pm.save_ply(os.path.join(a.out, "cloud.ply"))
    ids = [i for i in sorted(nodes) if i in poses]
    write_trajectory(os.path.join(a.out, "trajectory.txt"), rec, {i: poses[i] for i in ids}, node_frame)
    np.savez_compressed(os.path.join(a.out, "nodes.npz"), ids=np.array(ids), depth=np.stack([nodes[i][0] for i in ids]), bgr=np.stack([nodes[i][1] for i in ids]), poses=np.stack([poses[i] for i in ids]),
                        cal=json.dumps({"f": cal["f"], "B": cal["B"], "cx": cal["cx"], "cy": cal["cy"], "size": list(cal["size"])}))
    print(f"{rec.n if hasattr(rec, 'indices') else clip[1] - clip[0]} pairs in {time.perf_counter() - t0:.0f} s; {len(ids)} posed nodes; skipped for quality {st['skipped_quality']}, lost {st['lost']}, restarts {st['restarts']}, loop closures {st['loops']}; depth {np.median(ms):.0f} ms per node")
    bm = np.array(st["bridge_ms"]) if st["bridge_ms"] else None
    if bm is not None:
        print(f"RTAB-Map bridge, median ms per pair: odometry {np.median(bm[:, 0]):.0f}, map update {np.median(bm[bm[:, 1] > 0, 1]) if (bm[:, 1] > 0).any() else 0:.0f}, its own disparity {np.median(bm[bm[:, 2] > 0, 2]) if (bm[:, 2] > 0).any() else 0:.0f} (max {bm[:, 2].max():.0f})")
    dl = st["depth_delay_ms"]
    if dl:
        print(f"depth delay (the pair arrives -> its depth is ready): median {np.median(dl):.0f} ms, 90th {np.percentile(dl, 90):.0f} ms, max {np.max(dl):.0f} ms over {len(dl)} nodes")
    if view is not None and getattr(view, "cloud_ms", None):
        print(f"cloud rebuild (own thread): {len(view.cloud_ms)} builds, median {np.median(view.cloud_ms):.0f} ms, max {np.max(view.cloud_ms):.0f} ms")
    nv = st["nav"]
    if nv["ms"]:
        print(f"navigation: {len(nv['ms'])} updates, median {np.median(nv['ms']):.1f} ms, max {np.max(nv['ms']):.1f} ms; cues " + ", ".join(f"{k} {v}" for k, v in sorted(nv["cues"].items(), key=lambda kv: -kv[1])))
    print(f"cloud: {len(Pw)} points seen by >= {a.min_views} nodes (median {np.median(views) if len(views) else 0:.0f} views, confidence {np.mean(cf) if len(cf) else 0:.2f}); extent {np.ptp(Pw, axis=0).round(2) if len(Pw) else 0} m")
    if fusion is not None:
        report_semantics(fusion, a.out, a.flatten, (rec, poses, node_frame))
        fusion.close()
    if a.compare:
        rows = compare(nodes, poses, node_frame, cal, pm, extra=extra if a.semantic else None, flatten=a.flatten)
        json.dump(rows, open(os.path.join(a.out, "scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
