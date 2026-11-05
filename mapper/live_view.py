"""Live Rerun view of a mapper run: RTAB-Map (odometry, graph, loop closures), our depth + confidence, RTAB-Map's own disparity for comparison, the semi-dense cloud and the TSDF surface.

Reuses rtabmap_lab.dashboard.Dashboard for everything RTAB-Map and the TSDF mesh log (camera, optimised path, loop closures, status, time series, mesh); this adds our depth, confidence and cloud
and its own layout. Its own app id ("mapper"), nothing shared with the lab dashboard.

Entity tree additions: depth/ours (turbo, metres), depth/confidence (turbo, 0..1), world/cloud (semi-dense, radius = the --point-size viewer setting),
timing/depth_ms, cloud/points.
"""
import threading
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from rtabmap_lab.dashboard import Dashboard

CLOUD_EVERY_S = 2.0
EGO_KIND = np.array([[0, 0, 0], [150, 220, 80], [230, 60, 60], [255, 150, 0], [170, 60, 220]], np.uint8)     # grid kinds of mapper/navigate.py: -, floor, obstacle, overhead, drop
TRAIL_KEEP = 150          # newest nodes / sightings drawn in a trail; older ones are dropped
TRAIL_MIN_ALPHA = 15      # the oldest drawn segment is this transparent, the newest opaque


def fading(points, rgb, keep=TRAIL_KEEP):
    """One strip per segment of the newest `keep` points, alpha rising from TRAIL_MIN_ALPHA (oldest) to 255 (newest): (strips, colours)."""
    P = np.asarray(points, np.float32)[-keep:]
    if len(P) < 2:
        return [], []
    a = np.linspace(TRAIL_MIN_ALPHA, 255, len(P) - 1).astype(np.uint8)
    return [P[i:i + 2] for i in range(len(P) - 1)], [list(rgb) + [int(x)] for x in a]


def blueprint():
    cams = rrb.Grid(*[rrb.Spatial2DView(origin=o, name=n) for o, n in (
        ("cameras/left", "left"), ("cameras/right", "right"), ("depth/ours", "our depth (retinify by default; LAS2 or FoundationStereo with --depth)"), ("depth/confidence", "confidence 0..1"),
        ("stereo/disparity", "RTAB-Map's own disparity"))], grid_columns=2)
    maps = rrb.Tabs(                                                                 # the three maps share one big panel (tabs); the egocentric view gets the other half of the column
        rrb.Spatial3DView(origin="/", contents=["/world/cloud/**", "/world/camera/**", "/world/walkable/**", "/world/live/**", "/world/path/**", "/world/loops/**", "/world/nodes/**"], name="semi-dense cloud"),
        rrb.Spatial3DView(origin="/", contents=["/world/surfels/**", "/world/walkable/**", "/world/objects/**", "/world/camera/**", "/world/path/**", "/world/loops/**"], name="semantic surfels + tracked objects"),
        rrb.Spatial3DView(origin="/", contents=["/world/tsdf_mesh/**", "/world/walkable/**", "/world/camera/**", "/world/path/**", "/world/loops/**"], name="TSDF surface"))
    ego = rrb.Spatial3DView(origin="ego_view", name="egocentric: you at the centre, forward = +x; floor green, obstacle red, overhead orange, drop purple; rings 0.5 / 1.5 / 3 / 5 m")
    return rrb.Blueprint(rrb.Horizontal(
        rrb.Vertical(maps, ego, row_shares=[1, 1]),
        rrb.Vertical(cams, rrb.TimeSeriesView(origin="odom", name="odometry: features, inliers, lost"), rrb.TimeSeriesView(origin="map", name="map: nodes, loops, mesh"),
                     rrb.TimeSeriesView(origin="timing", name="milliseconds"), rrb.TimeSeriesView(origin="range", name="distance to each tracked object (m)"), row_shares=[5, 2, 2, 2, 2]),
        rrb.Vertical(rrb.TextDocumentView(origin="status", name="status"), rrb.TextDocumentView(origin="ego", name="egocentric: what is where around you"), rrb.TextLogView(origin="events", name="events"), row_shares=[2, 2, 3]),
        column_shares=[5, 5, 2]), rrb.TimePanel(timeline="frame"))


class LiveView:
    def __init__(self, cal, size, point_size=0.004, save=None, viewer=True, flatten=False, history=False):
        rr.init("mapper", spawn=False)
        if viewer:
            rr.spawn(memory_limit="2GiB")                                         # the viewer drops its oldest data instead of growing until it crawls
        self.static = not history
        rr.log("ego_view", rr.ViewCoordinates.FLU, static=True)                      # forward, left, up: the body frame of the poses
        rr.log("ego_view/you", rr.Points3D([[0, 0, 0]], colors=[255, 255, 255], radii=0.06, labels=["you"], show_labels=True), static=True)
        rr.log("ego_view/rings", rr.LineStrips3D([[(r * np.cos(a), r * np.sin(a), 0) for a in np.linspace(0, 2 * np.pi, 49)] for r in (0.5, 1.5, 3.0, 5.0)], colors=[90, 90, 90], radii=0.004), static=True)
        if save:
            rr.save(save)
        rr.send_blueprint(blueprint())
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world/camera", rr.Pinhole(focal_length=cal["f"], principal_point=(cal["cx"], cal["cy"]), resolution=size, camera_xyz=rr.ViewCoordinates.RDF), static=True)
        self.dash = Dashboard(cal, size, viewer=False)
        self.point_size, self._last_cloud, self.skipped, self.flatten = point_size, 0.0, 0, flatten

    def mesh_callback(self):
        return self.mesh

    def mesh(self, mesh, stats):
        """The TSDF worker's finished meshes (its own thread). Latest only: every mesh is the whole surface, so a history of them is what made the viewer crawl."""
        rr.set_time("frame", sequence=self.dash._frame)
        v, tri = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.uint32)
        if not len(v):
            return
        col = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None
        rr.log("world/tsdf_mesh", rr.Mesh3D(vertex_positions=v, triangle_indices=tri, vertex_normals=np.asarray(mesh.vertex_normals, np.float32) if mesh.has_vertex_normals() else None, vertex_colors=col), static=self.static)
        for path, key in (("map/tsdf_vertices", "vertices"), ("map/tsdf_rebuilds", "rebuilds"), ("timing/tsdf_mesh_ms", "mesh_ms"), ("timing/tsdf_integrate_ms", "integrate_ms")):
            if key in stats:
                rr.log(path, rr.Scalars(float(stats[key])))

    def at(self, idx, t):
        self.dash.at(idx, t)

    def pair(self, left, right):
        self.dash.pair(left, right)

    def message(self, m, left=None):
        t = m["type"]
        if t == "status":
            self.dash.status(m, left, self.skipped)
            if m["loop_id"] > 0 or m["proximity_id"] > 0:
                self.dash.event(f"loop closure: frame {m['id']} with node {m['loop_id'] or m['proximity_id']}")
            if m["lost"]:
                self.dash.event(f"odometry lost at frame {m['id']}", "WARN")
        elif t == "graph":
            self.dash.graph(m)
            ids = sorted(self.dash.poses)                                   # redraw the dashboard's whole-history path with its oldest part fading out
            strips, cols = fading([self.dash.poses[i][:3, 3] for i in ids], (80, 170, 255))
            rr.log("world/path", rr.LineStrips3D(strips, colors=cols, radii=0.012))
            rr.log("world/nodes", rr.Points3D(np.array([self.dash.poses[i][:3, 3] for i in ids[-TRAIL_KEEP:]]), colors=[255, 255, 255, 120], radii=0.015)) if ids else None
        elif t == "node":
            self.dash.node(m)
        elif t == "disparity":
            self.dash.disparity(m["image"])
        elif t == "log":
            self.dash.event(m["text"])

    def depth(self, r, node_id, frame=None):
        if frame is not None:
            rr.set_time("frame", sequence=frame)                             # called from the depth thread: Rerun's time is per thread
        rr.log("depth/ours", rr.Image(Dashboard._depth_rgb(r.depth)).compress(jpeg_quality=70))
        rr.log("depth/confidence", rr.Image(Dashboard._depth_rgb(r.conf, 1.0)).compress(jpeg_quality=70))
        rr.log("timing/depth_ms", rr.Scalars(float(r.ms)))
        self.dash.event(f"node {node_id}: depth {r.ms:.0f} ms, {np.mean(r.depth > 0) * 100:.0f}% of pixels, {np.mean(r.conf > 0.8) * 100:.0f}% above confidence 0.8")

    def surfels(self, m, stats):
        """Called by the surfel worker after each round: stable surfels coloured by their most probable class (dim real colour where unknown), thinned to 300k."""
        from mapper.semantics import PALETTE
        now = time.monotonic()
        if now - getattr(self, "_last_surf", 0.0) < (5.0 if self.flatten else 1.0):
            return
        self._last_surf = now
        P, _, Cb = m.points()
        if not len(P):
            return
        k, p = m.labels()
        if now - getattr(self, "_last_align", -99.0) > 5.0:                       # world alignment from the planes of the map so far, applied by Rerun as the transform of "world"
            from mapper.align import alignment
            st_ = m.stable()
            A, info = alignment(self.dash.poses, m.P[st_], m.N[st_], m.W[st_])
            self._last_align = now
            if A is not None:
                rr.log("world", rr.Transform3D(mat3x3=A[:3, :3]))
                self._A = A
                self.dash.event(f"world alignment from the camera-up prior ({info['refinement']})")
        if self.flatten:                                                          # the plane search is the expensive part, so with it the whole update runs at most every 5 s
            from mapper.flatten import flatten
            st = m.stable()
            F, keep, planes = flatten(m.P[st], m.N[st], m.W[st], k)
            self.dash.event(f"flattened {len(planes)} planes, {int((~keep).sum())} stacked-layer surfels hidden: " + ", ".join(f"{q['label']} {q['rms_before_mm']:.0f}->{q['rms_after_mm']:.0f} mm" for q in planes[:6]))
            P, Cb, k, p = F[keep], Cb[keep], k[keep], p[keep]
        col = np.where((k >= 0)[:, None], PALETTE[np.maximum(k, 0)], np.clip(Cb[:, ::-1], 0, 255).astype(np.uint8) // 3)
        sel = np.linspace(0, len(P) - 1, min(len(P), 150_000)).astype(int)
        rr.set_time("frame", sequence=self.dash._frame)
        rr.log("world/surfels", rr.Points3D(P[sel], colors=col[sel], radii=self.point_size), static=self.static)
        rr.log("cloud/surfels", rr.Scalars(float(len(P))))
        rr.log("timing/surfel_ms", rr.Scalars(float(stats["ms"])))
        self.walkable(P, now)
        self.egocentric(P, k, p, now)

    def people(self, tracks):
        """People tracked in the video, drawn in the world: solid-ish while seen, a faint ghost at the last place after they left (never an obstacle), gone after the tracker's ghost time."""
        rr.set_time("frame", sequence=self.dash._frame)
        now = {t.id for t in tracks}
        for gone in getattr(self, "_people", set()) - now:
            rr.log(f"world/objects/people/{gone}", rr.Clear(recursive=True), static=self.static)
        self._people = now
        for t in tracks:
            if t.points is None or not len(t.points):
                continue
            alpha = 150 if t.visible else 45
            rr.log(f"world/objects/people/{t.id}/body", rr.Points3D(t.points, colors=[255, 255, 255, alpha], radii=self.point_size * 2), static=self.static)
            tag = f"person {t.id}" + (f", {t.speed:.1f} m/s" if t.visible and t.hits >= 4 and t.speed >= 0.3 else "" if t.visible else ", left")
            rr.log(f"world/objects/people/{t.id}/label", rr.Points3D([np.median(t.points, axis=0)], labels=[tag], radii=0.02, colors=[255, 255, 255, alpha]), static=self.static)

    def walkable(self, P, now):
        """Walkable floor, holes, doorways and the path to the nearest doorway (mapper/walkable.py) in the aligned world, drawn under "world/walkable" (a child whose transform undoes the world alignment,
        so everything below is in aligned coordinates). Doorways also become objects for the egocentric view."""
        from mapper import walkable as wk
        A = getattr(self, "_A", None)
        if A is None or not self.dash.poses or len(P) < 5000:
            return
        R = A[:3, :3]
        g = wk.build(P @ R.T)
        doors = wk.doorways(g)
        T = self.dash.poses[max(self.dash.poses)]
        u, f = R @ T[:3, 3], R @ T[:3, 0]
        gd = wk.guide(g, u[:2], f[:2], doors)
        self._walk = {"g": g, "doors": doors, "guide": gd, "R": R}
        fz = g["floor_z"]
        rr.set_time("frame", sequence=self.dash._frame)
        rr.log("world/walkable", rr.Transform3D(mat3x3=R.T), static=self.static)
        rr.log("world/walkable/inferred", rr.Clear(recursive=False), static=self.static)              # merged into "safe" below; clears what an older build drew there
        for name, mask, col in (("safe", g["safe"], [150, 220, 80]), ("edge", g["walkable"] & ~g["safe"], [30, 110, 60])):     # light green: the walkable path (measured or inferred floor); dark green: too close to something
            r_, c_ = np.nonzero(mask)
            sel = np.linspace(0, len(r_) - 1, min(len(r_), 30_000)).astype(int) if len(r_) else r_
            x, y = wk.cell_xy(g, r_[sel], c_[sel])
            rr.log(f"world/walkable/{name}", rr.Points3D(np.stack([x, y, np.full(len(x), fz + 0.01)], 1), colors=col, radii=g["cell"] * 0.55) if len(x) else rr.Points3D([]), static=self.static)
        if gd:
            pa = np.c_[gd["path"], np.full(len(gd["path"]), fz + 0.06)]
            rr.log("world/walkable/path", rr.LineStrips3D([pa], colors=[0, 220, 255], radii=0.03), static=self.static)
        else:
            rr.log("world/walkable/path", rr.LineStrips3D([]), static=self.static)
        if doors:
            rr.log("world/walkable/doorways", rr.Boxes3D(centers=[[d["center"][0], d["center"][1], fz + 1.0] for d in doors], half_sizes=[[d["width"] / 2, 0.08, 1.0] for d in doors],
                                                         rotation_axis_angles=[rr.RotationAxisAngle([0, 0, 1], float(np.arctan2(d["axis"][1], d["axis"][0]))) for d in doors], colors=[255, 200, 0],
                                                         labels=[f"doorway {d['width']:.1f} m wide" for d in doors], show_labels=True), static=self.static)
        self._doors = [{"id": 10_000 + i, "label": "doorway", "p_w": np.array([d["center"][0], d["center"][1], fz + 1.0]) @ R, "dims": np.array([d["width"], 0.15, 2.0])} for i, d in enumerate(doors)]
        st = g["stats"]
        self.dash.event(f"walkable {st['walkable_m2']:.1f} m2 ({st['inferred_m2']:.1f} m2 inferred), {len(doors)} doorways" + (f", guide to the {gd['target']} {gd['dist']:.1f} m, bearing {gd['bearing_deg']:+.0f} deg" if gd else ", no guide (not on safe floor)"))

    def egocentric(self, P, k, p, now):
        """Slow part (every 1-5 s): the objects of the map with persistent ids (same class within 0.4 m of last time). Where they are relative to you is redone every node in navigation()."""
        from mapper.semantics import instances
        if not hasattr(self, "_ego"):
            from mapper.egocentric import Egocentric
            self._ego, self._ego_tracks, self._ego_objs = Egocentric(), [], []
        objs = []
        for d in instances(P, k, p):
            t = next((t for t in self._ego_tracks if t["label"] == d["label"] and np.linalg.norm(t["p_w"] - d["p_w"]) < 0.4), None)
            if t is None:
                t = {"id": len(self._ego_tracks), "label": d["label"], "trail": []}
                self._ego_tracks.append(t)
            t["p_w"] = d["p_w"]
            t["trail"] = (t["trail"] + [d["p_w"].copy()])[-TRAIL_KEEP:]
            objs.append({**t, "dims": d["dims"]})
        self._ego_objs = objs

    def navigation(self, res, T_wc, up_w):
        """Every node: the walking guidance (cue, free-distance fan, obstacle points) and the egocentric objects against the current pose, in the you-centred view and as sentences."""
        from gabai.net.lexicon import message, say
        from mapper.egocentric import bucket, direction
        from mapper.navigate import HEADINGS
        from mapper.semantics import CLASSES, PALETTE
        rr.set_time("frame", sequence=self.dash._frame)
        free = res["free"]
        lines, cols = [], []
        if free is not None:
            for th, f_ in zip(HEADINGS, free):
                if f_ >= 0:
                    lines.append([(0, 0, 0), (f_ * np.cos(th), f_ * np.sin(th), 0)]); cols.append([60, 200, 90] if f_ >= 3.0 else [230, 160, 40] if f_ >= 1.2 else [230, 60, 60])
        rr.log("ego_view/free_paths", rr.LineStrips3D(lines, colors=cols, radii=0.012) if lines else rr.LineStrips3D([]))
        gr = res.get("grid")
        if gr is not None:                                                  # every 0.1 m cell the picture covers: floor, obstacle, overhead, drop, drawn on the floor plane
            ix, iy = np.nonzero(gr["kind"])
            k = gr["kind"][ix, iy]
            xy = np.c_[(ix + 0.5) * gr["cell"], (iy + 0.5) * gr["cell"] - gr["half"]]
            fz = res["floor_h"]
            rr.log("ego_view/ground", rr.Points3D(np.c_[xy, np.full(len(xy), fz)], colors=EGO_KIND[k], radii=gr["cell"] * 0.45))
        for name, pts, col in (("drops", res.get("drops"), [170, 60, 220]), ("overhead", res.get("overhead"), [255, 150, 0])):
            rr.log(f"ego_view/{name}", rr.Points3D(pts[::2], colors=col, radii=0.03) if pts is not None and len(pts) else rr.Points3D([]))
        obs = res["obstacles"]
        rr.log("ego_view/obstacle_points", rr.Points3D(obs[::2], colors=[230, 60, 60], radii=0.025) if obs is not None and len(obs) else rr.Points3D([]))
        if res["heading_deg"] is not None:
            th = np.radians(res["heading_deg"])
            rr.log("ego_view/steer", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[2.0 * np.cos(th), 2.0 * np.sin(th), 0]], colors=[40, 255, 120], radii=0.04))
        else:
            rr.log("ego_view/steer", rr.Arrows3D(origins=[], vectors=[]))
        msg, rows = {}, []
        if hasattr(self, "_ego") and (self._ego_objs or getattr(self, "_doors", [])):
            msg = self._ego.update(self.dash._frame, T_wc, self._ego_objs + getattr(self, "_doors", []), up_w)
            rows = self._ego.rows
        if rows:
            col = np.array([PALETTE[CLASSES.index(r["o"]["label"])] if r["o"]["label"] in CLASSES else [255, 200, 0] for r in rows], np.uint8)
            label = [f"#{r['o']['id']} {r['word']} {r['d']:.1f} m {r['dir']}" + (f" {r['state']}" if r["state"] else "") for r in rows]
            half = np.array([np.maximum(np.asarray(r["o"]["dims"], float), 0.05) / 2 for r in rows])
            rr.log("world/objects/boxes", rr.Boxes3D(centers=[r["o"]["p_w"] for r in rows], half_sizes=half, colors=col, labels=label, show_labels=True), static=self.static)
            strips, cols = [], []
            for t in self._ego_tracks:
                a, b = fading(t["trail"], (200, 200, 60), 60)
                strips += a
                cols += b
            rr.log("world/objects/trails", rr.LineStrips3D(strips, colors=cols, radii=0.004), static=self.static)
            centres = [r["p"] for r in rows]
            rr.log("ego_view/objects", rr.Boxes3D(centers=centres, half_sizes=half, colors=col, labels=label, show_labels=True))
            for r in rows:
                rr.log(f"range/{r['o']['id']}_{r['word']}", rr.Scalars(float(r["d"])))
        close = res["obstacles"]
        pk = []
        if close is not None and len(close):
            near = close[np.argmin(np.hypot(close[:, 0], close[:, 1]) + 5.0 * (np.abs(close[:, 1]) > 0.5))]
            d = float(np.hypot(near[0], near[1]))
            if d < 3.0:
                word = next((r["word"] for r in rows if r["d"] < d + 0.5 and abs(r["p"][0] - near[0]) < 1.0 and abs(r["p"][1] - near[1]) < 1.0), "obstacle")
                pk = [(word, bucket(d), direction(near[0], near[1], near[2], d))]
        cue = res["cue"]
        walk = getattr(self, "_walk", None)
        gd = walk["guide"] if walk else None
        gtext = ""
        if gd:
            b_ = gd["bearing_deg"]
            gcue = "go straight" if abs(b_) < 15 else ("move left" if b_ > 0 else "move right") if abs(b_) < 30 else ("turn left" if b_ > 0 else "turn right")
            gtext = f"guide: to the {gd['target']}, {gd['dist']:.1f} m, bearing {b_:+.0f} deg -> {gcue}"
            if cue in ("path is clear", "careful", "go straight"):
                cue = gcue                                                      # the reactive check only overrides when something is in the way
        doors_ = sorted((r for r in rows if r["word"] == "doorway"), key=lambda r: r["d"])
        if doors_:
            pk.append(("doorway", bucket(doors_[0]["d"]), doors_[0]["dir"]))
        wire = message(pk, cue=[cue] if cue in __import__("gabai.net.lexicon", fromlist=["IDS"]).IDS["cue"] else [])
        sentences = [say(q) for q in wire.get("d", [])]
        nlnl = chr(10) * 2
        text = f"# {cue.upper()}" + (f"  (heading {res['heading_deg']:+.0f} deg)" if res["heading_deg"] is not None else "") + nlnl + (gtext + nlnl if gtext else "") + chr(10).join(sentences + [say(q) for q in msg.get("d", [])][:4]) + nlnl + f"{len(str(wire).replace(' ', ''))} bytes to the phone: {str(wire).replace(' ', '')}   (navigation {res['ms']:.1f} ms)"
        rr.log("ego", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN))
        rr.log("timing/nav_ms", rr.Scalars(float(res["ms"])))

    def cloud(self, pm, force=False):
        """The semi-dense cloud under the newest poses. The rebuild walks every stored point (seconds with a thousand nodes), so it runs on its own thread, never twice at once, and no more often than
        every CLOUD_EVERY_S s or 10 x its own duration (it shares the GPU with the depth network, so a long rebuild is a stalled depth frame): the main loop (depth, RTAB-Map) never waits for it. force=True waits for a running build and builds once more here (end of the run)."""
        th = getattr(self, "_cloud_thread", None)
        if force:
            if th is not None:
                th.join()
            self._cloud_job(pm)
            return
        if (th is not None and th.is_alive()) or time.monotonic() - self._last_cloud < max(CLOUD_EVERY_S, 10 * getattr(self, "_cloud_s", 0.0)):
            return
        self._last_cloud = time.monotonic()
        self._cloud_thread = threading.Thread(target=self._cloud_job, args=(pm,), daemon=True)
        self._cloud_thread.start()

    def _cloud_job(self, pm):
        t0 = time.monotonic()
        P, C, cf, views = pm.cloud()
        self._cloud_s = time.monotonic() - t0
        self.cloud_ms = getattr(self, "cloud_ms", []) + [self._cloud_s * 1000.0]
        self._last_cloud = time.monotonic()
        rr.set_time("frame", sequence=self.dash._frame)                       # Rerun's time is per thread
        if len(P):
            rr.log("world/cloud", rr.Points3D(P, colors=C, radii=self.point_size), static=self.static)
        rr.log("cloud/points", rr.Scalars(float(len(P))))
        rr.log("timing/cloud_rebuild_ms", rr.Scalars(self._cloud_s * 1000.0))
