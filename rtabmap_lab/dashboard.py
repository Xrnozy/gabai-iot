"""The Rerun dashboard of the RTAB-Map experiment (its own app id, nothing shared with the GabAi pipeline's viewer).

Entity tree
  cameras/left, cameras/right      the rectified pair RTAB-Map is given (the picture the odometry and the depth work on)
  stereo/disparity                 RTAB-Map's own stereo matching, coloured
  world/camera                     the camera now (Transform3D + Pinhole), with world/camera/left in its frustum
  world/path                       the optimised graph: one point per map node, joined in order
  world/live                       where the camera has been since the start, with the map correction applied
  world/loops                      a line for every loop closure in the graph (red)
  world/map                        every node's points, placed with its OPTIMISED pose: after a loop closure the whole map is laid again
  world/surfels                    surfel fusion (rtabmap_lab/surfel.py): only what at least 2 pictures agree on, the big planes snapped flat (rtabmap_lab/planes.py)
  world/tsdf_mesh                  the Open3D TSDF surface fused from every node's dense depth + colour (rtabmap_lab/tsdf.py), built again after a loop closure
  odom/*  map/*  timing/*          time series
  status  events                   markdown status and a text log (loop closures, odometry restarts, warnings)

Coordinates. RTAB-Map's base frame is x forward, y left, z up (right handed), the same as Rerun's RIGHT_HAND_Z_UP, so poses go in unchanged. The camera is the base frame
turned by CameraModel::opticalRotation(), which is the same constant the bridge uses: a point P in the left optical frame is at  T_world_base @ OPTICAL @ P.
A level, upright camera therefore gives a z-up map; a camera that is upside down in the pictures it is given gives an upside-down one (use the pictures' 180 deg turn).
"""
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from rtabmap_lab.bridge_client import LOOP_LINKS

OPTICAL = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])      # CameraModel::opticalRotation(): camera optical frame in the base frame
VOXEL, MAP_MAX = 0.03, 400_000


def blueprint():
    fusion = rrb.Grid(*[rrb.Spatial2DView(origin=o, name=n) for o, n in (
        ("cameras/left", "left picture"), ("cameras/right", "right picture"), ("fusion/mono_left", "monocular depth, left"), ("fusion/mono_right", "monocular depth, right"),
        ("fusion/rtabmap", "RTAB-Map stereo depth"), ("fusion/las2", "LAS2 stereo depth"), ("fusion/ffs", "Fast-FoundationStereo depth"), ("fusion/fused", "fused depth (what the map gets)"),
        ("fusion/confidence", "confidence 0..1"), ("fusion/class", "green: measured, agreed. yellow: one source. blue: estimated. red: rejected"))], grid_columns=4, name="Depth fusion")
    nav = rrb.Horizontal(rrb.Spatial2DView(origin="nav/map", name="egocentric map: grey unknown, green free, red obstacle, blue keep-out band, yellow path"),
                         rrb.Vertical(rrb.TextDocumentView(origin="nav/status", name="navigation"), rrb.TimeSeriesView(origin="nav/stats", name="navigation numbers"), row_shares=[1, 2]), column_shares=[3, 2], name="Navigation")
    return rrb.Blueprint(rrb.Tabs(_map_layout(), fusion, nav), rrb.TimePanel(timeline="frame"))


def _map_layout():
    return (
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(origin="world", contents=["/world/tsdf_mesh/**", "/world/camera/**", "/world/live/**", "/world/path/**", "/world/nav_path/**", "/world/loops/**"], name="3D: TSDF surface, camera, path, navigation route, loop closures"),
                rrb.Spatial3DView(origin="world", contents=["/world/surfels/**", "/world/map/**", "/world/camera/**", "/world/live/**", "/world/path/**", "/world/loops/**", "/world/nodes/**"], name="3D: surfels confirmed by 2+ pictures, walls flattened"),
                row_shares=[3, 2]),
            rrb.Vertical(
                rrb.Horizontal(rrb.Spatial2DView(origin="cameras/left", name="left camera"), rrb.Spatial2DView(origin="cameras/right", name="right camera")),
                rrb.Horizontal(rrb.Spatial2DView(origin="stereo/disparity", name="RTAB-Map stereo disparity"), rrb.Spatial2DView(origin="stereo/fused_depth", name="GabAi depth (--depth gabai): LAS2 + 2 monocular")),
                rrb.TimeSeriesView(origin="odom", name="odometry: features, inliers, lost"),
                rrb.TimeSeriesView(origin="map", name="map: nodes, loop closures, distance"),
                rrb.TimeSeriesView(origin="timing", name="milliseconds per frame"),
                row_shares=[2, 2, 2, 2, 2]),
            rrb.Vertical(rrb.TextDocumentView(origin="status", name="status"), rrb.TextLogView(origin="events", name="events"), row_shares=[2, 3]),
            column_shares=[5, 4, 2], name="Map"))


class Dashboard:
    def __init__(self, cal, size, viewer=True, save=None, app="rtabmap_lab"):
        self.cal, self.size = cal, size
        self.nodes = {}                          # id -> dict(points (n,3) optical, bgr (n,3), pose 4x4 or None)
        self.poses = {}                          # id -> optimised 4x4 base pose
        self.live, self.t0, self.n_loops, self._last_map = [], None, 0, 0.0
        self.tsdf, self.upgrader, self._frame, self._t = None, None, 0, 0.0
        self.sinks, self.raw_points, self.flatten, self._last_surf = [], False, True, 0.0       # objects with add_node / set_poses: the TSDF and the surfel fusion
        self._dirty, self._world = set(), {}     # nodes whose pose changed; id -> (pose used, points in the world)
        if viewer or save:
            rr.init(app, spawn=bool(viewer))
            if save:
                rr.save(save)
            rr.send_blueprint(blueprint())
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr.log("world/camera", rr.Pinhole(focal_length=cal["f"], principal_point=(cal["cx"], cal["cy"]), resolution=size, camera_xyz=rr.ViewCoordinates.RDF), static=True)

    def at(self, idx, t):
        self.t0 = t if self.t0 is None else self.t0
        self._frame, self._t = int(idx), float(t - self.t0)
        rr.set_time("frame", sequence=self._frame)
        rr.set_time("t", duration=self._t)

    # -- per frame ---------------------------------------------------------------------------------------------------------------------------------
    def pair(self, left_bgr, right_bgr):
        rr.log("cameras/left", rr.Image(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=70))
        rr.log("cameras/right", rr.Image(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=70))

    def disparity(self, u8):
        """u8 = disparity x 4, 0 where the stereo matching found nothing."""
        col = cv2.applyColorMap(np.clip(u8.astype(np.int32) * 255 // 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[u8 == 0] = (40, 40, 40)
        rr.log("stereo/disparity", rr.Image(cv2.cvtColor(col, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=70))

    def status(self, st, left_bgr=None, skipped=0):
        if st["odom"] is not None:
            T = st["map_correction"] @ st["odom"]                                  # the camera's base pose in the map, with the loop-closure correction
            Tc = T @ OPTICAL
            rr.log("world/camera", rr.Transform3D(translation=Tc[:3, 3], mat3x3=Tc[:3, :3]))
            if left_bgr is not None:
                rr.log("world/camera/left", rr.Image(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=50))
            self.live.append(T[:3, 3].copy())
            if len(self.live) > 1:
                tr = np.asarray(self.live)
                if len(tr) > 2000:
                    tr = tr[np.linspace(0, len(tr) - 1, 2000).astype(int)]
                rr.log("world/live", rr.LineStrips3D([tr], colors=[255, 200, 0], radii=0.008))
        for path, v in (("odom/features", st["features"]), ("odom/inliers", st["inliers"]), ("odom/matches", st["matches"]), ("odom/lost", 1.0 if st["lost"] else 0.0),
                        ("map/nodes", st["nodes"]), ("map/loop_closures", st["loops"]), ("map/distance_m", st["distance"]), ("map/skipped_frames", skipped),
                        ("timing/odometry_ms", st["odom_ms"]), ("timing/rtabmap_ms", st["rtabmap_ms"]), ("timing/disparity_ms", st["disparity_ms"])):
            rr.log(path, rr.Scalars(float(v)))
        md = ["### RTAB-Map, stereo only", f"- odometry: **{'LOST' if st['lost'] else 'tracking'}**, {st['features']} features, {st['inliers']} inliers, {st['restarts']} restarts",
              f"- map: {st['nodes']} nodes, {st['loops']} loop closures, {st['distance']:.2f} m travelled",
              f"- this frame: odometry {st['odom_ms']:.0f} ms, map update {st['rtabmap_ms']:.0f} ms" + (", **node added**" if st["added"] else ""),
              f"- frames skipped because the bridge was busy: {skipped}"]
        rr.log("status", rr.TextDocument("\n".join(md), media_type=rr.MediaType.MARKDOWN))
        if st["loop_id"] > 0 or st["proximity_id"] > 0:
            self.n_loops += 1

    # -- the map -----------------------------------------------------------------------------------------------------------------------------------
    def node(self, m):
        if not self.raw_points:
            if m["pose"] is not None:
                self.poses[m["id"]] = m["pose"]
            return
        self.nodes[m["id"]] = {"points": m["points"], "bgr": m["bgr"]}
        if m["pose"] is not None:
            self.poses[m["id"]] = m["pose"]
        self._dirty.add(m["id"])

    def depth(self, m):
        """The bridge's dense depth of a node. With --depth gabai the upgrader takes it (and gives the TSDF the GabAi depth instead, a moment later)."""
        if self.upgrader is not None:
            self.upgrader.node(m)
        else:
            self.add_depth(m["id"], m["depth_mm"], m["bgr"])

    def add_depth(self, node_id, depth_mm, bgr, extra=None):
        """extra (from the fusion): {cls, conf, ...}. The TSDF weights a pixel by its class, the surfel map by its confidence."""
        cls, conf = (extra or {}).get("cls"), (extra or {}).get("conf")
        for sink in self.sinks:
            sink.add_node(node_id, depth_mm, bgr, cls=cls, conf=conf)

    def surfels(self, points, normals, colors, stats):
        """Called by the surfel worker thread after each round: the confirmed surfels, big planes snapped flat, thinned and sent (at most once a second)."""
        import time
        now = time.monotonic()
        if now - self._last_surf < 1.0 or not len(points):
            return
        self._last_surf = now
        rr.set_time("frame", sequence=self._frame)
        rr.set_time("t", duration=self._t)
        P, share = points, 0.0
        if self.flatten:
            from rtabmap_lab.planes import planarize
            P, used, planes = planarize(points, normals)
            share = float(used.mean())
        if len(P) > MAP_MAX:
            sel = np.linspace(0, len(P) - 1, MAP_MAX).astype(int)
            P, colors = P[sel], colors[sel]
        rr.log("world/surfels", rr.Points3D(P, colors=np.clip(colors[:, ::-1], 0, 255).astype(np.uint8), radii=rr.Radius.ui_points(2.0)))
        for path, v in (("map/surfels_confirmed", stats["stable"]), ("map/surfels_all", stats["surfels"]), ("map/surfels_flattened_share", share), ("timing/surfel_ms", stats["ms"])):
            rr.log(path, rr.Scalars(float(v)))

    def set_points(self, node_id, points, bgr):
        """Replace a node's points by better ones (the GabAi depth) and lay it again."""
        if node_id in self.nodes:
            self.nodes[node_id] = {"points": points, "bgr": bgr}
            self._dirty.add(node_id)

    @staticmethod
    def _depth_rgb(z, vmax=4.0):
        z = np.nan_to_num(np.asarray(z, np.float32), nan=0.0)
        col = cv2.applyColorMap(np.clip(z * 255.0 / vmax, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[z <= 0] = (40, 40, 40)
        return cv2.cvtColor(col, cv2.COLOR_BGR2RGB)

    def fused_depth(self, depth_mm, ms, extra=None):
        """The depth just computed for a node, and with the fusion every source next to the result (worker thread: Rerun's time is per thread)."""
        rr.set_time("frame", sequence=self._frame)
        rr.set_time("t", duration=self._t)
        z = depth_mm.astype(np.float32) / 1000.0
        rr.log("stereo/fused_depth", rr.Image(self._depth_rgb(z)).compress(jpeg_quality=70))
        rr.log("timing/gabai_depth_ms", rr.Scalars(float(ms)))
        if not extra:
            return
        for name, zz in extra["parts"].items():
            if name != "ms":
                rr.log(f"fusion/{name}", rr.Image(self._depth_rgb(zz)).compress(jpeg_quality=70))
        for name, v in extra["parts"]["ms"].items():
            rr.log(f"timing/depth_{name}_ms", rr.Scalars(float(v)))
        f = extra["fused"]
        rr.log("fusion/fused", rr.Image(self._depth_rgb(extra["z"])).compress(jpeg_quality=70))
        rr.log("fusion/confidence", rr.Image(self._depth_rgb(f.conf, 1.0)).compress(jpeg_quality=70))
        cmap = np.array([[40, 40, 40], [60, 200, 90], [240, 200, 40], [60, 190, 240]], np.uint8)          # none, measured and agreed, measured by one source, estimated
        cls = cmap[f.cls]
        cls[f.rejected] = (230, 60, 60)
        rr.log("fusion/class", rr.Image(cls).compress(jpeg_quality=80))
        rr.log("fusion/rejected", rr.Image(np.where(f.rejected[..., None], (230, 60, 60), (30, 30, 30)).astype(np.uint8)).compress(jpeg_quality=80))
        for k, v in f.info.items():
            if k.startswith("share_"):
                rr.log(f"fusion/{k}", rr.Scalars(float(v)))

    def mesh(self, mesh, stats):
        """Called by the TSDF worker thread with every finished mesh: logged at the frame the main loop is at (Rerun's time is per thread)."""
        rr.set_time("frame", sequence=self._frame)
        rr.set_time("t", duration=self._t)
        v, tri = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.uint32)
        if not len(v):
            return
        col = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None
        rr.log("world/tsdf_mesh", rr.Mesh3D(vertex_positions=v, triangle_indices=tri, vertex_normals=np.asarray(mesh.vertex_normals, np.float32) if mesh.has_vertex_normals() else None, vertex_colors=col))
        for path, key in (("map/tsdf_vertices", "vertices"), ("map/tsdf_rebuilds", "rebuilds"), ("timing/tsdf_mesh_ms", "mesh_ms"), ("timing/tsdf_integrate_ms", "integrate_ms")):
            rr.log(path, rr.Scalars(float(stats[key])))

    def nav(self, res, nm):
        """Called by NavSink with every navigation result (the main loop's thread)."""
        rr.set_time("frame", sequence=self._frame)
        rr.set_time("t", duration=self._t)
        rr.log("nav/map", rr.Image(nm.image(res)[:, :, ::-1]))
        if len(res["path_world"]) > 1:
            rr.log("world/nav_path", rr.LineStrips3D([res["path_world"]], colors=[(0, 230, 255)], radii=0.015))
        else:
            rr.log("world/nav_path", rr.Clear(recursive=False))
        ahead = res["nearest_obstacle_ahead_m"]
        n_path = len(res["path_cells"])
        route = "none" if not n_path else "%.1f m, of which %.1f m over ground not yet seen" % (n_path * nm.grid.cell, res["unseen_path_m"])
        steer = "-" if res["steer_deg"] is None else "%+.0f deg (+ = left)" % res["steer_deg"]
        lines = ["nearest obstacle in the walking corridor: " + ("none within 3 m" if ahead is None else "%.2f m" % ahead),
                 "unknown share of the corridor ahead: %.0f%%" % (res["unknown_ahead"] * 100),
                 "route: " + route,
                 "steer: " + steer,
                 "floor found in %d pictures, missing in %d" % (nm.stats["floor_ok"], nm.stats["floor_missing"])]
        rr.log("nav/status", rr.TextDocument("\n".join(lines)))
        rr.log("nav/stats/update_ms", rr.Scalars(res["ms"]))
        if ahead is not None:
            rr.log("nav/stats/obstacle_ahead_m", rr.Scalars(float(ahead)))

    def graph(self, m):
        if self.upgrader is not None:
            self.upgrader.posed(m["poses"])
        for sink in self.sinks:
            sink.set_poses(m["poses"])
        for i, T in m["poses"].items():
            if T is None:
                continue
            old = self.poses.get(i)
            if old is None or np.abs(old - T).max() > 1e-4:
                self.poses[i] = T
                self._dirty.add(i)
        ids = sorted(self.poses)
        if len(ids) > 1:
            rr.log("world/path", rr.LineStrips3D([np.array([self.poses[i][:3, 3] for i in ids])], colors=[80, 170, 255], radii=0.012))
        rr.log("world/nodes", rr.Points3D(np.array([self.poses[i][:3, 3] for i in ids]), colors=[255, 255, 255], radii=0.02)) if ids else None
        loops = [(a, b) for a, b, t in m["links"] if t in LOOP_LINKS and a in self.poses and b in self.poses]
        if loops:
            rr.log("world/loops", rr.LineStrips3D([np.array([self.poses[a][:3, 3], self.poses[b][:3, 3]]) for a, b in loops], colors=[255, 60, 60], radii=0.02))
        self.log_map()

    def log_map(self, force=False, now=None):
        """Lay every node's points with its latest pose (only the nodes whose pose changed are transformed again), thin them on a voxel grid, send."""
        import time
        now = time.monotonic() if now is None else now
        if (not self._dirty and not force) or not self.raw_points:
            return
        if not force and now - self._last_map < 1.0:
            return
        for i in list(self._dirty):
            n, T = self.nodes.get(i), self.poses.get(i)
            if n is None or T is None:
                continue
            Tw = T @ OPTICAL
            self._world[i] = (n["points"].astype(np.float64) @ Tw[:3, :3].T + Tw[:3, 3]).astype(np.float32)
        self._dirty.clear()
        self._last_map = now
        if not self._world:
            return
        ids = sorted(self._world)
        pts = np.concatenate([self._world[i] for i in ids])
        col = np.concatenate([self.nodes[i]["bgr"][:, ::-1] for i in ids])
        key = np.floor(pts / VOXEL).astype(np.int64)
        _, first = np.unique(key[:, 0] * 73856093 ^ key[:, 1] * 19349663 ^ key[:, 2] * 83492791, return_index=True)
        pts, col = pts[first], col[first]
        if len(pts) > MAP_MAX:
            sel = np.linspace(0, len(pts) - 1, MAP_MAX).astype(int)
            pts, col = pts[sel], col[sel]
        rr.log("world/map", rr.Points3D(pts, colors=col, radii=rr.Radius.ui_points(1.4)))

    def event(self, text, level="INFO"):
        rr.log("events", rr.TextLog(text, level=level))
