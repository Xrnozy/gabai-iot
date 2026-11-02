"""Rerun layout for the pipeline. One place defines every entity path, so both pose modes and every source log to the same panels.

Entity tree (ARKitScenes / RGB-D style: images hang under the camera so the 3D view shows them in the camera frustum):

  world                         z up, metres (metric only when the calibration is real; see diag/status)
  world/camera                  Transform3D + Pinhole      the tracked camera
  world/camera/left             the left picture in the frustum
  world/trajectory              the path so far
  world/live                    this frame's semi-dense stereo points, placed with the pose
  world/map                     all frames fused (5 cm voxels, thin dense layer kept)
  world/hands/<id>, world/objects/<key>   3D hands (keypoints + bones) and object boxes, coloured by relation (held / contact / near ...)
  cameras/left, cameras/right   the two raw pictures, same frame
  stereo/disparity, stereo/depth, stereo/valid, stereo/correspondences
  annotated/left                picture with hand and object boxes, labels, scores and relations
  tracking/features             picture with the tracked feature points and their motion
  klt/*                         the Pi's KLT tracks: points, ids, trails, motion, lost points, counts (gabai/pipeline/viz_klt.py)
  filter/*                      the Kalman filter: innovations, gyro and accelerometer bias, uncertainty, refused measurements
  sync/*                        what the server received and released per frame: frame and KLT numbers, IMU interval, queue sizes, latency, losses, recovery
  imu/*  tracking/*  depth/*  sync/*   time series
  diag/status                   markdown: mode, every check, what is assumed
  diag/log                      text events (tracking lost, grab, warnings)

Time: every log call uses two timelines, "frame" (sequence number) and "t" (seconds since the first frame). A delayed pose (Basalt) is logged
at the time of the frame it belongs to, so the camera, the points and the pictures of one frame line up when you scrub.
"""
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

FINGER_CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
RELATION_COLOR = {"held": (255, 60, 60), "held_unconfirmed": (255, 140, 0), "contact": (255, 200, 0), "overlap_not_touching": (200, 200, 120),
                  "near": (120, 200, 255), "uncertain": (180, 120, 255), "apart": (150, 150, 150)}
OBJECT_COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128)]
STATUS_COLOR = {"pass": "green", "fail": "red", "suspect": "orange", "unknown": "gray"}


def blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(origin="world", contents=["/world/**", "- /world/tsdf_mesh/**", "- /world/pov_mesh/**"], name="3D: map, camera, trajectory, hands, objects"),
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="world", contents=["/world/tsdf_mesh/**", "/world/camera/**", "/world/trajectory/**", "/world/boxer/landmarks/**"],
                                      name="TSDF mesh: the fused surface, coloured (map.tsdf)"),
                    rrb.Spatial3DView(origin="world", contents=["/world/map/**", "/world/camera/**", "/world/trajectory/**", "/world/boxer/landmarks/**"],
                                      name="final map: every frame fused, laid again after a correction")),
                rrb.Spatial3DView(origin="world", contents=["/world/tsdf_mesh/**", "/world/map/**", "/world/boxer/**", "/world/pov_mesh/**"],
                                  eye_controls=rrb.EyeControls3D(tracking_entity="/world/camera"),
                                  name="camera POV: the reconstruction seen from the camera (the view follows it)"),
                row_shares=[3, 2, 2]),
            rrb.Vertical(
                rrb.Horizontal(rrb.Spatial2DView(origin="cameras/left", name="left camera"), rrb.Spatial2DView(origin="cameras/right", name="right camera")),
                rrb.Horizontal(rrb.Spatial2DView(origin="stereo/disparity", name="disparity"), rrb.Spatial2DView(origin="stereo/depth", name="depth")),
                rrb.Horizontal(rrb.Spatial2DView(origin="fusion/mono", name="monocular depth, left"), rrb.Spatial2DView(origin="fusion/mono_right", name="monocular depth, right"),
                               rrb.Spatial2DView(origin="fusion/fused", name="fused depth (stereo + monocular in the holes)")),
                rrb.Horizontal(rrb.Spatial2DView(origin="stereo/valid", name="valid / rejected"), rrb.Spatial2DView(origin="stereo/correspondences", name="stereo matches")),
                rrb.Horizontal(rrb.Spatial2DView(origin="annotated/left", name="hands and objects"), rrb.Spatial2DView(origin="tracking/features", name="tracked features")),
                rrb.Horizontal(rrb.Spatial2DView(origin="klt/left", name="Pi KLT tracks, left: ids, trails, motion, lost"), rrb.Spatial2DView(origin="klt/right", name="Pi KLT tracks, right"),
                               rrb.TimeSeriesView(origin="klt/count", name="KLT counts and Pi tracking time")),
                row_shares=[1, 1, 1, 1, 1, 1]),
            rrb.Vertical(
                rrb.TextDocumentView(origin="diag/status", name="status and checks"),
                rrb.TimeSeriesView(origin="imu", name="IMU and visual-inertial agreement"),
                rrb.TimeSeriesView(origin="tracking", name="tracking and map alignment"),
                rrb.TimeSeriesView(origin="depth", name="depth quality"),
                rrb.TimeSeriesView(origin="sync", name="synchronisation: frame / KLT / IMU, queues, latency, losses"),
                rrb.TimeSeriesView(origin="filter", name="Kalman filter: innovations, biases, uncertainty, refusals"),
                rrb.TextLogView(origin="diag/log", name="events"),
                row_shares=[2, 2, 2, 2, 2, 2, 1]),
            column_shares=[3, 3, 2]),
        rrb.TimePanel(timeline="frame"))


class Viz:
    """Logs to Rerun. Subclass and override _log to capture instead (tests)."""
    scale = 0.5
    map_max_points = 250000

    def __init__(self, cal, size, save=None, viewer=True, app="gabai_pipeline", scale=0.5, map_max_points=250000):
        self.cal, self.size = cal, size
        self.scale, self.map_max_points = float(scale), int(map_max_points)
        self.t0 = None
        self.n_logged = 0
        self._held_paths = set()
        from gabai.pipeline.viz_klt import KltPanel
        self.klt_panel = KltPanel(self._log, self._clear, lambda p, v: self.scalar(p, v), self.scale)
        self._hand_paths, self._obj_paths = set(), set()      # what is currently drawn, so that what is no longer tracked can be removed
        if viewer or save:
            rr.init(app, spawn=bool(viewer))
            if save:
                rr.save(save)
            rr.send_blueprint(blueprint())
            self._static()

    # -- low level ----------------------------------------------------------------------------------------------------------------
    def _log(self, path, arch, **kw):
        self.n_logged += 1
        rr.log(path, arch, **kw)

    def _clear(self, path, recursive=True):
        """Remove an entity from the scene. Rerun keeps the last value logged at a path until it is cleared, so anything that stops being tracked MUST be cleared
        or it stays behind as a stale copy."""
        self._log(path, rr.Clear(recursive=recursive))

    def reset_map(self):
        """The fused map was dropped: take the old one out of the scene so it is not left standing beside the new one."""
        self._clear("world/map")
        self._clear("world/live")

    def clear_tracked(self):
        """Remove every hand, hand-attached object and object position from the scene (the analysis is off, or this frame's pose cannot be trusted).
        Objects that were put down and frozen in the world stay: they are not tracked, they are a result."""
        for hid in self._hand_paths:
            self._clear(f"world/hands/{hid}", True)
        for key in self._obj_paths:
            self._clear(f"world/objects/{key}", False)
        for path in self._held_paths:
            self._clear(path, True)
        for path in ("annotated/left/objects", "annotated/left/hands", "annotated/left/keypoints", "annotated/left/bones"):
            self._clear(path, False)
        self._hand_paths, self._obj_paths, self._held_paths = set(), set(), set()

    def at(self, idx, t):
        self.t0 = t if self.t0 is None else self.t0
        rr.set_time("frame", sequence=int(idx))
        rr.set_time("t", duration=float(t - self.t0))

    def _static(self):
        self._log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        c = self.cal
        self._log("world/camera", rr.Pinhole(focal_length=c["f"], principal_point=(c["cx"], c["cy"]), resolution=self.size, camera_xyz=rr.ViewCoordinates.RDF), static=True)

    # -- per frame ----------------------------------------------------------------------------------------------------------------
    def stereo(self, left_bgr, right_bgr, disp, trusted, cal, corr_pairs=None):
        """The raw pair at full size (JPEG), and the disparity / depth / valid panels at `scale` of the size, compressed. Depth goes out as a 16-bit PNG in
        millimetres, about 20 times smaller than raw float32, which is what made the Rerun viewer slow."""
        rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        self._log("cameras/left", rr.Image(rgb(left_bgr)).compress(jpeg_quality=70))
        self._log("cameras/right", rr.Image(rgb(right_bgr)).compress(jpeg_quality=70))
        sc = self.scale
        small = (lambda x, interp: cv2.resize(x, None, fx=sc, fy=sc, interpolation=interp)) if sc < 1.0 else (lambda x, interp: x)
        valid = np.isfinite(disp) & (disp > 0.5)
        d = np.where(valid, disp, 0.0).astype(np.float32)
        top = max(float(np.percentile(d[valid], 98)), 1.0) if valid.any() else 1.0
        dv = cv2.applyColorMap(np.clip(small(d, cv2.INTER_NEAREST) / top * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        dv[~small(valid.astype(np.uint8), cv2.INTER_NEAREST).astype(bool)] = (40, 40, 40)
        self._log("stereo/disparity", rr.Image(rgb(dv)).compress(jpeg_quality=70))
        z_mm = np.where(valid, np.clip(cal["f"] * cal["B"] / np.where(valid, disp, 1.0) * 1000.0, 1, 65000), 0).astype(np.uint16)
        ok, png = cv2.imencode(".png", small(z_mm, cv2.INTER_NEAREST))
        if ok:
            self._log("stereo/depth", rr.EncodedDepthImage(png.tobytes(), media_type="image/png", meter=1000.0, colormap="turbo", depth_range=[200.0, 6000.0]))
        v8, t8 = small(valid.astype(np.uint8), cv2.INTER_NEAREST).astype(bool), small(np.asarray(trusted, np.uint8), cv2.INTER_NEAREST).astype(bool)
        flag = np.full(v8.shape + (3,), 50, np.uint8)                     # no data
        flag[v8 & t8] = (60, 200, 60)                                     # measured and left-right consistent
        flag[v8 & ~t8] = (230, 60, 60)                                    # produced by the model but rejected by the left-right check
        self._log("stereo/valid", rr.Image(flag).compress(jpeg_quality=70))
        if corr_pairs is not None:
            h, w = left_bgr.shape[:2]
            both = np.hstack([left_bgr, right_bgr]).copy()
            for (ul, vl), (ur, vr) in corr_pairs:
                cv2.line(both, (int(ul), int(vl)), (int(ur) + w, int(vr)), (0, 255, 255), 1)
                cv2.circle(both, (int(ul), int(vl)), 2, (0, 255, 0), -1)
            self._log("stereo/correspondences", rr.Image(rgb(small(both, cv2.INTER_AREA))).compress(jpeg_quality=65))

    def rejects(self, labels):
        """The cause of every missing depth pixel, in the 'valid / rejected' panel (colours: gabai.depthq.LABELS)."""
        from gabai.depthq import paint
        sc = self.scale
        img = paint(labels if sc >= 1.0 else cv2.resize(labels, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST))
        self._log("stereo/valid", rr.Image(img).compress(jpeg_quality=70))

    def _depth_png(self, path, disp, cal):
        """A disparity map as the same 16-bit millimetre PNG depth image the stereo panel uses."""
        valid = np.isfinite(disp) & (disp > 0.5)
        self._z_png(path, np.where(valid, cal["f"] * cal["B"] / np.where(valid, disp, 1.0), 0.0))

    def _z_png(self, path, z_m):
        """A depth map in metres (0 = unknown) as a 16-bit millimetre PNG."""
        z_mm = np.where(np.isfinite(z_m) & (z_m > 0), np.clip(z_m * 1000.0, 1, 65000), 0).astype(np.uint16)
        if self.scale < 1.0:
            z_mm = cv2.resize(z_mm, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        ok, png = cv2.imencode(".png", z_mm)
        if ok:
            self._log(path, rr.EncodedDepthImage(png.tobytes(), media_type="image/png", meter=1000.0, colormap="turbo", depth_range=[200.0, 6000.0]))

    def fusion(self, fused, mono, cal, mono_m=None, mono_r=None):
        """The monocular model's own depth (metres, every pixel, whether or not it could be fitted to stereo) next to the fused depth (stereo, plus the monocular
        pixels in the holes, only on real surfaces). `mono` = (aligned disparity, fit info) or None; its numbers go to the plots under fusion/."""
        info = mono[1] if mono is not None else {}
        shown = self.__dict__.setdefault("_shown", set())                      # fusion panels currently drawn
        for path, z in (("fusion/mono", mono_m), ("fusion/mono_right", mono_r)):
            if z is not None:
                self._z_png(path, z)
                shown.add(path)
            elif path in shown:                                          # drawn before and now gone: take it out; no call at all when it never was drawn
                self._clear(path)
                shown.discard(path)
        if "pair_agree" in info:
            self.scalar("fusion/left_right_mono_agree", info["pair_agree"])
        if "resid_px" in info:
            self.scalar("fusion/fit_residual_px", info["resid_px"])
            self.scalar("fusion/fit_correlation", info["corr"])
        if mono is not None:
            self.scalar("fusion/accepted", 1.0 if info.get("ok") else 0.0)
            self.scalar("fusion/filled_px", float(info.get("filled_px", 0)))
        self._depth_png("fusion/fused", fused, cal)

    def pov_mesh(self, left_bgr, disp, R_wc, t_wc, cal, step=4):
        """Textured surface of what the camera sees now (gabai/povmesh.py), placed with the current pose in its own entity. The world map is not touched."""
        from gabai.povmesh import build
        m = build(disp, cal, step=step)
        if m is None:
            self._clear("world/pov_mesh")
            return
        v, tris, uv = m
        R32 = np.asarray(R_wc, np.float32)
        v_w = v @ R32.T + np.asarray(t_wc, np.float32)
        self._log("world/pov_mesh", rr.Mesh3D(vertex_positions=v_w, triangle_indices=tris, vertex_texcoords=uv,
                                              albedo_texture=cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)))

    def tsdf_mesh(self, v, tris, colors):
        """The TSDF surface (gabai/tsdf.py) in its own entity; the voxel map is not touched."""
        self._log("world/tsdf_mesh", rr.Mesh3D(vertex_positions=v, triangle_indices=tris, vertex_colors=colors))

    def tsdf_clear(self):
        self._clear("world/tsdf_mesh")

    def objects3d(self, snap):
        """Boxer boxes (gabai/objects3d.py): yellow = this frame, coloured = landmarks seen often enough, dots = every look of a landmark (a smear = drift or a moved object)."""
        from scipy.spatial.transform import Rotation
        dets, marks = snap["current"], snap["landmarks"]
        if dets:
            q = Rotation.from_matrix(np.array([d["R_obj"] for d in dets])).as_quat()
            self._log("world/boxer/current", rr.Boxes3D(centers=[d["p_w"] for d in dets], half_sizes=[d["dims"] / 2 for d in dets], quaternions=[rr.Quaternion(xyzw=x) for x in q],
                                                        labels=[f"{d['label']} {d['conf']:.2f}" for d in dets], colors=[255, 220, 0], radii=0.004))
        else:
            self._clear("world/boxer/current")
        if marks:
            q = Rotation.from_matrix(np.array([m["R"] for m in marks])).as_quat()
            col = [OBJECT_COLORS[m["id"] % len(OBJECT_COLORS)] for m in marks]
            self._log("world/boxer/landmarks", rr.Boxes3D(centers=[m["center"] for m in marks], half_sizes=[m["dims"] / 2 for m in marks], quaternions=[rr.Quaternion(xyzw=x) for x in q],
                                                          labels=[f"#{m['id']} {m['label']} {m['conf']:.0%}  {m['looks']} looks  {m['sigma_cm']:.0f} cm   [{m['text']}]" for m in marks], colors=col, radii=0.006))
            self._log("world/boxer/looks", rr.Points3D(np.concatenate([m["recent"] for m in marks]), colors=np.concatenate([[col[i]] * len(m["recent"]) for i, m in enumerate(marks)]),
                                                       radii=rr.Radius.ui_points(2.5)))

    def objects3d_clear(self):
        self._clear("world/boxer")

    def pov_clear(self):
        self._clear("world/pov_mesh")

    def pose(self, R_wc, t_wc, trajectory, left_bgr=None):
        self._log("world/camera", rr.Transform3D(translation=np.asarray(t_wc, float), mat3x3=np.asarray(R_wc, float)))
        if left_bgr is not None:
            self._log("world/camera/left", rr.Image(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=60))
        if trajectory is not None and len(trajectory) > 1:
            tr = np.asarray(trajectory)
            if len(tr) > 1500:                                            # a long path is drawn thinned, so the amount sent per frame stays bounded
                tr = tr[np.linspace(0, len(tr) - 1, 1500).astype(int)]
            self._log("world/trajectory", rr.LineStrips3D([tr], colors=[255, 200, 0], radii=0.01))

    def cloud(self, live_points=None, live_colors=None, map_points=None):
        """Points are drawn a fixed number of screen pixels wide, not a fixed size in metres, so they stay fine and sharp however far the view is zoomed."""
        if live_points is not None and len(live_points):
            self._log("world/live", rr.Points3D(live_points, colors=live_colors, radii=rr.Radius.ui_points(1.2)))
        if map_points is not None:
            if len(map_points) > self.map_max_points:                       # an even stride through the voxel order thins the map the same way everywhere and the same way next time (a random draw shimmered)
                map_points = map_points[::int(np.ceil(len(map_points) / self.map_max_points))]
            self._log("world/map", rr.Points3D(map_points, radii=rr.Radius.ui_points(1.6), colors=[110, 170, 255]))

    def log(self, t, R_wc=None, t_wc=None, live_points=None, live_colors=None, trajectory=None, map_points=None, **_):
        """Same call SlamCloud makes, so it can drive this sink directly. The timeline position is set by at() before the call."""
        self.pose(R_wc, t_wc, trajectory)
        self.cloud(live_points, live_colors, map_points)

    def annotate(self, left_bgr, held):
        rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        self._log("annotated/left", rr.Image(rgb).compress(jpeg_quality=75))
        rels = {}
        for r in held["relations"]:
            for key in (r["obj"], r["hand"]):
                if key not in rels or _rank(r["relation"]) > _rank(rels[key]["relation"]):
                    rels[key] = r
        if held["objects"]:
            boxes = [o["box"] for o in held["objects"]]
            labels = [f"{o['key']} {o['conf']:.2f} [{rels.get(o['key'], {}).get('relation', 'no hand')}]" + ("" if o["center_cam"] is not None else " (no depth)") + (f" [stale {o['stale_frames']}f]" if o.get("stale_frames") else "") for o in held["objects"]]
            colors = [RELATION_COLOR.get(rels.get(o["key"], {}).get("relation"), (90, 220, 90)) for o in held["objects"]]
            self._log("annotated/left/objects", rr.Boxes2D(array=boxes, array_format=rr.Box2DFormat.XYXY, labels=labels, colors=colors))
        else:
            self._clear("annotated/left/objects", False)
        if held["hands"]:
            self._log("annotated/left/hands", rr.Boxes2D(array=[h["box"] for h in held["hands"]], array_format=rr.Box2DFormat.XYXY,
                                                         labels=[f"hand {h['id']} {h['state']} (score {h['side_score']:.2f}); 3D: " + (f"{h['solve']['n_stereo']} joints stereo, {h['solve']['n_model']} model" if h['solve']['ok'] else 'refused: ' + h['solve']['reason'][:60]) for h in held["hands"]],
                                                         colors=[(255, 90, 90) if h["state"] == "closed" else (90, 255, 140) for h in held["hands"]]))
            from gabai.hands import HAND_CONNECTIONS
            self._log("annotated/left/keypoints", rr.Points2D(np.vstack([h["kp2d"] for h in held["hands"]]), radii=2.5, colors=[255, 255, 0]))
            self._log("annotated/left/bones", rr.LineStrips2D([h["kp2d"][[a, b]] for h in held["hands"] for a, b in HAND_CONNECTIONS], colors=[255, 255, 0]))
        else:
            for path in ("annotated/left/hands", "annotated/left/keypoints", "annotated/left/bones"):
                self._clear(path, False)
        self.hands3d(held, rels)

    def hands3d(self, held, rels):
        """3D hand (joints coloured by where each depth came from, bones, palm surface), its frame, and the held object drawn INSIDE that frame, so the two
        move together by construction: the object's points are logged in hand coordinates under the hand-frame transform."""
        logged = set()
        present = {h["id"] for h in held["hands"]}
        for stale in self._hand_paths - present:                     # a hand that is no longer tracked: remove it and everything under it (bones, palm, frame, held object)
            self._clear(f"world/hands/{stale}", True)
        drawn = set()
        for h in held["hands"]:
            base = f"world/hands/{h['id']}"
            X = h["xyz_world"]
            ok = np.isfinite(X[:, 2])
            if not ok.any():
                self._clear(base, True)
                continue
            drawn.add(h["id"])
            cols = np.array([_src_color(s) for s in h["source"]], np.uint8)
            self._log(base, rr.Points3D(X[ok], radii=0.006, colors=cols[ok]))
            segs = []
            for chain in FINGER_CHAINS:
                for a_, b_ in zip(chain[:-1], chain[1:]):
                    if ok[a_] and ok[b_]:
                        segs.append(X[[a_, b_]])
            if segs:
                self._log(base + "/bones", rr.LineStrips3D(segs, colors=[(235, 190, 150)], radii=0.0045))
            else:
                self._clear(base + "/bones", False)
            palm = [0, 1, 5, 9, 13, 17]
            if ok[palm].all():
                self._log(base + "/palm", rr.Mesh3D(vertex_positions=X[palm], triangle_indices=[[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]], vertex_colors=[(235, 190, 150)] * 6))
            else:
                self._clear(base + "/palm", False)
            if h["frame"] is not None:
                Rh, th = h["frame"]
                self._log(base + "/frame", rr.Transform3D(translation=th, mat3x3=Rh))
                self._log(base + "/frame/axes", rr.Arrows3D(origins=[[0, 0, 0]] * 3, vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]], colors=[(255, 60, 60), (60, 220, 60), (60, 120, 255)]))
            else:
                self._clear(base + "/frame", True)                    # no frame this time: the object that hangs under it must not stay at the old one
        self._hand_paths = drawn
        for m in held.get("models", []):
            if m["frame"] is None or not m.get("n_points"):
                continue
            path = f"world/hands/{m['hand']}/frame/held/{m['key']}"
            seen = m["state"].startswith("observed")
            col = (60, 200, 255) if seen else (255, 150, 40)
            self._log(path, rr.Points3D(m["points_local"], radii=0.005, colors=col))
            logged.add(path)
            b = m["box_local"]
            if b is not None:
                from scipy.spatial.transform import Rotation as Rot
                self._log(path + "/box", rr.Boxes3D(centers=[b["center"]], half_sizes=[b["half_sizes"]], quaternions=[rr.Quaternion(xyzw=Rot.from_matrix(b["R"]).as_quat())],
                                                    labels=[f"{m['label']} [{m['state']}]"], colors=[col]))
        for r in held.get("released", []):
            base = f"world/objects/{r['key']}"
            col = (255, 150, 40) if r["unobserved"] else (120, 255, 120)
            self._log(base + "/model", rr.Points3D(r["points_world"], radii=0.005, colors=col))
            if r["box_world"] is not None:
                from scipy.spatial.transform import Rotation as Rot
                b = r["box_world"]
                self._log(base + "/box", rr.Boxes3D(centers=[b["center"]], half_sizes=[b["half_sizes"]], quaternions=[rr.Quaternion(xyzw=Rot.from_matrix(b["R"]).as_quat())],
                                                    labels=[f"{r['label']} " + ("(unobserved)" if r["unobserved"] else "(put down here)")], colors=[col]))
        for path in self._held_paths - logged:                       # no longer held: remove the hand-attached copy
            self._log(path, rr.Clear(recursive=True))
        self._held_paths = logged
        held_keys = {m["key"] for m in held.get("models", [])}
        shown = set()
        for o in held["objects"]:
            if o["center_world"] is None or o["key"] in held_keys:
                continue
            rel = rels.get(o["key"], {}).get("relation", "no hand")
            shown.add(o["key"])
            self._log(f"world/objects/{o['key']}", rr.Points3D([o["center_world"]], radii=0.03, labels=[f"{o['key']} [{rel}]"], colors=[RELATION_COLOR.get(rel, (90, 220, 90))]))
        for key in (self._obj_paths - shown) | held_keys:             # gone from view, or now held (drawn inside the hand instead): remove the old position
            self._clear(f"world/objects/{key}", False)                  # not recursive: a frozen object put down earlier (.../model) stays
        self._obj_paths = shown

    def features(self, left_bgr, pts=None, tracks=None):
        sc = self.scale
        img = cv2.resize(left_bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1.0 else left_bgr
        self._log("tracking/features", rr.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=65))
        if pts is not None and len(pts):
            self._log("tracking/features/points", rr.Points2D(np.asarray(pts, float).reshape(-1, 2) * sc, radii=2.0, colors=[0, 255, 0]))
        if tracks is not None and len(tracks):
            self._log("tracking/features/motion", rr.LineStrips2D([np.asarray(t, float) * sc for t in tracks], colors=[0, 220, 255]))

    def klt(self, left_bgr, obs, right_bgr=None):
        """The Pi's tracks over this frame's left (and right) picture (obs = gabai.kltfuse.KltObs or None)."""
        self.klt_panel.draw(left_bgr, obs, right_bgr)

    def sync(self, info, pose_lag_frames=None):
        from gabai.pipeline.viz_klt import log_sync
        log_sync(self.scalar, self.event, info, pose_lag_frames)

    def scalar(self, path, v):
        if v is not None and np.isfinite(v):
            self._log(path, rr.Scalars(float(v)))

    def status(self, mode, lines, checks):
        md = [f"### mode: `{mode}`"] + lines + ["", "| check | status | detail |", "|---|---|---|"]
        md += [f"| {c.name} | {c.status.upper()} | {c.detail} |" for c in checks]
        bad = [c for c in checks if c.status in ("fail", "suspect")]
        if bad:
            md += ["", "**Look at:**"] + [f"- **{c.name}** ({c.status}): {c.hint}" for c in bad if c.hint]
        self._log("diag/status", rr.TextDocument("\n".join(md), media_type=rr.MediaType.MARKDOWN))

    def event(self, text, level="INFO"):
        self._log("diag/log", rr.TextLog(text, level=level))


def _src_color(src):
    src = str(src)
    return (80, 220, 120) if src == "stereo" else ((255, 170, 60) if src.startswith("model") else (160, 160, 160))


def _rank(rel):
    order = ["apart", "near", "overlap_not_touching", "uncertain", "contact", "held_unconfirmed", "held"]
    return order.index(rel) if rel in order else -1
