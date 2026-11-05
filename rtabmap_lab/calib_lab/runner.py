"""One RTAB-Map run over a clip of a recording with ONE fixed calibration: the unit of every experiment.

A Run owns a fresh RTAB-Map bridge process and a fresh TSDF volume, both created from the calibration it is given and from nothing else. When the calibration changes, the lab stops the
run and starts another one: a map built with other numbers is never reused (the odometry, the poses and every depth in the volume depend on f, cx, cy and the baseline). The recording is
only read. Frames go in order, one at a time, the next only after RTAB-Map answered the last, exactly like  live.py replay.

emit(kind, payload) is called from the run's thread:  frame (idx, left, right) / camera (pose, status) / disparity (u8) / mesh (arrays) / progress / done (metrics).
"""
import argparse
import os
import threading
import time

import cv2
import numpy as np

from rtabmap_lab import flatness, live as livemod
from rtabmap_lab.bridge_client import Bridge
from rtabmap_lab.tools.pose_check import consistency
from rtabmap_lab.tsdf import OPTICAL


def default_settings():
    """live.py's own defaults for everything the run does not take from the calibration (so a lab run and `live.py replay` are configured identically)."""
    ns = livemod.add_args(argparse.ArgumentParser()).parse_args([])
    keys = ("cloud_stride", "min_range", "max_range", "disparity_every", "lost_reset", "max_depth_std", "tsdf_voxel", "tsdf_trunc", "tsdf_backend", "config")
    return {k: getattr(ns, k) for k in keys} | {"gyro": False, "shutter": False}       # pictures only by default; the gyro features are opt-in


class Run:
    def __init__(self, rec, params, clip, settings=None, emit=None, port=5710):
        self.rec, self.params, self.settings = rec, params.copy(), dict(settings or default_settings())
        self.clip = (int(clip[0]), int(min(clip[1], rec.n)))
        self.emit, self.port = emit or (lambda *a: None), port
        self.fingerprint = params.fingerprint()
        self.pause, self.stop_flag, self.thread = threading.Event(), False, None
        self.pos, self.state, self.error = self.clip[0], "starting", ""
        self.samples = {"row_err_px": [], "valid_share": [], "sigma_d_px": []}      # filled by the preview worker from the frames this run shows
        self.metrics, self.mesh, self.cameras, self.t0 = {}, None, [], None
        self.tsdf = self.bridge = None
        self.node_pair = {}                                                       # RTAB-Map node id -> the pair index it was made from
        self._last_mesh_emit = 0.0

    # -- control ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.thread = threading.Thread(target=self._guard, daemon=True)
        self.thread.start()
        return self

    def stop(self, wait=True):
        self.stop_flag = True
        self.pause.clear()
        if wait and self.thread:
            self.thread.join(timeout=90)

    def set_paused(self, on):
        if on:
            self.pause.set()
        else:
            self.pause.clear()

    def alive(self):
        return bool(self.thread and self.thread.is_alive())

    def _guard(self):
        try:
            self._run()
        except Exception as e:                                                   # the lab must survive a failed run; the reason goes to the page
            self.state, self.error = "failed", f"{type(e).__name__}: {e}"
            self.emit("done", {"failed": self.error})
        finally:
            for closer in (lambda: self.tsdf and self.tsdf.close(), lambda: self.bridge and self.bridge.close(save=False)):
                try:
                    closer()
                except Exception:
                    pass

    # -- the run -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _run(self):
        s, rec = self.settings, self.rec
        rect = self.params.rect()
        cal = rect.camera(flip=rec.flip, k=rec.k)
        w, h = cal["size"]
        self.cal = cal
        self.bridge = bridge = Bridge(db="", port=self.port, config=s["config"])
        bridge.init(cal["f"], cal["f"], cal["cx"], cal["cy"], cal["B"], w, h, s["cloud_stride"], s["min_range"], s["max_range"], s["disparity_every"], s["lost_reset"], s["max_depth_std"])
        K = (cal["f"], cal["f"], cal["cx"], cal["cy"])
        if s["tsdf_backend"] == "gpu":
            from rtabmap_lab.gpu_tsdf import GpuTsdfMap
            self.tsdf = GpuTsdfMap(K, (w, h), voxel=s["tsdf_voxel"], trunc=s["tsdf_trunc"], max_range=s["max_range"], fb=cal["f"] * cal["B"], on_mesh=self._on_mesh)
        else:
            from rtabmap_lab.tsdf import TsdfMap
            self.tsdf = TsdfMap(K, (w, h), voxel=s["tsdf_voxel"], trunc=s["tsdf_trunc"], max_range=s["max_range"], on_mesh=self._on_mesh)
        self.t0, self.state = time.time(), "running"
        self.st = {"lost": 0, "frames": 0, "inliers": [], "restarts": 0, "nodes": 0, "loops": 0, "distance": 0.0, "odom_ms": [], "skipped": 0}
        last_pair = None
        for idx in range(*self.clip):
            if self.stop_flag:
                break
            while self.pause.is_set() and not self.stop_flag:
                self.state = "paused"
                time.sleep(0.03)
            self.state, self.pos = "running", idx
            left, right = rec.prepare(idx, rect, shutter="default" if s.get("shutter", False) else None)
            bridge.wait_idle()
            for m in bridge.poll():
                self._handle(m)
            gyro = rec.gyro_rotvec(last_pair, idx) if (s.get("gyro", False) and last_pair is not None) else None
            if bridge.send_frame(idx, float(rec.t[idx] - rec.t[0]), left, cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), gyro):
                last_pair = idx
                self.emit("frame", idx, left, right)
            bridge.wait_idle()
            for m in bridge.poll():
                self._handle(m)
            if not bridge.alive:
                raise RuntimeError("the RTAB-Map bridge stopped: " + " | ".join(bridge.log[-4:]))
            self.emit("progress", idx)
        self.st["skipped"] = bridge.skipped
        self.state = "finishing"
        for m in bridge.poll():
            self._handle(m)
        self._finish(bridge)

    def _handle(self, m):
        t, st = m["type"], self.st
        if t == "status":
            st["frames"] += 1
            st["lost"] += int(m["lost"])
            st["restarts"], st["nodes"], st["loops"], st["distance"] = m["restarts"], m["nodes"], m["loops"], m["distance"]
            st["odom_ms"].append(m["odom_ms"])
            if not m["lost"]:
                st["inliers"].append(m["inliers"])
            if m["odom"] is not None:
                T = (m["map_correction"] @ m["odom"]) @ OPTICAL
                self.cameras.append(T[:3, 3].copy())
                self.camera_T = T
                self.emit("camera", T, {"lost": m["lost"], "features": m["features"], "inliers": m["inliers"], "nodes": m["nodes"], "loops": m["loops"], "distance": m["distance"], "restarts": m["restarts"], "odom_ms": m["odom_ms"]})
        elif t == "disparity":
            self.emit("disparity", m["image"])
        elif t == "depth":
            self.node_pair[m["id"]] = m["frame_id"]
            self.tsdf.add_node(m["id"], m["depth_mm"], m["bgr"])
        elif t == "graph":
            self.tsdf.set_poses(m["poses"])
        elif t == "log":
            self.emit("log", m["text"])

    def _on_mesh(self, mesh, stats):
        now = time.monotonic()
        if now - self._last_mesh_emit < 0.8 or not len(mesh.vertices):
            return
        self._last_mesh_emit = now
        self.mesh = mesh_arrays(mesh)
        self.emit("mesh", self.mesh, dict(stats))

    # -- the numbers that make two runs comparable -------------------------------------------------------------------------------------------------------------------------
    def _finish(self, bridge):
        tsdf, cal, st = self.tsdf, self.cal, self.st
        if self.stop_flag:                                                     # stopped (a new calibration, a restart): nothing of this half-built map is worth measuring
            self.state = "stopped"
            self.emit("done", {"stopped": True})
            return
        mesh = tsdf.flush()
        if mesh is not None and len(mesh.vertices):
            self.mesh = mesh_arrays(mesh)
            self.emit("mesh", self.mesh, dict(tsdf.stats))
        m = {"calibration": self.params.fingerprint(), "clip": list(self.clip), "frames_sent": bridge.sent, "frames_skipped": bridge.skipped,
             "odometry": {"lost_frames": st["lost"], "restarts": st["restarts"], "mean_inliers": float(np.mean(st["inliers"])) if st["inliers"] else 0.0,
                          "nodes": st["nodes"], "loop_closures": st["loops"], "distance_m": st["distance"], "median_odom_ms": float(np.median(st["odom_ms"])) if st["odom_ms"] else 0.0}}
        nodes = {i: (tsdf.frames[i][0], tsdf.poses[i]) for i in tsdf.frames if i in tsdf.poses}
        if len(nodes) >= 4:
            c = consistency(nodes, {"f": cal["f"], "cx": cal["cx"], "cy": cal["cy"], "B": cal["B"]}, max_pairs=120)
            m["depth_consistency"] = {"pairs": c["pairs"], "median_rel": c["median_rel"], "p90_rel": c["p90_rel"], "expected_from_noise": c["expected_rel_from_depth_noise"],
                                      "excess": (c["median_rel"] - c["expected_rel_from_depth_noise"]) if c["pairs"] else float("nan")}
        if self.mesh is not None and len(self.mesh["vertices"]) > 500:
            m["tsdf"] = {"vertices": int(len(self.mesh["vertices"])), "triangles": int(len(self.mesh["triangles"])), **flatness.roughness(self.mesh["vertices"].astype(np.float64))}
        sm = self.samples
        m["pictures"] = {k: (float(np.median(v)) if v else None) for k, v in sm.items()} | {"sampled_pairs": len(sm["row_err_px"])}
        m["seconds"] = time.time() - self.t0
        self.metrics, self.state = m, "done"
        self.emit("done", m)

    def save_mesh(self, path):
        """The final TSDF mesh as .ply (Open3D)."""
        return bool(self.tsdf and self.tsdf.save_mesh(path))


def mesh_arrays(mesh):
    v = np.asarray(mesh.vertices, np.float32)
    col = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8) if mesh.has_vertex_colors() else np.full((len(v), 3), 160, np.uint8)
    tri = np.asarray(mesh.triangles, np.uint32)
    if len(tri) > 450_000:                                                     # a very large mesh: the page keeps interactive; the saved .ply stays complete
        import open3d as o3d
        m2 = mesh.simplify_vertex_clustering(0.015, contraction=o3d.geometry.SimplificationContraction.Average)
        v, tri = np.asarray(m2.vertices, np.float32), np.asarray(m2.triangles, np.uint32)
        col = (np.clip(np.asarray(m2.vertex_colors), 0, 1) * 255).astype(np.uint8) if m2.has_vertex_colors() else np.full((len(v), 3), 160, np.uint8)
    return {"vertices": v, "colors": col, "triangles": tri}


def run_headless(rec, params, clip, settings=None, port=5710, progress=None):
    """Run a clip to the end without a viewer and return the finished Run (metrics in .metrics). The batch comparison of calibration files uses it."""
    r = Run(rec, params, clip, settings, emit=(lambda k, *a: progress(*a) if progress and k == "progress" else None), port=port).start()
    r.thread.join()
    if r.state == "failed":
        raise RuntimeError(r.error)
    return r
