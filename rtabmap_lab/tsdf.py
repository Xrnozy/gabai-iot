"""Open3D TSDF fusion of RTAB-Map's keyframes: dense depth + colour per node, placed with the node's OPTIMISED pose.

RTAB-Map gives the poses (odometry, loop closure, graph optimisation) and, through its own stereo matching, a depth image per node. Open3D integrates them into a truncated signed distance volume
and extracts a mesh. A TSDF cannot take a frame back out, so when the poses change (a loop closure moved the graph) the volume is built again from the stored depth + colour of all nodes with their new
poses, on a worker thread: the frame loop never waits for it, and the mesh drawn is always the last one finished.

Frames: pose = RTAB-Map base frame (x forward, y left, z up); the camera is the base frame turned by OPTICAL (the same constant as the bridge and the dashboard). The Open3D extrinsic is world -> camera,
the inverse of  T_world_base @ OPTICAL.
"""
import threading

import cv2
import time

import numpy as np
import open3d as o3d

OPTICAL = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
MOVED_M, MOVED_RAD = 0.02, np.radians(1.0)             # a node whose pose moved more than this since it was integrated makes the volume stale


def _moved(a, b):
    dt = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    c = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return dt > MOVED_M or float(np.arccos(c)) > MOVED_RAD


class TsdfMap:
    def __init__(self, intrinsics, size, voxel=0.02, trunc=0.08, max_range=6.0, max_nodes=600, mesh_every_s=2.0, on_mesh=None, threaded=True, blur_keep=0.75):
        """intrinsics = (fx, fy, cx, cy), size = (w, h). voxel / trunc in metres (trunc: at least 3 to 4 voxels and about the depth noise at the working range).
        on_mesh(mesh, stats) is called from the worker thread with every finished mesh.
        blur_keep: a node's depth is fused only if its picture is at least as sharp (variance of the Laplacian) as the sharpest blur_keep share of the last 60 nodes; 1.0 = fuse all.
        Blurred pairs have noisier disparity (walk_20261102: 1.21 px against 0.89 px for the sharpest quarter); skipping the blurriest quarter: held-out error 3.72 -> 3.41 px (docs/stereo-depth-diagnosis.md)."""
        self.K = o3d.camera.PinholeCameraIntrinsic(size[0], size[1], *intrinsics)
        self.voxel, self.trunc, self.max_range, self.max_nodes, self.mesh_every, self.on_mesh = voxel, trunc, max_range, max_nodes, mesh_every_s, on_mesh
        self.frames = {}                                  # node id -> (depth uint16 mm, bgr)
        self.blur_keep, self.sharp, self.blurred = blur_keep, {}, set()
        self.poses, self.used = {}, {}                    # latest optimised base pose; the pose each node was integrated with
        self.vol, self.mesh, self.stats = self._new_volume(), None, {"rebuilds": 0, "integrated": 0, "mesh_ms": 0.0, "integrate_ms": 0.0, "vertices": 0, "triangles": 0}
        self._lock, self._step_lock, self._wake, self._stop, self._last_mesh = threading.RLock(), threading.Lock(), threading.Event(), False, 0.0
        self._thread = threading.Thread(target=self._work, daemon=True) if threaded else None
        if self._thread:
            self._thread.start()

    def _new_volume(self):
        return o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=self.voxel, sdf_trunc=self.trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # -- input (frame loop thread) --------------------------------------------------------------------------------------------------------------------
    def add_node(self, node_id, depth_mm, bgr, cls=None, conf=None):
        """cls: optional per-pixel class of the fused depth (1 measured and agreed, 2 measured by one source, 3 estimated): the TSDF adds a pixel that many times (3, 2, 1), which is its weight in the average."""
        with self._lock:
            if self.blur_keep < 1.0:
                g = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                s = float(cv2.Laplacian(g, cv2.CV_32F).var())
                recent = [self.sharp[k] for k in sorted(self.sharp)[-60:]]
                if len(recent) >= 10 and s < np.quantile(recent, 1.0 - self.blur_keep):
                    self.blurred.add(node_id)
                self.sharp[node_id] = s
            self.frames[node_id] = (depth_mm, bgr, cls)
            while len(self.frames) > self.max_nodes:         # a very long session: the oldest nodes drop out of the volume (the next rebuild)
                old = min(self.frames)
                self.frames.pop(old)
                self.poses.pop(old, None)
                self.used.pop(old, None)
        self._wake.set()

    def set_poses(self, poses):
        """poses: node id -> 4x4 optimised base pose (None entries ignored)."""
        with self._lock:
            for i, T in poses.items():
                if T is not None:
                    self.poses[i] = T
        self._wake.set()

    # -- fusion ------------------------------------------------------------------------------------------------------------------------------------
    def step(self, force_mesh=False):
        """One round of work: integrate new nodes, or rebuild when a pose moved; extract the mesh when due. Returns True if a mesh was made."""
        with self._step_lock:                                # the worker and flush() never fuse at the same time
            return self._step(force_mesh)

    def _step(self, force_mesh):
        with self._lock:
            todo = {i: self.poses[i].copy() for i in self.frames if i in self.poses and i not in self.blurred}
            stale = any(i in self.used and _moved(self.used[i], T) for i, T in todo.items()) or any(i not in todo for i in self.used)
            new = [i for i in todo if i not in self.used]
            frames = {i: self.frames[i] for i in todo}
        if not todo:
            return False
        t0 = time.perf_counter()
        if stale:                                            # a loop closure (or any graph change) moved nodes already in the volume: build it again with the new poses
            vol, order = self._new_volume(), sorted(todo)
            self.stats["rebuilds"] += 1
        else:
            vol, order = self.vol, sorted(new)
        for i in order:
            self._integrate_from(vol, frames[i], todo[i])
        with self._lock:
            self.vol = vol
            for i in order:
                self.used[i] = todo[i]
            for i in [k for k in self.used if k not in todo]:
                del self.used[i]
        if order:
            self.stats["integrated"] = len(self.used)
            self.stats["integrate_ms"] = (time.perf_counter() - t0) * 1000.0
        if not (order or force_mesh) or (not force_mesh and time.monotonic() - self._last_mesh < self.mesh_every):
            return False
        return self._extract()

    WEIGHT = {1: 3, 2: 2, 3: 1}

    def _integrate_from(self, vol, frame, T_base):
        depth_mm, bgr, cls = frame
        color = o3d.geometry.Image(np.ascontiguousarray(bgr[:, :, ::-1]))
        extr = np.linalg.inv(T_base @ OPTICAL)
        layers = [(depth_mm, 1)] if cls is None else [(np.where(cls == c, depth_mm, 0).astype(np.uint16), m) for c, m in self.WEIGHT.items()]
        for d, times in layers:
            if not d.any():
                continue
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, o3d.geometry.Image(np.ascontiguousarray(d)), depth_scale=1000.0, depth_trunc=self.max_range, convert_rgb_to_intensity=False)
            for _ in range(times):
                vol.integrate(rgbd, self.K, extr)

    def _extract(self):
        t0 = time.perf_counter()
        with self._lock:
            mesh = self.vol.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        self.mesh, self._last_mesh = mesh, time.monotonic()
        self.stats.update(mesh_ms=(time.perf_counter() - t0) * 1000.0, vertices=len(mesh.vertices), triangles=len(mesh.triangles))
        if self.on_mesh:
            self.on_mesh(mesh, dict(self.stats))
        return True

    def _work(self):
        while not self._stop:
            self._wake.wait(0.5)
            self._wake.clear()
            try:
                self.step()
            except Exception as e:                           # the fusion must never take the viewer down; the reason is in stats
                self.stats["error"] = f"{type(e).__name__}: {e}"

    # -- output ------------------------------------------------------------------------------------------------------------------------------------
    def flush(self):
        """Finish all pending work and make a mesh now (end of a run, tests)."""
        self.step(force_mesh=True)
        return self.mesh

    def save_mesh(self, path):
        m = self.flush()
        if m is None or not len(m.vertices):
            return False
        return bool(o3d.io.write_triangle_mesh(path, m))

    def close(self):
        self._stop = True
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)
