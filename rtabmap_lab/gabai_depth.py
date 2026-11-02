"""Optional depth source for the TSDF and the point map: the main GabAi system's own depth, instead of RTAB-Map's stereo matching.  py -m rtabmap_lab live --depth gabai

The chain is the main pipeline's `default_depth` (gabai/pipeline/run.py, used read-only): LAS2 stereo on the rectified pair, Depth Anything V2 on the LEFT and the RIGHT picture, and the fusion in 'fill' mode:
stereo stays the only source of every pixel it measured, the two monocular maps (scaled to the stereo pixels) fill the holes, and only where they agree. The same speckle removal, depth correction,
fill reach and depth-edge mask as the main pipeline are applied.

RTAB-Map still does the odometry, loop closure and graph optimisation from the stereo pictures (it matches features, it does not use this depth), so the poses are unchanged; only the surface and the points
placed with those poses come from this depth. It runs on the GPU in a worker thread and only for the pairs RTAB-Map turns into nodes (a node comes about twice a second at most), so the frame loop does not wait for it.
"""
import queue
import threading
import time

import cv2
import numpy as np

from gabai.povmesh import fused_for_view, surface_mask


class GabaiDepth:
    def __init__(self, cfg, depth_fn=None):
        """cfg: a gabai.pipeline.config.PipelineConfig (stereo model, mono model and mode, speckle settings, depth correction, fill reach, surface filter). depth_fn: tests."""
        s = cfg.stereo
        if depth_fn is None:
            from gabai.pipeline.run import default_depth
            depth_fn = default_depth(s.model, s.speckle_px, s.speckle_diff, s.mono, s.mono_mode, s.scale, s.depth_correction, s.mono_pair)
            depth_fn.warm_up()
        self.fn, self.reach, self.surface = depth_fn, int(cfg.map.fill_reach_px), bool(cfg.map.surface_filter)

    def depth_mm(self, left_bgr, right_bgr, cal, min_range=0.3, max_range=6.0, rtab_mm=None):
        """(uint16 depth in millimetres, 0 = none; model milliseconds) for one rectified pair. cal: dict with f and B (the pair's rectified focal length and baseline)."""
        disp, trusted, ms = self.fn(left_bgr, right_bgr)
        fill = getattr(self.fn, "filled", None)
        if fill is not None and self.reach:                               # a hole far from any measured pixel stays empty
            near = cv2.dilate(trusted.astype(np.uint8), np.ones((2 * self.reach + 1,) * 2, np.uint8)).astype(bool)
            fill = (fill[0], fill[1] & near, fill[2])
        fused = fused_for_view(disp, fill)
        if self.surface:                                                  # no flying pixels between surfaces
            fused = np.where(surface_mask(fused, cal), fused, np.nan).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(np.isfinite(fused) & (fused > 0.5), cal["f"] * cal["B"] / fused, 0.0)
        z = np.where((z >= min_range) & (z <= max_range), z, 0.0)
        return (z * 1000.0).astype(np.uint16), float(ms)


def points_from_depth(depth_mm, bgr, fx, fy, cx, cy, stride=4):
    """Points in the left OPTICAL frame (x right, y down, z forward) and their colours, every `stride`-th pixel that has a depth."""
    d = depth_mm[::stride, ::stride]
    v, u = np.nonzero(d)
    z = d[v, u].astype(np.float32) / 1000.0
    u, v = u * stride, v * stride
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1).astype(np.float32), bgr[v, u]


class DepthUpgrader:
    """Gives each new node the GabAi depth instead of the bridge's: the pair that made the node is kept for a short while, the depth is computed on a worker thread, and only then is the node handed to the
    TSDF and the point map. A node whose pair is gone, or that arrives while the worker is far behind, takes the bridge's depth at once: no node is ever left out."""

    def __init__(self, depth, cal, size, add_node, set_points, on_depth=None, keep=60, max_queue=3, stride=4, min_range=0.3, max_range=6.0, wait_pose_s=0.0):
        """depth.depth_mm(left, right, cal, min, max, rtab_mm=) returns (uint16 mm, ms) or (uint16 mm, ms, extra): extra = {cls, conf, parts, fused}, handed on to add_node(id, depth, bgr, extra) and on_depth(depth, ms, extra)."""
        self.depth, self.cal, self.size, self.add_node, self.set_points, self.on_depth = depth, cal, size, add_node, set_points, on_depth
        self.ring, self.keep, self.stride, self.range = {}, keep, stride, (min_range, max_range)
        self.q, self.max_queue, self.stats = queue.Queue(), max_queue, {"gabai": 0, "fallback": 0, "ms": 0.0}
        self.pending = 0                                                  # nodes put in the queue and not finished: queued or being computed
        # wait_pose_s > 0: a node is only worth its depth once RTAB-Map's graph holds it with a pose (about 60% of the nodes it announces never get one: measured 44 of 115). Such a node is held,
        # and its models run when posed() reports its pose; after wait_pose_s seconds without one it is dropped (stats["skipped"]).
        self.wait_pose_s, self.held, self._hold_lock = wait_pose_s, {}, threading.Lock()
        self.stats["skipped"] = 0
        self._stop = False
        self._t = threading.Thread(target=self._work, daemon=True)
        self._t.start()

    def remember(self, frame_id, left, right):
        self.ring[frame_id] = (left, right)
        for old in sorted(self.ring)[:-self.keep]:
            del self.ring[old]

    def node(self, m):
        """m = the bridge's depth message (node id, frame id, its own depth and colour)."""
        pair = self.ring.get(m["frame_id"])
        if pair is None or self.q.qsize() >= self.max_queue:
            self.stats["fallback"] += 1
            self.pending -= 1
            self.add_node(m["id"], m["depth_mm"], m["bgr"])
            return
        self.pending += 1
        if self.wait_pose_s > 0:
            with self._hold_lock:
                self.held[m["id"]] = (m, pair, time.time())
            return
        self.q.put((m, pair))

    def posed(self, poses):
        """poses: node id -> pose (None = no pose), from the bridge's graph. Releases the held nodes that have one, drops the ones held too long."""
        if not self.held:
            return
        now = time.time()
        with self._hold_lock:
            for i in list(self.held):
                m, pair, t0 = self.held[i]
                if poses.get(i) is not None:
                    del self.held[i]
                    if self.q.qsize() >= self.max_queue:                  # busy: the bridge's depth now, as for an unheld node
                        self.stats["fallback"] += 1
                        self.pending -= 1
                        self.add_node(m["id"], m["depth_mm"], m["bgr"])
                    else:
                        self.q.put((m, pair))
                elif now - t0 > self.wait_pose_s:
                    del self.held[i]
                    self.stats["skipped"] += 1
                    self.pending -= 1

    def wait_idle(self, timeout=60.0):
        """Block until the worker has finished every node it was given (a recording is replayed faster than the models run, so it waits here; live it never needs to)."""
        end = time.time() + timeout
        while self.pending - len(self.held) > 0 and time.time() < end:        # a held node waits for the graph, which the caller's own loop delivers: waiting for it here would never end
            time.sleep(0.005)

    def _work(self):
        while not self._stop:
            try:
                m, (left, right) = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                t0 = time.perf_counter()
                res = self.depth.depth_mm(left, right, self.cal, *self.range, rtab_mm=m["depth_mm"])
                d, extra = res[0], (res[2] if len(res) > 2 else None)
                self.stats["ms"] = (time.perf_counter() - t0) * 1000.0
                self.stats["gabai"] += 1
                self.add_node(m["id"], d, left, extra)
                self.set_points(m["id"], *points_from_depth(d, left, self.cal["f"], self.cal["f"], self.cal["cx"], self.cal["cy"], self.stride))
                if self.on_depth:
                    self.on_depth(d, self.stats["ms"], extra)
            except Exception as e:                                        # the GPU chain must not take the viewer down: this node falls back to the bridge's depth
                self.stats["error"] = f"{type(e).__name__}: {e}"
                self.stats["fallback"] += 1
                self.add_node(m["id"], m["depth_mm"], m["bgr"])
            finally:
                self.pending -= 1

    def close(self):
        """Stops the worker. Nodes still waiting for their GabAi depth take the bridge's instead: none is dropped from the surface."""
        self._stop = True
        self._t.join(timeout=5)
        with self._hold_lock:                                             # never posed: nothing to give them to
            self.pending -= len(self.held)
            self.stats["skipped"] += len(self.held)
            self.held.clear()
        while True:
            try:
                m, _ = self.q.get_nowait()
            except queue.Empty:
                break
            self.stats["fallback"] += 1
            self.pending -= 1
            self.add_node(m["id"], m["depth_mm"], m["bgr"])
