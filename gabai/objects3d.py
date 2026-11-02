"""Static-object landmarks kept off the main loop, with persistent identity.

A 3D object detector (gabai/boxer_lift.make_lifter: OWLv2 + BoxerNet, about 135 ms a frame on the RTX 5060 Ti) must not run inside the frame loop: the loop would take that time every
frame. Objects3D runs it on its own thread, a few times a second, on the newest frame only:
  submit()  never waits. It copies the frame (about 2 ms) and hands it over, or returns False when the interval has not passed or the worker is still busy (the frame is dropped,
            never queued, so the worker never works on old pictures and the loop never builds up a backlog).
  poll()    never waits. The newest result (this frame's boxes and every landmark so far) once, then None until the next one.
Each frame is lifted with ITS OWN pose, so a result that arrives late still lands in the right place in the world.

ObjectMap is the memory (plain python, no threads): one entry per physical object with a fixed id.
  association  a detection joins the existing object whose 3D box overlaps it most (IoU of the world-axis boxes, any label), or, failing that, an object that has already been called the
               same thing (or a synonym: tv = monitor) within `gate` metres. Boxes of very different volume do not join (a cup on a table is not the table).
  label        every look adds its confidence to the object's evidence for that label. The label changes only when another label has 1.5x the evidence of the current one and at least
               3 looks of its own: one odd detection never renames an object, repeated stronger evidence does (tv -> monitor), and no second object is made by the change.
  duplicates   two objects whose boxes overlap (IoU >= merge_iou) are one object: the younger is folded into the older (its looks, evidence and id history), so a late-arriving
               detection that was first taken as new cannot leave two copies.
  history      nothing is replaced: every look is kept as an observation (keyframe it was seen from, its position in that keyframe's frame, world position then, confidence). The graph
               (gabai/posegraph.py, run from KeyframeVO when an object is seen again from a keyframe well apart from the earlier ones) takes them as factors and solves keyframe poses and object
               positions TOGETHER; apply_graph() then re-places every observation with the corrected keyframe poses and stores the solved position X as the object's position.
  forgetting   an object that is out of view, occluded or simply not found is KEPT. It is dropped only (a) as a one-off: fewer than min_looks looks and not seen for stale_s seconds, or
               (b) with positive evidence: it is inside the camera's view, within range, and Boxer ran on `miss_drop` frames in a row without finding anything of it (a moved or removed object).
"""
import queue
import threading
import time

import numpy as np

ALIAS = {"tv": "monitor", "television": "monitor", "bookcase": "shelf", "bookshelf": "shelf"}       # names the detector flips between for one object
SWITCH_RATIO, SWITCH_LOOKS = 1.5, 3                                                                 # label hysteresis
VOLUME_RATIO = 0.25                                                                                 # boxes of more different volume than 4x never join
MERGE_IOU, ASSOC_IOU = 0.4, 0.25
STALE_S, MISS_DROP = 20.0, 15
OBS_MAX = 3000                                                                                      # observations kept per object


def aabb(center, dims, R):
    """Axis-aligned world box (lo, hi) around an oriented box."""
    half = 0.5 * np.abs(np.asarray(R)) @ np.asarray(dims)
    c = np.asarray(center, float)
    return c - half, c + half


def iou(a, b):
    lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    va, vb = float(np.prod(a[1] - a[0])), float(np.prod(b[1] - b[0]))
    return inter / max(va + vb - inter, 1e-9), min(va, vb) / max(va, vb, 1e-9)


class ObjectMap:
    def __init__(self, gate=0.25, min_looks=5, stale_s=STALE_S, miss_drop=MISS_DROP, merge_iou=MERGE_IOU):
        self.gate, self.min_looks, self.stale_s, self.miss_drop, self.merge_iou = gate, min_looks, stale_s, miss_drop, merge_iou
        self.objs, self._next = [], 0

    @staticmethod
    def pos(T):
        """The object's position: the pose-graph solution when there is one, else the median of its looks."""
        return T["X"] if T.get("X") is not None else np.median(T["P"][-200:], axis=0)

    def box(self, T):
        return aabb(self.pos(T), np.median(T["dims"][-200:], axis=0), T["R"])

    def _canon(self, label):
        return ALIAS.get(label, label)

    def _match(self, d):
        db, best, bs = aabb(d["p_w"], d["dims"], d["R_obj"]), None, 0.0
        for T in self.objs:
            i, vr = iou(db, self.box(T))
            if i >= ASSOC_IOU and vr >= VOLUME_RATIO and i > bs:                    # any label: the same physical thing
                best, bs = T, i
        if best is None:
            canon, bd = self._canon(d["label"]), self.gate
            for T in self.objs:
                if canon in {self._canon(k) for k in T["ev"]}:
                    dist = np.linalg.norm(self.pos(T) - d["p_w"])
                    if dist < bd:
                        best, bd = T, dist
        return best

    def _relabel(self, T):
        cur = T["label"]
        top = max(T["ev"], key=T["ev"].get)
        if top != cur and T["ev"][top] > SWITCH_RATIO * T["ev"].get(cur, 0.0) and T["n_label"][top] >= SWITCH_LOOKS:
            T["label"] = top

    def update(self, t, dets, in_view):
        """One Boxer frame. dets: [dict(label, conf, p_w, dims, R_obj)]; in_view(p) -> True when the point is inside the camera's view and range now."""
        seen = set()
        for d in dets:
            T = self._match(d)
            if T is None:
                T = dict(id=self._next, label=d["label"], ev={}, n_label={}, P=[], dims=[], obs=[], X=None, R=d["R_obj"], first=t, last=t, miss=0)
                self._next += 1
                self.objs.append(T)
            T["P"].append(d["p_w"]), T["dims"].append(d["dims"])
            T["obs"].append(dict(kf=d.get("kf"), p_k=d.get("p_k"), sg=d.get("sg", 0.05), p_w=d["p_w"], p_w0=d["p_w"], t=t, conf=d["conf"], label=d["label"]))
            del T["obs"][:-OBS_MAX]
            T["R"], T["last"], T["miss"] = d["R_obj"], t, 0
            T["ev"][d["label"]] = T["ev"].get(d["label"], 0.0) + d["conf"]
            T["n_label"][d["label"]] = T["n_label"].get(d["label"], 0) + 1
            self._relabel(T)
            seen.add(T["id"])
        self._merge()
        for T in list(self.objs):
            if T["id"] in seen:
                continue
            if in_view(self.pos(T)):                           # looked for and not found
                T["miss"] += 1
            if len(T["P"]) < self.min_looks and t - T["last"] > self.stale_s:         # a one-off that never came back
                self.objs.remove(T)
            elif T["miss"] >= self.miss_drop:                                       # in view, in range, not there, many frames in a row
                self.objs.remove(T)

    def _merge(self):
        objs = sorted(self.objs, key=lambda T: T["id"])
        gone = set()
        for i, A in enumerate(objs):
            if A["id"] in gone:
                continue
            for B in objs[i + 1:]:
                if B["id"] in gone:
                    continue
                v, vr = iou(self.box(A), self.box(B))
                if v >= self.merge_iou and vr >= VOLUME_RATIO:
                    A["P"] += B["P"]
                    A["dims"] += B["dims"]
                    A["obs"] = sorted(A["obs"] + B["obs"], key=lambda o: o["t"])[-OBS_MAX:]
                    A["X"] = None                                                   # the solved position was for one of them: the next solve gives the merged object its own
                    A["last"], A["miss"] = max(A["last"], B["last"]), 0
                    for k, x in B["ev"].items():
                        A["ev"][k] = A["ev"].get(k, 0.0) + x
                        A["n_label"][k] = A["n_label"].get(k, 0) + B["n_label"][k]
                    self._relabel(A)
                    gone.add(B["id"])
        self.objs = [T for T in self.objs if T["id"] not in gone]

    def graph_obs(self):
        """Factors for the pose graph: [(object id, keyframe id, p_k, sigma)] of objects seen at least min_looks times. The looks from one keyframe are averaged (sigma / sqrt(n), not below 2 cm)."""
        out = []
        for T in self.objs:
            if len(T["P"]) < self.min_looks:
                continue
            by = {}
            for o in T["obs"]:
                if o["kf"] is not None and o["p_k"] is not None:
                    by.setdefault(o["kf"], []).append(o)
            for kf, os_ in by.items():
                out.append((T["id"], kf, np.mean([o["p_k"] for o in os_], axis=0), max(float(np.mean([o["sg"] for o in os_])) / np.sqrt(len(os_)), 0.02)))
        return out

    def relocate(self, node_pose, lm_pos):
        """The pose graph moved keyframes and objects. node_pose(kf) -> (R, t) now or None (forgotten: those looks stay where they were); lm_pos {object id: position now}."""
        for T in self.objs:
            for o in T["obs"]:
                n = None if o["kf"] is None else node_pose(o["kf"])
                if n is not None:
                    o["p_w"] = np.asarray(n[0]) @ o["p_k"] + np.asarray(n[1])
            T["P"] = [o["p_w"] for o in T["obs"]] or T["P"]
            if T["id"] in lm_pos:
                T["X"] = np.asarray(lm_pos[T["id"]], float)

    def landmarks(self):
        out = []
        for T in self.objs:
            if len(T["P"]) < self.min_looks:
                continue
            tot = sum(T["ev"].values())
            top = sorted(T["ev"].items(), key=lambda kv: -kv[1])[:2]
            out.append(dict(id=T["id"], label=T["label"], center=self.pos(T), dims=np.median(T["dims"], axis=0), R=T["R"], looks=len(T["P"]),
                            conf=T["ev"][T["label"]] / max(tot, 1e-9), text="  ".join(f"{k} {v / max(tot, 1e-9):.0%}" for k, v in top),
                            sigma_cm=float(np.linalg.norm(np.std(T["P"][-200:], axis=0)) * 100), recent=np.array(T["P"][-300:])))
        return out


class Objects3D:
    def __init__(self, make_lift, every_s=0.3, max_range=2.0, gate=0.25, min_looks=5, warm=True):
        """make_lift() -> lift(left, disp, trusted, cal, R_wc, t_wc) -> [dict(label, conf, p_w, dims, R_obj)]; it is called on the worker thread (loading the models takes seconds)."""
        self.make_lift, self.lift, self.every_s, self.max_range, self.warm = make_lift, None, every_s, max_range, warm
        self.map, self.error, self.ms = ObjectMap(gate, min_looks), None, []
        self._last, self._busy, self._snap, self._new, self._reset, self._dead = -1e9, False, None, False, False, False
        self._q, self._lock, self._mlock, self._ready = queue.Queue(maxsize=1), threading.Lock(), threading.Lock(), threading.Event()
        threading.Thread(target=self._loop, daemon=True, name="objects3d").start()

    @property
    def tracks(self):
        return self.map.objs

    def graph_obs(self):
        with self._mlock:
            return self.map.graph_obs()

    def apply_graph(self, node_pose, lm_pos):
        """After a pose-graph correction: re-place every observation with the corrected keyframe poses, take the solved object positions, and refresh what is drawn."""
        with self._mlock:
            self.map.relocate(node_pose, lm_pos)
            marks = self.map.landmarks()
        with self._lock:
            prev = self._snap
            self._snap, self._new = dict(t=prev["t"] if prev else None, current=prev["current"] if prev else [], landmarks=marks), True

    def submit(self, t, left, disp, trusted, cal, R_wc, t_wc, anchor=None):
        """Offer a frame (t in seconds). True when it was taken. Call only with a pose that is trusted. anchor = (keyframe id, its pose R, t in the same frame as R_wc, t_wc) from the
        odometry: what the object looks like from that keyframe is what the pose graph gets."""
        if self.lift is None or self._dead or self._busy or t - self._last < self.every_s:      # not loaded yet, failed to load, busy or too soon
            return False
        self._last, self._busy = t, True
        self._q.put((t, left.copy(), disp.copy(), trusted.copy(), cal, np.array(R_wc, float), np.array(t_wc, float), None if anchor is None else (anchor[0], np.array(anchor[1], float), np.array(anchor[2], float))))
        return True

    def wait_ready(self, timeout=300.0):
        """Block until the models are loaded and warmed up (or failed): the pipeline calls this before it opens its source, so nothing starts while a checkpoint is still loading."""
        t0 = time.perf_counter()
        while not self._ready.is_set() and time.perf_counter() - t0 < timeout:
            self._ready.wait(0.2)

    def reset(self):
        """The poses of the looks so far are no longer valid (a loop closure moved the map, the map was rebuilt): forget every landmark."""
        self._reset = True

    def poll(self):
        with self._lock:
            snap, self._new = (self._snap if self._new else None), False
        return snap

    def _loop(self):
        try:
            lift = self.make_lift()
            try:                                                             # (warm) one dummy frame: the first real call otherwise takes seconds (kernels, caches) while the stream is running
                if self.warm:
                    lift(np.zeros((360, 640, 3), np.uint8), np.full((360, 640), 30.0, np.float32), np.ones((360, 640), bool), dict(f=618.0, cx=320.0, cy=180.0, B=0.0417), np.eye(3), np.zeros(3))
            except Exception as e:
                self.error = e
            self.lift = lift
        except Exception as e:                                               # models or weights missing: stay off, keep the reason in .error
            self.error, self._dead = e, True
            self._ready.set()
            return
        self._ready.set()
        while True:
            t, left, disp, trusted, cal, R_wc, t_wc, anchor = self._q.get()
            t0 = time.perf_counter()
            if self._reset:
                self.map.objs, self._reset = [], False
            try:
                dets = [d for d in self.lift(left, disp, trusted, cal, R_wc, t_wc) if np.linalg.norm(R_wc.T @ (d["p_w"] - t_wc)) <= self.max_range]
                ok = True
            except Exception as e:                                           # a bad frame must not end the thread: keep the error, carry on
                self.error, dets, ok = e, [], False
            H, W = left.shape[:2]
            f, cx, cy = float(cal["f"]), float(cal["cx"]), float(cal["cy"])

            def in_view(p):
                if not ok:
                    return False                                             # no detector result: nothing was looked for
                c = R_wc.T @ (p - t_wc)
                return 0.2 < c[2] <= self.max_range and 0.1 * W < f * c[0] / c[2] + cx < 0.9 * W and 0.1 * H < f * c[1] / c[2] + cy < 0.9 * H
            fB = float(cal.get("f", 618.0)) * float(cal.get("B", 0.0417))
            for d in dets:
                if anchor is not None:
                    d["kf"], d["p_k"] = anchor[0], anchor[1].T @ (d["p_w"] - anchor[2])
                z = float(np.linalg.norm(R_wc.T @ (d["p_w"] - t_wc)))
                d["sg"] = float(np.hypot(0.04, 0.3 * z * z / fB))                  # a box centre is good to about 4 cm; stereo depth adds z^2 * 0.3 / (f B)
            with self._mlock:
                self.map.update(t, dets, in_view)
                marks = self.map.landmarks()
            self.ms.append((time.perf_counter() - t0) * 1000)
            with self._lock:
                self._snap, self._new = dict(t=t, current=dets, landmarks=marks), True
            self._busy = False
