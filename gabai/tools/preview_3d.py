"""3D preview: stereo depth as a point cloud, hands lifted to 3D, grab/release events, and the fused voxel map, shown in Rerun.

    py -m gabai.tools.preview_3d --left data/left.mp4 --right data/right.mp4 --start 60 --max-frames 150
    py -m gabai.tools.preview_3d --left data/left.mp4 --right data/right.mp4 --save data/preview.rrd --no-viewer

For two plain video files there is no calibration, no IMU and no camera pose. So: f and B are ASSUMED (--f-frac, --baseline), distances
are not metric, the camera is treated as fixed at the origin (head motion is NOT compensated, so the fused map smears when the camera
moves and a held object's path includes the camera's own motion), and nothing here is a measurement. It is a way to look at what the pipeline
sees. With the rig, the pose comes from Basalt and the calibration from calibrate_stereo (pass them to Preview3D.frame).

Needs checkpoints/hand_landmarker.task (hands) and models/yolo11n-seg.pt (people and objects); either missing turns that part off and
the output says so.
"""
import argparse
import os
import time

import cv2
import numpy as np

from gabai.cloud import VoxelMap, map_points, transform
from gabai.detect import PEOPLE, det_depth, person_mask
from gabai.grasp import GraspTracker, phrase
from gabai.hands import HAND_CONNECTIONS, finger_state, hand_mask, lift_hand

PALM = (0, 5, 9, 13, 17)


class ListSink:
    """Keeps what would be drawn; used by tests."""

    def __init__(self):
        self.n_frames, self.last = 0, None

    def log(self, t, **kw):
        self.n_frames += 1
        self.last = dict(kw, t=t)


class RerunSink:
    def __init__(self, save=None, viewer=True):
        import rerun as rr
        self.rr = rr
        rr.init("gabai_preview", spawn=viewer)
        if save:
            rr.save(save)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)       # camera frame: x right, y down, z forward

    def log(self, t, image=None, live_points=None, live_colors=None, hands=(), objects=(), events=(), map_points=None, **_):
        rr = self.rr
        rr.set_time("t", duration=float(t))
        if image is not None:
            rr.log("camera/image", rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        if live_points is not None and len(live_points):
            rr.log("world/live", rr.Points3D(live_points, colors=live_colors, radii=0.006))
        if map_points is not None and len(map_points):
            rr.log("world/map", rr.Points3D(map_points, radii=0.012, colors=[120, 160, 255]))
        for h in hands:
            ok = np.isfinite(h["xyz"][:, 2])
            rr.log(f"world/hand/{h['id']}", rr.Points3D(h["xyz"][ok], radii=0.008, colors=[255, 80, 80] if h["closed"] else [80, 255, 120]))
            segs = [h["xyz"][[a, b]] for a, b in HAND_CONNECTIONS if ok[a] and ok[b]]
            rr.log(f"world/hand/{h['id']}/bones", rr.LineStrips3D(segs) if segs else rr.LineStrips3D([]))
        for o in objects:
            rr.log(f"world/object/{o['key']}", rr.Points3D([o["center"]], radii=0.03, labels=[o["label"] + (" (held)" if o["held"] else "")], colors=[255, 200, 0]))
        for e in events:
            rr.log("events", rr.TextLog(f"{e['type']}: {e.get('label')} {e.get('reason') or ''}"))


class Preview3D:
    def __init__(self, f, B, cx, cy, detector=None, hand_tracker=None, sink=None, voxel=0.05, window_m=8.0, map_every=2):
        self.f, self.B, self.cx, self.cy = f, B, cx, cy
        self.det, self.hands, self.sink = detector, hand_tracker, sink
        self.map, self.grasp, self.map_every, self.window_m = VoxelMap(voxel, window_m), GraspTracker(), map_every, window_m
        self.n, self._objs = 0, {}

    def _lift_objects(self, dets, disp):
        out = []
        for d in dets:
            if d.label in PEOPLE:
                continue
            r = det_depth(d, disp, self.f, self.B)
            z = r["depth_m"]["v"]
            if z is None or r["weak"]:
                continue
            x0, y0, x1, y1 = d.box
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            out.append({"label": d.label.replace(" ", "_"), "center": np.array([(u - self.cx) / self.f * z, (v - self.cy) / self.f * z, z]), "det": d,
                        "radius": max(0.04, 0.5 * (x1 - x0) / self.f * z)})
        return out

    def _assoc(self, lifted):
        """Keep an object's key from frame to frame by matching to the nearest previous one of the same label (15 cm)."""
        keys, used, prev = {}, set(), self._objs
        for o in lifted:
            best = None
            for k, p in prev.items():
                if p["label"] == o["label"] and k not in used:
                    dist = float(np.linalg.norm(p["center"] - o["center"]))
                    if dist < 0.15 and (best is None or dist < best[0]):
                        best = (dist, k)
            if best:
                k = best[1]
            else:
                i = 0
                while f"{o['label']}{i}" in prev or f"{o['label']}{i}" in keys:
                    i += 1
                k = f"{o['label']}{i}"
            used.add(k)
            keys[k] = o
        return keys

    def frame(self, t, image, disp, trusted, pose=None):
        self.n += 1
        h, w = disp.shape
        dets = self.det.detect(image) if self.det is not None else []
        hands = self.hands.detect(image, int(t * 1000)) if self.hands is not None else []
        keys = self._assoc(self._lift_objects(dets, disp))
        held_keys = {v["obj"] for v in self.grasp.held().values()}
        excl = person_mask(dets, (h, w))
        if hands:
            excl |= hand_mask(hands, (h, w))
        for k in held_keys:
            if k in keys and keys[k]["det"].mask is not None:
                excl |= keys[k]["det"].mask
            elif k in keys:
                x0, y0, x1, y1 = (int(v) for v in keys[k]["det"].box)
                excl[max(y0, 0):y1, max(x0, 0):x1] = True
        pts, info = map_points(disp, trusted, self.f, self.B, self.cx, self.cy, exclude=excl, stride=3)
        R, tt = (np.eye(3), np.zeros(3)) if pose is None else pose
        pw = transform(pts, R, tt)
        if self.n % self.map_every == 0:
            self.map.integrate(pw)
            self.map.trim(tt)
        colors = None
        if len(pts):
            uv = np.stack([pts[:, 0] / pts[:, 2] * self.f + self.cx, pts[:, 1] / pts[:, 2] * self.f + self.cy], axis=1).astype(int)
            uv = np.clip(uv, [0, 0], [w - 1, h - 1])
            colors = image[uv[:, 1], uv[:, 0]][:, ::-1]
        hd_out, hd_in = [], []
        for i, hnd in enumerate(hands):
            lifted = lift_hand(hnd, disp, self.f, self.B, self.cx, self.cy)
            xyz = transform(lifted["xyz"], R, tt)
            st = finger_state(hnd)
            pv = xyz[list(PALM)]
            pv = pv[np.isfinite(pv[:, 2])]
            hid = hnd.side if sum(1 for q in hands if q.side == hnd.side) == 1 else f"{hnd.side}{i}"
            hd_out.append({"id": hid, "xyz": xyz, "closed": st["closed"], "open": st["open"], "ratio": st["ratio"], "valid_fraction": lifted["valid_fraction"]})
            hd_in.append({"id": hid, "palm": pv.mean(0) if len(pv) >= 3 else None, "closed": st["closed"], "open": st["open"]})
        obj_in = {k: {"label": o["label"], "center": transform(o["center"][None], R, tt)[0], "radius": o["radius"]} for k, o in keys.items()}
        events = self.grasp.update(t, hd_in, obj_in)
        for e in events:
            e["phrase"] = phrase(e)
        self._objs = {k: {"label": o["label"], "center": o["center"]} for k, o in keys.items()}
        held_now = {v["obj"] for v in self.grasp.held().values()}
        if self.sink is not None:
            self.sink.log(t, image=image, live_points=pts, live_colors=colors, hands=hd_out, events=events,
                          objects=[{"key": k, "label": o["label"], "center": o["center"], "held": k in held_now} for k, o in keys.items()],
                          map_points=self.map.points(min_obs=3, core=0.5) if self.n % (self.map_every * 5) == 0 else None)
        return {"events": events, "cloud_info": info, "n_hands": len(hands), "detector": "on" if self.det else "off",
                "hands_tracker": "on" if self.hands else "off", "map_voxels": len(self.map)}


def main():
    from gabai.spike.demo_viewer import load_rectification, video_frames
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="data/left.mp4")
    ap.add_argument("--right", default="data/right.mp4")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--skip", type=int, default=3, help="use every Nth frame")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--f-frac", type=float, default=0.6)
    ap.add_argument("--baseline", type=float, default=0.065)
    ap.add_argument("--model", default="m", choices=["s", "m", "h"])
    ap.add_argument("--save", default=None, help="write a Rerun recording (.rrd) you can open later with `rerun file.rrd`")
    ap.add_argument("--no-viewer", action="store_true", help="do not open the Rerun window")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--no-detect", action="store_true")
    a = ap.parse_args()
    from gabai.spike.las2_runner import Las2Depth
    det = hands = None
    if not a.no_detect:
        try:
            from gabai.detect import UltralyticsDetector
            det = UltralyticsDetector()
        except Exception as e:
            print("detector OFF:", e)
    if not a.no_hands:
        try:
            from gabai.hands import HandTracker
            hands = HandTracker()
        except Exception as e:
            print("hands OFF:", e)
    las = Las2Depth(a.model)
    rect, *_ = load_rectification(a.left, a.right, a.width, os.path.join("data", f"autorect_{a.width}.npz"))
    sink = RerunSink(a.save, viewer=not a.no_viewer)
    pv, n, t0 = None, 0, time.time()
    for label, t, l, r, f, B, cy in video_frames(a.left, a.right, a.width, a.start, a.skip, a.f_frac, a.baseline, rect):
        if pv is None:
            pv = Preview3D(f, B, l.shape[1] / 2, cy, det, hands, sink)
            print(label, "| f=%.0f px, B=%.3f m assumed" % (f, B))
        disp, trusted, _ = las.disparity_lr_batched(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        out = pv.frame(t, l, np.where(trusted, disp, np.nan), trusted)
        for e in out["events"]:
            print(f"t={t:6.1f}s  {e['type']:8s} {e.get('label')}  {e.get('reason') or ''}")
        n += 1
        if n >= a.max_frames:
            break
    print(f"{n} frames in {time.time() - t0:.0f} s; map voxels {len(pv.map) if pv else 0}" + (f"; saved {a.save}" if a.save else ""))


if __name__ == "__main__":
    main()
