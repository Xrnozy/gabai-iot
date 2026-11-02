"""Hands, objects, and whether an object is actually HELD. Builds on gabai.hands, gabai.detect and gabai.grasp; adds the 2D/3D relation between each
hand and each object, with the reason in words.

Relation names, strongest first:
  held               the grasp rules confirmed a grab, and the object was seen moving with the hand
  held_unconfirmed   grab confirmed but the object could not be seen after the hand closed on it (hidden by the hand); treated as held, flagged
  contact            hand and object overlap in the picture AND their trusted 3D positions are within CONTACT_M: touching, but not shown to move together
  near               within NEAR_M in 3D, no picture overlap
  overlap_not_touching   they overlap in the picture but are more than CONTACT_M apart in depth (a hand in front of an object, or behind it)
  uncertain          they overlap in the picture and there is no trusted depth for one of them: contact cannot be told from line-of-sight overlap
  apart              none of the above
The numbers are starting values. Confidence values shown are the models' own scores; they are not probabilities that a hand or an object is real or
held. "Held" comes from the grasp rules, not from a score.
"""
import numpy as np

from gabai.detect import PEOPLE, det_depth, person_mask
from gabai.grasp import GraspTracker, phrase
from gabai.hands import finger_state, hand_mask
from gabai.cloud import transform
from gabai.pipeline.hand3d import HandIdentity, HandSmoother, HeldObjectModel, extract_object_points, hand_frame, oriented_box, solve_hand_3d

CONTACT_M = 0.12
NEAR_M = 0.30
PALM = (0, 5, 9, 13, 17)
MIN_OVERLAP = 0.05


def _hand_box(kp):
    kp = np.asarray(kp, float)
    return [float(kp[:, 0].min()), float(kp[:, 1].min()), float(kp[:, 0].max()), float(kp[:, 1].max())]


def _overlap_frac(a, b):
    """Share of box b covered by box a."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0) * max(y1 - y0, 0)
    return inter / max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)


class HeldAnalyzer:
    def __init__(self, f, B, cx, cy, detector=None, hand_tracker=None, metric=True, detector_every=1):
        self.f, self.B, self.cx, self.cy, self.metric = f, B, cx, cy, metric
        self.detector_every, self._n, self._last_dets, self._since_det = max(int(detector_every), 1), 0, None, 0
        self.det, self.hands = detector, hand_tracker
        self.grasp = GraspTracker()
        self._prev_objs = {}
        self.smoother = HandSmoother()
        self.identity = HandIdentity()
        self.models = {}                       # object key -> HeldObjectModel, in the frame of the hand that holds it

    def _lift_objects(self, dets, disp):
        out = []
        for d in dets:
            if d.label in PEOPLE:
                continue
            r = det_depth(d, disp, self.f, self.B)
            z = r["depth_m"]["v"]
            x0, y0, x1, y1 = d.box
            o = {"label": d.label.replace(" ", "_"), "conf": float(d.conf), "box": [float(x0), float(y0), float(x1), float(y1)], "det": d, "depth": r,
                 "center_cam": None, "radius": 0.05, "stale_frames": self._since_det}
            if z is not None and not r["weak"]:
                u, v = (x0 + x1) / 2, (y0 + y1) / 2
                o["center_cam"] = np.array([(u - self.cx) / self.f * z, (v - self.cy) / self.f * z, z])
                o["radius"] = max(0.04, 0.5 * (x1 - x0) / self.f * z)
            out.append(o)
        return out

    def _assoc(self, objs, R, t):
        keys, used = {}, set()
        for o in objs:
            o["center_world"] = None if o["center_cam"] is None else transform(o["center_cam"][None], R, t)[0]
            best = None
            if o["center_world"] is not None:
                for k, p in self._prev_objs.items():
                    if p["label"] == o["label"] and k not in used and p["center_world"] is not None:
                        d = float(np.linalg.norm(p["center_world"] - o["center_world"]))
                        if d < 0.15 and (best is None or d < best[0]):
                            best = (d, k)
            if best:
                k = best[1]
            else:
                i = 0
                while f"{o['label']}{i}" in self._prev_objs or f"{o['label']}{i}" in keys:
                    i += 1
                k = f"{o['label']}{i}"
            used.add(k)
            o["key"] = k
            keys[k] = o
        return keys

    def frame(self, t, bgr, disp, trusted, R_wc, t_wc):
        h, w = disp.shape
        status = {"hands": "on" if self.hands is not None else "off", "objects": "on" if self.det is not None else "off"}
        self._n += 1
        dets = []
        if self.det is not None:
            # The detector is the costly part and objects move slowly, so it runs every Nth frame and the last detections are reused in between (marked stale).
            # It always runs while something is held: the held object's mask has to follow the hand.
            if self._last_dets is None or self._n % self.detector_every == 0 or self.grasp.held():
                self._last_dets, self._since_det = self.det.detect(bgr), 0
            else:
                self._since_det += 1
            dets = self._last_dets
        hands = self.hands.detect(bgr, int(t * 1000)) if self.hands is not None else []
        objs = self._assoc(self._lift_objects(dets, disp), R_wc, t_wc)
        hd_out, hd_in = [], []
        sols = []
        for hnd in hands:
            sol = solve_hand_3d(hnd, disp, self.f, self.B, self.cx, self.cy, metric=self.metric)
            raw_w = transform(sol["xyz"], R_wc, t_wc)
            pr = raw_w[list(PALM)]
            pr = pr[np.isfinite(pr[:, 2])]
            kp = np.asarray(hnd.kp2d)
            sols.append((sol, raw_w, pr.mean(0) if len(pr) >= 3 else None, (float(kp[:, 0].mean()), float(kp[:, 1].mean()))))
        # Identity comes from tracking, not from MediaPipe's flipping left/right label: one id per real hand, kept while it moves.
        ids, retired = self.identity.assign([{"palm": s_[2], "px": s_[3]} for s_ in sols])
        for rid in retired:
            self.smoother.drop(rid)
        for hnd, (sol, raw_w, _palm_raw, _px), hid in zip(hands, sols, ids):
            xyz_w = self.smoother.update(hid, raw_w) if sol["ok"] else raw_w
            st = finger_state(hnd)
            pv = xyz_w[list(PALM)]
            pv = pv[np.isfinite(pv[:, 2])]
            pc = sol["xyz"][list(PALM)]
            pc = pc[np.isfinite(pc[:, 2])]
            palm_w = pv.mean(0) if len(pv) >= 3 else None
            state = "closed" if st["closed"] else ("open" if st["open"] else "between")
            fr = hand_frame(xyz_w) if sol["ok"] else None
            hd_out.append({"id": hid, "label": hnd.side, "kp2d": np.asarray(hnd.kp2d), "xyz_cam": sol["xyz"], "xyz_world": xyz_w, "palm_world": palm_w,
                           "palm_cam": pc.mean(0) if len(pc) >= 3 else None, "state": state, "ratio": st["ratio"], "side_score": hnd.score,
                           "valid_fraction": sol["stereo_valid_fraction"], "box": _hand_box(hnd.kp2d), "source": sol["source"], "solve": {k: sol[k] for k in ("ok", "reason", "scale", "n_stereo", "n_model", "palm_depth_m")},
                           "frame": fr})
            hd_in.append({"id": hid, "palm": palm_w, "closed": st["closed"], "open": st["open"]})
        obj_in = {k: {"label": o["label"], "center": o["center_world"], "radius": o["radius"]} for k, o in objs.items()}
        events = self.grasp.update(t, hd_in, obj_in)
        for e in events:
            e["phrase"] = phrase(e)
        held = self.grasp.held()
        relations = []
        for hd in hd_out:
            hv = held.get(hd["id"])
            for k, o in objs.items():
                if hv is not None and hv["obj"] == k:
                    st_ = self.grasp.h[hd["id"]]
                    following = bool(st_.followed)
                    relations.append({"hand": hd["id"], "obj": k, "relation": "held" if following else "held_unconfirmed",
                                      "why": ("grab confirmed by the grasp rules and the object was seen moving with the hand" if following else
                                              "grab confirmed, but the object was not seen after the hand closed on it (hidden by the hand): treated as held, not verified")})
                    continue
                relations.append(self._relation(hd, o))
            if hv is not None and hv["obj"] not in objs:
                relations.append({"hand": hd["id"], "obj": hv["obj"], "relation": "held_unconfirmed",
                                  "why": "grab confirmed, the object is not seen in this frame (hidden by the hand): treated as held, not verified"})
        self._prev_objs = {k: {"label": o["label"], "center_world": o["center_world"]} for k, o in objs.items()}
        models, released = self._update_models(t, hd_out, hands, objs, held, events, disp, trusted, R_wc, t_wc, (h, w))
        return {"hands": hd_out, "objects": list(objs.values()), "relations": relations, "events": events, "status": status, "models": models, "released": released,
                "exclude_mask": self._exclude(dets, hands, (h, w), objs, held)}

    def _update_models(self, t, hd_out, hands, objs, held, events, disp, trusted, R_wc, t_wc, shape):
        """Fuse what is seen of each held object into its model in the hand frame; draw every held model with the CURRENT hand frame."""
        models, released = [], []
        by_id = {h["id"]: (h, hands[i]) for i, h in enumerate(hd_out)}
        for hid, hv in held.items():
            key = hv["obj"]
            hd, hnd = by_id.get(hid, (None, None))
            label = objs[key]["label"] if key in objs else (self.models[key].label if key in self.models else key)
            m = self.models.setdefault(key, HeldObjectModel(key, label))
            entry = {"key": key, "label": label, "hand": hid, "state": "", "n_views": m.n_views, "frame": None}
            if hd is None or hd["frame"] is None:
                entry["state"] = "hand not in 3D this frame: " + ("hand not detected" if hd is None else hd["solve"]["reason"])
                models.append(entry)
                continue
            Rh, th = hd["frame"]
            m.last_frame = (Rh, th)
            seen = False
            st_ = self.grasp.h[hid]
            if key in objs and bool(st_.followed):
                o = objs[key]
                zref = o["depth"]["depth_m"]["v"] if o["depth"]["depth_m"]["v"] is not None else (hd["palm_cam"][2] if hd["palm_cam"] is not None else None)
                if zref is not None:
                    det = o["det"]
                    if det.mask is not None:
                        region = np.asarray(det.mask, bool)
                    else:
                        region = np.zeros(shape, bool)
                        x0, y0, x1, y1 = (int(v) for v in o["box"])
                        region[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
                    pts, info = extract_object_points(disp, trusted, region, hand_mask([hnd], shape, grow_px=6), self.f, self.B, self.cx, self.cy, z_ref=float(zref))
                    if len(pts) >= 20:
                        m.add_world_points(transform(pts, R_wc, t_wc), Rh, th)
                        m.last_seen_t, seen = t, True
                        entry["extract"] = info
            pl = m.points_local(min_obs=2)
            entry.update(state="observed this frame" if seen else ("hidden: shown from earlier views, attached to the hand" if len(pl) else "never seen well enough to build points"),
                         n_views=m.n_views, n_points=int(len(pl)), points_local=pl, box_local=oriented_box(pl), frame=(Rh, th), seen=seen)
            models.append(entry)
        for e in events:
            if e["type"] in ("release", "lost"):
                key = e["obj"]
                m = self.models.pop(key, None)
                if m is not None and getattr(m, "last_frame", None) is not None and len(m.points_local(2)):
                    Rh, th = m.last_frame
                    b = m.box_local()
                    released.append({"key": key, "label": m.label, "points_world": m.points_world(Rh, th, 2), "event": e["type"], "unobserved": e["type"] == "lost",
                                     "box_world": None if b is None else {"center": Rh @ b["center"] + th, "R": Rh @ b["R"], "half_sizes": b["half_sizes"]}})
            elif e["type"] == "cancel":
                self.models.pop(e["obj"], None)
        return models, released

    def _relation(self, hd, o):
        ov = _overlap_frac(hd["box"], o["box"])
        p, c = hd["palm_cam"], o["center_cam"]
        base = {"hand": hd["id"], "obj": o["key"]}
        if p is None or c is None:
            if ov >= MIN_OVERLAP:
                missing = "hand" if p is None else "object"
                return dict(base, relation="uncertain", why=f"they overlap in the picture ({ov * 100:.0f}% of the object) but the {missing} has no trusted depth, so contact cannot be told from overlap along the line of sight")
            return dict(base, relation="apart", why="no overlap in the picture and no depth to compare")
        d = float(np.linalg.norm(p - c))
        if ov >= MIN_OVERLAP and d <= CONTACT_M + o["radius"]:
            return dict(base, relation="contact", distance_m=round(d, 3),
                        why=f"overlap in the picture and {d * 100:.0f} cm apart in 3D (within {100 * (CONTACT_M + o['radius']):.0f} cm): touching, not shown to move with the hand")
        if ov >= MIN_OVERLAP:
            return dict(base, relation="overlap_not_touching", distance_m=round(d, 3), why=f"they overlap in the picture but are {d * 100:.0f} cm apart in 3D (in front of or behind)")
        if d <= NEAR_M:
            return dict(base, relation="near", distance_m=round(d, 3), why=f"{d * 100:.0f} cm apart in 3D, no overlap in the picture")
        return dict(base, relation="apart", distance_m=round(d, 3), why=f"{d * 100:.0f} cm apart")

    @staticmethod
    def _exclude(dets, hands, shape, objs, held):
        m = person_mask(dets, shape)
        if hands:
            m = m | hand_mask(hands, shape)
        for hv in held.values():
            o = objs.get(hv["obj"])
            if o is not None:
                x0, y0, x1, y1 = (int(v) for v in o["box"])
                m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        return m
