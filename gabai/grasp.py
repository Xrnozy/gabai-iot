"""Grab and release rules (spec phase 3, after ETH's Lost & Found).

Input per frame, all in one common frame (world, so a moving head does not look like a moving hand):
  hands   [{"id", "palm": xyz or None, "closed": bool, "open": bool}]   palm = e.g. the mean of the wrist and the four knuckles
  objects {key: {"label", "center": xyz or None, "radius"}}              from detections lifted with the depth map
Output: a list of events {"t", "type": grab|release|cancel|lost, "hand", "obj", "label", "pos", "reason", ...}.

Rules, with the constants named at the top:
  grab     the hand is closed and within (radius + NEAR_M) of an object for CONFIRM_S, then moves MOVE_AWAY_M (8 cm) away with it. While it moves
           the object must either be seen following the hand, or not be seen at all (hidden by the hand). If the object is seen staying
           where it was, the candidate is cancelled with the reason. A confirmed grab says whether the object was seen following.
  held     the object's position is the palm plus the offset it had at the grab, every frame.
  release  the hand is open and slower than RELEASE_SPEED for RELEASE_S. The object stays at its last held position.
  lost     the hand is not seen for LOST_S while holding: the object keeps its last position and the event says it is unobserved.
These are starting values to tune on real pick-ups; nothing here has been measured against the spec's 9-of-10 target.
"""
import numpy as np

NEAR_M = 0.06
CONFIRM_S = 0.2
MOVE_AWAY_M = 0.08
FOLLOW_TOL_M = 0.03             # object this close to (palm + its offset at the start) is following the hand
STAY_TOL_M = 0.03               # object this close to where it started has stayed
RELEASE_SPEED = 0.15            # m/s
RELEASE_S = 0.3
LOST_S = 1.0


def phrase(ev):
    return {"grab": "holding: {}", "release": "put down: {}"}.get(ev["type"], "").format(ev["label"])


class _H:
    def __init__(self):
        self.state, self.obj, self.t0, self.p0, self.offset = "idle", None, 0.0, None, None
        self.prev, self.prev_t, self.slow_since, self.unseen_since, self.center = None, None, None, None, None
        self.followed = None


class GraspTracker:
    def __init__(self):
        self.h = {}

    def held(self):
        return {k: {"obj": v.obj, "center": v.center} for k, v in self.h.items() if v.state == "holding"}

    def update(self, t, hands, objects):
        ev = []
        seen = set()
        for hd in hands:
            seen.add(hd["id"])
            s = self.h.setdefault(hd["id"], _H())
            if hd["palm"] is None:
                self._unseen(t, hd["id"], s, objects, ev)
                continue
            s.unseen_since = None
            palm = np.asarray(hd["palm"], float)
            speed = 0.0 if s.prev is None else float(np.linalg.norm(palm - s.prev) / max(t - s.prev_t, 1e-6))
            if s.state == "idle":
                self._idle(t, hd, s, palm, objects)
            elif s.state == "candidate":
                self._candidate(t, hd, s, palm, objects, ev)
            else:
                self._holding(t, hd, s, palm, speed, objects, ev)
            s.prev, s.prev_t = palm, t
        for hid, s in self.h.items():
            if hid not in seen:
                self._unseen(t, hid, s, objects, ev)
        return ev

    # -- states -------------------------------------------------------------------------------------------------------------------
    def _nearest(self, palm, objects, taken=()):
        best = None
        for k, o in objects.items():
            if o["center"] is None or k in taken:
                continue
            d = float(np.linalg.norm(np.asarray(o["center"], float) - palm))
            if d <= o.get("radius", 0.05) + NEAR_M and (best is None or d < best[0]):
                best = (d, k)
        return best

    def _idle(self, t, hd, s, palm, objects):
        n = self._nearest(palm, objects, {x.obj for x in self.h.values() if x.state == "holding"}) if hd["closed"] else None
        if n:
            s.state, s.obj, s.t0, s.p0 = "candidate", n[1], t, palm.copy()
            s.label = objects[n[1]]["label"]                  # remembered: the detection may vanish once the hand covers the object
            s.obj_p0 = np.asarray(objects[n[1]]["center"], float).copy()

    def _candidate(self, t, hd, s, palm, objects, ev):
        o = objects.get(s.obj)
        if not hd["closed"]:
            return self._cancel(t, hd, s, objects, ev, None)          # opened again before leaving: nothing happened
        near = o is not None and o["center"] is not None and np.linalg.norm(np.asarray(o["center"], float) - palm) <= o.get("radius", 0.05) + NEAR_M
        moved = float(np.linalg.norm(palm - s.p0))
        if moved < MOVE_AWAY_M:
            if t - s.t0 > 2.0:
                return self._cancel(t, hd, s, objects, ev, "closed hand stayed at the object for over 2 s without moving it")
            if o is not None and o["center"] is not None and not near and t - s.t0 < CONFIRM_S:
                return self._cancel(t, hd, s, objects, ev, "hand left the object before it was held long enough")
            return
        if t - s.t0 < CONFIRM_S:
            return self._cancel(t, hd, s, objects, ev, "hand passed the object too fast to be a grab (under %.1f s near it)" % CONFIRM_S)
        c = None if o is None else o["center"]
        if c is None:
            s.followed = False
        else:
            c = np.asarray(c, float)
            if np.linalg.norm(c - s.obj_p0) <= STAY_TOL_M:
                return self._cancel(t, hd, s, objects, ev, "closed hand moved away but the object stayed where it was")
            if np.linalg.norm(c - (palm + s.obj_p0 - s.p0)) > FOLLOW_TOL_M:
                return                                                  # neither following nor staying: not decided yet
            s.followed = True
        s.state, s.offset = "holding", s.obj_p0 - s.p0
        s.center, s.slow_since = palm + s.offset, None
        ev.append({"t": t, "type": "grab", "hand": hd["id"], "obj": s.obj, "label": s.label, "pos": s.center.copy(),
                   "object_observed_following": s.followed})

    def _holding(self, t, hd, s, palm, speed, objects, ev):
        s.center = palm + s.offset
        if hd["open"] and speed < RELEASE_SPEED:
            s.slow_since = t if s.slow_since is None else s.slow_since
            if t - s.slow_since >= RELEASE_S:
                ev.append({"t": t, "type": "release", "hand": hd["id"], "obj": s.obj, "label": self._label(objects, s), "pos": s.center.copy()})
                self._reset(s)
        else:
            s.slow_since = None

    def _unseen(self, t, hid, s, objects, ev):
        if s.state == "candidate":
            return self._cancel(t, {"id": hid}, s, objects, ev, "hand lost while deciding")
        if s.state != "holding":
            return
        s.unseen_since = t if s.unseen_since is None else s.unseen_since
        if t - s.unseen_since >= LOST_S:
            ev.append({"t": t, "type": "lost", "hand": hid, "obj": s.obj, "label": self._label(objects, s), "pos": s.center.copy(),
                       "reason": "hand not seen for %.1f s while holding; the object is at its last held position, unobserved" % LOST_S})
            self._reset(s)

    # -- helpers ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _label(objects, s):
        return objects.get(s.obj, {}).get("label") or getattr(s, "label", None) or str(s.obj)

    @staticmethod
    def _reset(s):
        s.state, s.obj, s.offset, s.center, s.slow_since, s.followed = "idle", None, None, None, None, None

    def _cancel(self, t, hd, s, objects, ev, reason):
        if reason:
            ev.append({"t": t, "type": "cancel", "hand": hd["id"], "obj": s.obj, "label": self._label(objects, s), "pos": None, "reason": reason})
        self._reset(s)
