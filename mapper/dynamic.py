"""People as dynamic objects: tracked in the VIDEO (YOLO person masks, associated frame to frame by box overlap), kept out of the static maps.

A person is not geometry the map should remember: surfels, the semi-dense cloud and the TSDF would keep a ghost wherever someone walked through. `static_conf` zeroes the confidence of person pixels so
they never enter those maps. The tracker gives each person a persistent id, a position (median depth under the mask, lifted to the world with the node pose) and a velocity. A person who is no longer seen
stays as a GHOST (visible=False, drawn semi-transparent) for `ghost_s` seconds and then disappears; a ghost is never an obstacle. The walking guidance (mapper/navigate.py) still sees the person in the
current depth, so someone standing in the way now is an obstacle now.
Limit: velocity comes from the SLAM pose, so a loop-closure jump of the pose shows up as one frame of fake speed (smoothed by the EMA, and `moving` needs several hits).
"""
from dataclasses import dataclass, field

import numpy as np

from gabai.detect import person_mask
from mapper.align import OPTICAL


@dataclass
class Track:
    id: int
    box: tuple
    pos: object = None                  # world xyz of the person, None until a depth under the mask was trusted
    cam: object = None                  # camera optical frame: x right, y down, z forward (metres)
    vel: object = field(default_factory=lambda: np.zeros(3))
    hits: int = 1
    missed: int = 0
    seen_t: float = 0.0
    visible: bool = True
    points: object = None               # world points on the person (subsampled), for drawing

    @property
    def speed(self):
        return float(np.linalg.norm(self.vel))


def static_conf(conf, dets, grow_px=8):
    """conf with every person pixel set to 0 (copy): what the static maps may use."""
    m = person_mask(dets, conf.shape, grow_px=grow_px)
    return np.where(m, 0.0, conf).astype(conf.dtype) if m.any() else conf


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    def __init__(self, f, cx, cy, min_iou=0.15, max_missed=15, ghost_s=2.0, move_mps=0.3, min_hits=4, conf_min=0.5, max_points=300):
        self.f, self.cx, self.cy, self.min_iou, self.max_missed, self.ghost_s = f, cx, cy, min_iou, max_missed, ghost_s
        self.move_mps, self.min_hits, self.conf_min, self.max_points = move_mps, min_hits, conf_min, max_points
        self.tracks, self._next, self._cam_pos = [], 1, np.zeros(3)

    def moving(self, tr):
        return tr.visible and tr.hits >= self.min_hits and tr.speed >= self.move_mps

    def update(self, dets, depth, conf, T_wc, t):
        """dets: gabai.detect.Det list of one picture; depth (m) and conf at the picture size; T_wc the node pose; t seconds. Returns every track, ghosts included."""
        people = [d for d in dets if d.label == "person"]
        pairs = sorted(((_iou(tr.box, d.box), i, j) for i, tr in enumerate(self.tracks) if tr.visible for j, d in enumerate(people)), reverse=True)
        used_t, used_d, match = set(), set(), {}
        for s, i, j in pairs:
            if s >= self.min_iou and i not in used_t and j not in used_d:
                used_t.add(i)
                used_d.add(j)
                match[j] = i
        Tc = T_wc @ OPTICAL
        self._cam_pos = Tc[:3, 3]
        for tr in self.tracks:
            tr.missed += 1
            tr.visible = False
        for j, d in enumerate(people):
            if j in match:
                tr = self.tracks[match[j]]
                tr.hits += 1
            else:
                tr = Track(self._next, d.box)
                self._next += 1
                self.tracks.append(tr)
            tr.box, tr.missed, tr.visible = d.box, 0, True
            self._locate(tr, d, depth, conf, Tc, t)
            tr.seen_t = t
        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed and t - tr.seen_t <= self.ghost_s]
        return list(self.tracks)

    def _locate(self, tr, d, depth, conf, Tc, t):
        h, w = depth.shape
        if d.mask is not None:
            m = np.asarray(d.mask, bool)
        else:
            m = np.zeros((h, w), bool)
            x0, y0, x1, y1 = (int(round(v)) for v in d.box)
            m[max(y0, 0):y1, max(x0, 0):x1] = True
        ok = m & (depth > 0.2) & (conf >= self.conf_min)
        if ok.sum() < 30:
            return                                                              # no trusted depth under the person: keep the track, do not move it
        vs, us = np.nonzero(ok)
        z = depth[vs, us]
        zc = float(np.median(z))
        near = np.abs(z - zc) < 0.5                                              # the person, not the wall seen between an arm and the body
        vs, us, z = vs[near], us[near], z[near]
        opt = np.stack([(us - self.cx) * z / self.f, (vs - self.cy) * z / self.f, z], axis=1)
        cam = np.median(opt, axis=0)
        pos = Tc[:3, :3] @ cam + Tc[:3, 3]
        if tr.pos is not None and t > tr.seen_t + 1e-3:
            v = (pos - tr.pos) / (t - tr.seen_t)
            tr.vel = 0.6 * tr.vel + 0.4 * v if tr.hits > 1 else v
        tr.pos, tr.cam = pos, cam
        sel = np.linspace(0, len(z) - 1, min(len(z), self.max_points)).astype(int)
        tr.points = (Tc[:3, :3] @ opt[sel].T).T + Tc[:3, 3]

    def hazards(self, tracks, hazard_m=3.0):
        """Relay hazards for the VISIBLE people within hazard_m: zone from the heading, whether they move and whether they come towards you. Ghosts are not hazards."""
        from gabai.relay.results import _zone
        out = []
        for tr in tracks:
            if not tr.visible or tr.cam is None or tr.cam[2] > hazard_m:
                continue
            heading = float(np.degrees(np.arctan2(-tr.cam[0], tr.cam[2])))
            away = float(np.dot(tr.vel, tr.pos - self._cam_pos)) if tr.pos is not None else 0.0      # > 0: moving away from the camera
            out.append({"zone": _zone(heading), "distance_m": round(float(tr.cam[2]), 1), "kind": "person", "id": tr.id, "moving": self.moving(tr), "approaching": self.moving(tr) and away < 0,
                        "speed_mps": round(tr.speed, 1)})
        return out
