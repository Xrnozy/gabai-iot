"""Egocentric view of the map: objects in the world -> "what is where relative to the user now", as lexicon packets (gabai/net/lexicon.py).

Input per call: the camera pose T_wc (RTAB-Map body frame: x forward, y left, z up) and the map's objects (dicts with id, label, p_w, dims, as from mapper.semantics.instances).
For each object: distance bucket, direction from the azimuth (left positive), a state from how fast the distance changes between calls ("approaching" / "moving away"),
and one relation to the nearest other object ("on top of" when it rests on it, "next to" when the gap is small). Only the nearest `top` objects are reported, and with
delta=True only those whose packet changed since the last call. Distances are to the object's centre minus its half size, so a big table is not "far" while touching it.
"""
import numpy as np

from gabai.net.lexicon import IDS, packet

NAMES = {"person": "human", "cell phone": "phone", "potted plant": "plant"}     # mapper class -> lexicon word; others are the same word
DIST = ((0.5, "very close to you"), (1.5, "near you"), (3.0, "a few steps away"), (5.0, "far from you"), (1e9, "very far from you"))
SPEED = 0.3                                                                      # m/s of change in distance to call it approaching / moving away
NEXT_TO_GAP = 0.3


def direction(x, y, z, horiz):
    """Lexicon direction for a point in the body frame (x forward, y left, z up)."""
    if horiz < 0.7 and abs(z) > 1.0:
        return "above you" if z > 0 else "below you"
    a = abs(np.degrees(np.arctan2(y, x)))
    side = y >= 0                                                                # left
    if a < 30:
        return "in front of you"
    if a < 75:
        return "front left" if side else "front right"
    if a < 120:
        return "on your left" if side else "on your right"
    if a < 165:
        return "behind you on the left" if side else "behind you on the right"
    return "behind you"


def bucket(d):
    return next(w for lim, w in DIST if d < lim)


class Egocentric:
    def __init__(self, top=5, delta=False):
        self.top, self.delta, self.last, self.range, self.rows = top, delta, {}, {}, []

    def update(self, t, T_wc, objs, up_w=None):
        """Returns {"d": [packets]} (lexicon message body) for the objects worth saying now; [] packets are left out."""
        if up_w is not None:                                                     # gravity-levelled body frame: forward = the camera's forward made horizontal, up as given
            u = np.asarray(up_w, float) / np.linalg.norm(up_w)
            f = T_wc[:3, 0] - (T_wc[:3, 0] @ u) * u
            f = f / max(np.linalg.norm(f), 1e-9)
            L = np.stack([f, np.cross(u, f), u])
            to_body = lambda pw: L @ (np.asarray(pw, float) - T_wc[:3, 3])
        else:
            Tinv = np.linalg.inv(T_wc)
            to_body = lambda pw: Tinv[:3, :3] @ np.asarray(pw, float) + Tinv[:3, 3]
        rows = []
        for o in objs:
            word = NAMES.get(o["label"], o["label"])
            if word not in IDS["obj"]:
                continue
            p = to_body(o["p_w"])
            half = float(np.linalg.norm(np.asarray(o["dims"], float)[:2])) / 2
            horiz = float(np.hypot(p[0], p[1]))
            d = max(horiz - half, 0.0)
            prev = self.range.get(o["id"])
            self.range[o["id"]] = (t, d)
            v = (d - prev[1]) / (t - prev[0]) if prev and t > prev[0] else 0.0
            state = "approaching" if v < -SPEED else "moving away" if v > SPEED else None
            rows.append({"o": o, "word": word, "d": d, "dir": direction(*p, horiz), "state": state, "p": p})
        rows.sort(key=lambda r: r["d"])
        self.rows = rows                                                        # every mapped object with its body-frame position, for drawing
        out = []
        for r in rows[: self.top]:
            rel = self._relation(r["o"], [x["o"] for x in rows if x is not r])
            pk = packet(r["word"], bucket(r["d"]), r["dir"], *(rel if rel else (None, None)), r["state"])
            if self.delta and self.last.get(r["o"]["id"]) == pk:
                continue
            self.last[r["o"]["id"]] = pk
            out.append(pk)
        return {"d": out} if out else {}

    @staticmethod
    def _relation(o, others):
        """(relation, other word) for the nearest other object that touches or is next to this one, else None."""
        lo, hi = np.asarray(o["p_w"], float) - np.asarray(o["dims"], float) / 2, np.asarray(o["p_w"], float) + np.asarray(o["dims"], float) / 2
        best = None
        for q in others:
            w = NAMES.get(q["label"], q["label"])
            if w not in IDS["obj"]:
                continue
            qlo, qhi = np.asarray(q["p_w"], float) - np.asarray(q["dims"], float) / 2, np.asarray(q["p_w"], float) + np.asarray(q["dims"], float) / 2
            gap = np.maximum(np.maximum(qlo - hi, lo - qhi), 0.0)
            g = float(np.linalg.norm(gap[:2]))
            if abs(lo[2] - qhi[2]) < 0.15 and g == 0:
                cand = (0.0, "on top of", w)
            elif g < NEXT_TO_GAP:
                cand = (g, "next to", w)
            else:
                continue
            best = cand if best is None or cand[0] < best[0] else best
        return best[1:] if best else None
