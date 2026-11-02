"""Object memory (spec phase 5): where things were last seen, kept in SQLite, answered as "where are my keys?".

World frame: x forward, y left, z up (the frame the rest of Gabai uses). Nothing here invents confidence. Each answer carries what is
actually known: the state (seen / held / placed / unobserved), where the evidence came from (a detection, or a hand placing it), how many
times it was seen, how long ago, and whether the spot was later looked at and found empty (`mark_absent`). An object is never deleted just because it
was not seen again: "not seen there since ..." is said instead, because absence from view is not absence from the place.

What it does NOT do: detect objects. Detections come from a detector (people/objects from YOLO today; "keys" is not a class in the COCO model
that is installed, so finding keys needs an open-vocabulary detector such as YOLOE, which is not set up). Sightings must be given positions in
the world frame by the caller (camera pose from SLAM, depth from the stereo).
"""
import sqlite3
import time

import numpy as np

SAME_PLACE_M = 0.30

SCHEMA = """CREATE TABLE IF NOT EXISTS objects(
  id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT NOT NULL, x REAL, y REAL, z REAL, first_t REAL, last_t REAL, n_obs INTEGER,
  state TEXT, source TEXT, absent_checks INTEGER DEFAULT 0, last_absent_t REAL);
CREATE INDEX IF NOT EXISTS ix_label ON objects(label);"""


def heading(R_wc):
    """Unit ground direction a camera looks along: its forward axis (third column of the camera-to-world rotation) projected onto the floor.
    Looking straight up or down has no heading and raises ValueError, so a caller cannot get a made-up direction."""
    f = np.asarray(R_wc, float)[:2, 2]
    n = np.linalg.norm(f)
    if n < 0.2:
        raise ValueError("camera looks nearly straight up or down: no ground heading")
    return f / n


def bearing(pos, cam_pos, heading_xy):
    """Distance along the ground and the side of an object, seen from cam_pos facing heading_xy (a unit vector in the ground plane)."""
    v = np.asarray(pos, float) - np.asarray(cam_pos, float)
    fwd = np.asarray(heading_xy, float)
    h = v[:2]
    along = float(h @ fwd)
    left = float(h[0] * -fwd[1] + h[1] * fwd[0])
    ang = np.degrees(np.arctan2(left, along))
    side = "ahead" if abs(ang) <= 30 else ("behind" if abs(ang) > 150 else ("left" if ang > 0 else "right"))
    return {"dist_m": float(np.linalg.norm(h)), "angle_deg": float(ang), "side": side, "above_m": float(v[2])}


def _age(s):
    if s < 90:
        return f"{int(round(s))} seconds ago"
    if s < 5400:
        return f"{int(round(s / 60))} minutes ago"
    return f"{s / 3600:.1f} hours ago"


def describe(r, cam_pos, heading_xy):
    if cam_pos is None:
        where = ""
    else:
        b = bearing(r["pos"], cam_pos, heading_xy)
        where = f"about {b['dist_m']:.0f} meters, " if b["dist_m"] >= 0.75 else "within arm's reach, "
        where += b["side"] if b["side"] in ("ahead", "behind") else f"to your {b['side']}"
        if b["side"] == "ahead" and abs(b["angle_deg"]) > 10:
            where += " and to the " + ("left" if b["angle_deg"] > 0 else "right")
        where += ". "
    src = {"unobserved": "I last saw it in a hand and then lost the hand, so this spot is unobserved. ",
           "held": "It is in your hand. "}.get(r["state"], "")
    s = f"{r['label']}: {where}{src}Last {'placed' if r['state'] == 'placed' else 'seen'} {_age(r['age_s'])}."
    if r.get("absent_checks"):
        s += f" Not seen there since {_age(r['age_s'] if r['last_absent_t'] is None else r['now'] - r['last_absent_t'])}."
    return s


class ObjectMemory:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.commit()
        self.db.close()

    # -- writing ------------------------------------------------------------------------------------------------------------------
    def _near(self, label, pos):
        best = None
        for r in self.db.execute("SELECT * FROM objects WHERE label=?", (label,)):
            d = float(np.linalg.norm(np.array([r["x"], r["y"], r["z"]]) - np.asarray(pos, float)))
            if d <= SAME_PLACE_M and (best is None or d < best[0]):
                best = (d, r)
        return best[1] if best else None

    def observe(self, label, pos, t, state="seen", source="detected"):
        pos = np.asarray(pos, float)
        r = self._near(label, pos)
        if r is None:
            self.db.execute("INSERT INTO objects(label,x,y,z,first_t,last_t,n_obs,state,source) VALUES(?,?,?,?,?,?,1,?,?)",
                            (label, *pos, t, t, state, source))
        else:
            n = r["n_obs"]
            new = (np.array([r["x"], r["y"], r["z"]]) * n + pos) / (n + 1)
            self.db.execute("UPDATE objects SET x=?,y=?,z=?,last_t=?,n_obs=?,state=?,source=?,absent_checks=0,last_absent_t=NULL WHERE id=?",
                            (*new, t, n + 1, state, source, r["id"]))
        self.db.commit()

    def apply_event(self, e):
        """Take a grab/release/lost event from gabai.grasp.GraspTracker."""
        pos, t, label = e["pos"], e["t"], e["label"]
        if e["type"] == "grab":
            self._set_latest(label, pos, t, "held", "grabbed")
        elif e["type"] == "release":
            self._set_latest(label, pos, t, "placed", "placed by hand")
        elif e["type"] == "lost":
            self._set_latest(label, pos, t, "unobserved", "hand lost while holding")

    def _set_latest(self, label, pos, t, state, source):
        r = self.db.execute("SELECT * FROM objects WHERE label=? ORDER BY last_t DESC LIMIT 1", (label,)).fetchone()
        pos = np.asarray(pos, float)
        if r is None:
            self.db.execute("INSERT INTO objects(label,x,y,z,first_t,last_t,n_obs,state,source) VALUES(?,?,?,?,?,?,1,?,?)", (label, *pos, t, t, state, source))
        else:
            self.db.execute("UPDATE objects SET x=?,y=?,z=?,last_t=?,n_obs=n_obs+1,state=?,source=?,absent_checks=0,last_absent_t=NULL WHERE id=?",
                            (*pos, t, state, source, r["id"]))
        self.db.commit()

    def mark_absent(self, label, pos, t):
        """The spot was in view and the object was not there. Recorded, not deleted."""
        r = self._near(label, pos)
        if r is not None:
            self.db.execute("UPDATE objects SET absent_checks=absent_checks+1, last_absent_t=? WHERE id=?", (t, r["id"]))
            self.db.commit()

    # -- reading ------------------------------------------------------------------------------------------------------------------
    def _rows(self, rows, now):
        return [{"id": r["id"], "label": r["label"], "pos": np.array([r["x"], r["y"], r["z"]]), "first_t": r["first_t"], "last_t": r["last_t"],
                 "age_s": now - r["last_t"], "n_obs": r["n_obs"], "state": r["state"], "source": r["source"], "absent_checks": r["absent_checks"],
                 "last_absent_t": r["last_absent_t"], "now": now} for r in rows]

    def where(self, label, now=None):
        now = time.time() if now is None else now
        return self._rows(self.db.execute("SELECT * FROM objects WHERE label=? ORDER BY last_t DESC", (label,)).fetchall(), now)

    def all(self, now=None):
        now = time.time() if now is None else now
        return self._rows(self.db.execute("SELECT * FROM objects ORDER BY last_t DESC").fetchall(), now)

    def answer(self, label, cam_pos, heading_xy, now=None):
        r = self.where(label, now)
        if not r:
            return f"I have not seen {label}."
        return describe(r[0], cam_pos, heading_xy) + (f" I also have {len(r) - 1} older place{'s' if len(r) > 2 else ''}." if len(r) > 1 else "")
