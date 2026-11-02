"""State and rules of the caregiver relay: what each device last said, how old that is, and which safety events follow.

Everything a caregiver sees carries its age. Nothing is called "live" because it once was: connection state is derived from
when the server last HEARD from a device, never from the last status the device reported about itself.
"""
import json
import os
import sqlite3
import threading
import time

from .geo import distance_to_route_m, haversine_m

PHONE_STALE_S = 25            # phone posts every 5-10 s; 2.5 missed cycles = offline
PI_STALE_S = 6                # Pi frames arrive several times a second
NAV_STALE_S = 5
LOCATION_FRESH_S = 30
GOOD_ACC_M, WEAK_ACC_M = 20.0, 50.0
MAX_ODOMETER_ACC_M = 50.0     # fixes worse than this never add distance
DEVIATION_M, DEVIATION_FIXES = 30.0, 2
INACTIVITY_S, INACTIVITY_RADIUS_M = 10 * 60, 15.0
EMERGENCY_KINDS = ("fall", "sos")
PENDING_ESCALATE_S = 60       # a "pending" fall nobody resolved: the phone probably died
STATUS_RANK = {"pending": 0, "cancelled": 1, "confirmed": 1, "acknowledged": 2, "resolved": 3}
SCHEMA = """
CREATE TABLE IF NOT EXISTS locations(id TEXT PRIMARY KEY, t REAL, recv REAL, lat REAL, lon REAL, acc REAL, speed REAL, bearing REAL, provider TEXT, heading REAL);
CREATE INDEX IF NOT EXISTS ix_loc_t ON locations(t);
CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, kind TEXT, status TEXT, t REAL, lat REAL, lon REAL, confidence REAL, evidence TEXT,
  ack_by TEXT, ack_t REAL, updated REAL);
CREATE TABLE IF NOT EXISTS conn_events(n INTEGER PRIMARY KEY AUTOINCREMENT, device TEXT, state TEXT, reason TEXT, t REAL);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
"""


def _num(v):
    return float(v) if isinstance(v, (int, float)) and v == v else None


class Relay:
    def __init__(self, db_path=":memory:", clock=time.time):
        self.now = clock
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._seen = {}                      # device -> server time of last contact
        self._conn_state = {}                # device -> "online" | "offline"
        self._pi = {"health": {}, "hello": {}, "connected": False, "last_error": None}
        self._deviation_run = 0
        self._deviation_open = False
        self._inactive_open = False
        self._caregivers = {}                # client id -> last seen
        self._load()

    def _load(self):
        with self._lock:
            for k in ("phone", "nav", "route", "odometer", "pi_last"):
                row = self._db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
                setattr(self, "_" + k, json.loads(row["v"]) if row else None)
            self._odometer = self._odometer or {"m": 0.0, "since": self.now()}
            self._pi_last = self._pi_last or {}
            for d in ("phone", "pi"):                               # a restart is not proof a device is online
                self._conn_state[d] = "offline"

    def _put(self, k, v):
        self._db.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))

    # ---------- phone ----------
    def ingest_phone(self, batch):
        """One idempotent batch: replaying it after a dropped response changes nothing."""
        now = self.now()
        acks = {}
        with self._lock:
            self._touch("phone", now, "ingest")
            newest = None
            for loc in (batch.get("locations") or [])[:500]:
                row = self._store_location(loc, now)
                if row and (newest is None or row["t"] > newest["t"]):
                    newest = row
            if newest:
                self._on_newest_fix(newest)
            tel = batch.get("telemetry")
            if isinstance(tel, dict) and self._newer("phone", tel):
                self._phone = {**tel, "recv": now}
                self._put("phone", self._phone)
            nav = batch.get("nav")
            if isinstance(nav, dict):
                self.ingest_nav(nav, now)
            for ev in (batch.get("events") or [])[:50]:
                acks[str(ev.get("id"))] = self._upsert_event(ev, now)
            self._db.commit()
            row = self._db.execute("SELECT id,kind,status,ack_by FROM events WHERE kind IN ('fall','sos') AND t>? ORDER BY t DESC LIMIT 1",
                                   (now - 3600,)).fetchone()
        return {"ok": True, "server_time": now, "events": acks, "latest_emergency": dict(row) if row else None, **self.link_info()}

    def _newer(self, key, payload):
        old = getattr(self, "_" + key, None)
        t = _num(payload.get("t"))
        return old is None or t is None or t >= (_num(old.get("t")) or 0)

    def _store_location(self, loc, now):
        lat, lon, t = _num(loc.get("lat")), _num(loc.get("lon")), _num(loc.get("t"))
        if lat is None or lon is None or t is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        lid = str(loc.get("id") or f"{t:.3f}")
        cur = self._db.execute(
            "INSERT OR IGNORE INTO locations VALUES(?,?,?,?,?,?,?,?,?,?)",
            (lid, t, now, lat, lon, _num(loc.get("acc")), _num(loc.get("speed")), _num(loc.get("bearing")), str(loc.get("provider") or ""), _num(loc.get("heading"))))
        if cur.rowcount == 0:
            return None
        return {"t": t, "lat": lat, "lon": lon, "acc": _num(loc.get("acc")), "speed": _num(loc.get("speed"))}

    def _on_newest_fix(self, fix):
        latest = self._latest_fix()
        if latest and latest["t"] > fix["t"]:                        # a queued backlog arriving late: history only
            return
        prev = self._prev_fix(fix["t"])
        acc = fix["acc"]
        if prev and (acc is None or acc <= MAX_ODOMETER_ACC_M):
            step = haversine_m(prev["lat"], prev["lon"], fix["lat"], fix["lon"])
            if step > max(3.0, (acc or 0) * 0.5):                    # below the noise floor it is jitter, not walking
                self._odometer["m"] += step
                self._put("odometer", self._odometer)
        self._check_deviation(fix)

    def _latest_fix(self):
        r = self._db.execute("SELECT * FROM locations ORDER BY t DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    def _prev_fix(self, t):
        r = self._db.execute("SELECT * FROM locations WHERE t<? AND (acc IS NULL OR acc<=?) ORDER BY t DESC LIMIT 1",
                             (t, MAX_ODOMETER_ACC_M)).fetchone()
        return dict(r) if r else None

    def _check_deviation(self, fix):
        if not self._route or not self._route.get("points"):
            self._deviation_run, self._deviation_open = 0, False
            return
        d = distance_to_route_m(fix["lat"], fix["lon"], self._route["points"])
        limit = DEVIATION_M + (fix["acc"] or 0)
        if d is not None and d > limit:
            self._deviation_run += 1
            if self._deviation_run >= DEVIATION_FIXES and not self._deviation_open:
                self._deviation_open = True
                self._upsert_event({"id": f"dev-{int(fix['t'])}", "kind": "deviation", "status": "confirmed", "t": fix["t"],
                                    "lat": fix["lat"], "lon": fix["lon"], "evidence": {"off_route_m": round(d)}}, self.now())
        else:
            self._deviation_run, self._deviation_open = 0, False

    def ingest_nav(self, nav, now=None):
        now = now or self.now()
        with self._lock:
            if self._newer("nav", nav):
                self._nav = {**nav, "recv": now}
                self._put("nav", self._nav)
                self._db.commit()

    # ---------- events ----------
    def _upsert_event(self, ev, now):
        eid, kind, status = str(ev.get("id") or ""), str(ev.get("kind") or ""), str(ev.get("status") or "confirmed")
        if not eid or not kind or status not in STATUS_RANK:
            return "rejected"
        row = self._db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
        evidence = json.dumps(ev.get("evidence") or {})
        if row is None:
            self._db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,NULL,NULL,?)",
                             (eid, kind, status, _num(ev.get("t")) or now, _num(ev.get("lat")), _num(ev.get("lon")),
                              _num(ev.get("confidence")), evidence, now))
            return "created"
        # A replay or late phone update must never undo a caregiver's acknowledgement.
        if STATUS_RANK[status] >= STATUS_RANK[row["status"]] and status != row["status"] and row["status"] not in ("acknowledged", "resolved"):
            self._db.execute("UPDATE events SET status=?, updated=? WHERE id=?", (status, now, eid))
            return "updated"
        return "duplicate"

    def acknowledge(self, event_id, who="caregiver"):
        now = self.now()
        with self._lock:
            cur = self._db.execute("UPDATE events SET status='acknowledged', ack_by=?, ack_t=?, updated=? "
                                   "WHERE id=? AND status IN ('pending','confirmed')", (who, now, now, event_id))
            self._db.commit()
            return cur.rowcount == 1

    def events(self, limit=100, since=0.0):
        with self._lock:
            rows = self._db.execute("SELECT * FROM events WHERE t>? ORDER BY t DESC LIMIT ?", (since, min(int(limit), 500))).fetchall()
        return [self._event_out(r) for r in rows]

    @staticmethod
    def _event_out(r):
        d = dict(r)
        d["evidence"] = json.loads(d["evidence"] or "{}")
        return d

    # ---------- route, history ----------
    def set_route(self, name, points):
        pts = [[float(a), float(b)] for a, b in points][:2000]
        with self._lock:
            self._route = {"name": str(name)[:60], "points": pts, "set": self.now()}
            self._put("route", self._route)
            self._deviation_run, self._deviation_open = 0, False
            self._db.commit()

    def clear_route(self):
        with self._lock:
            self._route = None
            self._db.execute("DELETE FROM kv WHERE k='route'")
            self._deviation_run, self._deviation_open = 0, False
            self._db.commit()

    def history(self, since=0.0, limit=2000):
        with self._lock:
            rows = self._db.execute("SELECT t,lat,lon,acc,speed,bearing,heading FROM locations WHERE t>? ORDER BY t DESC LIMIT ?",
                                    (since, min(int(limit), 5000))).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ---------- Pi ----------
    def update_pi(self, stats):
        """Called by the Pi bridge with LiveReceiver.stats(). Liveness comes from the age of the newest frame, not from the
        socket still being open or the last health report being present."""
        now = self.now()
        with self._lock:
            h = stats.get("health") or {}
            age = stats.get("frame_age_s")
            self._pi.update(health=h, hello=stats.get("hello") or {}, connected=bool(stats.get("connected")),
                            last_error=stats.get("last_error"),
                            stats={k: stats.get(k) for k in ("fps", "bytes", "frames", "decode_errors", "overwritten", "frame_age_s")})
            if self._pi["connected"] and age is not None:
                self._seen["pi"] = now - age
            if h:
                self._pi_last = {"health": h, "hello": self._pi["hello"], "recv": now}
                self._put("pi_last", self._pi_last)
                self._db.commit()

    # ---------- connections ----------
    def _touch(self, device, now, reason):
        self._seen[device] = now
        if self._conn_state.get(device) != "online":
            self._conn_state[device] = "online"
            self._db.execute("INSERT INTO conn_events(device,state,reason,t) VALUES(?,?,?,?)", (device, "online", reason, now))

    def caregiver_seen(self, client_id):
        with self._lock:
            self._caregivers[client_id] = self.now()

    def tick(self):
        """Run every couple of seconds: turns silence into offline events, stale pending falls into confirmed, and long stillness into an alert."""
        now = self.now()
        with self._lock:
            for dev, limit in (("phone", PHONE_STALE_S), ("pi", PI_STALE_S)):
                seen = self._seen.get(dev)
                online = seen is not None and now - seen <= limit
                if dev == "pi":
                    online = online and self._pi["connected"]
                if not online and self._conn_state.get(dev) == "online":
                    reason = (self._pi["last_error"] if dev == "pi" else None) or "no data received"
                    self._conn_state[dev] = "offline"
                    self._db.execute("INSERT INTO conn_events(device,state,reason,t) VALUES(?,?,?,?)", (dev, "offline", reason, now))
                elif online and self._conn_state.get(dev) != "online":
                    self._touch(dev, now, "data received")
            for r in self._db.execute("SELECT id FROM events WHERE status='pending' AND ? - updated > ?", (now, PENDING_ESCALATE_S)).fetchall():
                self._db.execute("UPDATE events SET status='confirmed', updated=? WHERE id=?", (now, r["id"]))
            self._check_inactivity(now)
            self._db.commit()

    def _check_inactivity(self, now):
        if self._conn_state.get("phone") != "online":
            return
        rows = self._db.execute("SELECT lat,lon,t FROM locations WHERE t>? ORDER BY t", (now - INACTIVITY_S,)).fetchall()
        if len(rows) < 3 or rows[-1]["t"] - rows[0]["t"] < INACTIVITY_S * 0.9:
            self._inactive_open = False
            return
        far = max(haversine_m(rows[0]["lat"], rows[0]["lon"], r["lat"], r["lon"]) for r in rows)
        if far <= INACTIVITY_RADIUS_M:
            if not self._inactive_open:
                self._inactive_open = True
                last = rows[-1]
                self._upsert_event({"id": f"idle-{int(rows[0]['t'])}", "kind": "inactivity", "status": "confirmed", "t": now,
                                    "lat": last["lat"], "lon": last["lon"], "evidence": {"still_minutes": round(INACTIVITY_S / 60)}}, now)
        else:
            self._inactive_open = False

    def link_info(self):
        now = self.now()
        pi_age = None if self._seen.get("pi") is None else now - self._seen["pi"]
        watchers = sum(1 for t in self._caregivers.values() if now - t < 30)
        ever = self._db.execute("SELECT 1 FROM conn_events WHERE device='pi' AND state='online' LIMIT 1").fetchone() is not None
        return {"pi_connected": self._conn_state.get("pi") == "online", "pi_ever_online": ever, "pi_age_s": pi_age, "watchers": watchers}

    def conn_log(self, limit=30):
        with self._lock:
            rows = self._db.execute("SELECT device,state,reason,t FROM conn_events ORDER BY n DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---------- snapshot ----------
    def snapshot(self):
        now = self.now()
        with self._lock:
            conns = {d: self._conn_out(d, now) for d in ("phone", "pi")}
            conns["caregivers"] = sum(1 for t in self._caregivers.values() if now - t < 30)
            loc = self._location_out(now)
            nav = self._nav_out(now)
            pi = self._pi_out(now, conns["pi"])
            ph = self._phone_out(now)
            events = [self._event_out(r) for r in self._db.execute("SELECT * FROM events ORDER BY t DESC LIMIT 20")]
            active = next((e for e in events if e["kind"] in EMERGENCY_KINDS and e["status"] in ("pending", "confirmed")), None)
            return {
                "server_time": now, "connections": conns, "location": loc, "navigation": nav, "pi": pi, "phone": ph,
                "safety": {"active": active, "recent": events},
                "route": self._route and {"name": self._route["name"], "points": self._route["points"]},
                "last_known_safe": self._lkg(loc, nav, pi, conns),
            }

    def _conn_out(self, dev, now):
        seen = self._seen.get(dev)
        last = self._db.execute("SELECT state,reason,t FROM conn_events WHERE device=? ORDER BY n DESC LIMIT 1", (dev,)).fetchone()
        state = self._conn_state.get(dev, "offline")
        out = {"state": state, "last_seen_age_s": None if seen is None else round(now - seen, 1), "last_seen": seen}
        if state == "offline":
            off = self._db.execute("SELECT reason,t FROM conn_events WHERE device=? AND state='offline' ORDER BY n DESC LIMIT 1", (dev,)).fetchone()
            out["offline_since"], out["offline_reason"] = (off["t"], off["reason"]) if off else (None, None)
        return out

    def _location_out(self, now):
        r = self._latest_fix()
        if not r:
            return {"available": False, "quality": "none", "odometer_m": round(self._odometer["m"])}
        age = now - r["recv"]
        acc = r["acc"]
        quality = "none" if age > LOCATION_FRESH_S else "good" if acc is not None and acc <= GOOD_ACC_M else "weak" if acc is None or acc <= WEAK_ACC_M else "poor"
        moving = (r["speed"] or 0) > 0.4
        return {"available": True, "lat": r["lat"], "lon": r["lon"], "accuracy_m": acc, "speed_mps": r["speed"], "bearing_deg": r["bearing"], "heading_deg": r["heading"],
                "provider": r["provider"], "fix_t": r["t"], "age_s": round(age, 1), "stale": age > LOCATION_FRESH_S, "quality": quality,
                "moving": moving, "odometer_m": round(self._odometer["m"]),
                "off_route_m": self._off_route(r)}

    def _off_route(self, r):
        if not self._route:
            return None
        d = distance_to_route_m(r["lat"], r["lon"], self._route["points"])
        return None if d is None else round(d)

    def _nav_out(self, now):
        n = self._nav
        if not n:
            return {"available": False, "mode": "unknown", "stale": True}
        age = now - n["recv"]
        return {**{k: v for k, v in n.items() if k != "recv"}, "available": True, "age_s": round(age, 1), "stale": age > NAV_STALE_S}

    def _phone_out(self, now):
        p = self._phone
        if not p:
            return {"available": False, "stale": True}
        age = now - p["recv"]
        return {**{k: v for k, v in p.items() if k != "recv"}, "available": True, "age_s": round(age, 1), "stale": age > PHONE_STALE_S}

    def _pi_out(self, now, conn):
        live = conn["state"] == "online"
        src = self._pi if live and self._pi["health"] else self._pi_last
        h = src.get("health") or {}
        hello = src.get("hello") or {}
        if not h and not live:
            return {"available": False, "stale": True}
        recv = self._seen.get("pi") if live else src.get("recv")
        age = None if recv is None else round(now - recv, 1)
        out = {"available": True, "stale": not live, "age_s": age, "health": h,
               "camera": {"width": hello.get("width"), "height": hello.get("height"), "fps_left": h.get("cam_fps_left"),
                          "fps_right": h.get("cam_fps_right"), "pair_dt_ms": h.get("pair_dt_ms"), "dropped": h.get("cam_skipped")},
               "link": self._pi.get("stats") or {}}
        if "rtt_s" in h and h["rtt_s"] is not None:
            out["latency_ms"] = round(h["rtt_s"] * 1000)
        return out

    def _lkg(self, loc, nav, pi, conns):
        """Everything we last knew, each piece with its own age; used when a device is offline."""
        recv = [t for t in (self._seen.get("phone"), self._seen.get("pi"), self._nav and self._nav["recv"]) if t]
        down = {d: {"since": conns[d].get("offline_since"), "reason": conns[d].get("offline_reason")}
                for d in ("phone", "pi") if conns[d]["state"] == "offline"}
        return {"location": loc if loc.get("available") else None,
                "direction": self._nav and {k: self._nav.get(k) for k in ("direction", "instruction", "distance_m")},
                "obstacles": self._nav and self._nav.get("obstacles"),
                "path_safety": self._nav and {k: self._nav.get(k) for k in ("forward_safe", "path_confidence", "clearance_left_m", "clearance_right_m")},
                "nav_mode": self._nav and self._nav.get("mode"),
                "device_health": pi.get("health") if pi.get("available") else None,
                "last_update": max(recv) if recv else None, "disconnected": down}


def default_db_path():
    return os.environ.get("GABAI_RELAY_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "relay.sqlite3"))
