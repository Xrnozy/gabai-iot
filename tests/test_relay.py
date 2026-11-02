import numpy as np
import pytest
from fastapi.testclient import TestClient

from gabai.relay.app import create_app
from gabai.relay.bridge import PiBridge
from gabai.relay.core import (INACTIVITY_S, PENDING_ESCALATE_S, PHONE_STALE_S, PI_STALE_S, Relay)
from gabai.relay.frames import FrameStore
from gabai.relay.geo import distance_to_route_m, haversine_m

TOK = {"phone": "p", "caregiver": "c", "pi": "i"}
PH, CG = {"Authorization": "Bearer p"}, {"Authorization": "Bearer c"}


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


class FakeReceiver:
    def __init__(self):
        self.pair, self.s = None, {"connected": False, "health": {}, "hello": {}, "frame_age_s": None, "last_error": None}

    def latest(self):
        return self.pair

    def stats(self):
        return dict(self.s)


class Pair:
    def __init__(self, seq):
        self.seq = seq
        self.left = np.full((48, 64, 3), 90, np.uint8)
        self.right = np.full((48, 64, 3), 200, np.uint8)


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    relay = Relay(str(tmp_path / "r.db"), clock)
    frames = FrameStore(str(tmp_path), clock)
    return clock, relay, frames, TestClient(create_app(relay, frames, TOK))


def loc(t, lat=14.5995, lon=120.9842, acc=8.0, speed=1.2, i=None):
    return {"id": i or f"l{t}", "t": t, "lat": lat, "lon": lon, "acc": acc, "speed": speed, "bearing": 90.0, "provider": "gps"}


def test_auth_roles(env):
    _, _, _, c = env
    assert c.post("/api/phone/ingest", json={}).status_code == 401
    assert c.post("/api/phone/ingest", json={}, headers=CG).status_code == 403
    assert c.get("/api/state", headers=PH).status_code == 403
    assert c.get("/api/state", headers=CG).status_code == 200
    assert c.get("/api/state?token=c").status_code == 200


def test_ingest_idempotent_and_distance(env):
    clock, relay, _, c = env
    batch = {"locations": [loc(clock.t - 10), loc(clock.t, lat=14.5995 + 0.0001)]}
    r1 = c.post("/api/phone/ingest", json=batch, headers=PH).json()
    c.post("/api/phone/ingest", json=batch, headers=PH)
    assert r1["ok"]
    assert len(relay.history()) == 2
    assert relay.snapshot()["location"]["odometer_m"] == round(haversine_m(14.5995, 120.9842, 14.5996, 120.9842))


def test_jitter_does_not_add_distance(env):
    clock, relay, _, c = env
    for k in range(5):
        c.post("/api/phone/ingest", json={"locations": [loc(clock.t + k, lat=14.5995 + k * 1e-6, acc=15)]}, headers=PH)
    assert relay.snapshot()["location"]["odometer_m"] == 0


def test_backlog_does_not_overwrite_newer_state(env):
    clock, relay, _, c = env
    c.post("/api/phone/ingest", json={"locations": [loc(clock.t, lat=14.6)], "nav": {"t": clock.t, "mode": "pi", "direction": "forward"}}, headers=PH)
    c.post("/api/phone/ingest", json={"locations": [loc(clock.t - 60, lat=14.5)], "nav": {"t": clock.t - 60, "mode": "arcore"}}, headers=PH)
    s = relay.snapshot()
    assert s["location"]["lat"] == 14.6
    assert s["navigation"]["mode"] == "pi"
    assert len(relay.history()) == 2


def test_location_goes_stale_and_phone_offline_with_reason(env):
    clock, relay, _, c = env
    c.post("/api/phone/ingest", json={"locations": [loc(clock.t)]}, headers=PH)
    assert relay.snapshot()["connections"]["phone"]["state"] == "online"
    clock.t += PHONE_STALE_S + 10
    relay.tick()
    s = relay.snapshot()
    assert s["connections"]["phone"]["state"] == "offline"
    assert s["location"]["stale"] and s["location"]["quality"] == "none"
    assert s["last_known_safe"]["disconnected"]["phone"]["reason"] == "no data received"
    assert s["last_known_safe"]["location"]["lat"] == 14.5995
    c.post("/api/phone/ingest", json={"locations": [loc(clock.t)]}, headers=PH)
    assert relay.snapshot()["connections"]["phone"]["state"] == "online"
    assert [e["state"] for e in relay.conn_log() if e["device"] == "phone"][:3] == ["online", "offline", "online"]


def test_gps_quality_levels(env):
    clock, relay, _, c = env
    for acc, want in ((5, "good"), (35, "weak"), (120, "poor")):
        c.post("/api/phone/ingest", json={"locations": [loc(clock.t, acc=acc)]}, headers=PH)
        assert relay.snapshot()["location"]["quality"] == want
        clock.t += 1


def test_fall_flow_cancel_ack_and_escalation(env):
    clock, relay, _, c = env
    fall = {"id": "f1", "kind": "fall", "status": "pending", "t": clock.t, "lat": 14.6, "lon": 121.0, "confidence": 0.82,
            "evidence": {"peak_g": 3.4, "still_s": 4}}
    assert c.post("/api/phone/ingest", json={"events": [fall]}, headers=PH).json()["events"]["f1"] == "created"
    assert relay.snapshot()["safety"]["active"]["id"] == "f1"
    c.post("/api/phone/ingest", json={"events": [{**fall, "status": "cancelled"}]}, headers=PH)
    assert relay.events()[0]["status"] == "cancelled"
    assert relay.snapshot()["safety"]["active"] is None

    c.post("/api/phone/ingest", json={"events": [{**fall, "id": "f2", "status": "confirmed"}]}, headers=PH)
    assert c.post("/api/events/f2/ack?who=Ana", headers=CG).status_code == 200
    c.post("/api/phone/ingest", json={"events": [{**fall, "id": "f2", "status": "cancelled"}]}, headers=PH)
    ev = [e for e in relay.events() if e["id"] == "f2"][0]
    assert ev["status"] == "acknowledged" and ev["ack_by"] == "Ana"
    assert c.post("/api/events/f2/ack", headers=CG).status_code == 409

    c.post("/api/phone/ingest", json={"events": [{**fall, "id": "f3"}]}, headers=PH)
    clock.t += PENDING_ESCALATE_S + 1
    relay.tick()
    assert [e for e in relay.events() if e["id"] == "f3"][0]["status"] == "confirmed"


def test_event_rejects_bad_input(env):
    _, _, _, c = env
    r = c.post("/api/phone/ingest", json={"events": [{"id": "", "kind": "fall"}, {"id": "x", "kind": "sos", "status": "bogus"}]}, headers=PH)
    assert set(r.json()["events"].values()) == {"rejected"}


def test_route_deviation_once_per_episode(env):
    clock, relay, _, c = env
    route = [[14.5995, 120.9842], [14.5995, 120.9942]]
    assert c.put("/api/route", json={"name": "Home", "points": route}, headers=CG).status_code == 200
    on = lambda k: loc(clock.t + k, lat=14.5995, lon=120.9850)
    off = lambda k: loc(clock.t + k, lat=14.6050, lon=120.9850)
    c.post("/api/phone/ingest", json={"locations": [on(0)]}, headers=PH)
    for k in (1, 2, 3, 4):
        c.post("/api/phone/ingest", json={"locations": [off(k)]}, headers=PH)
    devs = [e for e in relay.events() if e["kind"] == "deviation"]
    assert len(devs) == 1 and devs[0]["evidence"]["off_route_m"] > 400
    assert relay.snapshot()["location"]["off_route_m"] > 400


def test_single_off_route_fix_is_not_an_alert(env):
    clock, relay, _, c = env
    c.put("/api/route", json={"name": "Home", "points": [[14.5995, 120.9842], [14.5995, 120.9942]]}, headers=CG)
    c.post("/api/phone/ingest", json={"locations": [loc(clock.t, lat=14.6050)]}, headers=PH)
    assert not [e for e in relay.events() if e["kind"] == "deviation"]


def test_inactivity_alert_after_long_stillness(env):
    clock, relay, _, c = env
    start = clock.t
    for k in range(0, INACTIVITY_S + 60, 30):
        clock.t = start + k
        c.post("/api/phone/ingest", json={"locations": [loc(clock.t, speed=0.0)]}, headers=PH)
        relay.tick()
    assert [e for e in relay.events() if e["kind"] == "inactivity"]


def test_walking_is_not_inactivity(env):
    clock, relay, _, c = env
    start = clock.t
    for k in range(0, INACTIVITY_S + 60, 30):
        clock.t = start + k
        c.post("/api/phone/ingest", json={"locations": [loc(clock.t, lat=14.5995 + k * 1e-6)]}, headers=PH)
        relay.tick()
    assert not [e for e in relay.events() if e["kind"] == "inactivity"]


def test_pi_bridge_liveness_uses_frame_age_not_open_socket(env):
    clock, relay, frames, c = env
    rx = FakeReceiver()
    bridge = PiBridge(relay, frames, rx)
    rx.pair = Pair(1)
    rx.s.update(connected=True, frame_age_s=0.1, health={"temp_c": 51.0, "cam_fps_left": 14.0, "cam_fps_right": 14.0, "rtt_s": 0.04, "cam_skipped": 3},
                hello={"width": 640, "height": 480})
    bridge.step()
    relay.tick()
    s = relay.snapshot()
    assert s["connections"]["pi"]["state"] == "online" and not s["pi"]["stale"]
    assert s["pi"]["camera"]["fps_left"] == 14.0 and s["pi"]["latency_ms"] == 40 and s["pi"]["camera"]["dropped"] == 3

    clock.t += PI_STALE_S + 3                         # socket still "connected", frames stopped
    rx.s["frame_age_s"] = PI_STALE_S + 3
    bridge.step()
    relay.tick()
    s = relay.snapshot()
    assert s["connections"]["pi"]["state"] == "offline" and s["pi"]["stale"]
    assert s["pi"]["health"]["temp_c"] == 51.0       # last health kept, labelled stale


def test_last_known_state_survives_restart(tmp_path):
    clock = Clock()
    db = str(tmp_path / "r.db")
    r1 = Relay(db, clock)
    r1.ingest_phone({"locations": [loc(clock.t)], "nav": {"t": clock.t, "mode": "pi", "direction": "left", "obstacles": [{"kind": "person"}]},
                     "telemetry": {"t": clock.t, "battery_pct": 80}})
    r2 = Relay(db, clock)
    s = r2.snapshot()
    assert s["connections"]["phone"]["state"] == "offline"          # a restart proves nothing about liveness
    assert s["location"]["lat"] == 14.5995 and s["location"]["stale"] is False
    assert s["navigation"]["direction"] == "left" and s["phone"]["battery_pct"] == 80
    assert s["last_known_safe"]["obstacles"] == [{"kind": "person"}]


def test_snapshot_and_mjpeg(env):
    clock, relay, frames, c = env
    assert c.get("/snapshot/left.jpg", headers=CG).status_code == 404
    frames.put(1, Pair(1).left, Pair(1).right)
    r = c.get("/snapshot/left.jpg", headers=CG)
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8" and float(r.headers["x-frame-age-s"]) == 0.0
    clock.t += 30
    stale = c.get("/snapshot/right.jpg", headers=CG)
    assert float(stale.headers["x-frame-age-s"]) == 30.0
    with c.stream("GET", "/stream/left.mjpg", headers=CG) as s:                # stale frame: stream ends instead of replaying it as live
        assert b"".join(s.iter_bytes()) == b""


def test_geo_route_distance():
    assert distance_to_route_m(14.0, 121.0, []) is None
    assert distance_to_route_m(14.0, 121.0, [[14.0, 121.0]]) == 0
    assert 100 < distance_to_route_m(14.001, 121.0, [[14.0, 120.99], [14.0, 121.01]]) < 120


def test_deviation_alert_is_not_an_emergency(env):
    clock, relay, _, c = env
    c.put("/api/route", json={"name": "Home", "points": [[14.5995, 120.9842], [14.5995, 120.9942]]}, headers=CG)
    for k in range(3):
        c.post("/api/phone/ingest", json={"locations": [loc(clock.t + k, lat=14.6050)]}, headers=PH)
    assert relay.snapshot()["safety"]["active"] is None


def test_phone_hears_back_when_caregiver_acknowledges(env):
    clock, relay, _, c = env
    ev = {"id": "s1", "kind": "sos", "status": "confirmed", "t": clock.t}
    assert c.post("/api/phone/ingest", json={"events": [ev]}, headers=PH).json()["latest_emergency"]["status"] == "confirmed"
    c.post("/api/events/s1/ack?who=Ana", headers=CG)
    r = c.post("/api/phone/ingest", json={}, headers=PH).json()
    assert r["latest_emergency"]["status"] == "acknowledged" and r["latest_emergency"]["ack_by"] == "Ana"


def test_phone_is_told_when_a_caregiver_is_watching(env):
    clock, relay, _, c = env
    assert c.post("/api/phone/ingest", json={}, headers=PH).json()["watchers"] == 0
    c.get("/api/state?client=ana", headers=CG)
    assert c.post("/api/phone/ingest", json={}, headers=PH).json()["watchers"] == 1
    clock.t += 40
    assert c.post("/api/phone/ingest", json={}, headers=PH).json()["watchers"] == 0


def test_heading_is_stored_and_reported(env):
    clock, relay, _, c = env
    c.post("/api/phone/ingest", json={"locations": [{**loc(clock.t), "heading": 271.5}]}, headers=PH)
    assert relay.snapshot()["location"]["heading_deg"] == 271.5
    assert relay.history()[0]["heading"] == 271.5


def test_phone_can_tell_a_lost_pi_from_a_pi_that_never_connected(env):
    clock, relay, _, c = env
    r = c.post("/api/phone/ingest", json={}, headers=PH).json()
    assert r["pi_connected"] is False and r["pi_ever_online"] is False
    rx = FakeReceiver()
    rx.s.update(connected=True, frame_age_s=0.1)
    PiBridge(relay, FrameStore(None, clock), rx).step()
    relay.tick()
    assert c.post("/api/phone/ingest", json={}, headers=PH).json()["pi_ever_online"] is True
    clock.t += PI_STALE_S + 3
    rx.s["frame_age_s"] = PI_STALE_S + 3
    PiBridge(relay, FrameStore(None, clock), rx).step()
    relay.tick()
    r = c.post("/api/phone/ingest", json={}, headers=PH).json()
    assert r["pi_connected"] is False and r["pi_ever_online"] is True
