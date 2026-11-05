import numpy as np
import pytest
from fastapi.testclient import TestClient

from gabai.relay.app import create_app
from gabai.relay.bridge import PiBridge
from gabai.relay.core import (INACTIVITY_S, LOCATION_FRESH_S, PENDING_ESCALATE_S, PHONE_STALE_S, PI_STALE_S, Relay)
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


def _bearer(person):
    return {"Authorization": f"Bearer {person['token']}"}


def _pair(c, handle="maria", relay=None):
    """Invite flow for people already registered; public registration is closed (see the access-request tests)."""
    user = relay.register("user", "Maria", handle, _internal=True)
    cg = relay.register("caregiver", "Ana", _internal=True)
    c.post("/api/links/invite", json={"handle": handle}, headers=_bearer(cg))
    link = c.get("/api/links", headers=_bearer(user)).json()["links"][0]
    return user, cg, link


def test_invite_gives_no_access_until_user_accepts(env):
    _, relay, _, c = env
    user, cg, link = _pair(c, relay=relay)
    assert link["status"] == "pending"
    assert c.get("/api/state", headers=_bearer(cg)).status_code == 403
    assert c.post(f"/api/links/{link['id']}/respond", json={"action": "accept"}, headers=_bearer(cg)).status_code == 403
    assert c.post(f"/api/links/{link['id']}/respond", json={"action": "accept"}, headers=_bearer(user)).status_code == 200
    assert c.get("/api/state", headers=_bearer(cg)).status_code == 200


def test_revoke_stops_access_and_second_caregiver_is_independent(env):
    _, relay, _, c = env
    user, cg, link = _pair(c, relay=relay)
    c.post(f"/api/links/{link['id']}/respond", json={"action": "accept"}, headers=_bearer(user))
    other = relay.register("caregiver", "Ben", _internal=True)
    assert c.get("/api/state", headers=_bearer(other)).status_code == 403
    assert c.post(f"/api/links/{link['id']}/revoke", headers=_bearer(user)).status_code == 200
    assert c.get("/api/state", headers=_bearer(cg)).status_code == 403


def test_invite_answer_does_not_reveal_unknown_handle_and_is_rate_limited(env):
    _, relay, _, c = env
    user, cg, _ = _pair(c, relay=relay)
    known = c.post("/api/links/invite", json={"handle": "maria"}, headers=_bearer(cg))
    unknown = c.post("/api/links/invite", json={"handle": "nobody"}, headers=_bearer(cg))
    assert known.status_code == unknown.status_code == 200 and known.json() == unknown.json()
    codes = [c.post("/api/links/invite", json={"handle": "nobody"}, headers=_bearer(cg)).status_code for _ in range(12)]
    assert 429 in codes


def test_block_is_invisible_to_caregiver_and_pending_invite_expires(env):
    clock, relay, _, c = env
    user, cg, link = _pair(c, relay=relay)
    c.post(f"/api/links/{link['id']}/respond", json={"action": "block"}, headers=_bearer(user))
    c.post("/api/links/invite", json={"handle": "maria"}, headers=_bearer(cg))            # re-invite after a block does nothing
    assert c.get("/api/links", headers=_bearer(cg)).json()["links"] == []              # no name, no sign of the block
    assert c.get("/api/state", headers=_bearer(cg)).status_code == 403
    user2, cg2, link2 = _pair(c, "joao", relay=relay)
    clock.t += 8 * 24 * 3600
    assert c.post(f"/api/links/{link2['id']}/respond", json={"action": "accept"}, headers=_bearer(user2)).status_code == 404


def test_legacy_shared_tokens_still_work(env):
    _, _, _, c = env
    assert c.get("/api/state", headers=CG).status_code == 200
    assert c.post("/api/phone/ingest", json={}, headers=PH).status_code == 200


def _request(c, name="Ben"):
    return c.post("/api/access/request", json={"name": name}).json()


def _primary(c):
    return c.post("/api/register", json={"role": "user", "name": "Maria", "handle": "maria"}).json()


def test_only_the_first_device_can_claim_primary_and_caregivers_cannot_self_register(env):
    _, _, _, c = env
    assert c.post("/api/register", json={"role": "caregiver", "name": "Eve"}).status_code == 403
    assert _primary(c)["token"]
    assert c.post("/api/register", json={"role": "user", "name": "Eve", "handle": "eve123"}).status_code == 409


def test_request_approve_gives_one_time_token_and_access(env):
    _, _, _, c = env
    owner = _primary(c)
    req = _request(c)
    assert len(req["code"]) == 4
    assert c.get(f"/api/access/request/{req['request_id']}?secret={req['secret']}").json() == {"status": "pending"}
    assert c.get("/api/access/pending", headers=CG).status_code == 401                       # shared caregiver token cannot approve
    pend = c.get("/api/access/pending", headers=_bearer(owner)).json()["requests"]
    assert pend[0]["code"] == req["code"] and pend[0]["name"] == "Ben"
    assert c.post(f"/api/access/{req['request_id']}/decide", json={"approve": True}, headers=_bearer(owner)).json()["status"] == "approved"
    got = c.get(f"/api/access/request/{req['request_id']}?secret={req['secret']}").json()
    assert got["status"] == "approved" and c.get("/api/state", headers=_bearer(got)).status_code == 200
    assert "token" not in c.get(f"/api/access/request/{req['request_id']}?secret={req['secret']}").json()    # handed over once


def test_unapproved_rejected_wrong_secret_and_expired_get_nothing(env):
    clock, _, _, c = env
    owner = _primary(c)
    a, b, d = _request(c, "A"), _request(c, "B"), _request(c, "D")
    assert c.get(f"/api/access/request/{a['request_id']}?secret=wrong").status_code == 404
    c.post(f"/api/access/{a['request_id']}/decide", json={"approve": False}, headers=_bearer(owner))
    assert c.get(f"/api/access/request/{a['request_id']}?secret={a['secret']}").json() == {"status": "rejected"}
    clock.t += 121
    assert c.get(f"/api/access/request/{b['request_id']}?secret={b['secret']}").json() == {"status": "expired"}
    assert c.post(f"/api/access/{d['request_id']}/decide", json={"approve": True}, headers=_bearer(owner)).status_code == 404


def test_request_flood_is_limited_and_unauthenticated_devices_see_no_data(env):
    _, _, _, c = env
    codes = [c.post("/api/access/request", json={"name": "x"}).status_code for _ in range(5)]
    assert 429 in codes
    for path in ("/api/state", "/snapshot/left.jpg", "/stream/left.mjpg", "/api/history/locations", "/api/history/events"):
        assert c.get(path).status_code == 401, path
    stranger = _bearer({"token": "not-a-token"})
    assert c.get("/api/state", headers=stranger).status_code == 401


def test_revoked_second_caregiver_loses_access_at_once_first_unaffected(env):
    _, relay, _, c = env
    owner = _primary(c)
    toks = []
    for n in ("A", "B"):
        r = _request(c, n)
        c.post(f"/api/access/{r['request_id']}/decide", json={"approve": True}, headers=_bearer(owner))
        toks.append(c.get(f"/api/access/request/{r['request_id']}?secret={r['secret']}").json())
    links = c.get("/api/links", headers=_bearer(owner)).json()["links"]
    b_link = next(l for l in links if l["name"] == "B")
    c.post(f"/api/links/{b_link['id']}/revoke", headers=_bearer(owner))
    assert c.get("/api/state", headers=_bearer(toks[1])).status_code == 403
    assert c.get("/api/state", headers=_bearer(toks[0])).status_code == 200


def test_after_navigation_ends_last_location_stays_and_ages_with_gps_off(env):
    clock, relay, _, c = env
    c.post("/api/phone/ingest", json={"locations": [loc(clock.t)], "telemetry": {"t": clock.t, "gps_state": "ok"}}, headers=PH)
    clock.t += LOCATION_FRESH_S + 5
    c.post("/api/phone/ingest", json={"telemetry": {"t": clock.t, "gps_state": "off"}}, headers=PH)   # Navigation closed: no more fixes
    s = relay.snapshot()
    assert s["phone"]["gps_state"] == "off"
    assert s["location"]["available"] and s["location"]["stale"] and s["location"]["lat"] == 14.5995


def test_pc_result_reaches_phone_and_caregiver_with_frame_id(env):
    from gabai.relay.results import build_result
    clock, relay, _, c = env
    relay.ingest_nav(build_result(42, clock.t, ("ahead", 1.5), ["ahead, about one and a half metres"]))
    g = c.get("/api/guidance", headers=PH).json()
    assert g["frame_id"] == 42 and g["hazards"] == [{"zone": "forward", "distance_m": 1.5, "kind": "obstacle"}]
    assert g["speech"]["text"].startswith("ahead") and g["summary"] == "ahead, about one and a half metres"
    assert c.get("/api/guidance", headers=CG).status_code == 403 and c.get("/api/guidance").status_code == 401
    assert relay.snapshot()["navigation"]["frame_id"] == 42
    relay.ingest_nav(build_result(43, clock.t + 1, None, []))
    assert relay.snapshot()["navigation"]["summary"] == "Path clear" and "speech" not in relay.snapshot()["navigation"]
    clock.t += 10
    assert c.get("/api/guidance", headers=PH).json()["stale"] is True


def test_result_for_drop_and_system_alerts():
    from gabai.relay.results import build_result
    assert build_result(1, 0, ("drop_ahead", None), ["drop ahead"])["speech"]["interrupt"] is True
    assert build_result(1, 0, ("cant_see_ahead", None), [])["hazards"][0]["zone"] == "forward"
    assert build_result(1, 0, ("left", 2.0), [])["hazards"][0]["zone"] == "left"


def test_status_strip_states(env):
    clock, relay, _, c = env
    s = relay.snapshot()["status"]
    assert s["patient"]["state"] == "bad" and s["video"]["state"] == "off" and s["gps"]["text"] == "No location yet" and s["emergency"]["state"] == "ok"
    c.post("/api/phone/ingest", json={"locations": [loc(clock.t, acc=8)], "telemetry": {"t": clock.t, "gps_state": "ok", "battery_pct": 80}}, headers=PH)
    s = relay.snapshot()["status"]
    assert s["gps"] == {"state": "ok", "text": "Live"} and s["battery"]["text"] == "80%" and s["patient"]["state"] == "ok"
    c.post("/api/phone/ingest", json={"locations": [loc(clock.t + 1, acc=120)]}, headers=PH)
    assert relay.snapshot()["status"]["gps"]["state"] == "warn"
    clock.t += LOCATION_FRESH_S + 1
    c.post("/api/phone/ingest", json={"telemetry": {"t": clock.t, "gps_state": "off", "battery_pct": 10}}, headers=PH)
    s = relay.snapshot()["status"]
    assert s["gps"]["state"] == "warn" and s["gps"]["text"].startswith("GPS off") and s["battery"]["state"] == "bad"
    c.post("/api/phone/ingest", json={"events": [{"id": "e1", "kind": "sos", "status": "pending", "t": clock.t}]}, headers=PH)
    assert relay.snapshot()["status"]["emergency"] == {"state": "bad", "text": "SOS"}


def test_tiles_cached_validated_and_authenticated(tmp_path):
    from gabai.relay.tiles import Tiles
    calls = []

    def fetch(url):
        calls.append(url)
        return b"\x89PNG-fake" if len(calls) < 3 else b"<html>Access blocked</html>"

    clock = Clock()
    relay = Relay(str(tmp_path / "r.db"), clock)
    c = TestClient(create_app(relay, FrameStore(str(tmp_path), clock), TOK, tiles=Tiles(str(tmp_path / "t"), fetch)))
    assert c.get("/tiles/3/2/1.png").status_code == 401
    assert c.get("/tiles/3/2/1.png", headers=CG).content.startswith(b"\x89PNG")
    assert c.get("/tiles/3/2/1.png?token=c").status_code == 200 and len(calls) == 1       # second hit served from disk
    assert c.get("/tiles/3/9/1.png", headers=CG).status_code == 404 and len(calls) == 1   # x out of range: never fetched
    assert c.get("/tiles/4/1/1.png", headers=CG).status_code == 200
    assert c.get("/tiles/4/2/2.png", headers=CG).status_code == 404                       # an HTML block page is not cached as a tile
    assert not (tmp_path / "t" / "4" / "2" / "2.png").exists()


def test_admin_page_is_local_only_and_hides_secrets(tmp_path):
    clock = Clock()
    relay = Relay(str(tmp_path / "r.db"), clock)
    app = create_app(relay, FrameStore(str(tmp_path), clock), TOK)
    outside = TestClient(app)                                      # client host is not loopback
    inside = TestClient(app, client=("127.0.0.1", 50000))
    for path in ("/admin", "/admin/state", "/admin/stream/left.mjpg"):
        assert outside.get(path).status_code == 403, path
        assert outside.get(path, headers=CG).status_code == 403, path       # a token does not unlock it
    page = inside.get("/admin")
    assert page.status_code == 200 and "GabAi relay" in page.text
    owner = inside.post("/api/register", json={"role": "user", "name": "Maria", "handle": "maria"}).json()
    req = inside.post("/api/access/request", json={"name": "Ben"}).json()
    st = inside.get("/admin/state").json()
    assert st["pending"][0]["code"] == req["code"] and st["people"][0]["name"] == "Maria"
    assert owner["token"] not in inside.get("/admin/state").text and req["secret"] not in inside.get("/admin/state").text


def test_pc_owner_approves_from_admin_even_without_a_primary_phone(tmp_path):
    clock = Clock()
    relay = Relay(str(tmp_path / "r.db"), clock)
    app = create_app(relay, FrameStore(str(tmp_path), clock), TOK)
    outside, inside = TestClient(app), TestClient(app, client=("127.0.0.1", 50000))
    req = inside.post("/api/access/request", json={"name": "Ben"}).json()
    assert outside.post(f"/admin/decide/{req['request_id']}", json={"approve": True}).status_code == 403
    assert inside.post(f"/admin/decide/{req['request_id']}", json={"approve": True}).json()["status"] == "approved"
    tok = inside.get(f"/api/access/request/{req['request_id']}?secret={req['secret']}").json()
    assert inside.get("/api/state", headers=_bearer(tok)).status_code == 200
    assert inside.get("/admin/state").json()["links"][0]["caregiver"] == "Ben"          # listed even with no primary phone
    owner = inside.post("/api/register", json={"role": "user", "name": "Maria", "handle": "maria"}).json()
    link = inside.get("/api/links", headers=_bearer(owner)).json()["links"][0]            # the new primary can now see and revoke Ben
    assert inside.post(f"/api/links/{link['id']}/revoke", headers=_bearer(owner)).status_code == 200
    assert inside.get("/api/state", headers=_bearer(tok)).status_code == 403


def test_slam_guidance_posted_by_the_mapper_reaches_the_phone(env):
    clock, _, _, c = env
    body = {"mode": "slam", "t": clock.t, "frame_id": 7, "summary": "Turn left", "hazards": [{"zone": "left", "distance_m": 1.0, "kind": "obstacle"}],
            "speech": {"text": "Turn left", "kind": "steer", "interrupt": False}}
    assert c.post("/api/pi/nav", json=body).status_code == 401
    assert c.post("/api/pi/nav", json=body, headers={"Authorization": "Bearer i"}).status_code == 200      # the pi token the mapper reads from relay_tokens.json
    g = c.get("/api/guidance", headers=PH).json()
    assert g["mode"] == "slam" and g["speech"]["text"] == "Turn left" and g["frame_id"] == 7


def test_egocentric_message_is_kept_and_decoded_for_the_admin_page(tmp_path):
    clock = Clock()
    relay = Relay(str(tmp_path / "r.db"), clock)
    app = create_app(relay, FrameStore(str(tmp_path), clock), TOK)
    inside = TestClient(app, client=("127.0.0.1", 50000))
    inside.post("/api/pi/nav", json={"mode": "slam", "t": clock.t, "frame_id": 1, "summary": "Path is clear", "ego": {"d": [[1, 1, 2]], "c": [1]}}, headers={"Authorization": "Bearer i"})
    assert inside.get("/api/guidance", headers=PH).json()["ego"] == {"d": [[1, 1, 2]], "c": [1]}      # the phone gets the numbers and decodes them itself
    assert inside.get("/admin/state").json()["ego_text"][0] == "Human near you on your right."
    inside.post("/api/pi/nav", json={"t": clock.t + 1, "ego": "garbage"}, headers={"Authorization": "Bearer i"})
    assert inside.get("/admin/state").json()["ego_text"] == []
