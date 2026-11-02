"""HTTP surface of the caregiver relay.

Roles: `phone` posts GPS/telemetry/events, `pi` posts navigation, `caregiver` reads everything and acknowledges alerts.
The Pi camera pair reaches the relay through the existing TCP link (see bridge.py), so the Pi sender needs no change.
"""
import asyncio
import hmac
import json
import secrets
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .core import LinkError

MAX_BODY = 1_000_000
BOUNDARY = "gabaiframe"


def load_tokens(path):
    """Tokens persist next to the database; first run generates them and prints where to read them."""
    try:
        with open(path, encoding="utf-8") as f:
            tokens = json.load(f)
    except (OSError, ValueError):
        tokens = {}
    changed = False
    for role in ("phone", "caregiver", "pi"):
        if not tokens.get(role):
            tokens[role] = secrets.token_urlsafe(24)
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=2)
    return tokens


def create_app(relay, frames, tokens):
    app = FastAPI(title="Gabai caregiver relay")

    def role_for(token):
        for role, value in tokens.items():
            if token and hmac.compare_digest(token, value):
                return role
        return None

    def actor(authorization: str = Header(default=""), token: str = Query(default="")):
        """Who is calling: a legacy shared token (role only) or a registered person (their own token)."""
        raw = authorization[7:] if authorization.startswith("Bearer ") else token
        role = role_for(raw)
        if role:
            return {"role": role, "person": None}
        person = relay.person_for_token(raw)
        if person:
            return {"role": "phone" if person["role"] == "user" else "caregiver", "person": person}
        return {"role": None, "person": None}

    def need(*roles, linked=True):
        """A person-token caregiver reads data only through an accepted link; shared tokens keep their old reach."""
        def dep(who=Depends(actor)):
            if who["role"] not in roles:
                raise HTTPException(401 if who["role"] is None else 403, "invalid token" if who["role"] is None else "role not allowed")
            p = who["person"]
            if linked and p and p["role"] == "caregiver" and not relay.has_access(p["id"]):
                raise HTTPException(403, "no active caregiver link")
            return who
        return dep

    def need_person(role):
        def dep(who=Depends(actor)):
            p = who["person"]
            if p is None:
                raise HTTPException(401, "a registered person token is required")
            if role and p["role"] != role:
                raise HTTPException(403, "role not allowed")
            return p
        return dep

    def link_call(fn, *args):
        try:
            return fn(*args)
        except LinkError as e:
            raise HTTPException(e.code, str(e))

    async def body(request: Request):
        raw = await request.body()
        if len(raw) > MAX_BODY:
            raise HTTPException(413, "body too large")
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            raise HTTPException(400, "body is not JSON")
        if not isinstance(data, dict):
            raise HTTPException(400, "body must be a JSON object")
        return data

    @app.get("/api/health")
    def health():
        return {"ok": True, "time": time.time()}

    @app.post("/api/phone/ingest", dependencies=[Depends(need("phone"))])
    async def phone_ingest(data=Depends(body)):
        return relay.ingest_phone(data)

    @app.post("/api/pi/nav", dependencies=[Depends(need("pi", "phone"))])
    async def nav(data=Depends(body)):
        relay.ingest_nav(data)
        return {"ok": True}

    @app.get("/api/state", dependencies=[Depends(need("caregiver"))])
    def state(client: str = Query(default="")):
        if client:
            relay.caregiver_seen(client)
        return relay.snapshot()

    @app.get("/api/history/locations", dependencies=[Depends(need("caregiver"))])
    def history(since: float = 0.0, limit: int = 2000):
        return {"points": relay.history(since, limit)}

    @app.get("/api/history/events", dependencies=[Depends(need("caregiver"))])
    def events(since: float = 0.0, limit: int = 100):
        return {"events": relay.events(limit, since), "connections": relay.conn_log()}

    @app.post("/api/register")
    async def register(data=Depends(body)):
        return link_call(relay.register, data.get("role"), data.get("name"), data.get("handle"))

    @app.get("/api/links")
    def links(p=Depends(need_person(None))):
        return {"person": p, "links": relay.links_for(p)}

    @app.post("/api/links/invite")
    async def invite(data=Depends(body), p=Depends(need_person("caregiver"))):
        return link_call(relay.invite, p["id"], data.get("handle"))

    @app.post("/api/links/{link_id}/respond")
    async def respond(link_id: str, data=Depends(body), p=Depends(need_person("user"))):
        return link_call(relay.respond, p["id"], link_id, data.get("action"))

    @app.post("/api/links/{link_id}/revoke")
    def revoke(link_id: str, p=Depends(need_person(None))):
        return link_call(relay.revoke, p["id"], link_id)

    @app.post("/api/events/{event_id}/ack")
    def ack(event_id: str, who: str = Query(default="caregiver", max_length=40), a=Depends(need("caregiver"))):
        if a["person"]:
            who = a["person"]["name"]                           # a registered caregiver cannot sign an ack as someone else
        if not relay.acknowledge(event_id, who):
            raise HTTPException(409, "event not found or already handled")
        return {"ok": True}

    @app.put("/api/route", dependencies=[Depends(need("caregiver"))])
    async def put_route(data=Depends(body)):
        try:
            relay.set_route(data.get("name", "Route"), data["points"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "points must be a list of [lat, lon]")
        return {"ok": True}

    @app.delete("/api/route", dependencies=[Depends(need("caregiver"))])
    def del_route():
        relay.clear_route()
        return {"ok": True}

    @app.get("/snapshot/{side}.jpg", dependencies=[Depends(need("caregiver"))])
    async def snapshot(side: str):
        if side not in ("left", "right"):
            raise HTTPException(404)
        data, age = await run_in_threadpool(frames.jpeg, side)
        if data is None:
            raise HTTPException(404, "no frame received yet")
        return Response(data, media_type="image/jpeg", headers={"X-Frame-Age-S": f"{age:.1f}", "Cache-Control": "no-store"})

    @app.get("/stream/{side}.mjpg", dependencies=[Depends(need("caregiver"))])
    async def stream(side: str, fps: float = Query(default=6.0, ge=1.0, le=12.0)):
        if side not in ("left", "right"):
            raise HTTPException(404)

        async def gen():
            last, wait = None, 1.0 / fps
            while True:
                age = frames.age_s()
                if age is None or age > 5.0:
                    return                                       # stale: end the stream so the client labels the last frame stale
                seq = frames.seq()
                if seq != last:
                    data, _ = await run_in_threadpool(frames.jpeg, side)
                    if data:
                        last = seq
                        yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n").encode() + data + b"\r\n"
                await asyncio.sleep(wait)

        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                                 headers={"Cache-Control": "no-store"})

    @app.exception_handler(Exception)
    async def boom(_, exc):
        return JSONResponse({"error": type(exc).__name__}, status_code=500)

    return app
