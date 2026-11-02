"""Live telemetry page and CSV log. The page is the IoT view of the system: link health, frame rate, latency, Pi
temperature and power flag, nearest distance per zone, and a scrolling log of every spoken alert.

    open http://127.0.0.1:8000/ while the viewer runs (also reachable from a phone on the same Wi-Fi).
Plain HTML + a WebSocket, no external libraries, so it works offline.
"""
import asyncio
import csv
import json
import threading
import time

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gabai telemetry</title><style>
:root{--bg:#101418;--card:#1a2027;--ink:#e6edf3;--mute:#8b98a5;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--line:#2d3640}
@media (prefers-color-scheme:light){:root{--bg:#f4f6f8;--card:#fff;--ink:#14202b;--mute:#5b6b7a;--line:#d5dce3}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 system-ui,sans-serif;padding:16px}
h1{font-size:18px;margin:0 0 4px}p.note{color:var(--mute);margin:0 0 14px;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.k{color:var(--mute);font-size:12px}.v{font-size:22px;font-weight:600}.u{font-size:12px;color:var(--mute);font-weight:400}
.v.bad{color:var(--bad)}
.zones{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}.zone .v{font-size:28px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}@media(max-width:560px){.row{grid-template-columns:1fr}}
canvas{width:100%;height:80px;display:block}.log{max-height:220px;overflow:auto;font-size:14px}.log div{padding:3px 0;border-bottom:1px solid var(--line)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;background:var(--bad)}body.on .dot{background:var(--ok)}
</style></head><body>
<h1><span class="dot"></span>Gabai telemetry</h1>
<p class="note">Proof of concept, supervised tests only. Distances are rounded; unknown is never reported as clear.</p>
<div class="grid" id="cards"></div>
<div class="zones" id="zones"></div>
<div class="row"><div class="card"><div class="k">Frame rate in (FPS)</div><canvas id="c_fps"></canvas></div>
<div class="card"><div class="k">Latency capture to PC (ms)</div><canvas id="c_lat"></canvas></div></div>
<div class="card"><div class="k">Spoken alerts (newest first)</div><div class="log" id="log"></div></div>
<script>
const H={fps:[],lat:[]};const $=id=>document.getElementById(id);
function card(k,v,u,cls){return '<div class="card"><div class="k">'+k+'</div><div class="v '+(cls||'')+'">'+v+'<span class="u"> '+(u||'')+'</span></div></div>'}
function spark(id,a,lo,hi,color){const c=$(id),x=c.getContext('2d');c.width=c.clientWidth*devicePixelRatio;c.height=80*devicePixelRatio;x.clearRect(0,0,c.width,c.height);
if(a.length<2)return;const mx=Math.max(hi,...a),mn=Math.min(lo,...a);x.strokeStyle=color;x.lineWidth=2*devicePixelRatio;x.beginPath();
a.forEach((v,i)=>{const px=i/(a.length-1)*c.width,py=c.height-(v-mn)/(mx-mn||1)*(c.height-6)-3;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}
function z(name,v){return '<div class="card zone"><div class="k">'+name+'</div><div class="v">'+(v==null?'?':v.toFixed(1))+'<span class="u"> m</span></div></div>'}
function show(s){document.body.classList.toggle('on',!!s.connected);
 const f=(v,d=1)=>v==null?'-':Number(v).toFixed(d);
 $('cards').innerHTML=card('Link',s.connected?'connected':'LOST','',s.connected?'':'bad')+card('FPS in',f(s.fps_in),'')+card('Pi sends',f(s.pi_fps),'FPS')
 +card('Latency',f(s.latency_ms,0),'ms')+card('Model',f(s.model_ms,0),'ms')+card('Pi temp',f(s.pi_temp),'C')+card('Pi power flag',s.throttled||'-','')
 +card('Turn speed',f(s.turn_dps,0),'deg/s')+card('L/R sync offset',f(s.pair_dt,1),'ms');
 $('zones').innerHTML=z('LEFT',s.zones.left)+z('AHEAD',s.zones.ahead)+z('RIGHT',s.zones.right);
 if(s.fps_in!=null){H.fps.push(s.fps_in);if(H.fps.length>120)H.fps.shift()}if(s.latency_ms!=null){H.lat.push(s.latency_ms);if(H.lat.length>120)H.lat.shift()}
 spark('c_fps',H.fps,0,20,'#3fb950');spark('c_lat',H.lat,0,300,'#d29922');
 $('log').innerHTML=s.events.slice().reverse().map(e=>'<div>'+new Date(e.t*1000).toLocaleTimeString()+' &nbsp; <b>'+e.text+'</b></div>').join('')||'<div class="k">nothing spoken yet</div>'}
function go(){const w=new WebSocket((location.protocol=='https:'?'wss://':'ws://')+location.host+'/ws');w.onmessage=m=>show(JSON.parse(m.data));w.onclose=()=>setTimeout(go,1000)}go();
</script></body></html>"""


def clean(v):
    """JSON-safe copy: numpy numbers become Python numbers, NaN and infinity become None."""
    import math
    import numpy as np
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class Telemetry:
    def __init__(self, port=8000, host="0.0.0.0"):
        self.port, self.host = port, host
        self.state = {"connected": False, "zones": {"left": None, "ahead": None, "right": None}, "events": []}
        self._lock = threading.Lock()
        self.app = FastAPI()
        self._server = None

        @self.app.get("/", response_class=HTMLResponse)
        def index():
            return PAGE

        @self.app.get("/state")
        def get_state():
            with self._lock:
                return JSONResponse(self.state)

        @self.app.websocket("/ws")
        async def ws(sock: WebSocket):
            await sock.accept()
            try:
                while True:
                    with self._lock:
                        payload = json.dumps(self.state)
                    await sock.send_text(payload)
                    await asyncio.sleep(0.2)
            except (WebSocketDisconnect, RuntimeError):
                pass

    def update(self, **kv):
        with self._lock:
            self.state.update(clean(kv))

    def event(self, text, t=None):
        with self._lock:
            self.state["events"].append({"t": t or time.time(), "text": text})
            del self.state["events"][:-30]

    def start(self):
        cfg = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self._server = uvicorn.Server(cfg)
        threading.Thread(target=self._server.run, daemon=True).start()

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True


FIELDS = ["t", "fps_in", "pi_fps", "latency_ms", "model_ms", "loop_ms", "left", "ahead", "right", "valid_pct", "alert",
          "turn_dps", "pair_dt_ms", "pi_temp", "throttled", "speech_ms", "spoken", "mem_mb", "pi_load", "rtt_ms", "gpu_temp"]


class TelemetryLog:
    """One CSV row per processed frame, for the Phase M graphs."""

    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS, extrasaction="ignore")
        self.w.writeheader()

    def write(self, **row):
        self.w.writerow({k: ("" if v is None else v) for k, v in row.items()})
        self.f.flush()

    def close(self):
        self.f.close()


class MarksLog:
    """The supervisor presses a key when the system warned about something that was not there (false_alarm) or stayed
    silent about something that was (missed). One CSV row per press, stamped on the same wall clock as the run log."""

    KINDS = ("false_alarm", "missed")

    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "kind"])
        self.f.flush()

    def mark(self, kind, t=None):
        assert kind in self.KINDS
        self.w.writerow([round(time.time() if t is None else t, 3), kind])
        self.f.flush()

    def close(self):
        self.f.close()


def gpu_temp_c(max_age_s=2.0):
    """GPU temperature in C, from the background reader in gabai.diag.sysmon (never blocks the frame loop). None when unreadable."""
    from gabai.diag.sysmon import gpu_read
    return gpu_read()[0]
