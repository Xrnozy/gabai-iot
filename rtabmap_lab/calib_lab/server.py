"""The calibration lab's web server: one page (static/index.html), one WebSocket. py -m rtabmap_lab calib-lab

  py -m rtabmap_lab calib-lab --folder data/recordings/walk_20261102_230115 --calib data/stereo_calib_charuco_45.npz --clip 0 300
Opens http://127.0.0.1:8765 . The page needs no internet: three.js is in static/.
"""
import argparse
import asyncio
import json
import os
import struct
import threading
import webbrowser

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rtabmap_lab.calib_lab.lab import Lab

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def build_app(lab):
    app = FastAPI()
    clients, loop_ref = set(), {}

    def broadcast(data):
        loop = loop_ref.get("loop")
        if loop is None:
            return
        def put():
            for q in list(clients):
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    pass
        loop.call_soon_threadsafe(put)

    lab.sink = broadcast

    @app.on_event("startup")
    async def _start():
        loop_ref["loop"] = asyncio.get_running_loop()

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q = asyncio.Queue(maxsize=400)
        h = json.dumps(lab.doc()).encode()
        await sock.send_bytes(struct.pack("<I", len(h)) + h)
        for data in lab.snapshot():
            await sock.send_bytes(data)
        lab.push_state()
        clients.add(q)

        async def pump():
            while True:
                await sock.send_bytes(await q.get())
        task = asyncio.create_task(pump())
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                await asyncio.get_running_loop().run_in_executor(None, command, lab, msg)
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(q)
            task.cancel()

    return app


def command(lab, m):
    """One command from the page. Errors go back to the page's log, never into a crash."""
    c = m.get("cmd")
    try:
        if c == "set_param":
            lab.set_param(m["name"], m["value"])
        elif c == "reset_param":
            lab.reset_param(m["name"])
        elif c == "reset_all":
            lab.reset_all()
        elif c == "play":
            lab.play()
        elif c == "pause":
            lab.pause()
        elif c == "restart":
            lab.restart()
        elif c == "seek":
            lab.seek(m["idx"])
        elif c == "set_clip":
            lab.set_clip(m["start"], m["end"])
        elif c == "set_flag":
            setattr(lab, m["name"], bool(m["value"]))
            lab.push_state()
        elif c == "pin":
            lab.pin_reference(m.get("name", ""))
        elif c == "unpin":
            lab.clear_reference()
        elif c == "auto":
            lab.auto_calibrate(int(m.get("n", 40)), bool(m.get("whole", True)))
        elif c == "apply_auto":
            lab.apply_auto()
        elif c == "save":
            lab.save_config(m["name"], m.get("kind", "manual"), m.get("notes", ""))
        elif c == "load":
            lab.load_config(m["name"])
        elif c == "export_npz":
            lab.export_npz(m["name"])
        elif c == "compare":
            lab.compare(m["a"], m["b"])
        elif c == "batch":
            lab.batch(m["names"])
        elif c == "depth_at":
            lab.depth_at(m["x"], m["y"])
        elif c == "export":
            lab.export_measurements(m["points"], m.get("note", ""))
        elif c == "scale_to_distance":
            measured, true = float(m["measured_m"]), float(m["true_m"])
            if measured <= 0 or true <= 0:
                raise ValueError("both distances must be above 0")
            old = lab.manual.values["baseline_mm"]
            lab.set_param("baseline_mm", old * true / measured)
            lab.note(f"baseline {old:.2f} -> {lab.manual.values['baseline_mm']:.2f} mm so that the picked point reads {true:.3f} m instead of {measured:.3f} m (this assumes the focal length is right)")
        else:
            raise ValueError(f"unknown command {c!r}")
    except Exception as e:
        lab.note(f"{c}: {type(e).__name__}: {e}", "error")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="data/recordings/walk_20261102_230115", help="recorded pair folder (never written to)")
    ap.add_argument("--calib", default="data/stereo_calib_charuco_45.npz", help="the saved calibration the session starts from ('original' for reset)")
    ap.add_argument("--clip", type=int, nargs=2, default=[0, 300], metavar=("START", "END"), help="pairs RTAB-Map processes (every restart replays this clip)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default="data/calib_lab", help="where configs, experiments and measurements are saved")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    lab = Lab(a.folder, a.calib, a.root, a.clip)
    app = build_app(lab)
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}")).start()
    print(f"calibration lab: http://127.0.0.1:{a.port}   recording {a.folder}   saved calibration {a.calib}", flush=True)
    try:
        uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
    finally:
        lab.close()


if __name__ == "__main__":
    main()
