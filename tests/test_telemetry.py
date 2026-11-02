import csv
import json
import os
import socket
import tempfile
import time
import urllib.request

from gabai.pc.telemetry import Telemetry, TelemetryLog


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_state_page_and_websocket():
    port = free_port()
    tel = Telemetry(port=port)
    tel.start()
    try:
        tel.update(fps_in=14.9, latency_ms=171, zones={"left": 1.5, "ahead": 0.8, "right": None}, alert="ahead, about one metre")
        tel.event("ahead, about one metre")
        deadline = time.time() + 5
        html = None
        while time.time() < deadline:
            try:
                html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1).read().decode()
                break
            except Exception:
                time.sleep(0.1)
        assert html and "Gabai" in html and "proof of concept" in html.lower()
        state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=2).read())
        assert state["fps_in"] == 14.9 and state["zones"]["ahead"] == 0.8
        assert state["events"][-1]["text"] == "ahead, about one metre"
        import asyncio
        import websockets

        async def one():
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
                return json.loads(await asyncio.wait_for(ws.recv(), 3))
        got = asyncio.run(one())
        assert got["latency_ms"] == 171
    finally:
        tel.stop()


def test_csv_log_writes_one_row_per_call():
    path = os.path.join(tempfile.mkdtemp(), "log.csv")
    log = TelemetryLog(path)
    log.write(t=1.0, fps_in=15.0, latency_ms=170, model_ms=30, left=1.2, ahead=0.9, right=None, alert="ahead", turn_dps=3)
    log.write(t=1.1, fps_in=14.0, latency_ms=190, model_ms=31, left=1.2, ahead=0.8, right=None, alert="", turn_dps=4)
    log.close()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2 and rows[0]["ahead"] == "0.9" and rows[1]["right"] == ""


def test_numpy_values_and_nan_do_not_break_the_page():
    import numpy as np
    port = free_port()
    tel = Telemetry(port=port)
    tel.start()
    try:
        tel.update(fps_in=np.float32(14.5), zones={"left": np.float32(1.5), "ahead": float("nan"), "right": None})
        deadline = time.time() + 5
        while True:
            try:
                state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=1).read())
                break
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(0.1)
        assert abs(state["fps_in"] - 14.5) < 1e-6 and state["zones"]["ahead"] is None and state["zones"]["left"] == 1.5
    finally:
        tel.stop()


def test_marks_log_records_supervisor_key_presses_on_the_wall_clock():
    from gabai.pc.telemetry import MarksLog
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "run.csv.marks.csv")
        m = MarksLog(p)
        m.mark("false_alarm", t=1000.5)
        m.mark("missed", t=1001.25)
        m.close()
        rows = list(csv.DictReader(open(p)))
    assert [(r["t"], r["kind"]) for r in rows] == [("1000.5", "false_alarm"), ("1001.25", "missed")]


def test_log_has_the_phase_m_columns():
    from gabai.pc.telemetry import FIELDS
    for c in ("speech_ms", "mem_mb", "pi_load", "rtt_ms", "gpu_temp", "spoken"):
        assert c in FIELDS
