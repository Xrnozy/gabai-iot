"""Run the relay inside the pipeline process, on the Pi receiver the pipeline already owns (port 5600 has one owner).

    relay, frames = embed.start(rx, port=8090)

Frames then reach the caregiver from the same receiver the pipeline reads, and results go in with `relay.ingest_nav`
without an HTTP hop, so a result's frame_id is exactly the id of a frame in `frames`.
"""
import os
import threading

import uvicorn

from . import discovery
from .app import create_app, load_tokens
from .bridge import PiBridge, Ticker
from .core import Relay, default_db_path
from .frames import FrameStore


def start(receiver, port=8090, host="0.0.0.0", db=None):
    db = os.path.abspath(db or default_db_path())
    os.makedirs(os.path.dirname(db), exist_ok=True)
    tokens = load_tokens(os.path.join(os.path.dirname(db), "relay_tokens.json"))
    relay, frames = Relay(db), FrameStore(save_dir=os.path.dirname(db))
    Ticker(relay).start()
    PiBridge(relay, frames, receiver, poll_s=0.1).start()
    discovery.serve(port)
    server = uvicorn.Server(uvicorn.Config(create_app(relay, frames, tokens), host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="relay-http").start()
    print("relay inside the pipeline; phones can reach it at: " + ", ".join(f"http://{ip}:{port}" for ip in discovery.lan_addresses()), flush=True)
    return relay, frames
