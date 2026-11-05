"""Run the caregiver relay:  py -m gabai.relay --port 8090

The Pi keeps sending to the PC receiver exactly as before (docs/live-link-guide.md); this process owns that receiver
unless --no-pi-bridge is given because the navigation pipeline already holds the port.
"""
import argparse
import os

import uvicorn

from . import discovery
from .app import create_app, load_tokens
from .bridge import PiBridge, Ticker
from .core import Relay, default_db_path
from .frames import FrameStore


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0", help="bind address; 0.0.0.0 serves the hotspot")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--db", default=default_db_path())
    ap.add_argument("--pi-port", type=int, default=5600)
    ap.add_argument("--pi-token", default="gabai", help="token the Pi sender uses (sender --token)")
    ap.add_argument("--no-pi-bridge", action="store_true", help="do not open the Pi receiver port")
    a = ap.parse_args()

    db = os.path.abspath(a.db)
    os.makedirs(os.path.dirname(db), exist_ok=True)
    tokens_path = os.path.join(os.path.dirname(db), "relay_tokens.json")
    tokens = load_tokens(tokens_path)
    relay = Relay(db)
    frames = FrameStore(save_dir=os.path.dirname(db))
    Ticker(relay).start()
    if not a.no_pi_bridge:
        from gabai.pc.receiver import LiveReceiver
        receiver = LiveReceiver(port=a.pi_port, token=a.pi_token)
        receiver.start()
        PiBridge(relay, frames, receiver).start()
    discovery.serve(a.port)
    print(f"relay on {a.host}:{a.port}; tokens in {tokens_path}")
    print("phones can reach it at: " + ", ".join(f"http://{ip}:{a.port}" for ip in discovery.lan_addresses()))
    uvicorn.run(create_app(relay, frames, tokens), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
