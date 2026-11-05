"""LAN discovery: phones broadcast b"GABAI?" to UDP DISCOVERY_PORT and the relay answers with its HTTP port.

No internet, no multicast, no extra dependency. The reply's source address is the relay's address, so the phone never has to be told it.
"""
import json
import socket
import threading

DISCOVERY_PORT = 8091
QUERY = b"GABAI?"


def reply_for(data: bytes, relay_port: int):
    """The answer to one datagram, or None when it is not a GabAI query (anything else on the port is ignored)."""
    if data.strip() != QUERY:
        return None
    return json.dumps({"gabai": 1, "relay_port": relay_port}).encode()


def serve(relay_port: int, udp_port: int = DISCOVERY_PORT) -> socket.socket:
    """Start the responder thread and return its socket (close it to stop)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", udp_port))

    def loop():
        while True:
            try:
                data, addr = sock.recvfrom(64)
            except OSError:
                return
            out = reply_for(data, relay_port)
            if out:
                sock.sendto(out, addr)

    threading.Thread(target=loop, daemon=True, name="gabai-discovery").start()
    return sock


def lan_addresses() -> list:
    """Private IPv4 addresses of this machine, for the start-up banner (the hotspot is usually 192.168.137.1)."""
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    return sorted({i[4][0] for i in infos if not i[4][0].startswith("127.")})
