"""Standalone LiDAR test, Pi side (not part of the pipeline): same link as gabai.pi.sender, own port 5601.

11-byte frames at 115200 baud (distance only, no angle). Connects OUT to the PC and sends a batch of distances every 0.1 s.
  PC:  py -m gabai.tools.lidar_view
  Pi:  python3 -m gabai.pi.lidar_sender --host <PC_IP>
The motor must already spin (about 300 RPM); this script does not drive it.
"""
import argparse
import os
import socket
import time

from gabai.net import protocol as P

LIDAR = 7  # json {"d": [dist_mm, ...]}, the readings of the last ~0.1 s
PKT = 11  # 0xAA, 00 09 01 61 ae 00 01, dist(2, little endian, mm), checksum


def checksum_ok(p):
    return (sum(p[:9]) & 0xFF) == p[10]


def parse(p):
    return p[8] | p[9] << 8


def packets(read):
    buf = bytearray()
    while True:
        buf += read(512)
        while len(buf) >= PKT:
            if buf[0] != 0xAA:
                del buf[0]
                continue
            p = bytes(buf[:PKT])
            if checksum_ok(p):
                del buf[:PKT]
                yield p
            else:
                del buf[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="PC address: LAN IP or Tailscale 100.x.y.z")
    ap.add_argument("--port", type=int, default=5601)
    ap.add_argument("--token", default=os.environ.get("GABAI_TOKEN", "gabai"))
    ap.add_argument("--serial", default="/dev/serial0")
    a = ap.parse_args()

    import serial
    ser = serial.Serial(a.serial, 115200, timeout=0.1)
    stats = {"bytes": 0, "packets": 0, "last": time.time()}

    def read(n):
        b = ser.read(n)
        stats["bytes"] += len(b)
        if time.time() - stats["last"] > 2:
            print(f"serial bytes {stats['bytes']}, valid packets {stats['packets']}", flush=True)
            stats["last"] = time.time()
        return b

    while True:
        try:
            sock = socket.create_connection((a.host, a.port), timeout=10)
            sock.sendall(P.pack_json(P.HELLO, time.monotonic(), {"token": a.token, "name": socket.gethostname()}))
            print(f"connected to {a.host}:{a.port}", flush=True)
            d, last = [], time.time()
            for p in packets(read):
                stats["packets"] += 1
                d.append(parse(p))
                if time.time() - last > 0.1:
                    sock.sendall(P.pack_json(LIDAR, time.monotonic(), {"d": d}))
                    d, last = [], time.time()
        except OSError as e:
            print(f"link down ({e}); retrying in 2 s", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
