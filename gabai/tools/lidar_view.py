"""Standalone LiDAR test, PC side (not part of the pipeline): listens for gabai.pi.lidar_sender and draws each scan in Rerun.

    py -m gabai.tools.lidar_view [--port 5601] [--token gabai]
"""
import argparse
import hmac
import json
import socket
import time

import collections
import math

import numpy as np
import rerun as rr

from gabai.net import protocol as P
from gabai.pi.lidar_sender import LIDAR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5601)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--rpm", type=float, default=300, help="assumed head speed; the frames carry no angle, so angle = time x rpm (approximate, phase is arbitrary)")
    ap.add_argument("--window", type=float, default=2.0, help="seconds of points kept in the cloud")
    a = ap.parse_args()

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", a.port))
    srv.listen(1)
    rr.init("gabai_lidar_test", spawn=True)
    t0 = time.time()
    cloud = collections.deque()  # (t, x, y)
    print(f"listening on 0.0.0.0:{a.port} (token '{a.token}'). Ctrl+C to stop.", flush=True)
    while True:
        conn, addr = srv.accept()
        try:
            t, _, payload = P.read_message(conn)
            if t != P.HELLO or not hmac.compare_digest(str(json.loads(payload).get("token", "")), a.token):
                print(f"rejected {addr[0]}: not a HELLO or token mismatch", flush=True)
                continue
            print(f"connected: {addr[0]}", flush=True)
            while True:
                t, _, payload = P.read_message(conn)
                if t != LIDAR:
                    continue
                d = json.loads(payload)["d"]
                now = time.time() - t0
                for i, v in enumerate(d):
                    rr.set_time("t", duration=now - 0.1 + 0.1 * i / len(d))
                    rr.log("lidar/dist_mm", rr.Scalars(v))
                    ts = now - 0.1 + 0.1 * i / len(d)
                    ang = 2 * math.pi * ts * a.rpm / 60
                    cloud.append((ts, v / 1000 * math.cos(ang), v / 1000 * math.sin(ang)))
                while cloud and cloud[0][0] < now - a.window:
                    cloud.popleft()
                if cloud:
                    rr.log("lidar/cloud", rr.Points2D(np.array(cloud)[:, 1:], radii=0.01))
        except (OSError, P.ProtocolError) as e:
            print(f"link lost ({e}); waiting for the Pi", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
