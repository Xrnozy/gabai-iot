"""Starts rtabmap_bridge.exe and talks to it over a local socket (protocol: rtabmap_lab/bridge/main.cpp).

The bridge does everything RTAB-Map does (stereo odometry, stereo depth, loop closure, the persistent map); this class only feeds it rectified pairs and turns what it
sends back into plain Python objects. Frames are sent one at a time: the next one goes out only after the bridge has answered the last (STATUS), so the bridge is never
behind; callers that have frames to spare simply skip them (counted in `skipped`).
"""
import os
import queue
import socket
import struct
import subprocess
import threading
import time

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(LAB, "bin", "rtabmap_bridge.exe")
RTABMAP_DIR = os.environ.get("RTABMAP_DIR", r"C:\Program Files\RTABMap")
INIT, FRAME, QUIT = 1, 2, 3
STATUS, DISPARITY, NODE, GRAPH, LOG, READY, DEPTH = 101, 102, 103, 104, 105, 106, 107
LOOP_LINKS = (1, 2, 3, 4)                         # Link::kGlobalClosure, kLocalSpaceClosure, kLocalTimeClosure, kUserClosure


def pose4(f):
    """12 floats (row-major 3x4) -> 4x4, or None for the all-zero pose RTAB-Map uses for 'null'."""
    a = np.asarray(f, np.float64).reshape(3, 4)
    if not np.any(a):
        return None
    return np.vstack([a, [0, 0, 0, 1.0]])


class Bridge:
    def __init__(self, db="", port=5710, localize=False, config=None, extra=(), exe=EXE):
        if not os.path.exists(exe):
            raise SystemExit(f"{exe} not found: build it first with  py -m rtabmap_lab setup")
        env = dict(os.environ, PATH=os.path.join(RTABMAP_DIR, "bin") + os.pathsep + os.environ.get("PATH", ""))
        cmd = [exe, "--port", str(port)] + (["--db", db] if db else []) + (["--config", config] if config else []) + (["--localize"] if localize else []) + list(extra)
        self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.out = queue.Queue()                  # decoded messages, oldest first
        self.log = []                             # the bridge's own console lines
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        self.sock = None
        end = time.time() + 20
        while time.time() < end:
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except OSError:
                if self.proc.poll() is not None:
                    raise SystemExit("the bridge exited at start:\n" + "\n".join(self.log[-10:]))
                time.sleep(0.1)
        if self.sock is None:
            raise SystemExit("could not connect to the bridge")
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.busy, self.sent, self.skipped, self.alive, self.ready = False, 0, 0, True, False
        self._lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _pump_stdout(self):
        for line in self.proc.stdout:
            self.log.append(line.rstrip())

    def _recv(self, n):
        buf = bytearray()
        while len(buf) < n:
            c = self.sock.recv(min(n - len(buf), 1 << 20))
            if not c:
                raise ConnectionError("bridge closed")
            buf += c
        return bytes(buf)

    def _read(self):
        try:
            while True:
                t, n = struct.unpack("<II", self._recv(8))
                p = self._recv(n) if n else b""
                msg = decode(t, p)
                if t == STATUS:
                    self.busy = False
                elif t == READY:
                    self.ready = True
                self.out.put(msg)
        except (ConnectionError, OSError):
            self.alive = False
            self.out.put({"type": "closed"})

    def _send(self, t, payload):
        with self._lock:
            self.sock.sendall(struct.pack("<II", t, len(payload)) + payload)

    def init(self, fx, fy, cx, cy, baseline, w, h, stride=4, min_range=0.3, max_range=6.0, disp_every=3, lost_reset=15, max_depth_std=0.08):
        self._send(INIT, struct.pack("<5d2ii2f2if", fx, fy, cx, cy, baseline, w, h, stride, min_range, max_range, disp_every, lost_reset, max_depth_std))

    def send_frame(self, frame_id, stamp, left_bgr, right_gray, gyro=None):
        """True if sent; False when the bridge is still working on the previous frame (the caller skips this one).
        gyro: the camera's rotation since the previous SENT frame (rotation vector, rad, optical frame), from an IMU; the bridge uses it to bridge lost odometry."""
        if self.busy or not self.alive:
            self.skipped += 1
            return False
        self.busy = True
        self.sent += 1
        tail = b"" if gyro is None else struct.pack("<3d", *gyro)
        self._send(FRAME, struct.pack("<id", frame_id, stamp) + np.ascontiguousarray(left_bgr).tobytes() + np.ascontiguousarray(right_gray).tobytes() + tail)
        return True

    def poll(self):
        """Every message that has arrived so far."""
        out = []
        while True:
            try:
                out.append(self.out.get_nowait())
            except queue.Empty:
                return out

    def wait_idle(self, timeout=30.0):
        end = time.time() + timeout
        while self.busy and self.alive and time.time() < end:
            time.sleep(0.002)

    def close(self, save=True):
        try:
            if self.alive:
                self._send(QUIT, b"")
        except OSError:
            pass
        try:
            self.proc.wait(timeout=60 if save else 5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def decode(t, p):
    if t == STATUS:
        sid, stamp, lost = struct.unpack_from("<idi", p, 0)
        pose, corr = pose4(struct.unpack_from("<12f", p, 16)), pose4(struct.unpack_from("<12f", p, 64))
        features, inliers, matches, local_map = struct.unpack_from("<4i", p, 112)
        odom_ms, map_ms, disp_ms = struct.unpack_from("<3f", p, 128)
        added, loop, prox = struct.unpack_from("<3i", p, 140)
        nposes, nloops, nres = struct.unpack_from("<3i", p, 152)
        dist = struct.unpack_from("<f", p, 164)[0]
        return {"type": "status", "id": sid, "stamp": stamp, "lost": bool(lost), "odom": pose, "map_correction": corr if corr is not None else np.eye(4),
                "features": features, "inliers": inliers, "matches": matches, "local_map": local_map, "odom_ms": odom_ms, "rtabmap_ms": map_ms, "disparity_ms": disp_ms,
                "added": bool(added), "loop_id": loop, "proximity_id": prox, "nodes": nposes, "loops": nloops, "restarts": nres, "distance": dist}
    if t == DISPARITY:
        r, c = struct.unpack_from("<ii", p, 0)
        return {"type": "disparity", "image": np.frombuffer(p, np.uint8, r * c, 8).reshape(r, c)}
    if t == NODE:
        nid = struct.unpack_from("<i", p, 0)[0]
        pose = pose4(struct.unpack_from("<12f", p, 4))
        n = struct.unpack_from("<i", p, 52)[0]
        rec = np.frombuffer(p, np.dtype([("xyz", "<f4", 3), ("bgr", "u1", 3)]), n, 56)
        return {"type": "node", "id": nid, "pose": pose, "points": rec["xyz"].copy(), "bgr": rec["bgr"].copy()}
    if t == GRAPH:
        n = struct.unpack_from("<i", p, 0)[0]
        poses, o = {}, 4
        for _ in range(n):
            i = struct.unpack_from("<i", p, o)[0]
            poses[i] = pose4(struct.unpack_from("<12f", p, o + 4))
            o += 52
        nl = struct.unpack_from("<i", p, o)[0]
        links = [struct.unpack_from("<3i", p, o + 4 + 12 * k) for k in range(nl)]
        return {"type": "graph", "poses": poses, "links": links}
    if t == DEPTH:
        nid, fid, r, c = struct.unpack_from("<iiii", p, 0)
        depth = np.frombuffer(p, np.uint16, r * c, 16).reshape(r, c)
        bgr = np.frombuffer(p, np.uint8, r * c * 3, 16 + r * c * 2).reshape(r, c, 3)
        right = np.frombuffer(p, np.uint8, r * c, 16 + r * c * 5).reshape(r, c)
        return {"type": "depth", "id": nid, "frame_id": fid, "depth_mm": depth.copy(), "bgr": bgr.copy(), "right": right.copy()}
    if t == LOG:
        return {"type": "log", "text": p.decode("utf-8", "replace")}
    if t == READY:
        return {"type": "ready", "nodes": struct.unpack_from("<i", p, 0)[0]}
    return {"type": "unknown", "code": t}
