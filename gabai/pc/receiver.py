"""PC-side TCP receiver for the Pi sender. Keeps only the newest stereo pair (older frames are dropped).

Run standalone to check the link:   py -m gabai.pc.receiver --port 5600

Everything that can go wrong on this link is counted and logged (component "link", see gabai.diag.log): rejected connections, bad
messages and why a connection ended, undecodable frames, frames replaced before anyone used them (`overwritten`), long gaps
between frames, IMU samples lost to a full queue. Nothing here fails silently.
"""
import argparse
import hmac
import json
import os
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from gabai.diag import log as dlog
from gabai.net import protocol as P
from gabai.pc.syncbuf import SyncBuffer

_log = dlog.get("link")
GAP_WARN_S = 0.30               # a gap between frames longer than this is logged
IMU_QUEUE, MAG_QUEUE = 6000, 3000


@dataclass
class PairFrame:
    seq: int
    left: np.ndarray
    right: np.ndarray
    t_capture: float  # sender's monotonic clock
    t_recv: float     # this PC's monotonic clock
    dt_ms: Optional[float] = None   # right capture time minus left, measured on the Pi; None if the sender does not send it


@dataclass
class KltPoints:
    """Feature points the Pi tracked in one pair (from `sender --klt`). t_capture equals PairFrame.t_capture. Pixels of the sent picture, OpenCV convention."""
    t_capture: float
    frame_seq: int              # the Pi's camera frame counter of that pair: the points belong to this pair and no other
    ids: np.ndarray
    xy: np.ndarray
    status: np.ndarray          # P.KLT_TRACKED / KLT_NEW / KLT_LOST
    quality: np.ndarray         # LK patch error x 8, 255 = worst
    xy_right: Optional[np.ndarray]
    klt_ms: float
    motion: Optional[dict] = None   # the Pi's own frame-to-frame motion estimate (sender --klt-motion)


_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _rotate(img, deg):
    return cv2.rotate(img, _ROT[deg % 360]) if deg % 360 else img


def _pi_view(h, age_s):
    """What the Pi last said about itself, for the log line of a stall: under-voltage, a camera below its rate, or a slow round trip names the cause, and a health
    report that is itself old (age) says the Pi's loop was blocked, not only the picture."""
    from gabai.diag.pi_status import active_flags
    now = [f for f in (active_flags(h.get("throttled")) or []) if f.endswith("_now")]
    cams = [h.get("cam_fps_left"), h.get("cam_fps_right")]
    rtt = h.get("rtt_s")
    out = dict(pi_power_now=now or None, pi_send_fps=h.get("fps"), pi_cam_fps=cams if any(c is not None for c in cams) else None,
               pi_rtt_ms=None if rtt is None else round(rtt * 1000), pi_load1=h.get("load1"), pi_temp_c=h.get("temp_c"),
               pi_health_age_s=None if age_s is None else round(age_s, 1))
    return {k: v for k, v in out.items() if v is not None}


def _reason(e):
    """Why a connection ended, in words an investigator can use."""
    if isinstance(e, socket.timeout):
        return "no data for 10 s (receive timeout): the Pi stopped sending or the link stalled"
    if isinstance(e, P.ProtocolError):
        return f"protocol error: {e}"
    if isinstance(e, ConnectionError):
        return "peer closed or reset the connection" + (f" ({e})" if str(e) and str(e) != "connection closed" else "")
    if isinstance(e, OSError):
        return f"socket error: {type(e).__name__}: {e}"
    return f"bad message content: {type(e).__name__}: {e}"


class LiveReceiver:
    def __init__(self, port=5600, token="gabai", host="0.0.0.0", rotate_left=0, rotate_right=0):
        self.host, self.token = host, token
        self.rot = (rotate_left, rotate_right)  # degrees: 0, 90, 180, 270; fixes a camera mounted upside down
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if os.name == "nt":
            # Windows lets a second process share a port with SO_REUSEADDR, so the Pi would connect to the stale
            # instance and the new window would stay empty. Exclusive bind makes the second one fail loudly.
            self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._srv.bind((host, port))
        except OSError as e:
            _log.error("link.bind_failed", port=port, error=str(e))
            raise SystemExit(f"port {port} is already in use ({e}). Another Gabai window (viewer, calibration or "
                             "receiver) is still running; close it first.")
        self._srv.listen(2)
        self._srv.settimeout(0.5)
        self.port = self._srv.getsockname()[1]
        self._lock = threading.Lock()
        self._latest = None
        self._consumed_seq = 0
        self._stop = threading.Event()
        self._conn = None
        self._times = deque(maxlen=60)
        self._gaps = deque(maxlen=60)     # seconds between consecutive frames
        self._mag_q = deque(maxlen=MAG_QUEUE)  # (t, x, y, z) raw counts, Pi clock
        self._imu_q = deque(maxlen=IMU_QUEUE)  # (t, gx, gy, gz, ax, ay, az), Pi clock
        self._klt = deque(maxlen=16)           # KltPoints of the latest pairs
        self.sync = SyncBuffer()               # the SLAM path: every pair, KLT set and IMU sample, off until enable_sync() (the display path below keeps only the newest pair)
        self._imu_next = None                  # first IMU sample number not received yet (sent back to the Pi after a reconnect, so it resends what the old link lost)
        self._s = dict(frames=0, bytes=0, rejected=0, connections=0, connected=False, health={}, hello={}, imu={},
                       decode_errors=0, overwritten=0, imu_overflow=0, mag_overflow=0, msg_counts={}, last_error=None)
        _log.info("link.listening", port=self.port, host=host, rotate=list(self.rot))

    def start(self):
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def stop(self):
        self._stop.set()
        for s in (self._conn, self._srv):
            try:
                s.close()
            except Exception:
                pass

    def latest(self):
        with self._lock:
            if self._latest is not None:
                self._consumed_seq = self._latest.seq
            return self._latest

    def drain_imu(self):
        """All IMU samples received since the last call, oldest first."""
        with self._lock:
            out = list(self._imu_q)
            self._imu_q.clear()
        return out

    def klt_for(self, t_capture):
        """The feature points that belong to the pair with this t_capture, or None (no --klt, or they have not arrived yet: they follow their pair by a few ms)."""
        with self._lock:
            return next((k for k in reversed(self._klt) if k.t_capture == t_capture), None)

    def enable_sync(self, **kw):
        """Keep the full measurement record for VIO/SLAM (see gabai/pc/syncbuf.py); kw = SyncBuffer settings."""
        self.sync.enable(**kw)
        return self.sync

    def _imu_seq_filter(self, seq0, rows):
        """(rows not seen before, samples missing before them). Sample numbers come from the Pi: a repeat (a resend after a reconnect) is dropped, a jump is counted."""
        n = len(rows)
        if self._imu_next is None or seq0 > self._imu_next:
            miss = 0 if self._imu_next is None else seq0 - self._imu_next
        else:
            miss = 0
            if seq0 + n <= self._imu_next:
                self._s["imu_dup"] = self._s.get("imu_dup", 0) + n
                return [], 0
            rows = rows[self._imu_next - seq0:]
            self._s["imu_dup"] = self._s.get("imu_dup", 0) + (n - len(rows))
        self._imu_next = seq0 + n
        return rows, miss

    def drain_mag(self):
        """All compass readings received since the last call, oldest first."""
        with self._lock:
            out = list(self._mag_q)
            self._mag_q.clear()
        return out

    def stats(self):
        with self._lock:
            s = dict(self._s)
            s["msg_counts"] = dict(self._s["msg_counts"])
            now = time.monotonic()
            ts = list(self._times)
            s["fps"] = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] and now - ts[-1] < 2 else 0.0
            s["frame_age_s"] = (now - ts[-1]) if ts else None
            s["inter_arrival_max_ms"] = round(max(self._gaps) * 1000, 1) if self._gaps else None
            s["imu_queue"], s["mag_queue"] = len(self._imu_q), len(self._mag_q)
            s["imu_next"] = self._imu_next
            s["sync"] = self.sync.stats()
            return s

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except (socket.timeout, OSError):
                continue
            if self._conn is not None:  # a new Pi connection replaces the old one
                _log.warning("link.replaced", peer=addr[0], note="a new connection replaces the previous one")
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = conn
            conn.settimeout(10)
            _log.info("link.accept", peer=addr[0], port=addr[1])
            threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()

    def _count(self, t):
        with self._lock:
            self._s["msg_counts"][str(t)] = self._s["msg_counts"].get(str(t), 0) + 1

    def _handle(self, conn, addr):
        t_start, frames, reason = time.monotonic(), 0, None
        try:
            t, _, payload = P.read_message(conn)
            hello = json.loads(payload) if t == P.HELLO else {}
            if t != P.HELLO or not hmac.compare_digest(str(hello.get("token", "")), self.token):
                with self._lock:
                    self._s["rejected"] += 1
                _log.warning("link.rejected", peer=addr[0], reason="first message was not a HELLO" if t != P.HELLO else "token mismatch",
                             msg_type=t)
                conn.close()
                return
            hello = {k: v for k, v in hello.items() if k != "token"}
            with self._lock:
                self._s.update(connections=self._s["connections"] + 1, connected=True, hello=hello, last_error=None)
                self._s["peer"] = addr[0]
            _log.info("link.hello", peer=addr[0], **{k: hello.get(k) for k in ("name", "width", "height", "fps")})
            if self._imu_next is not None:                                      # a reconnect: tell the Pi where the IMU record ends so it sends the rest again
                conn.sendall(P.pack_json(P.ACK, 0.0, {"imu_next": self._imu_next}))
                _log.info("link.ack", imu_next=self._imu_next)
            seq, last_arrival, health_t = 0, None, None
            while not self._stop.is_set():
                t, tc, payload = P.read_message(conn)
                self._count(t)
                if t == P.FRAME:
                    lb, rb, dt_ms, fseq = P.unpack_frame_seq(payload)
                    self.sync.add_frame(None if fseq is None else fseq[0], None if fseq is None else fseq[1], tc, lb, rb, dt_ms)
                    left = cv2.imdecode(np.frombuffer(lb, np.uint8), cv2.IMREAD_COLOR)
                    right = cv2.imdecode(np.frombuffer(rb, np.uint8), cv2.IMREAD_COLOR)
                    now = time.monotonic()
                    if left is None or right is None:
                        with self._lock:
                            self._s["decode_errors"] += 1
                        _log.warning_every("decode", 5.0, "link.decode_error", t_capture=tc, left_bytes=len(lb), right_bytes=len(rb),
                                           left_is_jpeg=lb[:2] == b"\xff\xd8", right_is_jpeg=rb[:2] == b"\xff\xd8",
                                           left_decoded=left is not None, right_decoded=right is not None)
                        continue
                    left, right = _rotate(left, self.rot[0]), _rotate(right, self.rot[1])
                    seq += 1
                    frames += 1
                    if last_arrival is not None and now - last_arrival > GAP_WARN_S:
                        with self._lock:
                            pi = _pi_view(self._s["health"], None if health_t is None else now - health_t)
                        _log.warning_every("gap", 2.0, "link.gap", gap_ms=round((now - last_arrival) * 1000), t_capture=tc, frames_so_far=frames, **pi)
                    with self._lock:
                        if last_arrival is not None:
                            self._gaps.append(now - last_arrival)
                        if self._latest is not None and self._latest.seq > self._consumed_seq:
                            self._s["overwritten"] += 1          # the previous pair was never taken by the consumer
                        self._latest = PairFrame(seq, left, right, tc, now, dt_ms)
                        self._s["frames"] += 1
                        self._s["bytes"] += len(payload)
                        self._times.append(now)
                    last_arrival = now
                elif t == P.HEALTH:
                    health_t = time.monotonic()
                    with self._lock:
                        self._s["health"] = json.loads(payload)
                elif t == P.IMU:
                    msg = json.loads(payload)
                    with self._lock:
                        self._s["imu"] = {k: v for k, v in msg.items() if k not in ("samples", "mag")}
                        new_i, new_m = msg.get("samples", []), msg.get("mag", [])
                        miss = 0
                        if msg.get("seq") is not None and new_i:
                            new_i, miss = self._imu_seq_filter(int(msg["seq"]), new_i)
                        self.sync.add_imu(new_i, miss)
                        self._s["imu_overflow"] += max(0, len(self._imu_q) + len(new_i) - IMU_QUEUE)
                        self._s["mag_overflow"] += max(0, len(self._mag_q) + len(new_m) - MAG_QUEUE)
                        self._imu_q.extend(tuple(x) for x in new_i)
                        self._mag_q.extend(tuple(x) for x in new_m)
                        if "mag_overflows" in msg:
                            self._s["mag_overflows"] = msg["mag_overflows"]
                elif t == P.KLT:
                    fs, ids, xy, st, q, xr, ms, mo = P.unpack_klt(payload)
                    kp = KltPoints(tc, fs, ids, xy, st, q, xr, ms, mo)
                    with self._lock:
                        self._klt.append(kp)
                    self.sync.add_klt(kp)
                elif t == P.PING:
                    conn.sendall(P.pack(P.PONG, tc, json.dumps({"t1": json.loads(payload)["t1"], "t2": time.monotonic()}).encode()))
                else:
                    _log.warning_every(f"unk{t}", 5.0, "link.unknown_message", msg_type=t, payload_bytes=len(payload))
        except (ConnectionError, P.ProtocolError, socket.timeout, OSError, ValueError, KeyError) as e:
            reason = _reason(e)
            with self._lock:
                self._s["last_error"] = reason
        finally:
            with self._lock:
                if self._conn is conn:
                    self._s["connected"] = False
            if reason is not None:
                _log.warning("link.disconnected", peer=addr[0], reason=reason, frames=frames, duration_s=round(time.monotonic() - t_start, 1))
            try:
                conn.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--rotate-left", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--rotate-right", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING (default INFO)")
    a = ap.parse_args()
    run = dlog.configure(run_name="receiver", level=a.log_level)
    print(f"diagnostic log: {run}")
    rx = LiveReceiver(a.port, a.token, rotate_left=a.rotate_left, rotate_right=a.rotate_right)
    rx.start()
    print(f"listening on 0.0.0.0:{rx.port}  (token '{a.token}'). Ctrl+C to stop.")
    last = 0
    try:
        while True:
            time.sleep(1)
            s = rx.stats()
            mbps = (s["bytes"] - last) * 8 / 1e6
            last = s["bytes"]
            h = s["health"]
            print(f"connected={s['connected']} frames={s['frames']} fps={s['fps']:.1f} {mbps:.1f} Mbit/s "
                  f"rejected={s['rejected']} pi_temp={h.get('temp_c')} throttled={h.get('throttled')} "
                  f"overwritten={s['overwritten']} decode_errors={s['decode_errors']} last_error={s['last_error']}", flush=True)
    except KeyboardInterrupt:
        rx.stop()
        dlog.shutdown()


if __name__ == "__main__":
    main()
