"""Wire format between the Pi sender and the PC receiver (plain TCP, works over LAN or a Tailscale IP).

Message = header + payload.  header = magic "GBAI", type (1 byte), t_capture (float64, sender's
monotonic clock, seconds), payload length (uint32), all big-endian.

  HELLO  json {"token":..., "name":..., "width":..., "height":..., "fps":...}   first message, required
  FRAME  payload = uint32 len_left, uint32 len_right, left JPEG bytes, right JPEG bytes
  HEALTH json {"throttled":..., "temp_c":..., "mem_available_mb":..., "load1":..., "fps":..., "clock_offset_s":...}
  IMU    json {"gyro_dps": [x,y,z], "peak_dps": recent peak turn speed, "accel_g": [x,y,z],
               "samples": [[t, gx, gy, gz, ax, ay, az], ...], "seq": number of the first sample (every sample has one, counted on the Pi), "mag": [[t, x, y, z], ...]}
               sent BEFORE each frame pair, so the samples up to the frame's t are already there when it arrives; t is the Pi clock, like FRAME
  FRAME  trailer (optional): float32 dt_ms (NaN = unknown), uint32 seq_left, uint32 seq_right: the cameras' own frame counters, so a gap is a frame the Pi skipped
  ACK    PC -> Pi, right after HELLO: json {"imu_next": first IMU sample seq the PC still needs}. The Pi sends the samples from there again (the ones the broken link lost)
  PING   json {"t1": sender clock}          PC replies PONG json {"t1":..., "t2": receiver clock}
  KLT    optional, only from `sender --klt`. t_capture equals the FRAME it belongs to. Payload: uint32 frame_seq (the FRAME's seq_left), uint8 flags
         (bit0 = right-image points present, bit1 = the Pi's own motion estimate follows), uint16 n, float32 klt_ms, [motion block: uint32 prev_seq, float32 dt (s since
         that pair), 3 float32 rotation vector and 3 float32 translation (this camera in the previous camera's frame, metres, rectified left camera), uint16 inliers,
         float32 rms px], then n records of: uint32 id, uint16 x, uint16 y, uint8 status, uint8 quality
         (LK patch error x 8, 255 = worst; 0 for a new point) [, uint16 xr, uint16 yr]. x, y are
         pixels * 16 in the full sent image, OpenCV convention (pixel centre at the integer); 0xFFFF in xr = not seen in the right image.
"""
import json
import struct

import numpy as np

MAGIC = b"GBAI"
HEADER = struct.Struct("!4sBdI")
MAX_PAYLOAD = 8 * 1024 * 1024
HELLO, FRAME, HEALTH, PING, PONG, IMU, KLT, ACK = 1, 2, 3, 4, 5, 6, 7, 8
KLT_TRACKED, KLT_NEW, KLT_LOST = 0, 1, 2
KLT_HEAD = struct.Struct("!IBHf")
KLT_MOTION = struct.Struct("!IfffffffHf")
KLT_SUBPX, KLT_NONE = 16, 0xFFFF
_KLT_REC = [("id", ">u4"), ("x", ">u2"), ("y", ">u2"), ("st", "u1"), ("q", "u1")]
_KLT_REC_R = _KLT_REC + [("xr", ">u2"), ("yr", ">u2")]


class ProtocolError(Exception):
    pass


def pack(msg_type, t_capture, payload=b""):
    return HEADER.pack(MAGIC, msg_type, t_capture, len(payload)) + payload


def pack_json(msg_type, t_capture, obj):
    return pack(msg_type, t_capture, json.dumps(obj).encode())


def pack_frame(left_jpeg, right_jpeg, dt_ms=None, seq=None):
    """Two JPEGs. dt_ms (right capture time minus left, in ms) rides after them as one float32 when the sender measured it; seq = (left, right) camera frame counters
    adds two uint32 (and then dt_ms is always present, NaN when unknown)."""
    if seq is not None:
        tail = struct.pack("!fII", float("nan") if dt_ms is None else dt_ms, *seq)
    else:
        tail = b"" if dt_ms is None else struct.pack("!f", dt_ms)
    return struct.pack("!II", len(left_jpeg), len(right_jpeg)) + left_jpeg + right_jpeg + tail


def unpack_frame_seq(payload):
    """(left jpeg, right jpeg, dt_ms or None, (seq_left, seq_right) or None). An older sender sends no trailer, or only dt_ms."""
    if len(payload) < 8:
        raise ProtocolError("frame payload too short")
    nl, nr = struct.unpack("!II", payload[:8])
    extra = len(payload) - 8 - nl - nr
    if extra not in (0, 4, 12):
        raise ProtocolError("frame lengths do not match payload")
    dt, seq = None, None
    if extra:
        dt = struct.unpack("!f", payload[8 + nl + nr:12 + nl + nr])[0]
        dt = None if dt != dt else dt
    if extra == 12:
        seq = struct.unpack("!II", payload[12 + nl + nr:])
    return payload[8:8 + nl], payload[8 + nl:8 + nl + nr], dt, seq


def unpack_frame_dt(payload):
    """(left jpeg, right jpeg, dt_ms or None)."""
    return unpack_frame_seq(payload)[:3]


def unpack_frame(payload):
    return unpack_frame_dt(payload)[:2]


def _q(a):
    return np.where(np.isfinite(a), np.clip(np.rint(a * KLT_SUBPX), 0, KLT_NONE - 1), KLT_NONE)


def pack_klt(frame_seq, ids, xy, status, quality=None, xy_right=None, klt_ms=0.0, motion=None):
    """ids (n,), xy (n,2) pixels, status (n,), quality (n,) uint8, xy_right (n,2) with NaN where the point is not in the right image,
    motion = dict(prev_seq, dt, rvec, t, n_inl, rms) or None."""
    n = len(ids)
    rec = np.zeros(n, np.dtype(_KLT_REC if xy_right is None else _KLT_REC_R))
    rec["id"], rec["x"], rec["y"], rec["st"] = ids, _q(xy[:, 0]), _q(xy[:, 1]), status
    if quality is not None:
        rec["q"] = quality
    if xy_right is not None:
        rec["xr"], rec["yr"] = _q(xy_right[:, 0]), _q(xy_right[:, 1])
    flags = (0 if xy_right is None else 1) | (0 if motion is None else 2)
    mb = b"" if motion is None else KLT_MOTION.pack(motion["prev_seq"], motion["dt"], *motion["rvec"], *motion["t"], motion["n_inl"], motion["rms"])
    return KLT_HEAD.pack(frame_seq, flags, n, klt_ms) + mb + rec.tobytes()


def unpack_klt(payload):
    """(frame_seq, ids, xy float32 (n,2), status, quality, xy_right float32 (n,2) with NaN for unseen or None, klt_ms, motion dict or None)."""
    if len(payload) < KLT_HEAD.size:
        raise ProtocolError("klt payload too short")
    seq, flags, n, ms = KLT_HEAD.unpack_from(payload)
    dt = np.dtype(_KLT_REC_R if flags & 1 else _KLT_REC)
    off = KLT_HEAD.size + (KLT_MOTION.size if flags & 2 else 0)
    if len(payload) != off + n * dt.itemsize:
        raise ProtocolError("klt lengths do not match payload")
    motion = None
    if flags & 2:
        m = KLT_MOTION.unpack_from(payload, KLT_HEAD.size)
        motion = {"prev_seq": m[0], "dt": m[1], "rvec": np.array(m[2:5], np.float64), "t": np.array(m[5:8], np.float64), "n_inl": m[8], "rms": m[9]}
    rec = np.frombuffer(payload, dt, n, off)
    xy = np.stack([rec["x"], rec["y"]], 1).astype(np.float32) / KLT_SUBPX
    right = None
    if flags & 1:
        right = np.stack([rec["xr"], rec["yr"]], 1).astype(np.float32) / KLT_SUBPX
        right[rec["xr"] == KLT_NONE] = np.nan
    return seq, rec["id"].astype(np.int64), xy, rec["st"].copy(), rec["q"].copy(), right, ms, motion


def recv_exact(sock, n):
    chunks, got = [], 0
    while got < n:
        b = sock.recv(n - got)
        if not b:
            raise ConnectionError("connection closed")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def read_message(sock):
    magic, msg_type, t_capture, length = HEADER.unpack(recv_exact(sock, HEADER.size))
    if magic != MAGIC:
        raise ProtocolError("bad magic; not a Gabai sender")
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"payload {length} bytes exceeds limit")
    return msg_type, t_capture, recv_exact(sock, length) if length else b""
