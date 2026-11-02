"""Runs on the Raspberry Pi. Captures both webcams and streams JPEG pairs to the PC over TCP.

The Pi connects OUT to the PC, so it works over the same Wi-Fi (use the PC's LAN IP) or over Tailscale
(use the PC's 100.x.y.z address) with no change. If the link drops it reconnects by itself.

  python3 -m gabai.pi.sender --host 192.168.1.20
  python3 -m gabai.pi.sender --host 100.64.0.5 --token mysecret
  python3 -m gabai.pi.sender --host 192.168.1.20 --test-pattern      (no cameras needed, checks the link)
  python3 -m gabai.pi.sender --host 192.168.1.20 --log-level DEBUG   (one log line per frame and raw sensor values)

Needs only: python3-opencv v4l-utils (apt). Cameras' own MJPEG bytes are forwarded without decoding when
the driver allows it; otherwise frames are re-encoded (slower on a Pi 3B).

Diagnostics: a structured log goes to ~/gabai_logs/pi_<time>/gabai.jsonl (see docs/debugging.md). The frame id `fid` in every
frame-related line is this Pi's monotonic capture time in milliseconds; the PC logs the same number for the same frame.
"""
import argparse
import json
import os
import socket
import subprocess
import threading
import time

import cv2
import numpy as np

from gabai.diag import log as dlog
from gabai.diag.pi_status import active_flags
from gabai.diag.trace import frame_id
from gabai.net import protocol as P
from gabai.pi.cams import SyncedCameras
from gabai.pi.pacing import RateGovernor, wifi_power_save

log = dlog.get("pi")
SEND_SLOW_MS, RTT_HIGH_MS = 200.0, 300.0


def v4l2(dev, **ctrls):
    for k, v in ctrls.items():
        subprocess.run(["v4l2-ctl", "-d", dev, "-c", f"{k}={v}"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_cam(dev, w, h, fps):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # ask for the camera's raw MJPEG bytes
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {dev}")
    return cap


def as_jpeg(frame, quality):
    """Raw MJPEG buffers come back as a 1-row uint8 array; decoded frames are HxWx3."""
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[0] == 1):
        raw = frame.tobytes()
        if raw[:2] == b"\xff\xd8":
            return raw
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)  # raw buffer was not MJPEG
        if frame is None:
            raise RuntimeError("camera returned an unreadable buffer; run v4l2-ctl --list-formats-ext")
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


class Cameras(SyncedCameras):
    """Both webcams in their own threads, paired by capture time (see cams.py)."""

    def __init__(self, a):
        super().__init__(a.left, a.right, a.width, a.height, a.capture_fps, a.exposure, a.gain, a.brightness,
                         a.quality, a.fps, a.max_exposure_ms, a.settle_s)


class TestPattern:
    """Moving shapes, left/right shifted like real disparity. Lets you check the link with no cameras."""

    def __init__(self, a):
        self.w, self.h, self.fps, self.q, self.k = a.width, a.height, a.fps, a.quality, 0
        self.next, self.last_seq = time.monotonic(), None

    def read(self):
        self.next += 1 / self.fps
        time.sleep(max(0, self.next - time.monotonic()))
        self.k += 1
        self.last_seq = (self.k, self.k)
        imgs = []
        for shift in (0, -12):
            im = np.full((self.h, self.w, 3), 40, np.uint8)
            x = (self.k * 6) % self.w
            cv2.rectangle(im, (x + shift, 150), (x + 120 + shift, 330), (0, 180, 255), -1)
            cv2.putText(im, f"TEST PATTERN {self.k}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            imgs.append(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, self.q])[1].tobytes())
        return time.monotonic(), imgs[0], imgs[1]


class SlowCache:
    """Values that cost a process spawn to read (vcgencmd) are refreshed every `max_age` seconds, not every health message."""

    def __init__(self, fn, max_age):
        self.fn, self.max_age, self.t, self.v = fn, max_age, None, None

    def get(self, now):
        if self.t is None or now - self.t >= self.max_age:
            self.v, self.t = self.fn(), now
        return self.v


def low_power_settings(low_power, capture_fps):
    """What --low-power changes. Numbers are the normal defaults otherwise. Lower polling means fewer wake-ups and
    fewer USB transfers, so lower average and peak current on a power bank; the cost is a coarser left/right pairing."""
    if not low_power:
        return dict(capture_fps=capture_fps, imu_hz=200, mag_hz=50, health_every=1.0)
    return dict(capture_fps=min(capture_fps, 20), imu_hz=100, mag_hz=25, health_every=5.0)


def read_pi_slow():
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout.strip()
        except Exception:
            return None
    t = sh(["vcgencmd", "get_throttled"])
    temp = sh(["vcgencmd", "measure_temp"])
    return {"throttled": t.split("=")[1] if t and "=" in t else None,
            "temp_c": float(temp.split("=")[1].rstrip("'C")) if temp and "=" in temp else None}


def pi_health(fps, offset, slow=None):
    slow = slow if slow is not None else read_pi_slow()
    mem = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                mem = int(line.split()[1]) // 1024
    except OSError:
        pass
    return {**slow, "mem_available_mb": mem, "load1": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
            "fps": round(fps, 1), "clock_offset_s": offset}


class SenderStats:
    """Counters for one reporting window of the sender loop."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = time.monotonic()
        self.sent = self.bytes = self.imu_n = self.mag_n = 0
        self.send_ms = []
        self.loop_gap_max = 0.0
        self.klt_ms, self.klt_tracked, self.klt_new, self.klt_lost, self.klt_bytes = [], 0, 0, 0, 0
        self.cam_skipped = 0

    def add_klt(self, r, nbytes):
        self.klt_ms.append(r.ms)
        self.klt_tracked, self.klt_new, self.klt_lost, self.klt_bytes = r.n_tracked, self.klt_new + r.n_new, self.klt_lost + r.n_lost, self.klt_bytes + nbytes

    def add(self, send_ms, nbytes, imu_n, mag_n, gap_s):
        self.sent += 1
        self.bytes += nbytes
        self.imu_n += imu_n
        self.mag_n += mag_n
        self.send_ms.append(send_ms)
        self.loop_gap_max = max(self.loop_gap_max, gap_s)

    def report(self, now):
        w = max(now - self.t0, 1e-6)
        ms = sorted(self.send_ms)

        def pick(q):
            return round(ms[min(len(ms) - 1, int(q * len(ms)))], 1) if ms else None
        out = {"window_s": round(w, 1), "sent": self.sent, "sent_fps": round(self.sent / w, 2), "mbit_s": round(self.bytes * 8 / w / 1e6, 2),
               "send_ms_p50": pick(0.5), "send_ms_p95": pick(0.95), "send_ms_max": round(ms[-1], 1) if ms else None,
               "imu_samples": self.imu_n, "imu_hz": round(self.imu_n / w, 1), "mag_samples": self.mag_n, "mag_hz": round(self.mag_n / w, 1),
               "loop_gap_max_ms": round(self.loop_gap_max * 1000, 1), "cam_skipped": self.cam_skipped}
        if self.klt_ms:                                           # present only with --klt, so the line is unchanged without it
            k = sorted(self.klt_ms)
            out.update(klt_ms_p50=round(k[len(k) // 2], 2), klt_ms_max=round(k[-1], 2), klt_tracked=self.klt_tracked, klt_new=self.klt_new,
                       klt_lost=self.klt_lost, klt_kbit_s=round(self.klt_bytes * 8 / w / 1e3, 1))
        return out


def _end_reason(e):
    """Why a connection attempt or session ended, in words an investigator can act on."""
    if isinstance(e, ConnectionRefusedError):
        return "connection refused: nothing is listening on that host and port (start the PC viewer or receiver first, check the IP and port)"
    if isinstance(e, socket.timeout):
        return "timed out: the PC did not answer (wrong IP, firewall, or a different network)"
    if isinstance(e, (BrokenPipeError, ConnectionResetError)):
        return f"the PC closed the connection ({type(e).__name__})"
    if isinstance(e, ConnectionError):
        return f"connection error: {e}"
    if isinstance(e, OSError):
        return f"network error: {type(e).__name__}: {e}"
    return f"{type(e).__name__}: {e}"


IMU_CHUNK = 400                                   # samples per IMU message (2 s at 200 Hz); a longer backlog goes out as several messages before the frame


def send_imu(sock, t, a, state):
    """All IMU samples not yet sent (oldest first, none skipped), then the compass, BEFORE the frame: when the frame arrives its interval is already complete.
    Returns (imu samples sent, compass samples sent)."""
    imu, mag, imu_n, mag_n = a.imu_obj, a.mag_obj, 0, 0
    if imu is not None:
        ack, state["ack"] = state["ack"], None
        if ack is not None:                                      # a new connection: the PC says where its IMU record ends, the samples the old link lost go out again
            imu.rewind(ack)
            log.info("pi.imu_resume", imu_next=ack)
        while True:
            seq0, rows = imu.drain_seq(IMU_CHUNK)
            msg, last = {"samples": rows, "seq": seq0}, len(rows) < IMU_CHUNK
            if last:
                msg.update(imu.snapshot())
            if last and mag is not None:
                msg["mag"] = mag.drain()
                msg["mag_overflows"] = mag.overflows
                mag_n = len(msg["mag"])
            sock.sendall(P.pack_json(P.IMU, t, msg))
            imu_n += len(rows)
            if last:
                return imu_n, mag_n
    if mag is not None:
        m = mag.drain()
        sock.sendall(P.pack_json(P.IMU, t, {"mag": m, "mag_overflows": mag.overflows}))
        mag_n = len(m)
    return imu_n, mag_n


def run_session(a, source):
    log.info("pi.connecting", host=a.host, port=a.port)
    t_conn = time.monotonic()
    try:
        sock = socket.create_connection((a.host, a.port), timeout=10)
    except OSError as e:
        log.warning("pi.connect_failed", host=a.host, port=a.port, reason=_end_reason(e), errno=getattr(e, "errno", None))
        raise
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # A small send buffer stops frames piling up inside the Pi's TCP stack on a slow Wi-Fi link (that queue,
    # not the network itself, was adding over a second of delay). When the link is slow, capture waits and
    # then grabs the newest frame instead of sending old ones.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, a.sndbuf)
    sock.settimeout(5)
    sock.sendall(P.pack_json(P.HELLO, time.monotonic(), {
        "token": a.token, "name": socket.gethostname(), "width": a.width, "height": a.height, "fps": a.fps, "left": getattr(a, "left", None), "right": getattr(a, "right", None)}))
    print(f"connected to {a.host}:{a.port}", flush=True)
    log.info("pi.connected", host=a.host, port=a.port, local=sock.getsockname()[0],
             connect_ms=round((time.monotonic() - t_conn) * 1000, 1), sndbuf=a.sndbuf)
    state = {"offset": None, "rtt": None, "reader_end": None, "ack": None}

    def reader():  # PONG replies -> clock offset estimate (NTP style)
        try:
            while True:
                t, _, payload = P.read_message(sock)
                if t == P.PONG:
                    d = json.loads(payload)
                    t3 = time.monotonic()
                    state["rtt"] = t3 - d["t1"]
                    state["offset"] = d["t2"] - (d["t1"] + t3) / 2  # receiver clock minus sender clock
                elif t == P.ACK:
                    state["ack"] = int(json.loads(payload)["imu_next"])
        except Exception as e:
            state["reader_end"] = _end_reason(e)
            log.debug("pi.reader_ended", reason=state["reader_end"])

    threading.Thread(target=reader, daemon=True).start()
    n, t_fps, last_health, last_ping = 0, time.monotonic(), 0.0, 0.0
    fps = 0.0
    slow = SlowCache(read_pi_slow, a.health_every)
    stats, h = SenderStats(), {}
    last_report = last_return = t_session = time.monotonic()
    last_flags, total, last_seq = None, 0, None
    governor = None if (getattr(a, "no_adapt_rate", True) or not hasattr(source, "min_gap")) else RateGovernor(a.fps)
    last_rate = float(a.fps)
    klt = getattr(a, "klt_obj", None)
    motion = getattr(a, "motion_obj", None)
    if klt is not None:
        klt.reset()                                            # a new connection must not track from a picture seen before the link went down
    if motion is not None:
        motion.reset()
    try:
        while True:
            got = source.read()
            now = time.monotonic()
            if got is None:
                continue
            gap, last_return = now - last_return, now
            t, lj, rj = got
            imu_n, mag_n = send_imu(sock, t, a, state)
            seq = getattr(source, "last_seq", None)
            if seq is None or None in seq:
                seq = (total, total)                       # a source without camera counters: the sender's own count, which has no gaps
            if last_seq is not None:
                stats.cam_skipped += max(0, seq[0] - last_seq - 1)
            last_seq = seq[0]
            t_s = time.monotonic()
            sock.sendall(P.pack(P.FRAME, t, P.pack_frame(lj, rj, getattr(source, "last_dt_ms", None), seq)))
            send_ms = (time.monotonic() - t_s) * 1000
            if klt is not None:                                # after the pair and the IMU are out, so tracking never delays them; the same t ties points to the pair
                try:
                    t_k = time.perf_counter()
                    r = klt.update(lj, rj)
                    mo = motion.update(r, t, seq[0]) if motion is not None else None
                    r.ms = (time.perf_counter() - t_k) * 1e3         # tracking and, with --klt-motion, the motion estimate
                    kb = P.pack_klt(seq[0], r.ids, r.xy, r.status, r.quality, r.xy_right, r.ms, mo)
                except Exception:
                    klt.reset()
                    if motion is not None:
                        motion.reset()
                    log.error("pi.klt_failed", exc=True, hint="tracking skipped for this pair; the stream is unaffected")
                else:
                    sock.sendall(P.pack(P.KLT, t, kb))
                    stats.add_klt(r, len(kb))
                    if r.ms > 500.0 / a.fps:
                        log.warning_every("kltslow", 10.0, "pi.klt_slow", klt_ms=round(r.ms, 1), stage_ms=list(r.stage_ms),
                                          hint="tracking takes over half a frame period: lower --klt-max or --klt-scale, or leave --klt-fb-px at 0")
            n += 1
            total += 1
            stats.add(send_ms, len(lj) + len(rj), imu_n, mag_n, gap)
            log.debug("pi.frame", fid=frame_id(t), send_ms=round(send_ms, 1), left_bytes=len(lj), right_bytes=len(rj), imu_n=imu_n, mag_n=mag_n)
            if governor is not None:
                new_fps = governor.update(send_ms, now)
                if abs(new_fps - last_rate) > 0.01:
                    log.warning("pi.rate", send_fps=round(new_fps, 1), was=round(last_rate, 1), send_ms=round(send_ms, 1),
                                hint="sends are slow (Wi-Fi, or a Pi starved by its supply): the send rate follows the link so it is not kept saturated")
                    print(f"send rate {last_rate:.1f} -> {new_fps:.1f} pairs/s (the link took {send_ms:.0f} ms for one pair)", flush=True)
                    last_rate = new_fps
                    if hasattr(source, "min_gap"):
                        source.min_gap = 1.0 / new_fps
            if send_ms > SEND_SLOW_MS:
                log.warning_every("slowsend", 5.0, "pi.send_slow", fid=frame_id(t), send_ms=round(send_ms, 1), bytes=len(lj) + len(rj),
                                  sndbuf=a.sndbuf, hint="the network cannot keep up: Wi-Fi signal, PC busy, or the Pi is starved of CPU")
            if now - t_fps >= 2:
                fps, n, t_fps = n / (now - t_fps), 0, now
            if now - last_ping >= 2:
                sock.sendall(P.pack_json(P.PING, now, {"t1": now}))
                last_ping = now
            if now - last_health >= 1:
                h = pi_health(fps, state["offset"], slow.get(now))
                h["rtt_s"] = state["rtt"]
                if hasattr(source, "stats"):
                    h.update(source.stats())  # camera fps and left/right pairing offset
                sock.sendall(P.pack_json(P.HEALTH, now, h))
                last_health = now
                flags = active_flags(h.get("throttled"))
                if flags != last_flags:
                    bad = [f for f in (flags or []) if f.endswith("_now")]
                    (log.warning if bad else log.info)("pi.power", throttled=h.get("throttled"), flags=flags, temp_c=h.get("temp_c"),
                                                       hint="under-voltage or throttling: see docs/pi-power-guide.md" if bad else None)
                    last_flags = flags
            if now - last_report >= a.stats_every:
                r = stats.report(now)
                rtt_ms = None if state["rtt"] is None else round(state["rtt"] * 1000, 1)
                cam = source.stats() if hasattr(source, "stats") else {}
                log.info("pi.stats", **r, clock_offset_s=state["offset"], rtt_ms=rtt_ms, temp_c=h.get("temp_c"), throttled=h.get("throttled"),
                         mem_available_mb=h.get("mem_available_mb"), load1=h.get("load1"),
                         i2c_errors=getattr(a.imu_obj, "errors", None), mag_errors=getattr(a.mag_obj, "errors", None),
                         mag_overflows=getattr(a.mag_obj, "overflows", None), **cam)
                if rtt_ms is not None and rtt_ms > RTT_HIGH_MS:
                    log.warning_every("rtt", 10.0, "pi.rtt_high", rtt_ms=rtt_ms, hint="Wi-Fi latency spikes: the PC-side delay will follow")
                if a.imu_obj is not None and r["window_s"] >= 2 and r["imu_hz"] < 100:
                    log.warning_every("imuhz", 10.0, "pi.imu_rate_low", imu_hz=r["imu_hz"], i2c_errors=getattr(a.imu_obj, "errors", None),
                                      hint="the IMU thread is starved or the I2C bus is erroring; sync and SLAM need about 160-200 Hz")
                if r["loop_gap_max_ms"] > 500:
                    log.warning_every("loopgap", 10.0, "pi.loop_gap", gap_ms=r["loop_gap_max_ms"], hint="no camera pair for a long time: see cam.no_pair")
                if a.imu_obj is not None and a.imu_obj.full:
                    s = a.imu_obj.full[-1]
                    log.debug("pi.imu_raw", t=round(s[0], 4), gyro_dps=list(s[1:4]), accel_g=list(s[4:7]),
                              accel_norm_g=round(sum(v * v for v in s[4:7]) ** 0.5, 4),
                              frame="IMU axes as wired; gyro is bias-corrected, accel is raw")
                if a.mag_obj is not None and a.mag_obj.buf:
                    m = a.mag_obj.buf[-1]
                    log.debug("pi.mag_raw", t=round(m[0], 4), counts=list(m[1:4]), norm=round(sum(v * v for v in m[1:4]) ** 0.5, 1),
                              frame="compass axes as wired, raw counts (not calibrated)")
                stats.reset()
                last_report = now
    except Exception as e:
        log.warning("pi.session_end", reason=_end_reason(e), frames_sent=total, duration_s=round(time.monotonic() - t_session, 1),
                    reader_saw=state["reader_end"])
        raise


def run_forever(a, source, sleep=time.sleep, max_attempts=None):
    delay, attempts = 1, 0
    while True:
        attempts += 1
        try:
            run_session(a, source)
        except KeyboardInterrupt:
            print("stopped")
            log.info("pi.stop", reason="Ctrl+C")
            return
        except (OSError, ConnectionError, P.ProtocolError) as e:
            print(f"link down ({e}); retrying in {delay}s", flush=True)
            log.warning("pi.link_down", reason=_end_reason(e), retry_in_s=delay, attempt=attempts)
            if max_attempts is not None and attempts >= max_attempts:
                return
            sleep(delay)
            delay = min(delay * 2, 10)
        except Exception:
            log.error("pi.fatal", exc=True)       # a bug: log it with the traceback, then let the process fail visibly
            raise
        else:
            delay = 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="PC address: LAN IP or Tailscale 100.x.y.z")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default=os.environ.get("GABAI_TOKEN", "gabai"))
    ap.add_argument("--left", default="/dev/video2", help="swapped vs video0/video2: the cameras are mounted upside down, this order gives upright pictures")
    ap.add_argument("--right", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360, help="640x360 matches data/stereo_calib.npz")
    ap.add_argument("--fps", type=int, default=15, help="pairs sent to the PC per second")
    ap.add_argument("--capture-fps", type=int, default=30, help="camera frame rate; higher = closer left/right pairing")
    ap.add_argument("--exposure", default="fast", help="'fast' (default): both cameras get the same manual exposure, never above --max-exposure-ms, the gain makes up the light; "
                    "'auto': the camera decides (long, smeared exposures in a dim room); or exposure_time_absolute in 0.1 ms units, e.g. 300")
    ap.add_argument("--settle-s", type=float, default=10.0, help="with --exposure fast: seconds the exposure may adapt before it is locked (it locks earlier once it stops changing)")
    ap.add_argument("--no-align-phase", action="store_true", help="do not restart the right camera's stream to bring the constant left/right capture offset near zero")
    ap.add_argument("--phase-target-ms", type=float, default=4.0, help="the left/right offset the phase alignment aims for")
    ap.add_argument("--no-adapt-rate", action="store_true", help="send --fps pairs a second whatever the link does (default: slow down while sends are slow, speed up again when they are quick)")
    ap.add_argument("--max-exposure-ms", type=float, default=16.0, help="with --exposure fast: the longest exposure; the smear of a moving scene is turn rate x focal length x this")
    ap.add_argument("--gain", type=int, default=None, help="camera gain (see v4l2-ctl --list-ctrls for the range)")
    ap.add_argument("--brightness", type=int, default=None, help="camera brightness (see v4l2-ctl --list-ctrls)")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality if frames must be re-encoded")
    ap.add_argument("--sndbuf", type=int, default=65536, help="TCP send buffer in bytes; smaller = lower delay on slow Wi-Fi")
    ap.add_argument("--low-power", action="store_true", help="for a power bank: capture at 20 FPS, IMU 100 Hz, compass 25 Hz, "
                    "Pi health read every 5 s (see docs/pi-power-guide.md; pairing gets slightly coarser)")
    ap.add_argument("--klt", action="store_true", help="track Shi-Tomasi corners with pyramidal Lucas-Kanade on the Pi and send their coordinates (off by default; "
                    "view them with python -m gabai.tools.klt_view). Costs CPU: see docs/pi-klt.md")
    ap.add_argument("--klt-stereo", action="store_true", help="with --klt: also follow each left point into the right picture (one more decode and LK per pair)")
    ap.add_argument("--klt-motion", action="store_true", help="with --klt: also estimate the camera's own motion between two sent pairs ON THE PI from the stereo tracks and send it with the points "
                    "(implies --klt-stereo; needs --klt-calib; one PnP of about 100 points per pair)")
    ap.add_argument("--klt-calib", default=os.environ.get("GABAI_CALIB", "data/stereo_calib_new.npz"), help="the stereo calibration file (copy it to the Pi), for --klt-motion")
    ap.add_argument("--klt-scale", type=float, default=0.5, choices=[1.0, 0.5, 0.25, 0.125], help="tracking size relative to the sent size; 0.5 = 320x180 from 640x360")
    ap.add_argument("--klt-max", type=int, default=200, help="most features followed at once")
    ap.add_argument("--klt-levels", type=int, default=2, help="LK pyramid levels")
    ap.add_argument("--klt-win", type=int, default=15, help="LK window size in tracking pixels")
    ap.add_argument("--klt-min-dist", type=int, default=10, help="smallest gap between features in tracking pixels")
    ap.add_argument("--klt-quality", type=float, default=0.01, help="Shi-Tomasi quality level")
    ap.add_argument("--klt-fb-px", type=float, default=0.0, help="forward-backward check: drop tracks that do not return within this many tracking pixels (0 = off; it doubles the LK cost)")
    ap.add_argument("--klt-max-err", type=float, default=20.0, help="drop a track whose patch changed by more than this many grey levels per pixel")
    ap.add_argument("--klt-min-frac", type=float, default=0.6, help="look for new features only when fewer than this fraction of --klt-max is left")
    ap.add_argument("--no-mag", action="store_true", help="do not read the HMC5883L compass")
    ap.add_argument("--mag-bus", type=int, default=3, help="I2C bus of the compass (3 with the i2c-gpio overlay)")
    ap.add_argument("--no-imu", action="store_true", help="do not read the BMI160 gyro")
    ap.add_argument("--test-pattern", action="store_true", help="send synthetic frames instead of cameras")
    ap.add_argument("--log-level", default=None, help="DEBUG, INFO or WARNING (default INFO). DEBUG adds one line per frame and raw sensor values")
    ap.add_argument("--log-dir", default=os.environ.get("GABAI_LOG_DIR", os.path.expanduser("~/gabai_logs")), help="where the Pi writes its diagnostic log")
    ap.add_argument("--stats-every", type=float, default=5.0, help="seconds between pi.stats log lines")
    a = ap.parse_args()
    run = dlog.configure(log_dir=a.log_dir, run_name="pi", level=a.log_level)
    print(f"diagnostic log: {run}", flush=True)
    log.info("pi.start", args={k: v for k, v in vars(a).items() if k != "token"}, host=socket.gethostname(), cv2=cv2.__version__,
             devices={d: os.path.exists(d) for d in (a.left, a.right)}, note="frame id (fid) = Pi monotonic capture time in ms")
    lp = low_power_settings(a.low_power, a.capture_fps)
    a.capture_fps, a.imu_hz, a.mag_hz, a.health_every = lp["capture_fps"], lp["imu_hz"], lp["mag_hz"], lp["health_every"]
    if a.low_power:
        print(f"low-power mode: capture {a.capture_fps} FPS, IMU {a.imu_hz} Hz, compass {a.mag_hz} Hz, health every {a.health_every:.0f} s", flush=True)
        log.info("pi.low_power", **lp)
    source = TestPattern(a) if a.test_pattern else Cameras(a)
    ps = wifi_power_save()
    if ps == "on":
        log.warning("pi.wifi_power_save", state=ps, hint="Wi-Fi power saving makes frames arrive in bursts with stalls of hundreds of ms: sudo iw dev wlan0 set power_save off")
        print("WARNING: Wi-Fi power saving is ON (stalls of hundreds of ms): run  sudo iw dev wlan0 set power_save off  (see scripts/pi_power_save.sh)", flush=True)
    if not a.test_pattern and not a.no_align_phase:
        print("aligning the two cameras' frame phase (a few seconds) ...", flush=True)
        hist = source.align_phase(a.phase_target_ms)
        if hist:
            print("camera offset (right minus left): " + " -> ".join(f"{h['offset_ms']:+.1f} ms" for h in hist) +
                  ("  (aligned)" if abs(hist[-1]["offset_ms"]) <= a.phase_target_ms else f"  (could not get below {a.phase_target_ms:g} ms, kept the best)"), flush=True)
    a.imu_obj = None
    if not a.no_imu:
        try:
            from gabai.pi.imu import Bmi160
            imu = Bmi160(rate_hz=a.imu_hz)
            for attempt in range(1, 6):
                print(f"IMU found. Keep the rig STILL for 3 s (measuring gyro bias, try {attempt}) ...", flush=True)
                bias, spread = imu.calibrate(3.0)
                print(f"gyro bias {[round(v, 2) for v in bias]} deg/s, movement during calibration {spread:.2f} deg/s"
                      + ("  <- the rig moved; trying again" if spread > 1.2 and attempt < 5 else ""), flush=True)
                if spread <= 1.2:                                  # a still rig scatters well under 1 deg/s; more than that means it was touched
                    break
            (log.warning if spread > 1.2 else log.info)(
                "pi.imu_ready", rate_hz=a.imu_hz, gyro_bias_dps=[round(v, 3) for v in bias], movement_dps=round(spread, 3), rig_moved=spread > 1.2,
                hint="a moving rig during calibration leaves a wrong gyro bias: restart the sender" if spread > 1.2 else None)
            imu.start()
            a.imu_obj = imu
        except Exception as e:  # no smbus2, wrong wiring, I2C off: carry on without turn flagging
            print(f"IMU not used ({e}); continuing without turn flagging", flush=True)
            log.error("pi.imu_init_failed", error=f"{type(e).__name__}: {e}", exc=True,
                      hint="check wiring (SDA/SCL, 3.3 V), `sudo i2cdetect -y 1` should show 0x68, and that I2C is enabled")
    a.mag_obj = None
    if not a.no_mag:
        try:
            from gabai.pi.mag import Hmc5883l
            mag = Hmc5883l(bus=a.mag_bus, rate_hz=a.mag_hz)
            mag.start()
            a.mag_obj = mag
            print(f"compass found (bus {a.mag_bus}, 0x1E)", flush=True)
            log.info("pi.mag_ready", bus=a.mag_bus, rate_hz=a.mag_hz)
        except Exception as e:  # overlay missing, wrong wiring, or a QMC5883L clone
            print(f"compass not used ({e})", flush=True)
            log.error("pi.mag_init_failed", error=f"{type(e).__name__}: {e}", exc=True,
                      hint="check the i2c-gpio overlay (bus 3), wiring, and that the chip is an HMC5883L (id 'H43') and not a QMC5883L clone")
    a.klt_obj, a.motion_obj = None, None
    if a.klt:
        from gabai.pi.klt import KltTracker
        a.klt_stereo = a.klt_stereo or a.klt_motion
        cv2.setNumThreads(1)                                   # one core for tracking at most: the capture threads keep theirs
        a.klt_obj = KltTracker(a.width, a.height, a.klt_scale, a.klt_max, a.klt_levels, a.klt_win, a.klt_min_dist, a.klt_quality,
                               a.klt_fb_px, a.klt_min_frac, max_err=a.klt_max_err, stereo=a.klt_stereo)
        k = a.klt_obj
        if a.klt_motion:
            from gabai.pi.klt import MotionEstimator
            if not os.path.exists(a.klt_calib):
                raise SystemExit(f"--klt-motion needs the stereo calibration: {a.klt_calib} not found (copy data/stereo_calib_new.npz to the Pi, or pass --klt-calib)")
            a.motion_obj = MotionEstimator(a.klt_calib, (a.width, a.height))
        mode = (f"KLT ON: Shi-Tomasi + pyramidal LK at {k.tw}x{k.th}, up to {k.max} features, {a.klt_levels} levels, window {a.klt_win}, "
                f"forward-backward {'off' if not a.klt_fb_px else f'{a.klt_fb_px:g} px'}, right picture {'tracked' if a.klt_stereo else 'not tracked'}, "
                f"own motion estimate {'ON (calibration ' + a.klt_calib + ')' if a.klt_motion else 'off'}")
    else:
        mode = "KLT off (video, IMU and compass only)"
    print(f"tracking mode: {mode}", flush=True)
    log.info("pi.tracking_mode", klt=a.klt, mode=mode)
    run_forever(a, source)


if __name__ == "__main__":
    main()
