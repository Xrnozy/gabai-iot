"""Two free-running USB webcams, paired by capture time instead of by call order.

The webcams cannot be hardware-synchronised. If you grab left then right in one loop, the right frame can be
anywhere from 0 to one frame period old (66 ms at 15 FPS), and a moving scene gets false disparity. Instead each
camera runs in its own thread at a high rate (30 FPS), every frame gets a timestamp the moment it arrives, and
we pair each left frame with the right frame nearest in time. Worst-case offset is then about half a frame
period of the capture rate (about 17 ms at 30 FPS), typically half of that. Use identical manual exposure on both
cameras for the best result, since exposure length shifts the moment actually captured.
"""
import re
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np

from gabai.diag import log as dlog
from gabai.pi.pacing import nearest_offsets_ms, phase_delay_ms

_log = dlog.get("cam")

def choose_pair(left, right, now, last_tl=None, max_wait=0.04):
    """left/right: lists of (t, frame), oldest first. Returns (tl, fl, tr, fr) or None if nothing new yet."""
    if not left or not right:
        return None
    tl, fl = left[-1]
    if last_tl is not None and tl <= last_tl:
        return None
    if right[-1][0] < tl and now - tl < max_wait:
        return None  # a closer right frame is probably about to arrive
    tr, fr = min(right, key=lambda x: abs(x[0] - tl))
    return tl, fl, tr, fr


CTRL_LINE = re.compile(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\(\w+\)\s*:\s*min=(-?\d+)\s+max=(-?\d+)", re.M)


def control_limits(dev):
    """{'exposure_time_absolute': (min, max), 'gain': (min, max), ...} from `v4l2-ctl --list-ctrls`; {} when the tool or the control is missing."""
    try:
        out = subprocess.run(["v4l2-ctl", "-d", dev, "--list-ctrls"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return {m.group(1): (int(m.group(2)), int(m.group(3))) for m in CTRL_LINE.finditer(out)}


def next_settings(mean, exposure, gain, lim, target=105.0, band=12.0, max_exposure_ms=16.0):
    """One step of the motion-safe exposure rule. Exposure is in 0.1 ms units (exposure_time_absolute). When the picture is too dark the exposure
    grows to the cap first and the gain takes over from there; when it is too bright the gain falls first. The exposure never goes above the cap,
    because the smear of a moving scene is (turn rate x focal length x exposure). Returns (exposure, gain, starved): starved = both at their limit and still dark."""
    e_lo, e_hi = lim["exposure"]
    g_lo, g_hi = lim.get("gain", (0, 0))
    cap = min(e_hi, max(e_lo, int(max_exposure_ms * 10)))
    step = max(1, (g_hi - g_lo) // 5)
    starved = False
    if mean < target - band:
        if exposure < cap:
            exposure = min(cap, max(exposure + 1, int(exposure * 1.5)))
        elif gain < g_hi:
            gain = min(g_hi, gain + step)
        else:
            starved = True
    elif mean > target + band:
        if gain > g_lo:
            gain = max(g_lo, gain - step)
        elif exposure > e_lo:
            exposure = max(e_lo, int(exposure / 1.5))
    return min(exposure, cap), gain, starved


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
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # raw MJPEG bytes, no decoding on the Pi
    if not cap.isOpened():
        _log.error("cam.open_failed", dev=dev, requested=[w, h, fps], hint="check the USB cable and that /dev/video* exists (ls /dev/video*); "
                   "another program may hold the camera")
        raise SystemExit(f"cannot open camera {dev}")
    got_w, got_h, got_fps = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)) if code else "?"
    _log.info("cam.open", dev=dev, requested=[w, h, fps], got=[got_w, got_h, got_fps], fourcc=fourcc)
    if (got_w, got_h) != (w, h) or fourcc != "MJPG":
        _log.warning("cam.mode_mismatch", dev=dev, requested=[w, h, "MJPG"], got=[got_w, got_h, fourcc],
                     hint="the driver changed the mode: pairing, calibration size and bandwidth all assume the requested one")
    return cap


def as_jpeg(frame, quality=80):
    """Raw MJPEG buffers come back as a 1-row uint8 array; decoded frames are HxWx3."""
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[0] == 1):
        raw = frame.tobytes()
        if raw[:2] == b"\xff\xd8":
            return raw
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("camera returned an unreadable buffer; run v4l2-ctl --list-formats-ext")
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def wait_until_due(now, last_out, min_gap, cap=0.05):
    """Seconds the sender loop may sleep before the next pair is due (0 when it is due). Capped so a stop is noticed.
    The old loop woke up every millisecond all the time, about 1000 wake-ups a second on a Pi that is on a power bank."""
    return min(max(last_out + min_gap - now, 0.0), cap)


class SyncedCameras:
    fast, applied, brightness_seen, last_dt_ms, last_seq = False, None, {}, None, None      # defaults for an instance built without __init__ (tests)

    def __init__(self, left, right, width=640, height=480, fps=30, exposure="auto", gain=None, brightness=None,
                 quality=80, send_fps=15, max_exposure_ms=16.0, settle_s=10.0):
        """exposure: 'auto' = the camera decides (in a dim room it picks exposures as long as the frame period, and a moving scene smears);
        'fast' = both cameras get the SAME manual exposure, never above max_exposure_ms, raised and lowered with brightness, the gain doing the rest
        (a thread re-applies it every few seconds, because a webcam may reset its controls when the stream opens); a number = fixed manual exposure in 0.1 ms."""
        self.fast, self.max_exp_ms, self.limits, self.cur = exposure == "fast", max_exposure_ms, {}, {}
        self.settle_s, self.locked = settle_s, False
        applied = {}
        for dev in (left, right):
            ctrls = dict(white_balance_automatic=0, focus_automatic_continuous=0, exposure_dynamic_framerate=0)
            if self.fast:
                lim = control_limits(dev)
                e = lim.get("exposure_time_absolute")
                if e is None:
                    raise SystemExit(f"{dev} has no exposure_time_absolute control: use --exposure auto, or a number if the camera accepts one")
                g = lim.get("gain")
                self.limits[dev] = {"exposure": e, **({"gain": g} if g else {})}
                self.cur[dev] = (min(max(int(max_exposure_ms * 10), e[0]), e[1]), (g[0] + 2 * (g[1] - g[0]) // 5) if g else 0)   # start at the cap with 40% gain: dark pictures cost more than noise
                ctrls.update(auto_exposure=1, exposure_time_absolute=self.cur[dev][0], **({"gain": self.cur[dev][1]} if g else {}))
            elif exposure == "auto":
                ctrls["auto_exposure"] = 3
            else:
                ctrls.update(auto_exposure=1, exposure_time_absolute=int(exposure))
            if gain is not None:
                ctrls["gain"] = gain
            if brightness is not None:
                ctrls["brightness"] = brightness
            v4l2(dev, **ctrls)
            applied[dev] = ctrls
            got = subprocess.run(["v4l2-ctl", "-d", dev, "--get-ctrl=auto_exposure,exposure_time_absolute,gain,brightness"],
                                 capture_output=True, text=True).stdout.split()
            print(f"{dev}: {' '.join(got)}", flush=True)
            _log.info("cam.controls", dev=dev, applied=ctrls, readback=" ".join(got),
                      note="readback must match what was asked: a control the camera ignores shows here")
        self.q, self.min_gap = quality, 1.0 / send_fps
        self.bufs = {"L": deque(maxlen=8), "R": deque(maxlen=8)}
        self.idx = {"L": deque(maxlen=16), "R": deque(maxlen=16)}     # (t, camera frame counter): the pictures stay plain bytes, no copy
        self.caps = {"L": open_cam(left, width, height, fps), "R": open_cam(right, width, height, fps)}
        for dev, ctrls in applied.items():           # a webcam may reset its controls when the stream opens: set them again and show what it kept
            v4l2(dev, **ctrls)
            got = subprocess.run(["v4l2-ctl", "-d", dev, "--get-ctrl=auto_exposure,exposure_time_absolute,gain,brightness"], capture_output=True, text=True).stdout.split()
            print(f"{dev} with the stream open: {' '.join(got)}", flush=True)
            _log.info("cam.controls_after_open", dev=dev, readback=" ".join(got))
        self.cam_args, self.applied_ctrls, self.capture_fps = {"L": (left, width, height, fps), "R": (right, width, height, fps)}, applied, fps
        self._reopen, self._reopened = {}, {"L": threading.Event(), "R": threading.Event()}
        self.counts = {"L": deque(maxlen=60), "R": deque(maxlen=60)}
        self.dts = deque(maxlen=60)
        self.last_dt_ms = None          # signed right-minus-left capture offset of the pair read last (sent with the frame)
        self._last_tl, self._last_out, self._stop = None, 0.0, False
        self.cnt = {k: {"frames": 0, "grab_fail": 0, "bad_buf": 0, "last_bytes": None} for k in "LR"}
        self.threads = {k: threading.Thread(target=self._reader, args=(k,), daemon=True) for k in "LR"}
        for t in self.threads.values():
            t.start()
        self.brightness_seen = {}
        self.dev = {"L": left, "R": right}
        if self.fast:
            threading.Thread(target=self._exposure_loop, daemon=True).start()

    def _brightness(self, k):
        if not self.bufs[k]:
            return None
        g = cv2.imdecode(np.frombuffer(self.bufs[k][-1][1], np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        return None if g is None else float(np.median(g))

    def _apply(self, exposure, gain):
        for d in self.dev.values():
            c = dict(auto_exposure=1, exposure_dynamic_framerate=0, exposure_time_absolute=exposure)
            if "gain" in self.limits[d]:
                c["gain"] = gain
            v4l2(d, **c)

    def _exposure_loop(self):
        """Both cameras always get identical settings (identical exposure keeps their capture instants comparable)."""
        d0 = self.dev["L"]
        exposure, gain = self.cur[d0]
        lim, last_apply = self.limits[d0], 0.0
        # the right camera's limits may differ in theory; the tighter of the two is used
        e_lo, e_hi = lim["exposure"]
        for d in self.dev.values():
            e_lo, e_hi = max(e_lo, self.limits[d]["exposure"][0]), min(e_hi, self.limits[d]["exposure"][1])
        lim = dict(lim, exposure=(e_lo, e_hi))
        time.sleep(1.0)
        t0, quiet = time.monotonic(), 0
        while not self._stop:
            b = [self._brightness(k) for k in "LR"]
            b = [x for x in b if x is not None]
            if b:
                mean = sum(b) / len(b)
                self.brightness_seen = {"mean": round(mean, 1), "starved": False}
                if self.locked:                                                          # settled: the values stay fixed, only re-asserted against a camera reset
                    e2, g2, starved = exposure, gain, False
                else:
                    e2, g2, starved = next_settings(mean, exposure, gain, lim, max_exposure_ms=self.max_exp_ms)
                    quiet = quiet + 1 if (e2, g2) == (exposure, gain) else 0
                    if quiet >= 6 or time.monotonic() - t0 > self.settle_s:              # 3 s without a change, or out of time: freeze
                        self.locked = True
                        _log.info("cam.exposure_locked", exposure_ms=exposure / 10, gain=gain, picture_mean=round(mean, 1))
                        print(f"cameras LOCKED: exposure {exposure / 10:g} ms, gain {gain}, picture mean {mean:.0f}. Brightness no longer changes while you move.", flush=True)
                self.brightness_seen["starved"] = starved
                if starved:
                    _log.warning_every("dark", 10.0, "cam.too_dark", mean=round(mean, 1), exposure_ms=exposure / 10, gain=gain,
                                       hint=f"at the {self.max_exp_ms:g} ms exposure cap and the gain limit the picture is still dark: add light, or raise --max-exposure-ms (more smear)")
                if (e2, g2) != (exposure, gain) or time.monotonic() - last_apply > 5.0:       # also a periodic re-assert
                    exposure, gain = e2, g2
                    self.applied = (exposure, gain)
                    self._apply(exposure, gain)
                    last_apply = time.monotonic()
                    _log.info("cam.exposure", exposure_ms=exposure / 10, gain=gain, picture_mean=round(mean, 1))
                    print(f"cameras: exposure {exposure / 10:g} ms, gain {gain}, picture mean {mean:.0f} (target 105)", flush=True)
            time.sleep(0.5)

    def _restart_stream(self, k, wait_ms):
        """Close camera k's stream and open it again after wait_ms: the stream's frames then start wait_ms later inside the 33 ms frame period."""
        dev, w, h, fps = self.cam_args[k]
        self.caps[k].release()
        time.sleep(max(wait_ms, 0.0) / 1000.0)
        for attempt in range(3):
            try:
                cap = open_cam(dev, w, h, fps)
                break
            except SystemExit:
                _log.error("cam.reopen_failed", cam=k, attempt=attempt + 1)
                time.sleep(0.5)
        else:
            raise RuntimeError(f"camera {k} did not reopen")
        self.caps[k] = cap
        if dev in self.applied_ctrls:
            v4l2(dev, **self.applied_ctrls[dev])
        self.bufs[k].clear()
        return cap

    def _offset_samples(self, seconds):
        """Signed right-minus-left offsets (ms) of the pairs formed over the next `seconds`."""
        out, seen, end = [], set(), time.monotonic() + seconds
        while time.monotonic() < end:
            left, right = [t for t, _ in self.bufs["L"]], [t for t, _ in self.bufs["R"]]
            fresh = [t for t in left[:-1] if t not in seen]                    # the newest left frame may still lack its partner
            seen.update(fresh)
            out += nearest_offsets_ms(fresh, right)
            time.sleep(0.05)
        return out

    def align_phase(self, target_ms=4.0, tries=6, settle_s=1.5):
        """Restart the right stream until the constant offset between the two cameras is within target_ms (closed loop: measure, wait (-offset mod period), restart, measure).
        Returns the list of (offset measured, wait applied) and keeps the best result."""
        period = 1000.0 / self.capture_fps
        history, best = [], None
        time.sleep(1.0)
        for attempt in range(tries + 1):
            s = self._offset_samples(settle_s + 0.5)
            if len(s) < 8:
                _log.warning("cam.phase_unmeasured", samples=len(s))
                break
            med = float(sorted(s)[len(s) // 2])
            history.append({"offset_ms": round(med, 2), "spread_ms": round(float(max(s) - min(s)), 2)})
            if best is None or abs(med) < abs(best):
                best = med
            if abs(med) <= target_ms or attempt == tries:
                break
            wait = phase_delay_ms(med, period)
            history[-1]["restart_right_after_ms"] = round(wait, 1)
            _log.info("cam.phase", attempt=attempt + 1, offset_ms=round(med, 2), restart_right_after_ms=round(wait, 1), period_ms=round(period, 1))
            self._reopen["R"] = wait
            while "R" in self._reopen and not self._stop:                       # the reader thread picks the request up
                time.sleep(0.02)
            time.sleep(0.5)
        _log.info("cam.phase_done", history=history, target_ms=target_ms)
        return history

    def _reader(self, k):
        cap, c, fails = self.caps[k], self.cnt[k], 0
        idx = self.__dict__.setdefault("idx", {"L": deque(maxlen=16), "R": deque(maxlen=16)})    # (an instance built without __init__ in a test has none)
        while not self._stop:
            req = getattr(self, "_reopen", {}).pop(k, None)
            if req is not None:
                try:
                    cap = self._restart_stream(k, req)
                except RuntimeError as e:
                    _log.error("cam.reopen_failed", cam=k, error=str(e))
                    return
                continue
            try:
                if not cap.grab():
                    c["grab_fail"] += 1
                    fails += 1
                    _log.warning_every(f"grab{k}", 5.0, "cam.grab_failed", cam=k, consecutive=fails, total=c["grab_fail"],
                                       hint="the camera stopped delivering: USB power/cable, or the Pi is overloaded (see pi_power in health)")
                    time.sleep(0.005)
                    continue
                if fails:
                    _log.info("cam.recovered", cam=k, after_failures=fails)
                    fails = 0
                t = time.monotonic()  # the frame has just arrived; this is our capture timestamp
                ok, fr = cap.retrieve()
                if not ok or fr is None:
                    c["bad_buf"] += 1
                    _log.warning_every(f"retr{k}", 5.0, "cam.retrieve_failed", cam=k, total=c["bad_buf"])
                    continue
                jpg = as_jpeg(fr, self.q)
                idx[k].append((t, c["frames"]))               # before the picture is visible, so a chosen picture always has its counter
                self.bufs[k].append((t, jpg))
                self.counts[k].append(t)
                c["frames"] += 1
                c["last_bytes"] = len(jpg)
                if c["frames"] == 1:
                    _log.info("cam.first_frame", cam=k, bytes=len(jpg), t=round(t, 3))
            except RuntimeError as e:
                c["bad_buf"] += 1
                _log.warning_every(f"buf{k}", 5.0, "cam.bad_buffer", cam=k, total=c["bad_buf"], error=str(e))
            except Exception:
                c["bad_buf"] += 1       # a bug or driver error must not silently kill this camera's thread
                _log.error("cam.reader_error", cam=k, exc=True)
                time.sleep(0.05)

    def read(self):
        """Blocks until a new pair is due (at most send_fps per second). Returns (t, left_jpeg, right_jpeg)."""
        while True:
            now = time.monotonic()
            if now - self._last_out > 2.0 and self._last_out > 0:
                age = {k: (round(now - self.bufs[k][-1][0], 2) if self.bufs[k] else None) for k in "LR"}
                _log.warning_every("nopair", 5.0, "cam.no_pair", seconds_without_pair=round(now - self._last_out, 1), newest_frame_age_s=age,
                                   alive={k: self.threads[k].is_alive() for k in "LR"}, grab_fail={k: self.cnt[k]["grab_fail"] for k in "LR"},
                                   hint="one camera stopped: the age that keeps growing is the dead one")
            wait = wait_until_due(now, self._last_out, self.min_gap)
            if wait > 0:
                time.sleep(wait)
                continue
            got = choose_pair(list(self.bufs["L"]), list(self.bufs["R"]), now, self._last_tl)
            if got:
                tl, fl, tr, fr = got
                self._last_tl, self._last_out = tl, now
                self.dts.append(abs(tl - tr))
                self.last_dt_ms = (tr - tl) * 1000.0
                self.last_seq = tuple(next((n for t_, n in list(self.__dict__.get("idx", {}).get(k, ())) if t_ == t), None) for k, t in (("L", tl), ("R", tr)))     # a gap between two sent pairs = pictures the Pi skipped
                return (tl + tr) / 2, fl, fr
            time.sleep(0.004)   # due, but the best partner frame has not arrived yet (at most about 40 ms)

    def stats(self):
        def fps(c):
            return (len(c) - 1) / (c[-1] - c[0]) if len(c) > 2 and c[-1] > c[0] else 0.0
        d = sorted(self.dts)
        cam = {}
        if self.fast and getattr(self, "applied", None):
            cam = {"exposure_ms": self.applied[0] / 10, "gain": self.applied[1], "picture_mean": self.brightness_seen.get("mean"), "exposure_cap_ms": self.max_exp_ms, "exposure_locked": self.locked}
        return {**cam, "cam_fps_left": round(fps(self.counts["L"]), 1), "cam_fps_right": round(fps(self.counts["R"]), 1),
                "pair_dt_ms": round(1000 * d[len(d) // 2], 1) if d else None,
                "pair_dt_max_ms": round(1000 * d[-1], 1) if d else None,
                "frames_left": self.cnt["L"]["frames"], "frames_right": self.cnt["R"]["frames"],
                "grab_fail_left": self.cnt["L"]["grab_fail"], "grab_fail_right": self.cnt["R"]["grab_fail"],
                "bad_buf_left": self.cnt["L"]["bad_buf"], "bad_buf_right": self.cnt["R"]["bad_buf"],
                "jpeg_bytes_left": self.cnt["L"]["last_bytes"], "jpeg_bytes_right": self.cnt["R"]["last_bytes"],
                "cam_thread_alive": all(t.is_alive() for t in self.threads.values())}

    def close(self):
        self._stop = True
