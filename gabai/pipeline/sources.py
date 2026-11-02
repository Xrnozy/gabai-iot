"""Frame sources for the Rerun pipeline: two plain video files, or a recorded EuRoC folder. Both give the same Frame.

A Frame carries a frame id (fid = whole milliseconds, the same convention the Pi and the depth viewer log), the time in seconds, the left and
right pictures, the stereo calibration (and whether it is metric), the IMU samples since the previous frame, and what is known about
left/right synchronisation. Nothing here invents a synchronisation result: for plain video the two files are paired by frame index and that
is stated as an assumption; for a recording the pair shares one stamp but the real capture offset between the two cameras was not recorded.

The IMU file of a recording is opened only when with_imu is True, so stereo-only runs never touch it.
"""
import itertools
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from gabai.diag import log as dlog
from gabai.skewcomp import SkewCompensator

G = 9.80665


log = dlog.get("link")


@dataclass
class Frame:
    idx: int
    fid: int
    t: float
    left: object                    # BGR uint8
    right: object
    cal: dict                       # f, B, cx, cy, metric (bool), source (str)
    imu: list = field(default_factory=list)      # (t_s, gyro rad/s (3), accel m/s^2 (3)) with t_s <= this frame's stamp, after the previous frame's
    sync: dict = field(default_factory=dict)
    klt: object = None              # gabai.kltfuse.KltObs: the Pi's tracked points in this left picture (live stream with sender --klt and klt.enabled), else None


def video_source(cfg):
    from gabai.spike.demo_viewer import load_rectification, video_frames
    s, st = cfg.source, cfg.stereo
    rect, before, after = load_rectification(s.left, s.right, st.width, os.path.join("data", f"autorect_{st.width}.npz"))
    for i, (label, t, l, r, f, B, cy) in enumerate(video_frames(s.left, s.right, st.width, s.start, s.skip, st.f_frac, st.baseline, rect)):
        h, w = l.shape[:2]
        yield Frame(i, int(round(t * 1000)), float(t), l, r,
                    {"f": float(f), "B": float(B), "cx": w / 2.0, "cy": float(cy), "metric": False,
                     "source": f"ASSUMED f={f:.0f}px B={B:.3f}m (no calibration for plain video)"},
                    [], {"kind": "assumed", "pair_dt_ms": None, "note": "left/right paired by frame index; not verified", "row_alignment_px_before": float(before),
                         "row_alignment_px_after": float(after)})


def load_recording_calib(folder):
    c = json.load(open(os.path.join(folder, "calib.json")))["value0"]
    k = c["intrinsics"][0]["intrinsics"]
    T = c["T_imu_cam"]
    B = float(np.linalg.norm([T[1]["px"] - T[0]["px"], T[1]["py"] - T[0]["py"], T[1]["pz"] - T[0]["pz"]]))
    return {"f": float(k["fx"]), "B": B, "cx": float(k["cx"]), "cy": float(k["cy"]), "metric": True,
            "source": "calib.json of the recording (stereo calibration)", "size": tuple(c["resolution"][0])}, c


def recording_source(cfg, with_imu):
    folder, s = cfg.source.folder, cfg.source
    cal, c = load_recording_calib(folder)
    info_p = os.path.join(folder, "recording_info.json")
    info = json.load(open(info_p)) if os.path.exists(info_p) else {}
    delay_ms = float(info.get("cam_delay_ms", 0.0))
    cam = [l.strip().split(",") for l in open(os.path.join(folder, "mav0", "cam0", "data.csv")) if not l.startswith("#")]
    imu = imu_t = None
    if with_imu:
        imu = np.loadtxt(os.path.join(folder, "mav0", "imu0", "data.csv"), delimiter=",", comments="#")
        imu_t = imu[:, 0] * 1e-9
    last_t, n_out = -1e18, 0
    for i, (ts, name) in enumerate(cam):
        if s.skip > 1 and i % s.skip:
            continue
        t = int(ts) * 1e-9
        rows = []
        if with_imu:                       # all IMU samples since the previous KEPT frame, so skipping frames never drops inertial data
            sel = (imu_t > last_t) & (imu_t <= t)
            rows = [(float(a), np.asarray(g, float), np.asarray(ac, float)) for a, g, ac in zip(imu_t[sel], imu[sel, 1:4], imu[sel, 4:7])]
        last_t = t
        l = cv2.imread(os.path.join(folder, "mav0", "cam0", "data", name), cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(os.path.join(folder, "mav0", "cam1", "data", name), cv2.IMREAD_GRAYSCALE)
        if l is None or r is None:
            continue
        yield Frame(n_out, int(round(t * 1000 + delay_ms)), t, cv2.cvtColor(l, cv2.COLOR_GRAY2BGR), cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), cal, rows,
                    {"kind": "shared_stamp", "pair_dt_ms": None, "note": "left and right share one stamp in the recording; their true capture offset was not stored",
                     "cam_delay_removed_ms": delay_ms})
        n_out += 1


def flipped_principal_point(c, length):
    """Principal point coordinate after the picture is turned 180 deg (pixel centres at integer coordinates)."""
    return (length - 1) - c


def prepare_live(cfg):
    """Live Pi stream: write the calibration file Basalt reads (the same one live_basalt writes) into a new folder and point source.folder at it.
    Returns the rectifier, the camera delay and the scale applied after rectification (live_width / picture width, at most 1); open_source starts the stream."""
    import time
    from gabai.slam.euroc import basalt_calib, load_accel_calib, load_imu_extrinsics
    from gabai.tools.calibrate_stereo import Rectifier
    s = cfg.source
    rect, ext = Rectifier(s.calib), load_imu_extrinsics(s.imu_sync)
    folder = os.path.abspath(s.folder or os.path.join("data", "slam", time.strftime("live_%Y%m%d_%H%M%S")))
    os.makedirs(folder, exist_ok=True)
    k = min(1.0, s.live_width / rect.size[0])
    size = (int(round(rect.size[0] * k)), int(round(rect.size[1] * k)))
    cx, cy, R_ic = float(np.load(s.calib)["cx"]) * k, rect.cy * k, ext["R_ic"]
    if s.flip180:                                                  # the camera Basalt sees is the old one turned 180 deg about its optical axis
        cx, cy, R_ic = flipped_principal_point(cx, size[0]), flipped_principal_point(cy, size[1]), R_ic @ np.diag([-1.0, -1.0, 1.0])
    json.dump(basalt_calib(rect.f * k, cx, cy, size, rect.B, R_ic, accel_params=load_accel_calib(s.accel_calib)),
              open(os.path.join(folder, "calib.json"), "w"), indent=2)
    s.folder = folder
    return rect, ext["delay_s"], k


def prepare_replay(cfg):
    """The rectifier, the camera delay (none: a replay has no IMU to line up) and the shrink factor for a recorded pair folder."""
    from gabai.tools.calibrate_stereo import Rectifier
    rect = Rectifier(cfg.source.calib)
    return rect, 0.0, min(1.0, cfg.source.live_width / rect.size[0])


def live_turns(src, calib_turns):
    """(left, right) picture turns for the live stream and a warning text ('' when they are what the calibration was made with).
    source.rotate_left / rotate_right of -1 mean 'as stored in the calibration'."""
    rot = (calib_turns[0] if src.rotate_left < 0 else src.rotate_left, calib_turns[1] if src.rotate_right < 0 else src.rotate_right)
    if rot == tuple(calib_turns):
        return rot, ""
    return rot, (f"WARNING: pictures are turned {rot[0]} / {rot[1]} deg (left / right) but the calibration was made with {calib_turns[0]} / {calib_turns[1]}: the "
                 f"rectification only fits if the cameras are mounted that way. Recalibrate with --rotate-left {rot[0]} --rotate-right {rot[1]} if the result looks wrong.")


def calibration_banner(path):
    """What was loaded, in lines a person can compare with what they expect."""
    z = np.load(path)
    g = lambda k, f=str: f(z[k]) if k in z.files else "not recorded"
    T = z["T"].ravel() * 1000
    return [f"stereo calibration: {path}",
            f"  made {g('date')}  layout {g('layout')}  lens model {g('lens_model')}  {int(z['size'][0])}x{int(z['size'][1])}  pictures turned {int(z['rotate_left'])} / {int(z['rotate_right'])} deg",
            f"  cameras: left {g('left_dev')}, right {g('right_dev')}",
            f"  baseline {float(z['baseline']) * 1000:.2f} mm (T = {T[0]:+.1f} {T[1]:+.1f} {T[2]:+.1f} mm), focal {float(z['focal']):.1f} px, reprojection {float(z['rms_stereo']):.3f} px, "
            f"row error {float(z['epipolar_err_px']):.3f} px, {int(z['n_views'])} board views, square {g('square_mm')} mm"]


def check_devices(path, hello):
    """(ok, text): does the sender read the cameras in the order the calibration was made with? Same turns with the other order give 3 px row errors."""
    z = np.load(path)
    want = (str(z["left_dev"]), str(z["right_dev"])) if "left_dev" in z.files else None
    sent = (hello.get("left"), hello.get("right"))
    if want is None:
        return True, "this calibration does not record its camera order, so the sender's order cannot be checked"
    if None in sent:
        return True, f"the sender is older and does not say which cameras it reads; {path} needs --left {want[0]} --right {want[1]}"
    if tuple(sent) == want:
        return True, f"camera order confirmed: the sender reads left {sent[0]}, right {sent[1]}, as the calibration was made"
    return False, (f"WRONG CAMERA ORDER: {path} was made with left {want[0]} / right {want[1]} but the sender reads left {sent[0]} / right {sent[1]}. Rectification would be off "
                   f"by pixels. Restart the sender with --left {want[0]} --right {want[1]}, or pick the calibration made for the other order.")


def power_warning(throttled):
    """Text when the Pi says it is under-voltage or throttled RIGHT NOW (flags set only since boot are not reported here), else ''. A starved Pi drops and delays
    frames, which also makes the capture times the left/right correction relies on unreliable."""
    from gabai.diag.pi_status import active_flags
    now = [f for f in (active_flags(throttled) or []) if f.endswith("_now")]
    if not now:
        return ""
    return (f"PI POWER: {', '.join(now)}. The Pi is starved (power bank or cable): it drops frames, stalls for hundreds of ms, and its frame times jitter, which spoils the "
            "left/right correction. Use a 5 V 3 A supply and a short thick cable; see docs/pi-power-guide.md")


def smear_px(exposure_ms, f_px, turn_dps):
    """How far a point is dragged across the picture during one exposure while the rig turns: f * omega * t."""
    return float(f_px) * np.radians(turn_dps) * exposure_ms / 1000.0


def exposure_line(health, f_px):
    """The Pi's camera exposure and what it means for a moving scene. The sender reports it only with --exposure fast."""
    e = health.get("exposure_ms")
    if e is None:
        return ("camera exposure: not reported (older sender, or --exposure auto). In a dim room auto exposure runs as long as the frame period and a moving scene "
                "smears; start the sender with --exposure fast")
    return (f"camera exposure {e:g} ms, gain {health.get('gain')}, picture mean {health.get('picture_mean')}: a turn of 30 deg/s smears {smear_px(e, f_px, 30):.1f} px, "
            f"90 deg/s {smear_px(e, f_px, 90):.1f} px")


def row_alignment_verdict(left, right):
    """(ok, text) from the rows of the rectified pair itself: matched features of the room should sit on the same row (about 0.1 px with a right calibration).
    This catches what the sender cannot report: an old sender, or the cameras read in the other order."""
    from gabai.spike.autorect import vertical_residual
    px = float(vertical_residual(cv2.resize(left, None, fx=0.5, fy=0.5), cv2.resize(right, None, fx=0.5, fy=0.5)))
    if not np.isfinite(px):
        return True, "row alignment of the first pair could not be measured (too little texture)"
    if px > 0.5:
        return False, (f"row alignment of the first pair is {px:.2f} px (a good calibration gives about 0.1): the calibration does not fit these pictures. "
                       "Most likely the sender reads the cameras in the other order (see the camera order line above), or a camera moved.")
    return True, f"row alignment of the first pair is {px:.2f} px: the calibration fits the pictures"


def layout_needs_swap(rect, left, right, n):
    """(swap, row error as received, row error swapped) for pair number n: True = both pictures must be turned 180 deg and traded, False = as calibrated, None = not decided yet
    (look at the next pair). After 9 pairs without a clear answer the calibrated layout stands. The calibration tools use it too, so the pictures they work on are the live ones."""
    from gabai.spike.autorect import vertical_residual
    half = lambda im: cv2.resize(im, None, fx=0.5, fy=0.5)
    la, ra = rect.apply(left, right)
    a = vertical_residual(half(la), half(ra))
    lb, rb = rect.apply(cv2.rotate(right, cv2.ROTATE_180), cv2.rotate(left, cv2.ROTATE_180))
    b = vertical_residual(half(lb), half(rb))
    if np.isfinite(a) and np.isfinite(b):
        if a > 0.8 and b < 0.5 * a and b < 0.6:
            return True, a, b
        if a <= 0.8 or n >= 9:
            return False, a, b
        return None, a, b
    return (False, a, b) if n >= 9 else (None, a, b)


class PairPrep:
    """A raw pair from the cameras (as the receiver hands it over: turned by the calibration's rotation, not yet rectified) to a pipeline Frame: size check, rectify,
    shrink, optional 180 deg turn, left/right offset correction. The live stream and a recorded one go through the same code, so a recording tests what the live system does."""

    def __init__(self, cfg, rect, k, delay_s, kind="live"):
        self.s, self.rect, self.k, self.delay_s, self.kind = cfg.source, rect, k, delay_s, kind
        s = self.s
        self.skew = SkewCompensator() if s.skew_comp else None
        self.swapped, self.layout_decided = False, not s.auto_layout
        self.size = (int(round(rect.size[0] * k)), int(round(rect.size[1] * k)))
        cx, cy = float(np.load(s.calib)["cx"]) * k, float(rect.cy) * k
        if s.flip180:
            cx, cy = flipped_principal_point(cx, self.size[0]), flipped_principal_point(cy, self.size[1])
            print("source.flip180: both pictures are turned 180 deg, the old right picture is now the left one (geometry stays valid, no recalibration)", flush=True)
        self.cal = {"f": float(rect.f) * k, "B": float(rect.B), "cx": cx, "cy": cy, "metric": True,
                    "source": "stereo calibration " + s.calib + (f", pictures shrunk x{k:.2f}" if k < 1 else "") + (", turned 180 deg" if s.flip180 else ""), "size": self.size}

    def _decide_layout(self, left, right, n):
        """Which camera is which is only a /dev/video number: after a replug or a reboot the two can trade places, and the calibration then fits badly (2-3 px of row
        error) although the device names look right. The rows of the first pairs say which layout fits: matched features must lie on the same row (about 0.1-0.3 px)."""
        swap, a, b = layout_needs_swap(self.rect, left, right, n)
        if swap is None:
            return
        self.swapped, self.layout_decided = swap, True
        if swap:
            print(f"CAMERA LAYOUT: the pictures fit {self.s.calib} only after both are turned 180 deg and swapped (row error {a:.2f} px as received, {b:.2f} px swapped): the two cameras "
                  "came out in the other order, probably the /dev/video numbers moved. Using the swapped layout; the pictures come out upright.", flush=True)
        else:
            print(f"camera layout: as calibrated (row error {a:.2f} px)" if np.isfinite(a) else "camera layout: could not be measured (too little texture in the first pairs), as calibrated", flush=True)

    def frame(self, n, left, right, t_capture, dt_ms, t, rows, health):
        rect, s, size, cal = self.rect, self.s, self.size, self.cal
        got = (left.shape[1], left.shape[0])
        if got != tuple(rect.size) or (right.shape[1], right.shape[0]) != tuple(rect.size):
            raise RuntimeError(f"the pictures are {got[0]}x{got[1]} but {s.calib} was made at {rect.size[0]}x{rect.size[1]}: depth would be wrong. "
                               f"Start the sender with --width {rect.size[0]} --height {rect.size[1]} (or recalibrate at the size you send).")
        if not self.layout_decided:
            self._decide_layout(left, right, n)
        if self.swapped:                                       # the cameras came out in the other order: turn both pictures 180 deg and trade them (what the calibration expects)
            left, right = cv2.rotate(right, cv2.ROTATE_180), cv2.rotate(left, cv2.ROTATE_180)
            dt_ms = None if dt_ms is None else -dt_ms
        l, r = rect.apply(left, right)
        if n == 0:
            ok_row, text = row_alignment_verdict(l, r)
            print(("OK: " if ok_row else "WARNING: ") + text, flush=True)
            if dt_ms is None:
                print("WARNING: no left/right capture offset for this stream, so it is NOT corrected: the Pi sender is old. Copy the current gabai/pi folder to the "
                      "Pi and restart the sender (a moving scene gives 2-4 px of false disparity per 15 ms of offset).", flush=True)
            print(exposure_line(health, cal["f"]), flush=True)
            if power_warning(health.get("throttled")):
                print("WARNING: " + power_warning(health.get("throttled")), flush=True)
        if self.k < 1:
            l, r = cv2.resize(l, size, interpolation=cv2.INTER_AREA), cv2.resize(r, size, interpolation=cv2.INTER_AREA)
        dt_signed = dt_ms
        if s.flip180:                                          # turned 180 deg: rotated right picture becomes the left one (disparity stays positive), so the offset changes sign
            l, r = cv2.rotate(r, cv2.ROTATE_180), cv2.rotate(l, cv2.ROTATE_180)
            dt_signed = None if dt_ms is None else -dt_ms
        comp = {"applied": False, "why": "switched off (source.skew_comp)"}
        if self.skew is not None:
            r, comp = self.skew.apply(l, r, float(t_capture), dt_signed)
        dt = None if dt_signed is None else abs(dt_signed)
        note = ("left and right arrive as one pair; their true capture offset is not known (the sender is older than this PC)" if dt is None else
                f"right camera is {dt_signed:+.1f} ms from the left; " + (f"corrected for the motion ({comp['shift_px']:.1f} px)" if comp["applied"] else f"not corrected: {comp['why']}"))
        if n % 150 == 0 and n and power_warning(health.get("throttled")):
            print("WARNING: " + power_warning(health.get("throttled")), flush=True)
        return Frame(n, int(round(t_capture * 1000)), t, l, r, cal, rows,
                     {"kind": self.kind, "layout": "swapped" if self.swapped else "as calibrated", "pair_dt_ms": dt, "note": note, "cam_delay_removed_ms": self.delay_s * 1000, "skew_comp": comp,
                      "cam": {k_: health.get(k_) for k_ in ("exposure_ms", "gain", "picture_mean", "exposure_cap_ms")}, "power": power_warning(health.get("throttled"))})


def live_source(cfg, rect, delay_s, k=1.0):
    """Frames from the Pi. With sync.enabled (default) every measurement group of gabai/pc/syncbuf.py comes out in order: the pair, the KLT points of that very pair, and ALL
    IMU samples of the interval since the previous group; a pair is skipped only when the pipeline falls more than sync.max_lag_s behind (counted, its IMU kept). With it off:
    the newest pair first, IMU read before it (the older behaviour)."""
    import time
    from gabai.pc.receiver import LiveReceiver, _rotate
    s = cfg.source
    rot, warn = live_turns(s, tuple(rect.rotation))
    if warn:
        print(warn, flush=True)
    rx = LiveReceiver(s.port, s.token, rotate_left=rot[0], rotate_right=rot[1])
    sb = rx.enable_sync(max_groups=cfg.sync.max_groups, max_lag_s=cfg.sync.max_lag_s, keep_lag_s=cfg.sync.keep_lag_s, imu_wait_s=cfg.sync.imu_wait_ms / 1000.0,
                        klt_wait_s=cfg.sync.klt_wait_ms / 1000.0) if cfg.sync.enabled else None
    rx.start()
    for line in calibration_banner(s.calib):
        print(line, flush=True)
    prep = PairPrep(cfg, rect, k, delay_s)
    print(f"waiting for the Pi on port {rx.port} ... hold the rig still for the first seconds", flush=True)

    def newest():
        buf, seq = [], 0
        while True:
            for t, gx, gy, gz, ax, ay, az in rx.drain_imu():
                buf.append((float(t), gx, gy, gz, ax, ay, az))
            pair = rx.latest()
            if pair is None or pair.seq == seq:
                time.sleep(0.003)
                continue
            seq = pair.seq
            t = float(pair.t_capture) - delay_s
            rows, buf = [r for r in buf if r[0] <= t], [r for r in buf if r[0] > t]
            kp = rx.klt_for(pair.t_capture)
            end = time.monotonic() + (0.1 if kp is None and cfg.klt.enabled else 0.0)           # its points follow the pair by a few ms
            while kp is None and time.monotonic() < end:
                time.sleep(0.002)
                kp = rx.klt_for(pair.t_capture)
            yield pair.left, pair.right, pair.t_capture, pair.dt_ms, rows, kp, {}

    def grouped():
        while True:
            g = sb.pop(delay_s)
            if g is None:
                time.sleep(0.003)
                continue
            left, right = cv2.imdecode(np.frombuffer(g.left, np.uint8), cv2.IMREAD_COLOR), cv2.imdecode(np.frombuffer(g.right, np.uint8), cv2.IMREAD_COLOR)
            if left is None or right is None:
                log.warning_every("sync_decode", 5.0, "link.decode_error", seq=g.seq, hint="a pair of the SLAM record could not be decoded: it is skipped, its IMU interval goes out with the next one")
                continue
            hello = rx.stats()
            off = (hello.get("health") or {}).get("clock_offset_s")
            g.info["latency_ms"] = None if off is None else round((g.t_recv - (g.t + off)) * 1000, 1)       # capture on the Pi to release here (offset: PC clock minus Pi clock)
            yield _rotate(left, rx.rot[0]), _rotate(right, rx.rot[1]), g.t, g.dt_ms, g.imu, g.klt, g.info

    n, checked, pmap, pmap_swap = 0, False, None, None
    try:
        for left, right, t_capture, dt_ms, raw_rows, kp, info in (grouped() if sb is not None else newest()):
            if not checked:
                checked = True
                ok_dev, text = check_devices(s.calib, rx.stats().get("hello") or {})
                print(("OK: " if ok_dev else "ERROR: ") + text, flush=True)
                if not ok_dev:
                    raise RuntimeError(text)
            t = float(t_capture) - delay_s
            rows = [(float(r[0]), np.radians(r[1:4]), np.array(r[4:7]) * G) for r in raw_rows]
            if not rows:                                       # no gyro for this frame: the rotation of a held pose and the gyro prior both stop, with no other sign
                health = rx.stats().get("health") or {}
                log.warning_every("imu_silent", 5.0, "link.imu_silent", pi_imu_hz=health.get("imu_hz"), sync=info or None,
                                  hint="no IMU samples before this frame: none = the Pi stopped sending them (BMI160 I2C error / under-voltage: see pi.stats i2c_errors); a rising imu_gap_s = samples lost on the link")
            fr = prep.frame(n, left, right, t_capture, dt_ms, t, rows, rx.stats().get("health") or {})
            fr.sync["buf"] = info
            if kp is not None or cfg.klt.enabled:                  # the points are drawn in Rerun whenever the sender sends them; only the fusion needs klt.enabled
                if kp is not None:
                    if pmap is None or pmap_swap != prep.swapped:                                # the layout is decided during the first pairs
                        from gabai.kltfuse import PointMap
                        hello = rx.stats().get("hello") or {}
                        sent = (int(hello.get("width") or rect.size[0]), int(hello.get("height") or rect.size[1]))
                        pmap, pmap_swap = PointMap(s.calib, sent, rot, prep.k, s.flip180, prep.swapped), prep.swapped
                    fr.klt = pmap.obs(kp)
                elif cfg.klt.enabled:
                    log.warning_every("klt_missing", 10.0, "link.klt_missing", hint="klt.enabled is on but no feature points came with this pair: start the Pi sender with --klt (the pose then uses the PC tracker alone)")
            yield fr
            n += 1
    finally:
        rx.stop()


def replay_source(cfg, rect, delay_s, k=1.0):
    """Frames from a folder written by gabai.tools.record_pairs (left.avi, right.avi, pairs.csv), in order and as fast as the pipeline takes them. The pairs go through
    the same preparation as live ones, with the recorded capture times and left/right offsets."""
    import csv
    s = cfg.source
    folder = s.folder
    meta = json.load(open(os.path.join(folder, "meta.json")))
    for line in calibration_banner(s.calib):
        print(line, flush=True)
    print(f"replaying {folder}: {meta['pairs_recorded']} pairs, recorded {meta['created']} with {meta['calibration']['file']}", flush=True)
    if os.path.basename(meta["calibration"]["file"]) != os.path.basename(s.calib):
        print(f"WARNING: the recording was made with {meta['calibration']['file']} but {s.calib} is used for the replay", flush=True)
    rows_csv = list(csv.DictReader(open(os.path.join(folder, "pairs.csv"))))
    capl, capr = cv2.VideoCapture(os.path.join(folder, "left.avi")), cv2.VideoCapture(os.path.join(folder, "right.avi"))
    prep = PairPrep(cfg, rect, k, delay_s, kind="replay")
    try:
        for n, row in enumerate(rows_csv):
            okl, left = capl.read()
            okr, right = capr.read()
            if not (okl and okr):
                break
            if s.replay_skip > 1 and n % s.replay_skip:
                continue
            dt = None if row["dt_ms"] == "" else float(row["dt_ms"])
            health = {k_: (None if row[k_] == "" else float(row[k_])) for k_ in ("exposure_ms", "gain", "picture_mean")}
            health["exposure_cap_ms"] = None
            yield prep.frame(n, left, right, float(row["t_capture"]), dt, float(row["t_capture"]) - delay_s, [], health)
    finally:
        capl.release(), capr.release()


def open_source(cfg):
    """The configured source, stopped after source.max_frames frames (0 = all)."""
    if cfg.source.kind == "live":
        gen = live_source(cfg, *prepare_live(cfg))      # eager: source.folder must hold calib.json before the pose provider is built
    elif cfg.source.kind == "replay":
        gen = replay_source(cfg, *prepare_replay(cfg))
    else:
        gen = video_source(cfg) if cfg.source.kind == "video" else recording_source(cfg, with_imu=cfg.pipeline.mode == "stereo_imu" or (cfg.tracking.gyro_prior and os.path.exists(os.path.join(cfg.source.folder, "mav0", "imu0", "data.csv"))))
    return itertools.islice(gen, cfg.source.max_frames) if cfg.source.max_frames else gen
