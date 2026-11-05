"""Replay stereo footage, run LAS2-S depth + the front-zone check, and show video | depth | the calculation.

Sources:
  --source sample                 LiteAnyStereo's 5 ZED sample pairs (already rectified), each shown for a few seconds
  --source euroc --path DIR       an unzipped EuRoC sequence (the folder containing cam0/, cam1/, imu0/); rectified here
Keys: q or Esc quit, space pause/resume.
"""
import argparse
import csv
import glob
import os
import time

import cv2
import numpy as np
import yaml

from gabai.diag import log as dlog
from gabai.diag.trace import frame_id
from gabai.spike.las2_runner import Las2Depth, ROOT
from gabai.zones import analyze, BANDS

log = dlog.get("viewer")

SAY = {"ahead": "ahead", "left": "left", "right": "right", "drop_ahead": "drop ahead",
       "cant_see_ahead": "can't see ahead"}


def sample_frames(width, hold_s=4.0, fps=15):
    for p in sorted(glob.glob(os.path.join(ROOT, "assets", "Explorer_*.png"))):
        img = cv2.imread(p)
        W = img.shape[1] // 2
        s = width / W
        size = (width, int(img.shape[0] * s))
        l = cv2.resize(img[:, :W], size, interpolation=cv2.INTER_AREA)
        r = cv2.resize(img[:, W:2 * W], size, interpolation=cv2.INTER_AREA)
        f, B, cy = 1068.63 * s, 0.120134, 632.34 * s
        for i in range(int(hold_s * fps)):
            yield os.path.basename(p), i / fps, l, r, f, B, cy


def load_rectification(left, right, width, cache):
    """Estimate (or load) row-alignment homographies from the footage itself; see autorect.py."""
    from gabai.spike.autorect import estimate_row_alignment
    if os.path.exists(cache):
        z = np.load(cache)
        if int(z["width"]) == width and "M" in z:
            return z["M"], float(z["before"]), float(z["after"])
    pairs = []
    cl, cr = cv2.VideoCapture(left), cv2.VideoCapture(right)
    fps = cl.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cl.get(cv2.CAP_PROP_FRAME_COUNT))
    for k in np.linspace(0.02, 0.95, 12) * total:
        for c in (cl, cr):
            c.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        okl, a = cl.read()
        okr, b = cr.read()
        if okl and okr:
            s = width / a.shape[1]
            size = (width, int(a.shape[0] * s))
            pairs.append((cv2.resize(a, size, interpolation=cv2.INTER_AREA),
                          cv2.resize(b, size, interpolation=cv2.INTER_AREA)))
    est = estimate_row_alignment(pairs)
    np.savez(cache, M=est.M, width=width, before=est.residual_before, after=est.residual_after)
    return est.M, est.residual_before, est.residual_after


def video_frames(left, right, width, start_s, skip, f_frac, baseline, rect=None):
    """Two synced video files. Calibration is unknown, so f and B are ASSUMED; distances are not metric.
    `rect` = 2x3 right-image row-alignment affine from load_rectification."""
    cl, cr = cv2.VideoCapture(left), cv2.VideoCapture(right)
    fps = cl.get(cv2.CAP_PROP_FPS) or 30.0
    for c in (cl, cr):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    k = int(start_s * fps)
    while True:
        okl, a = cl.read()
        okr, b = cr.read()
        if not (okl and okr):
            return
        k += 1
        if k % skip:
            continue
        s = width / a.shape[1]
        size = (width, int(a.shape[0] * s))
        l, r = cv2.resize(a, size, interpolation=cv2.INTER_AREA), cv2.resize(b, size, interpolation=cv2.INTER_AREA)
        if rect is not None:
            r = cv2.warpAffine(r, rect, size, flags=cv2.INTER_LINEAR)  # rows only; disparity untouched
        yield "UNCALIBRATED video (assumed f, B)", k / fps, l, r, f_frac * width, baseline, size[1] / 2


LINK = {}  # filled by live_frames, shown on the panel


def load_mag_cfg(cal_path, sync_path):
    """Compass correction + delay + axis map, or None if the compass has not been synced."""
    from gabai.mag import load_mag_cfg as _load
    return _load(cal_path, sync_path)


def load_imu_cfg(path, delay_ms=None):
    import json
    cfg = {"delay_s": 0.045, "R_ic": np.eye(3), "calibrated": False}
    if os.path.exists(path):
        d = json.load(open(path))
        cfg.update(delay_s=float(d["delay_s"]), R_ic=np.array(d["R_ci"], float).T, calibrated=True)
    if delay_ms is not None:
        cfg["delay_s"] = delay_ms / 1000.0
    return cfg


def live_frames(port, token, f_frac, baseline, gray=False, gamma=1.0, rectifier=None, rotation=(0, 0), imu_cfg=None, mag_cfg=None):
    """Newest stereo pair from the Pi over the network. Row alignment is estimated from the first pairs."""
    from gabai.pc.receiver import LiveReceiver
    rx = LiveReceiver(port, token, rotate_left=rotation[0], rotate_right=rotation[1])
    rx.start()
    print(f"waiting for the Pi on port {rx.port} (token '{token}') ...", flush=True)
    seq, calib, M = 0, [], None
    from gabai.fusion import SensorFusion
    from gabai.orient import q_to_R
    imu_cfg = imu_cfg or {"delay_s": 0.045, "R_ic": np.eye(3), "calibrated": False}
    from gabai.mag import load_accel_cal
    fusion = SensorFusion(mag_cfg=mag_cfg, accel_cal=load_accel_cal())
    t_last_calib = 0.0
    LINK["rx"] = rx
    while True:
        pair = rx.latest()
        LINK["connected"] = rx.stats()["connected"]
        if pair is None or pair.seq == seq:
            time.sleep(0.005)
            yield None  # lets the caller keep its window alive
            continue
        seq = pair.seq
        LINK["seq"] = seq
        st, h = rx.stats(), rx.stats()["health"]
        LINK["rx"] = rx
        imu_batch = rx.drain_imu()
        if LINK.get("watch") is not None:
            LINK["watch"].note_imu(imu_batch)
        fusion.add_imu(imu_batch)               # gyro + accel, stamped on the same Pi clock as the frames
        fusion.add_mag(rx.drain_mag())          # compass, aligned by its own measured delay
        fusion.step()
        if fusion.track.t:
            # the frame was stamped after the motion it shows, so look the orientation up `delay` earlier
            R_wi = q_to_R(fusion.track.at(pair.t_capture - imu_cfg["delay_s"]))
            LINK["R_wc"] = R_wi @ imu_cfg["R_ic"]
            LINK["imu_cal"] = bool(imu_cfg["calibrated"])
            src = "gyro + gravity + COMPASS" if (mag_cfg is not None and fusion.used_compass) else (
                "gyro + gravity only (compass " + ("waiting)" if mag_cfg is not None else "not synced: run sync_sensors)"))
            LINK["imu_sync"] = (f"SYNC: camera {imu_cfg['delay_s'] * 1000:.0f} ms"
                                + (f", compass {mag_cfg['delay_s'] * 1000:.0f} ms" if mag_cfg is not None else "")
                                + f" behind the gyro ({'calibrated' if imu_cfg['calibrated'] else 'DEFAULT'}); orientation from {src}")
        off = h.get("clock_offset_s")
        LINK.update(fps=st["fps"], mbit=st["bytes"] * 8 / 1e6, age=st["frame_age_s"], temp=h.get("temp_c"),
                    throttled=h.get("throttled"), pi_fps=h.get("fps"), rtt=h.get("rtt_s"), peer=st.get("peer"),
                    imu=st.get("imu") or {},
                    pair_dt=h.get("pair_dt_ms"), pair_max=h.get("pair_dt_max_ms"), cam_fps=h.get("cam_fps_left"),
                    mem=h.get("mem_available_mb"), load=h.get("load1"), t_recv=pair.t_recv,
                    lat=(pair.t_recv - (pair.t_capture + off)) if off is not None else None)
        l, r = pair.left, pair.right
        if gray:
            l, r = (cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) for x in (l, r))
        if gamma != 1.0:
            lut = (255 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
            l, r = cv2.LUT(l, lut), cv2.LUT(r, lut)
        LINK["raw"] = (l, r)
        if rectifier is not None:
            if (l.shape[1], l.shape[0]) != rectifier.size:
                raise SystemExit(f"calibration is for {rectifier.size} but the Pi sends {l.shape[1]}x{l.shape[0]}; "
                                 "recalibrate at this size or use --no-calib")
            l, r = rectifier.apply(l, r)
            LINK["calib"] = f"CALIBRATED: B={rectifier.B*100:.1f} cm  f={rectifier.f:.0f} px (metric)"
            LINK["cx"] = rectifier.cx            # the calibrated principal point, used by the floor model
            yield "LIVE from Pi (calibrated)", pair.t_capture, l, r, rectifier.f, rectifier.B, rectifier.cy
            continue
        if M is None:
            if pair.t_recv - t_last_calib > 0.7:
                calib.append((l.copy(), r.copy()))
                t_last_calib = pair.t_recv
            LINK["calib"] = f"calibrating row alignment {len(calib)}/12 (move the rig around the scene)"
            if len(calib) >= 12:
                from gabai.spike.autorect import estimate_row_alignment
                try:
                    est = estimate_row_alignment(calib)
                    M = est.M
                    LINK["calib"] = f"row error {est.residual_before:.1f}px -> {est.residual_after:.1f}px"
                except ValueError as e:
                    print("alignment failed:", e, "- retrying")
                    calib = calib[6:]
            yield "LIVE from Pi (assumed f, B)", pair.t_capture, l, r, f_frac * l.shape[1], baseline, l.shape[0] / 2
            continue
        LINK["raw"] = (l, r)
        r = cv2.warpAffine(r, M, (r.shape[1], r.shape[0]))
        yield "LIVE from Pi (assumed f, B)", pair.t_capture, l, r, f_frac * l.shape[1], baseline, l.shape[0] / 2


def _mat(node):
    return np.array(node["data"], dtype=np.float64).reshape(node["rows"], node["cols"])


def euroc_frames(path, skip=1):
    cams = []
    for c in ("cam0", "cam1"):
        y = yaml.safe_load(open(os.path.join(path, c, "sensor.yaml")))
        fu, fv, cu, cv = y["intrinsics"]
        cams.append((np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]]), np.array(y["distortion_coefficients"]),
                     _mat(y["T_BS"])))
    (K0, D0, T0), (K1, D1, T1) = cams
    T10 = np.linalg.inv(T1) @ T0  # cam0 -> cam1
    size = (752, 480)
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K0, D0, K1, D1, size, T10[:3, :3], T10[:3, 3], alpha=0)
    m0 = cv2.initUndistortRectifyMap(K0, D0, R1, P1, size, cv2.CV_32FC1)
    m1 = cv2.initUndistortRectifyMap(K1, D1, R2, P2, size, cv2.CV_32FC1)
    f, B, cy = float(P1[0, 0]), float(np.linalg.norm(T10[:3, 3])), float(P1[1, 2])
    rows = list(csv.reader(open(os.path.join(path, "cam0", "data.csv"))))[1:]
    t0 = int(rows[0][0])
    for k, (ts, name) in enumerate(rows):
        if k % skip:
            continue
        a = cv2.imread(os.path.join(path, "cam0", "data", name.strip()), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(os.path.join(path, "cam1", "data", name.strip()), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        l = cv2.cvtColor(cv2.remap(a, *m0, cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
        r = cv2.cvtColor(cv2.remap(b, *m1, cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
        yield "EuRoC MH_01_easy", (int(ts) - t0) / 1e9, l, r, f, B, cy


def band_nearest(disp, f, B):
    h, w = disp.shape
    out = {}
    for name, a, b in BANDS:
        d = disp[:, int(a * w):int(b * w)]
        d = d[np.isfinite(d) & (d > 0.5)]
        out[name] = f * B / np.percentile(d, 95) if d.size else None  # 95th pct disparity = nearest 5% of pixels
    return out


PROBE = {}  # last mouse click on the window: {'x':..., 'y':...} in image coordinates


def probe_depth(disp, f, B, x, y):
    """Median disparity in a 9x9 patch around (x, y), with its spread, and the depth Z = f*B/d."""
    h, w = disp.shape
    patch = disp[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5]
    patch = patch[np.isfinite(patch) & (patch > 0.5)]
    if not patch.size:
        return None
    d = float(np.median(patch))
    return d, float(patch.min()), float(patch.max()), f * B / d


def _turn_lines(info, f, B, near):
    imu = LINK.get("imu") or {}
    if not imu:
        return ["TURN: no IMU data (start the sender with the IMU wired, or ignore)"]
    from gabai.motion import turn_flag, sync_error_px, SYNC_TYPICAL_S, SYNC_WORST_S
    peak = imu.get("peak_dps", 0.0)
    flag = turn_flag(peak)
    zs = [z for z in near.values() if z]
    d_near = f * B / min(zs) if zs else None
    typ, worst = sync_error_px(f, peak, SYNC_TYPICAL_S), sync_error_px(f, peak, SYNC_WORST_S)
    pct = f"  = {worst / d_near * 100:.0f}% of the nearest disparity ({d_near:.0f} px)" if d_near else ""
    lines = [f"TURN: {peak:.0f} deg/s ({flag.upper()})  gyro {imu.get('gyro_dps')}   "
             f"sync-induced disparity error: typical {typ:.1f} px, worst {worst:.1f} px{pct}"]
    if flag == "fast":  # spec: say "reduced accuracy" once in degraded mode
        lines.append('ALERT: "reduced accuracy" (turning fast, depth less reliable until you slow down)')
    return lines


def render(left, disp, f, B, cy, info):
    h, w = disp.shape
    ok = np.isfinite(disp) & (disp > 0.5)
    vis = np.where(ok, disp, 0)
    # stretch contrast over the scene's own 2nd..98th percentile so a close-range scene is not one orange blob
    lo, hi = (np.percentile(vis[ok], [2, 98]) if ok.any() else (0.0, 8.0))
    hi = max(float(hi), float(lo) + 4.0)
    dep = cv2.applyColorMap(np.clip((vis - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dep[~ok] = 0
    for name, a, b in BANDS:
        for x in (int(a * w), int(b * w)):
            cv2.line(dep, (x, 0), (x, h), (255, 255, 255), 1)
    cx, cyp = w // 2, h // 2
    patch = disp[cyp - 4:cyp + 5, cx - 4:cx + 5]
    patch = patch[np.isfinite(patch) & (patch > 0.5)]
    dc = float(np.median(patch)) if patch.size else None
    cv2.drawMarker(dep, (cx, cyp), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    probe_line = None
    if PROBE:
        px, py = PROBE['x'] % w, min(PROBE['y'], h - 1)
        for img_ in (dep,):
            cv2.drawMarker(img_, (px, py), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 22, 3)
            cv2.drawMarker(img_, (px, py), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 1)
        left = left.copy()
        cv2.drawMarker(left, (px, py), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        pr = probe_depth(disp, f, B, px, py)
        probe_line = (f"CLICKED ({px},{py}): d={pr[0]:.1f}px (patch {pr[1]:.0f}-{pr[2]:.0f})  ->  Z = {f:.0f} x {B:.3f} / {pr[0]:.1f} = {pr[3]:.2f} m"
                      if pr else f"CLICKED ({px},{py}): no depth there")
    near = band_nearest(disp, f, B)
    for name, a, b in BANDS:
        z = near[name]
        txt = f"{name.upper()} {z:.1f} m" if z else f"{name.upper()} ?"
        cv2.putText(dep, txt, (int(a * w) + 6, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if info["src"].startswith("LIVE") and LINK.get("R_wc") is not None:
        from gabai.spike.cam3d import draw_camera_3d
        left = left.copy()
        inset = draw_camera_3d(LINK["R_wc"], size=min(220, h // 2), title=("camera (axes OK)" if LINK.get("imu_cal") else "camera (axes NOT calibrated)") + (", compass" if "COMPASS" in LINK.get("imu_sync", "") else ""))
        left[6:6 + inset.shape[0], 6:6 + inset.shape[1]] = inset
    top = np.hstack([left, dep])
    lines = [
        f"{info['src']}   t={info['t']:.1f}s   {w}x{h}   f={f:.0f}px  baseline B={B*100:.1f}cm   depth: Z = f x B / d",
        (f"centre pixel: d={dc:.1f}px  ->  Z = {f:.0f} x {B:.3f} / {dc:.1f} = {f*B/dc:.2f} m" if dc
         else "centre pixel: no depth"),
        *([probe_line] if probe_line else ["click anywhere on the picture to read the depth there"]),
        "nearest 5% of pixels:  " + "   ".join(f"{n} {near[n]:.1f} m" if near[n] else f"{n} ?" for n, _, _ in BANDS),
        f"trusted depth in lower-centre: {info['valid']*100:.0f}%   (left-right check {'ON: black = untrusted' if info['lr'] else 'OFF: every pixel gets a value; use --lr-check'})",
        f"{ {'ffs': 'Fast-FoundationStereo', 'defom': 'DEFOM-Stereo', 'prompt': 'PromptStereo'}.get(info['model'], 'LAS2-' + info['model'].upper()) } model {info['ms']:.1f} ms   loop {info['loop']:.1f} ms  ({1000/max(info['loop'], 1):.0f} FPS)   "
        f"frames dropped to stay real-time: {info['dropped']}   {info['rect']}",
        *([("LINK  Pi " + str(LINK.get("peer")) + f"  {LINK['fps']:.1f} FPS in  {LINK.get('pi_fps')} FPS sent  "
            f"{LINK.get('temp')} C  throttled={LINK.get('throttled')}  "
            + (f"capture->PC latency about {LINK['lat']*1000:.0f} ms (+-{(LINK.get('rtt') or 0)*500:.0f})" if LINK.get("lat") is not None else "latency: clock sync pending")
            + (f"   L/R pairing offset median {LINK['pair_dt']} ms (worst {LINK['pair_max']}) at {LINK['cam_fps']} cam FPS" if LINK.get("pair_dt") is not None else "")
            + f"   {LINK.get('calib', '')}")] if info["src"].startswith("LIVE") else []),
        *(_turn_lines(info, f, B, near) if info["src"].startswith("LIVE") else []),
        *([LINK["imu_sync"]] if info["src"].startswith("LIVE") and LINK.get("imu_sync") else []),
        *([LINK["fuse"]] if LINK.get("fuse") else []),
        *([f"SPOKEN (last): {LINK['spoken']}"] if info["src"].startswith("LIVE") and LINK.get("spoken") else []),
        "ALERT: " + (f"\"{SAY[info['alert'][0]]}\"" + (f", about {info['alert'][1]:g} m" if info['alert'][1] is not None else "")
                     if info['alert'] else "none (clear)"),
    ]
    pad = np.zeros((26 * len(lines) + 12, top.shape[1], 3), np.uint8)
    for i, s in enumerate(lines):
        col = (0, 200, 255) if s.startswith("ALERT") and info["alert"] else (230, 230, 230)
        cv2.putText(pad, s, (10, 26 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return np.vstack([top, pad])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["sample", "euroc", "video", "live"], default="sample")
    ap.add_argument("--path", help="EuRoC sequence folder containing cam0/ cam1/")
    ap.add_argument("--width", type=int, default=640, help="sample source width")
    ap.add_argument("--skip", type=int, default=1, help="EuRoC/video: use every Nth frame")
    ap.add_argument("--port", type=int, default=5600, help="live: TCP port to listen on")
    ap.add_argument("--token", default="gabai", help="live: shared token the Pi must send")
    ap.add_argument("--calib", default="data/stereo_calib_new.npz", help="live: calibration file from calibrate_stereo")
    ap.add_argument("--rotate-left", type=int, default=None, choices=[0, 90, 180, 270], help="live: default is whatever the calibration used")
    ap.add_argument("--rotate-right", type=int, default=None, choices=[0, 90, 180, 270])
    ap.add_argument("--imu-sync", default="data/imu_sync.json", help="live: delay and axis map from sync_imu_camera")
    ap.add_argument("--mag-cal", default="data/mag_cal.json", help="live: compass calibration")
    ap.add_argument("--mag-sync", default="data/mag_sync.json", help="live: compass delay and axis map")
    ap.add_argument("--no-mag", action="store_true", help="live: do not use the compass in the orientation")
    ap.add_argument("--imu-delay-ms", type=float, default=None, help="live: override the camera-IMU delay")
    ap.add_argument("--no-calib", action="store_true", help="live: ignore the saved calibration")
    ap.add_argument("--gray", action="store_true", help="live: convert both images to grayscale (default is full colour)")
    ap.add_argument("--gamma", type=float, default=1.0, help="live: brighten dark footage (0.5 = much brighter, 1 = off)")
    ap.add_argument("--left", default="data/left.mp4")
    ap.add_argument("--right", default="data/right.mp4")
    ap.add_argument("--start", type=float, default=0.0, help="video: start time in seconds")
    ap.add_argument("--f-frac", type=float, default=0.6, help="video: assumed focal length as fraction of image width")
    ap.add_argument("--baseline", type=float, default=0.065, help="video: assumed camera baseline in metres")
    ap.add_argument("--cam-h", type=float, default=None, help="camera height in m, enables floor/drop check")
    ap.add_argument("--model", choices=["s", "m", "l", "h", "ffs", "defom", "prompt"], default="m", help="LAS2 size (needs checkpoints/LAS2_<size>.pth), ffs = Fast-FoundationStereo, defom = DEFOM-Stereo, prompt = PromptStereo")
    ap.add_argument("--relay-port", type=int, default=0, help="live: also serve the caregiver relay on this port (8090), sharing this process's Pi link; 0 = off")
    ap.add_argument("--no-audio", action="store_true", help="live: do not speak alerts (they are still logged)")
    ap.add_argument("--telemetry-port", type=int, default=8000, help="live: port of the telemetry web page, 0 = off")
    ap.add_argument("--log", default="data/telemetry_log.csv", help="live: per-frame CSV log for the graphs, '' = off")
    ap.add_argument("--fuse", default="", choices=["", "dav2", "moge-s", "moge-b", "moge-l"], help="fuse stereo with this monocular metric model to cut noise (see docs/fusion-guide.md)")
    ap.add_argument("--mono-check", default=None, metavar="MODEL", help="with --lr-check: a monocular model (dav2) vetoes stereo pixels that disagree grossly with it; never blends its depth in")
    ap.add_argument("--near", action="store_true", help="with --lr-check: add a half-resolution pass for objects closer than 0.23 m (the model stops at 192 px). Costs about 55 ms with H, 20 ms with M")
    ap.add_argument("--smooth", action="store_true", help="live: average steady pixels over time (OFF by default: it ghosted on a hand-held rig, see docs/noise-guide.md)")
    ap.add_argument("--no-smooth", action="store_true", help="live: kept for old commands; smoothing is already off unless --smooth is given")
    ap.add_argument("--smooth-alpha", type=float, default=0.3, help="live: how fast the temporal average follows (1 = no smoothing)")
    ap.add_argument("--lr-check", action="store_true", help="mark untrusted pixels using a left-right consistency check (runs the model twice)")
    ap.add_argument("--no-autorect", action="store_true", help="video: skip the row-alignment fix")
    ap.add_argument("--no-realtime", action="store_true", help="process every frame as fast as possible instead of at video speed")
    ap.add_argument("--out", help="also save an mp4 here")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--log-level", default=None, help="diagnostic log level: DEBUG, INFO or WARNING (default INFO; see docs/debugging.md)")
    ap.add_argument("--log-dir", default=None, help="where the diagnostic log goes (default logs/)")
    ap.add_argument("--diag-save", default=None, help="save intermediate pictures for later inspection: any of depth,frame,disp (default none)")
    ap.add_argument("--no-diag", action="store_true", help="switch the per-frame diagnostics off (the log still records start-up and link events)")
    ap.add_argument("--grid", action="store_true", help="also run the rolling obstacle grid as a CROSS-CHECK of the zone alerts (needs --cam-h and the IMU calibration); it logs disagreements and never changes an alert")
    ap.add_argument("--governor", action="store_true", help="let the thermal/lane governor switch off optional extras (--near, --mono-check) when the GPU is hot or a lane is slow; without it the governor only logs its advice")
    a = ap.parse_args()
    run_dir = dlog.configure(log_dir=a.log_dir, run_name="viewer", level=a.log_level)
    print(f"diagnostic log: {run_dir}", flush=True)
    from gabai.diag.env import calibration_summary, environment_info
    log.info("viewer.start", args={k: v for k, v in vars(a).items() if k != "token"}, environment=environment_info(), calibration=calibration_summary())
    import sys as _sys
    import traceback as _tb

    def _crash(et, ev, tb):
        log.error("viewer.crash", error=f"{et.__name__}: {ev}", exc="".join(_tb.format_exception(et, ev, tb)))
        try:
            diag.sink.dump_ring(os.path.join(run_dir, "traces_ring.jsonl"))
        except Exception:
            pass
        dlog.shutdown()
        _sys.__excepthook__(et, ev, tb)
    _sys.excepthook = _crash
    diag = None

    if a.model == "ffs":
        from gabai.spike.ffs_runner import FfsDepth
        las = FfsDepth()
    elif a.model == "defom":
        from gabai.spike.other_stereo import DefomDepth
        las = DefomDepth()
    elif a.model == "prompt":
        from gabai.spike.other_stereo import PromptStereoDepth
        las = PromptStereoDepth()
    else:
        las = Las2Depth(a.model)
    if a.source == "sample":
        frames = sample_frames(a.width)
    elif a.source == "euroc":
        frames = euroc_frames(a.path, a.skip)
    elif a.source == "live":
        rectifier = None
        if not a.no_calib and os.path.exists(a.calib):
            from gabai.tools.calibrate_stereo import Rectifier
            rectifier = Rectifier(a.calib)
            print(f"using calibration {a.calib}: baseline {rectifier.B*100:.2f} cm, focal {rectifier.f:.0f} px", flush=True)
        elif not a.no_calib:
            print(f"no calibration at {a.calib}; distances are guesses. Run: py -m gabai.tools.calibrate_stereo --square-mm <mm>", flush=True)
        rotation = (a.rotate_left if a.rotate_left is not None else (rectifier.rotation[0] if rectifier else 0),
                    a.rotate_right if a.rotate_right is not None else (rectifier.rotation[1] if rectifier else 0))
        print(f"image rotation left/right: {rotation}", flush=True)
        frames = live_frames(a.port, a.token, a.f_frac, a.baseline, a.gray, a.gamma, rectifier, rotation,
                             load_imu_cfg(a.imu_sync, a.imu_delay_ms),
                             None if a.no_mag else load_mag_cfg(a.mag_cal, a.mag_sync))
    else:
        rect, rect_txt = None, "rectification: off"
        if not a.no_autorect:
            rect, before, after = load_rectification(a.left, a.right, a.width, os.path.join("data", f"autorect_{a.width}.npz"))
            rect_txt = f"auto-rectified: row error {before:.1f}px -> {after:.1f}px"
            print(rect_txt)
        frames = video_frames(a.left, a.right, a.width, a.start, a.skip, a.f_frac, a.baseline, rect)
    WIN = "Gabai depth demo (q quits, space pauses, click = read depth)"
    hooked = False
    writer, n, dropped, paused = None, 0, 0, False
    controller = tel = tlog = None
    relay_ref = {}
    lr_pass = getattr(las, "disparity_lr_batched", las.disparity_lr)   # one batched call when the model supports it
    mono_model = fuser = smoother = None
    veto_model = None
    marks = None
    if a.mono_check:
        from gabai.spike.mono_depth import MonoDepth
        veto_model = MonoDepth(a.mono_check)
    if a.smooth and not a.no_smooth and a.source == "live":
        from gabai.fuse_depth import TemporalFilter
        smoother = TemporalFilter(alpha=a.smooth_alpha)
    if a.fuse:
        from gabai.fuse_depth import DepthFuser
        from gabai.spike.mono_depth import MonoDepth
        mono_model, fuser = MonoDepth(a.fuse), DepthFuser()
    if a.source == "live":
        from gabai.live_alerts import AlertController
        from gabai.speaker import AlertSpeaker
        sp = AlertSpeaker(player=(lambda path: None) if a.no_audio else None)
        controller = AlertController(sp, obstacle_audio=a.cam_h is not None)
        print("spoken alerts:", "OFF (--no-audio)" if a.no_audio else "on",
              "| obstacle speech:", "on" if a.cam_h is not None else "OFF until you give --cam-h (camera height in metres)", flush=True)
        if a.telemetry_port:
            from gabai.pc.telemetry import Telemetry
            tel = Telemetry(port=a.telemetry_port)
            tel.start()
            print(f"telemetry page: http://127.0.0.1:{a.telemetry_port}/  (from a phone: http://<this PC's IP>:{a.telemetry_port}/)", flush=True)
        if a.log:
            from gabai.pc.telemetry import TelemetryLog
            os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
            tlog = TelemetryLog(a.log)
            from gabai.pc.telemetry import MarksLog
            marks = MarksLog(a.log + ".marks.csv")
            print("supervisor keys in the window: F = false alarm (it warned, nothing was there), X = missed (it stayed silent, something was there)", flush=True)
    wall0 = t_first = None
    rt = a.source in ("sample", "video") and not a.no_realtime
    watch = None
    if not a.no_diag:
        from gabai.diag.artifacts import ArtifactSaver
        from gabai.diag.session import PerceptionDiag
        diag = PerceptionDiag(source=a.source, f=0.0, B=0.0, cx=0.0, cy=0.0,
                              saver=ArtifactSaver(os.path.join(run_dir, "artifacts"), what=None if a.diag_save is None else
                                                  [w.strip() for w in a.diag_save.split(",") if w.strip()]))
        if a.source == "live":
            from gabai.diag.watch import LinkWatcher
            watch = LINK["watch"] = LinkWatcher()
    grid = gov = None
    if diag is not None and a.source == "live":
        from gabai.governor import Governor
        gov = Governor()
        if a.grid:
            if a.cam_h is None:
                log.warning("viewer.grid_off", reason="--grid needs --cam-h (the floor height)")
            else:
                from gabai.grid import RollingGrid
                grid = RollingGrid()
    gov_set, last_loop_ms = None, 0.0
    model_name = f"{a.model}" if a.model in ("ffs", "defom", "prompt") else f"las2-{a.model}"
    log.info("viewer.ready", source=a.source, model=model_name, lr_check=a.lr_check, near=a.near, mono_check=a.mono_check, fuse=a.fuse, smooth=a.smooth,
             cam_h=a.cam_h, diag=diag is not None, note="frame id (fid) = Pi monotonic capture time in ms; same number appears in the Pi log")
    from gabai.navguard import NavHeartbeat
    heartbeat = NavHeartbeat().start()      # tells the Rerun pipeline (and other optional tools) on this PC to slow down while navigation runs
    n_dropped_rt = 0
    for item in frames:
        if item is None:  # live: nothing new yet, keep the window responsive
            if watch is not None and LINK.get("rx") is not None:
                s_ = LINK["rx"].stats()
                watch.update(time.monotonic(), s_, s_["health"])
            if controller is not None:
                for ph in controller.on_idle(time.monotonic()):  # a dead stream must still be announced
                    LINK["spoken"] = ph
                    if tel:
                        tel.event(ph)
                if tel:
                    tel.update(connected=bool(LINK.get("connected")))
            if not a.no_window and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            continue
        src, t, l, r, f, B, cy = item
        if rt:  # play at video speed: wait if early, drop the frame if more than 100 ms late
            if wall0 is None:
                wall0, t_first = time.perf_counter(), t
            due = wall0 + (t - t_first)
            now = time.perf_counter()
            if now - due > 0.1:
                dropped += 1
                log.warning_every("rtdrop", 5.0, "viewer.frame_dropped_late", late_ms=round((now - due) * 1000), dropped_total=dropped,
                                  note="video playback fell behind real time; this frame was skipped (not a camera problem)")
                continue
            if due > now:
                time.sleep(due - now)
        t0 = time.perf_counter()
        tr = None
        if diag is not None:
            diag.set_geometry(f, B, LINK.get("cx", l.shape[1] / 2.0), cy)
            live = a.source == "live"
            tr = diag.begin(frame_id(t), t, LINK.get("t_recv") if live else None,
                            {"age_s": LINK.get("age"), "pair_dt_ms": LINK.get("pair_dt"), "turn_dps": (LINK.get("imu") or {}).get("peak_dps"),
                             "latency_s": LINK.get("lat")} if live else {}, left=l)
            if live and LINK.get("rx") is not None:
                s_ = LINK["rx"].stats()
                tr.notes["link"] = {"overwritten_total": s_.get("overwritten"), "decode_errors": s_.get("decode_errors"), "recv_fps": round(s_.get("fps", 0), 1),
                                    "imu_queue": s_.get("imu_queue")}
        trusted_d = None
        rgb_l, rgb_r = cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        fuse_txt = ""
        if a.fuse:  # stereo + monocular metric depth, blended by each one's noise (gabai/fuse_depth.py)
            disp, trusted, ms = lr_pass(rgb_l, rgb_r)
            trusted_d = trusted
            zm = np.clip(mono_model.depth(rgb_l, f_px=f), 0.2, 30.0)
            disp, finfo = fuser.fuse(disp, trusted, f * B / zm)
            fuse_txt = (f"fused with {a.fuse}: mono {mono_model.last_ms:.0f} ms, scale a={finfo['a']:.2f} b={finfo['b']:+.1f}px, "
                        f"{finfo['from_mono'].mean() * 100:.0f}% of pixels filled by mono only")
            LINK["fuse"] = fuse_txt
        elif a.lr_check:  # untrusted pixels (occlusions, textureless, glare) become NaN = unknown, shown black
            skip_extras = a.governor and gov_set is not None and gov_set["tier"] != "normal"      # hot or slow: optional passes are dropped, depth is not
            if a.near and not skip_extras:   # a second half-resolution pass reaches objects closer than 0.23 m (the full-size model stops at 192 px)
                disp, trusted, ms, near_mask = las.disparity_lr_near(rgb_l, rgb_r)
            else:
                disp, trusted, ms = lr_pass(rgb_l, rgb_r)
            trusted_d = trusted
            if veto_model is not None and not skip_extras:   # the monocular map only referees: grossly disagreeing stereo pixels become unknown
                from gabai.fuse_depth import mono_veto
                zm = np.clip(veto_model.depth(rgb_l, f_px=f), 0.2, 30.0)
                trusted, vetoed, _ = mono_veto(disp, trusted, f * B / zm)
                LINK["fuse"] = f"mono check ({a.mono_check}): {veto_model.last_ms:.0f} ms, {vetoed.mean() * 100:.1f}% of pixels vetoed"
                if tr is not None:
                    tr.put("mono_check", model=a.mono_check, vetoed_fraction=round(float(vetoed.mean()), 4), model_ms=round(veto_model.last_ms, 1),
                           note="the monocular model only vetoes pixels; its depth is never blended in")
            disp = np.where(trusted, disp, np.nan)
        else:
            disp, ms = las.disparity(rgb_l, rgb_r)
        if smoother is not None:  # average steady pixels over time; real changes pass through at once
            turn = (LINK.get("imu") or {}).get("peak_dps", 0.0)
            if turn >= 30.0:
                smoother.reset()           # turning: the whole picture moves, averaging would only smear it
            disp = smoother.update(disp, np.isfinite(disp) & (disp > 0.5), alpha=1.0 if turn >= 6.0 else None)
        up_cam = None
        if a.cam_h is not None and LINK.get("imu_cal") and LINK.get("R_wc") is not None:
            up_cam = LINK["R_wc"][2, :]   # world up in camera coordinates: the floor plane follows the real tilt
        if tr is not None:
            diag.after_depth(tr, disp, trusted_d, ms * 1000, model_name, lr_check=a.lr_check or bool(a.fuse),
                             options={"near": a.near, "mono_check": a.mono_check, "fuse": a.fuse, "smooth": smoother is not None})
        res = analyze(np.where(disp > 0.5, disp, np.nan), f, B, a.cam_h, cy, up_cam=up_cam, cx=LINK.get("cx"), evidence=tr is not None)
        said, speech_ms = [], None
        if controller is not None:
            from gabai.motion import turn_flag as _turn_flag
            peak = (LINK.get("imu") or {}).get("peak_dps", 0.0)
            now_pc = time.monotonic()
            said = controller.on_frame(now_pc, res.alert, _turn_flag(peak), LINK.get("lat"))
            # capture -> the moment the clip is handed to the sound card (not the sound itself: see docs/phase-M-manual-test.md)
            from gabai.report.metrics import speech_latency_ms
            speech_ms = speech_latency_ms(time.monotonic(), LINK.get("t_recv"), LINK.get("lat")) if said else None
            if a.relay_port and "relay" not in relay_ref and LINK.get("rx") is not None:     # the Pi link exists once frames flow
                from gabai.relay.embed import start as start_relay
                relay_ref["relay"], _ = start_relay(LINK["rx"], a.relay_port)
            if relay_ref.get("relay") and LINK.get("seq") is not None:
                from gabai.relay.results import build_result
                relay_ref["relay"].ingest_nav(build_result(LINK["seq"], time.time(), res.alert, said))
            for ph in said:
                LINK["spoken"] = ph
                if tel:
                    tel.event(ph)
            if tel or tlog:
                from gabai.pc.telemetry import gpu_temp_c
                near = band_nearest(disp, f, B)
                row = dict(t=round(time.time(), 3), fps_in=LINK.get("fps"), pi_fps=LINK.get("pi_fps"),
                           latency_ms=None if LINK.get("lat") is None else round(LINK["lat"] * 1000),
                           model_ms=round(ms * 1000, 1), loop_ms=round((time.perf_counter() - t0) * 1000, 1),
                           left=near["left"], ahead=near["ahead"], right=near["right"],
                           valid_pct=round(res.valid_fraction * 100), alert=(res.alert or ("",))[0],
                           turn_dps=peak, pair_dt_ms=LINK.get("pair_dt"), pi_temp=LINK.get("temp"), throttled=LINK.get("throttled"),
                           speech_ms=speech_ms, spoken=said[0] if said else "", mem_mb=LINK.get("mem"), pi_load=LINK.get("load"),
                           rtt_ms=None if LINK.get("rtt") is None else round(LINK["rtt"] * 1000, 1), gpu_temp=gpu_temp_c())
                if tlog:
                    tlog.write(**row)
                if tel:
                    tel.update(connected=bool(LINK.get("connected")), zones={k: near[k] for k in ("left", "ahead", "right")},
                               pair_dt=row["pair_dt_ms"], alert=row["alert"], turn_dps=peak, **{k: row[k] for k in
                               ("fps_in", "pi_fps", "latency_ms", "model_ms", "loop_ms", "valid_pct", "pi_temp", "throttled")})
        loop = (time.perf_counter() - t0) * 1000
        grid_info = None
        if grid is not None and tr is not None and LINK.get("R_wc") is not None and LINK.get("imu_cal"):
            R_wc = LINK["R_wc"]
            gst = grid.update(time.monotonic(), np.where(disp > 0.5, disp, np.nan), f, B, LINK.get("cx", l.shape[1] / 2.0), cy, R_wc,
                              (0.0, 0.0, a.cam_h), floor_z=0.0)
            fwd = R_wc @ np.array([0.0, 0.0, 1.0])
            heading = (float(fwd[0]), float(fwd[1]))
            grid_info = {"frames": grid.frames, "update": gst, "corridor": grid.corridor(heading) if np.hypot(*heading) > 0.2 else None,
                         "note": "orientation from the IMU, position fixed at the origin: the grid shows what has been in view, it does not track walking",
                         "image": lambda: grid.to_image(scale=4, camera_xy=(0.0, 0.0), heading_xy=heading)}
        if gov is not None:
            s_gpu = __import__("gabai.diag.sysmon", fromlist=["gpu_read"]).gpu_read()
            gov_set = gov.update(time.monotonic(), gpu_c=s_gpu[0], cpu_c=None, lane_ms={"depth": ms * 1000, "loop": last_loop_ms},
                                 user_moving=((LINK.get("imu") or {}).get("peak_dps", 0.0) > 6.0))
        last_loop_ms = loop
        if tr is not None:
            diag.after_decision(tr, res, controller, said, loop_ms=loop, speech_ms=speech_ms, cam_h=a.cam_h, imu_calibrated=bool(LINK.get("imu_cal")),
                                have_up_vector=up_cam is not None, up_cam_source="IMU orientation (gyro + gravity)" if up_cam is not None else None,
                                grid=grid_info, governor=gov_set)
            if watch is not None and LINK.get("rx") is not None:
                s_ = LINK["rx"].stats()
                watch.update(time.monotonic(), s_, s_["health"])
        img = render(l, disp, f, B, cy, dict(src=src, t=t, ms=ms * 1000, loop=loop,
                                             valid=res.valid_fraction, alert=res.alert, dropped=dropped,
                                             rect=locals().get("rect_txt", ""), model=a.model, lr=a.lr_check))
        if a.out:
            if writer is None:
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (img.shape[1], img.shape[0]))
            writer.write(img)
        if not a.no_window:
            if not hooked:
                cv2.namedWindow(WIN)
                cv2.setMouseCallback(WIN, lambda e, x, y, fl, pm: PROBE.update(x=x, y=y) if e == cv2.EVENT_LBUTTONDOWN else None)
                hooked = True
            cv2.imshow(WIN, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s") and "raw" in LINK:
                os.makedirs("data/live_snap", exist_ok=True)
                tag = int(time.time())
                cv2.imwrite(f"data/live_snap/{tag}_L.png", LINK["raw"][0])
                cv2.imwrite(f"data/live_snap/{tag}_R.png", LINK["raw"][1])
                cv2.imwrite(f"data/live_snap/{tag}_window.png", img)
                print("saved snapshot", tag, flush=True)
            if k == ord("d") and diag is not None:
                p_ = diag.sink.dump_ring(os.path.join(run_dir, f"traces_ring_{frame_id(t)}.jsonl"))
                log.info("viewer.dump", fid=frame_id(t), path=p_, note="last frame traces saved on request")
                print("saved the last frame traces:", p_, flush=True)
            if marks is not None and k in (ord("f"), ord("x")):
                kind_ = "false_alarm" if k == ord("f") else "missed"
                marks.mark(kind_)
                print("marked", "false alarm" if k == ord("f") else "missed obstacle", flush=True)
                if diag is not None:   # keep the evidence behind the operator's judgement
                    os.makedirs(os.path.join(run_dir, "marks"), exist_ok=True)
                    p_ = diag.sink.dump_ring(os.path.join(run_dir, "marks", f"{kind_}_{frame_id(t)}.jsonl"))
                    log.warning("viewer.mark", kind=kind_, fid=frame_id(t), evidence=p_, alert=res.alert,
                                note="operator pressed the key: the file holds the last frame traces so the cause can be traced")
            if k == ord(" "):
                while cv2.waitKey(50) & 0xFF != ord(" "):  # paused: clicks still update the readout
                    cv2.imshow(WIN, render(l, disp, f, B, cy, dict(src=src, t=t, ms=ms * 1000, loop=loop,
                               valid=res.valid_fraction, alert=res.alert, dropped=dropped,
                               rect=locals().get("rect_txt", ""), model=a.model, lr=a.lr_check)))
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
    if writer:
        writer.release()
        print("saved", a.out)
    print("frames:", n)
    heartbeat.stop()
    log.info("viewer.stop", frames=n, dropped_late=dropped)
    if diag is not None:
        diag.sink.dump_ring(os.path.join(run_dir, "traces_ring.jsonl"))
    dlog.shutdown()
    print(f"diagnostic log: {run_dir}\n  summary:  py -m gabai.tools.diag_report {run_dir}", flush=True)


if __name__ == "__main__":
    main()
