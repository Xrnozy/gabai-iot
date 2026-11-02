"""Run the depth -> zone check -> alert chain, with all its diagnostics, WITHOUT the hardware.

    py -m gabai.tools.replay_pipeline data/slam/rec_<time> --cam-h 1.2          a recorded sequence through the real depth model
    py -m gabai.tools.replay_pipeline --sim box_2m                              a ground-truth scenario through the same chain
    py -m gabai.tools.replay_pipeline --list                                    the scenarios

Recorded folders are the EuRoC layout written by record_basalt / live_basalt (rectified pictures, IMU, calib.json). Everything the live
viewer logs is logged here with the same event names, so gabai.tools.diag_report reads a replay exactly like a live run.

Scenarios render a floor and boxes from a known camera pose and feed the PERFECT disparity to the chain. They check the decision logic and
the floor model against geometry whose answer is known (the depth model itself is not exercised), and show what a configuration error
looks like in the logs (clear_floor_wrong_height: a floor seen with the wrong camera height produces a false alarm).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from gabai.diag import log as dlog
from gabai.diag.session import PerceptionDiag
from gabai.diag.trace import frame_id
from gabai.live_alerts import AlertController
from gabai.sim import Box, Camera, look_pose, render_disparity
from gabai.speaker import AlertSpeaker
from gabai.zones import analyze

log = dlog.get("replay")
CAM = Camera(f=881.0, B=0.0507, cx=319.5, cy=239.5, size=(640, 480))
TRUE_H, PITCH = 1.2, 35.0
FPS = 14.0
WARMUP = 3


def _box(d, depth=0.4, h=0.4):
    return Box(lo=(d, -0.4, 0.0), hi=(d + depth, 0.4, h))


SCENARIOS = {
    "clear_floor": dict(n=30, expect="none", believed_h=TRUE_H, about="empty floor, correct camera height: no alert expected"),
    "box_2m": dict(n=30, expect="ahead", box=_box(2.0), truth=2.0, believed_h=TRUE_H, about="a 40 cm box 2.0 m ahead: an 'ahead' alert near 2 m"),
    "approach": dict(n=70, expect="ahead", wall=4.0, speed=0.5, believed_h=TRUE_H, about="walking toward a wall at 0.5 m/s: the distance shrinks with the truth"),
    "clear_floor_wrong_height": dict(n=30, expect="none", believed_h=0.6, about="empty floor but the system believes the camera is 0.6 m high (it is 1.2 m): a false alarm is the expected failure signature"),
}


def list_scenarios():
    return {k: v["about"] for k, v in SCENARIOS.items()}


# -- sources ----------------------------------------------------------------------------------------------------------------------
def sim_frames(sc):
    for k in range(sc["n"]):
        x = sc.get("speed", 0.0) * k / FPS
        boxes = []
        truth = None
        if "box" in sc:
            boxes, truth = [sc["box"]], sc["truth"]
        if "wall" in sc:
            boxes, truth = [Box(lo=(sc["wall"], -5, 0.0), hi=(sc["wall"] + 0.2, 5, 3.0))], sc["wall"] - x
        R, t = look_pose(np.array([x, 0.0, TRUE_H]), pitch_deg=PITCH)
        yield {"t": k / FPS, "disp": render_disparity(CAM, R, t, boxes=boxes, noise_px=0.2, seed=k), "R_wc": R, "truth": truth, "imu": []}


def recording_frames(folder, max_frames=None, skip=1):
    if not os.path.exists(os.path.join(folder, "calib.json")):  # checked here, not inside the generator, so the error is immediate
        raise SystemExit(f"recording not found or incomplete: {folder} (needs calib.json and mav0/). Record one with py -m gabai.tools.record_basalt")
    return _recording_frames(folder, max_frames, skip)


def _recording_frames(folder, max_frames, skip):
    c = json.load(open(os.path.join(folder, "calib.json")))["value0"]
    k = c["intrinsics"][0]["intrinsics"]
    p0, p1 = c["T_imu_cam"][0], c["T_imu_cam"][1]
    B = float(np.linalg.norm([p1["px"] - p0["px"], p1["py"] - p0["py"], p1["pz"] - p0["pz"]]))
    cal = {"f": k["fx"], "B": B, "cx": k["cx"], "cy": k["cy"], "size": c["resolution"][0]}
    info = {}
    if os.path.exists(os.path.join(folder, "recording_info.json")):
        info = json.load(open(os.path.join(folder, "recording_info.json")))
    imu_p = os.path.join(folder, "mav0", "imu0", "data.csv")
    imu = np.loadtxt(imu_p, delimiter=",", comments="#") if os.path.exists(imu_p) else np.zeros((0, 7))
    imu_t = imu[:, 0] * 1e-9 if len(imu) else np.zeros(0)
    cam = [l.strip().split(",") for l in open(os.path.join(folder, "mav0", "cam0", "data.csv")) if not l.startswith("#")]
    delay = info.get("cam_delay_ms", 0.0) / 1000.0
    last_t = -1e18
    for i, (ts, name) in enumerate(cam):
        if skip > 1 and i % skip:
            continue
        t = int(ts) * 1e-9
        rows = imu[(imu_t > last_t) & (imu_t <= t)] if len(imu) else []
        last_t = t
        l = cv2.imread(os.path.join(folder, "mav0", "cam0", "data", name), cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(os.path.join(folder, "mav0", "cam1", "data", name), cv2.IMREAD_GRAYSCALE)
        if l is None or r is None:
            log.error("replay.frame_missing", name=name, folder=folder)
            continue
        yield {"t": t, "left": cv2.cvtColor(l, cv2.COLOR_GRAY2BGR), "right": cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "cal": cal, "delay": delay,
               "imu": [(x[0] * 1e-9, *np.degrees(x[1:4]), *(x[4:7] / 9.80665)) for x in rows]}
        if max_frames and (i // max(skip, 1) + 1) >= max_frames:
            return


# -- the chain --------------------------------------------------------------------------------------------------------------------
class Chain:
    """What the live viewer does for one frame, minus the window."""

    def __init__(self, cal, cam_h, imu_cal, R_ic=None, model_name="perfect depth (simulation)", diag=None, sample_every=1):
        from gabai.fusion import SensorFusion
        self.cal, self.cam_h, self.imu_cal, self.R_ic = cal, cam_h, imu_cal, np.eye(3) if R_ic is None else R_ic
        self.fusion = SensorFusion()
        self.ctl = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=cam_h is not None)
        self.diag = diag or PerceptionDiag("replay", cal["f"], cal["B"], cal["cx"], cal["cy"], sample_every=sample_every, system_every=0, image_every=5)
        self.model_name = model_name

    def frame(self, fid, t_clock, disp, trusted, model_ms, left=None, R_wc=None, imu=()):
        c = self.cal
        t0 = time.perf_counter()
        self.diag.set_geometry(c["f"], c["B"], c["cx"], c["cy"])
        tr = self.diag.begin(fid, t_clock, None, {}, left=left)
        self.diag.after_depth(tr, disp, trusted, model_ms, self.model_name, lr_check=trusted is not None)
        source = None
        if R_wc is None and len(imu):
            from gabai.orient import q_to_R
            self.fusion.add_imu(imu)
            self.fusion.step()
            if self.fusion.track.t:
                R_wc = q_to_R(self.fusion.track.at(t_clock)) @ self.R_ic
                source = "IMU orientation (gyro + gravity)"
        elif R_wc is not None:
            source = "simulation ground truth"
        up = R_wc[2, :] if (R_wc is not None and self.cam_h is not None) else None
        res = analyze(np.where(disp > 0.5, disp, np.nan), c["f"], c["B"], self.cam_h, c["cy"], up_cam=up, cx=c["cx"], evidence=True)
        said = self.ctl.on_frame(1000.0 + t_clock, res.alert, "steady", None)
        loop = (time.perf_counter() - t0) * 1000
        self.diag.after_decision(tr, res, self.ctl, said, loop_ms=loop, speech_ms=None, cam_h=self.cam_h, imu_calibrated=self.imu_cal, have_up_vector=up is not None,
                                 up_cam_source=source)
        return res, said


def _setup_logging(log_dir, name):
    return dlog.configure(log_dir=log_dir, run_name=name, level=os.environ.get("GABAI_LOG_LEVEL", "INFO"))


def run_scenario(name, out_dir=None, log_dir=None, quiet=True):
    if name not in SCENARIOS:
        raise SystemExit(f"unknown scenario '{name}'. Choose one of: " + ", ".join(SCENARIOS))
    sc = SCENARIOS[name]
    out_dir = out_dir or os.path.join("data", "replay")
    os.makedirs(out_dir, exist_ok=True)
    run = _setup_logging(log_dir or "logs", f"replay_{name}")
    log.info("replay.start", scenario=name, about=sc["about"], true_cam_h=TRUE_H, believed_cam_h=sc["believed_h"], pitch_down_deg=PITCH)
    chain = Chain({"f": CAM.f, "B": CAM.B, "cx": CAM.cx, "cy": CAM.cy}, sc["believed_h"], imu_cal=True)
    rows = []
    for k, fr in enumerate(sim_frames(sc)):
        fid = frame_id(fr["t"])
        res, said = chain.frame(fid, fr["t"], fr["disp"], np.isfinite(fr["disp"]), 0.0, R_wc=fr["R_wc"])
        spoken_m = res.alert[1] if (res.alert and res.alert[0] in ("ahead", "left", "right")) else None  # a wide object also fills the side bands
        rows.append({"frame": k, "fid": fid, "alert": list(res.alert) if res.alert else None, "spoken": said, "spoken_m": spoken_m, "truth_m": fr["truth"]})
    after = rows[WARMUP:]
    alerts = [r for r in after if r["alert"]]
    s = {"scenario": name, "about": sc["about"], "frames": len(rows), "believed_cam_h": sc["believed_h"], "true_cam_h": TRUE_H, "timeline": rows,
         "alerts_after_warmup": len(alerts), "alert_fraction": round(len(alerts) / max(len(after), 1), 3), "log": run}
    if sc["expect"] == "none":
        s["verdict"] = "PASS" if not alerts else "FALSE_ALARM"
        if alerts:
            s["verdict_detail"] = f"{len(alerts)} alert(s) on a floor with nothing on it; first: {alerts[0]['alert']}"
    else:
        # an obstacle alert is only expected once the object is inside the warning range (3 m of optical depth; on this 35 degree tilt that is about 2.4 m along the ground, so the alert comes late by that much)
        need = [r for r in after if r["truth_m"] is not None and r["truth_m"] <= 2.3]
        hit = [r for r in need if r["spoken_m"] is not None]
        s["truth_horizontal_m"] = sc.get("truth")
        s["required_frames"] = len(need)
        s["spoken_m_median"] = float(np.median([r["spoken_m"] for r in after if r["spoken_m"] is not None])) if any(r["spoken_m"] is not None for r in after) else None
        s["alert_fraction"] = round(len(hit) / max(len(need), 1), 3)
        if s["alert_fraction"] < 0.9:
            s["verdict"], s["verdict_detail"] = "MISSED", f"alert in only {len(hit)} of {len(need)} frames where the object was within 2.3 m"
        elif any(r["spoken_m"] is not None and abs(r["spoken_m"] - r["truth_m"]) > 0.6 for r in after):
            s["verdict"], s["verdict_detail"] = "WRONG_DISTANCE", "spoken distance differs from the truth by more than 0.6 m in some frames"
        else:
            s["verdict"] = "PASS"
    log.info("replay.verdict", scenario=name, verdict=s["verdict"], detail=s.get("verdict_detail"))
    json.dump(s, open(os.path.join(out_dir, "replay_summary.json"), "w", encoding="utf-8"), indent=2, default=float)
    chain.diag.sink.dump_ring(os.path.join(run, "traces_ring.jsonl"))
    dlog.shutdown()
    return s


def run_recording(folder, model="m", cam_h=None, max_frames=None, skip=1, out_dir=None, log_dir=None, imu_sync="data/imu_sync.json"):
    from gabai.slam.euroc import load_imu_extrinsics
    out_dir = out_dir or os.path.join("data", "replay")
    os.makedirs(out_dir, exist_ok=True)
    run = _setup_logging(log_dir or "logs", "replay_rec")
    ext = load_imu_extrinsics(imu_sync)
    from gabai.spike.las2_runner import Las2Depth
    las = Las2Depth(model)
    lr = getattr(las, "disparity_lr_batched", las.disparity_lr)
    log.info("replay.start", folder=folder, model=f"las2-{model}", cam_h=cam_h, imu_extrinsics_calibrated=ext["calibrated"], max_frames=max_frames, skip=skip)
    chain, n, errors, rows = None, 0, 0, []
    for fr in recording_frames(folder, max_frames, skip):
        try:
            if chain is None:
                chain = Chain(fr["cal"], cam_h, ext["calibrated"], ext["R_ic"], model_name=f"las2-{model}")
            disp, trusted, dt = lr(cv2.cvtColor(fr["left"], cv2.COLOR_BGR2RGB), cv2.cvtColor(fr["right"], cv2.COLOR_BGR2RGB))
            disp = np.where(trusted, disp, np.nan)
            fid = int(round(fr["t"] * 1000 + fr["delay"] * 1000))
            res, said = chain.frame(fid, fr["t"], disp, trusted, dt * 1000, left=fr["left"], imu=fr["imu"])
            rows.append({"fid": fid, "alert": list(res.alert) if res.alert else None, "spoken": said, "valid_fraction": round(res.valid_fraction, 3)})
        except Exception:
            errors += 1
            log.error("replay.frame_error", exc=True, frame=n)
        n += 1
    s = {"folder": folder, "frames": n, "errors": errors, "model": f"las2-{model}", "cam_h": cam_h, "imu_extrinsics_calibrated": ext["calibrated"], "timeline": rows, "log": run}
    json.dump(s, open(os.path.join(out_dir, "replay_summary.json"), "w", encoding="utf-8"), indent=2, default=float)
    if chain is not None:
        chain.diag.sink.dump_ring(os.path.join(run, "traces_ring.jsonl"))
    dlog.shutdown()
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="a recording folder (EuRoC layout)")
    ap.add_argument("--sim", help="a ground-truth scenario instead of a recording")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--model", default="m", choices=["s", "m", "h"])
    ap.add_argument("--cam-h", type=float, default=None, help="camera height in m (needed for the floor model and obstacle alerts)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--skip", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--log-dir", default=None)
    a = ap.parse_args()
    if a.list:
        for k, v in list_scenarios().items():
            print(f"{k:28s} {v}")
        return
    if a.sim:
        s = run_scenario(a.sim, log_dir=a.log_dir)
        print(f"{a.sim}: {s['verdict']} {s.get('verdict_detail', '')}  ({s['frames']} frames, {s['alerts_after_warmup']} alerts after warm-up)")
    elif a.folder:
        s = run_recording(a.folder, a.model, a.cam_h, a.max_frames or None, a.skip, log_dir=a.log_dir)
        alerts = sum(1 for r in s["timeline"] if r["alert"])
        print(f"{s['frames']} frames replayed ({s['errors']} errors), {alerts} with an alert")
    else:
        ap.error("give a recording folder, or --sim <scenario>, or --list")
    print(f"log: {s['log']}\n  report: py -m gabai.tools.diag_report {s['log']}")


if __name__ == "__main__":
    main()
