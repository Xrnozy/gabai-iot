"""Start-up self-check: is this machine, and are these files, in a state where Gabai can give trustworthy results?

    py -m gabai.tools.selfcheck                 quick (a few seconds)
    py -m gabai.tools.selfcheck --deep          also loads the depth model and runs it once on a synthetic pair
    py -m gabai.tools.selfcheck --pi 192.168.1.63    also checks the Pi answers on the network

Each line is PASS, WARN (works, but results may be poor), FAIL (will not work or will give wrong answers) or SKIP (could not be
checked). Every WARN and FAIL says what it means and the command that fixes it. The result is also saved as JSON in the log folder.
Exit code 1 if anything FAILED.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Result:
    name: str
    status: str            # PASS WARN FAIL SKIP
    detail: str = ""
    fix: str = ""


def summarize(results):
    s = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        s[r.status] += 1
    return s


def exit_code(results):
    return 1 if any(r.status == "FAIL" for r in results) else 0


# -- software and hardware on this PC ---------------------------------------------------------------------------------------------
def check_libraries():
    from gabai.diag.env import environment_info
    info, out = environment_info(), []
    for lib, need, why in (("numpy", True, ""), ("cv2", True, ""), ("scipy", True, "orientation and calibration maths"), ("torch", True, "the depth model"),
                           ("fastapi", False, "the telemetry web page"), ("uvicorn", False, "the telemetry web page"),
                           ("psutil", False, "CPU and memory readings in the diagnostics"), ("pandas", False, "reports"), ("matplotlib", False, "report graphs")):
        v = info["libs"].get(lib, "MISSING")
        if str(v).startswith("MISSING"):
            out.append(Result(f"library {lib}", "FAIL" if need else "WARN", f"{lib} is not installed ({why or 'required'})", f"py -m pip install {'opencv-python' if lib == 'cv2' else lib}"))
        else:
            out.append(Result(f"library {lib}", "PASS", str(v)))
    return out


def check_gpu():
    from gabai.diag.env import environment_info
    info = environment_info()
    if not info.get("cuda_available"):
        return Result("GPU (CUDA)", "FAIL", "torch cannot see a CUDA GPU: depth would run on the CPU, far too slowly for walking",
                      "install a CUDA build of torch for this GPU and check `nvidia-smi` works")
    g = info.get("gpu", {})
    return Result("GPU (CUDA)", "PASS", f"{g.get('name')} {g.get('vram_gb')} GB, compute {g.get('capability')}, torch CUDA {g.get('torch_cuda')}")


def check_models(root="."):
    base = os.path.join(root, "third_party", "LiteAnyStereo", "checkpoints")
    have = {s: os.path.exists(os.path.join(base, f"LAS2_{s}.pth")) for s in "SMH"}
    if not any(have.values()):
        return Result("depth model weights", "FAIL", f"no LAS2 weights in {base}", "put LAS2_S.pth / LAS2_M.pth / LAS2_H.pth there (see docs/spike-report.md)")
    missing = [s for s, v in have.items() if not v]
    return Result("depth model weights", "PASS" if not missing else "WARN", "found " + ", ".join(f"LAS2-{s}" for s, v in have.items() if v)
                  + (f"; missing {', '.join(missing)}" if missing else ""), "" if not missing else "only the missing sizes are unavailable")


def check_optional_models(root="."):
    out = []
    y = os.environ.get("GABAI_YOLO") or os.path.join(root, "models", "yolo11n-seg.pt")
    out.append(Result("people detector weights", "PASS" if os.path.exists(y) else "WARN",
                      y if os.path.exists(y) else "yolo11n-seg.pt not found: people are not masked out of the grid and no detection evidence is recorded",
                      "" if os.path.exists(y) else "download yolo11n-seg.pt from github.com/ultralytics/assets/releases into models/ (Gabai never downloads models itself)"))
    return out


def check_calibration(root="."):
    from gabai.diag.env import calibration_summary
    c, out = calibration_summary(root), []
    s = c["stereo_calib"]
    if not s.get("exists"):
        out.append(Result("stereo calibration", "FAIL", "data/stereo_calib.npz is missing: every distance would be a guess", "py -m gabai.tools.calibrate_stereo --square-mm <size of one checkerboard square>"))
    elif s.get("swapped"):
        out.append(Result("stereo calibration", "FAIL", "the cameras were swapped when this was calibrated (left and right are exchanged)", "recalibrate: py -m gabai.tools.calibrate_stereo ..."))
    elif s.get("warnings"):
        out.append(Result("stereo calibration", "WARN", "; ".join(s["warnings"]), "recalibrate with more varied, flatter board views"))
    elif s.get("size") and list(s["size"]) != [640, 480]:
        out.append(Result("stereo calibration", "WARN", f"calibrated at {s['size']}, but the sender sends 640x480 by default", "recalibrate at the size you stream"))
    else:
        out.append(Result("stereo calibration", "PASS", f"baseline {s.get('baseline', 0) * 100:.2f} cm, focal {s.get('focal', 0):.0f} px, reprojection {s.get('rms_stereo', 0):.2f} px, "
                                                   f"{s.get('age_days')} days old"))
    i = c["imu_sync"]
    if not i.get("exists"):
        out.append(Result("IMU-camera rotation", "WARN", "data/imu_sync.json is missing: the camera-IMU rotation is a guess, tilt-aware floor and Basalt will be wrong (Basalt diverges)",
                          "py -m gabai.tools.sync_imu_camera, then py -m gabai.tools.check_extrinsics <recording> --write"))
    elif i.get("rotation_orthonormal_error", 0) > 1e-3 or abs(i.get("rotation_det", 1.0) - 1.0) > 1e-3:
        out.append(Result("IMU-camera rotation", "FAIL", "the saved rotation is not a proper rotation matrix", "redo sync_imu_camera"))
    else:
        note = "; an exact axis swap (no fine fit): Basalt wants the fitted rotation, run check_extrinsics --write" if i.get("rotation_is_exact_axis_swap") else ""
        out.append(Result("IMU-camera rotation", "PASS" if not note else "WARN", f"delay {1000 * (i.get('delay_s') or 0):.0f} ms, quality {i.get('quality')}{note}", "" if not note else "py -m gabai.tools.check_extrinsics <recording> --write"))
    a = c["accel_calib"]
    out.append(Result("accelerometer calibration", "PASS", f"{a.get('n_orientations')} orientations, residual {a.get('rms_after')} m/s^2") if a.get("exists") else
               Result("accelerometer calibration", "WARN", "data/accel_calib.json is missing: the accelerometer reads gravity 2-4% off and SLAM drifts metres",
                      "py -m gabai.tools.calibrate_accel (hold the rig still on each of its six sides)"))
    for key, name, why in (("mag_cal", "compass calibration", "compass heading"), ("mag_sync", "compass sync", "compass heading")):
        out.append(Result(name, "PASS" if c[key].get("exists") else "SKIP", "present" if c[key].get("exists") else f"not set up (only needed for {why})", ""))
    return out


def check_alert_audio(root="."):
    d = os.path.join(root, "gabai", "audio", "alerts")
    n = len([f for f in os.listdir(d) if f.endswith(".wav")]) if os.path.isdir(d) else 0
    if n < 23:
        return Result("spoken alert clips", "FAIL", f"{n} of 23 clips found in {d}: missing alerts would be silent", "py -m gabai.tools.make_alert_audio")
    return Result("spoken alert clips", "PASS", "23 of 23 clips present")


def check_port(port, what, exclusive=True):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if exclusive and os.name == "nt":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        s.bind(("0.0.0.0", port))
        return Result(f"port {port} ({what})", "PASS", "free")
    except OSError as e:
        return Result(f"port {port} ({what})", "FAIL", f"in use ({e}): another Gabai window or program holds it", "close the other viewer/receiver/recorder, or `netstat -ano | findstr :%d`" % port)
    finally:
        s.close()


def check_log_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".probe")
        open(probe, "w").write("x")
        os.remove(probe)
        free = shutil.disk_usage(path).free / 1e9
    except (OSError, ValueError) as e:
        return Result("log folder", "FAIL", f"cannot write to {path!r}: {e}", "set GABAI_LOG_DIR to a writable folder")
    if free < 2.0:
        return Result("log folder", "WARN", f"{path}: only {free:.1f} GB free", "free disk space; logs rotate but recordings and artifacts do not")
    return Result("log folder", "PASS", f"{path} writable, {free:.0f} GB free")


def check_basalt():
    from gabai.slam.vit import BASALT_DLL
    if not os.path.exists(BASALT_DLL):
        return Result("Basalt library", "WARN", f"{BASALT_DLL} not found: live SLAM is unavailable (depth and alerts still work)", "build Basalt: see docs/slam-record-guide.md")
    try:
        from gabai.slam.vit import load_dll
        import ctypes as C
        dll = load_dll()
        v = [C.c_uint32() for _ in range(3)]
        ok = dll.vit_api_get_version(*[C.byref(x) for x in v]) == 0
        return Result("Basalt library", "PASS" if ok and v[0].value == 2 else "FAIL", f"VIT {v[0].value}.{v[1].value}.{v[2].value}, built {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(BASALT_DLL)))}")
    except Exception as e:
        return Result("Basalt library", "FAIL", f"cannot load: {type(e).__name__}: {e}", "rebuild Basalt or check D:\\slam\\vcpkg\\installed\\x64-windows\\bin is on the DLL path")


def check_pi(ip):
    t0 = time.perf_counter()
    try:
        r = subprocess.run(["ping", "-n" if os.name == "nt" else "-c", "2", ip], capture_output=True, text=True, timeout=10)
        ok = r.returncode == 0
    except Exception as e:
        return Result(f"Pi at {ip}", "SKIP", f"could not run ping: {e}")
    if not ok:
        return Result(f"Pi at {ip}", "FAIL", "no answer to ping: Pi off, wrong IP, or a different network", "check the Pi power/Wi-Fi; `ssh <user>@" + ip + "`")
    ssh = False
    try:
        socket.create_connection((ip, 22), timeout=2).close()
        ssh = True
    except OSError:
        pass
    return Result(f"Pi at {ip}", "PASS", f"answers ping ({(time.perf_counter() - t0) * 1000:.0f} ms incl. startup); ssh port {'open' if ssh else 'closed'}")


def check_software_chain():
    """No camera and no model: render a scene with known geometry, run the zone check and the alert controller, and compare with the truth."""
    from gabai.live_alerts import AlertController
    from gabai.sim import Box, Camera, look_pose, render_disparity
    from gabai.speaker import AlertSpeaker
    from gabai.zones import analyze
    cam = Camera(881.0, 0.0507, 319.5, 239.5, (640, 480))
    R, t = look_pose(np.array([0.0, 0.0, 1.2]), pitch_deg=35.0)
    disp = render_disparity(cam, R, t, boxes=[Box(lo=(2.0, -0.4, 0.0), hi=(2.4, 0.4, 1.0))])
    up = R[2, :]
    res = analyze(disp, cam.f, cam.B, 1.2, cam.cy, up_cam=up, cx=cam.cx)
    if not res.alert or res.alert[0] != "ahead":
        return Result("decision chain (synthetic)", "FAIL", f"a box 2 m ahead gave {res.alert}, expected an 'ahead' alert", "a regression in gabai/zones.py or gabai/sim.py: run py scripts/run_tests.py zones")
    ctl = AlertController(AlertSpeaker(player=lambda p: None, must_exist=False), obstacle_audio=True)
    said = ctl.on_frame(1.0, res.alert, "steady", 0.05)
    err = abs(res.distances["ahead"] - 2.0)
    st = "PASS" if said and err < 0.15 else "FAIL"
    return Result("decision chain (synthetic)", st, f"box at 2.0 m -> 'ahead' at {res.distances['ahead']:.2f} m (error {err * 100:.0f} cm), spoken: {said}",
                  "" if st == "PASS" else "the floor model or alert controller changed behaviour")


def check_depth_model_deep(size="m"):
    try:
        from gabai.spike.las2_runner import Las2Depth
        t0 = time.perf_counter()
        las = Las2Depth(size)
        load_s = time.perf_counter() - t0
        rng = np.random.default_rng(0)
        left = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        right = np.roll(left, -20, axis=1)                      # a known 20 px shift
        las.disparity(left, right)
        d, dt = las.disparity(left, right)
        med = float(np.median(d[:, 100:600]))
        ok = 17.0 < med < 23.0
        return Result("depth model run (deep)", "PASS" if ok else "FAIL", f"LAS2-{size.upper()} loaded in {load_s:.1f} s, one pass {dt * 1000:.0f} ms, a 20 px shift read as {med:.1f} px",
                      "" if ok else "the model returns wrong disparities: wrong weights file or a CUDA/driver problem")
    except Exception as e:
        return Result("depth model run (deep)", "FAIL", f"{type(e).__name__}: {e}", "see the traceback with GABAI_LOG_LEVEL=DEBUG; check CUDA and the weights")


def run(root=".", deep=False, pi=None, log_dir="logs"):
    res = check_libraries()
    res += [check_gpu(), check_models(root)] + check_optional_models(root) + check_calibration(root)
    res += [check_alert_audio(root), check_port(5600, "Pi stream"), check_port(8000, "telemetry page"), check_log_dir(log_dir), check_basalt(), check_software_chain()]
    if pi:
        res.append(check_pi(pi))
    if deep:
        res.append(check_depth_model_deep())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--pi", help="Pi IP address to check")
    ap.add_argument("--root", default=".")
    ap.add_argument("--log-dir", default=os.environ.get("GABAI_LOG_DIR", "logs"))
    a = ap.parse_args()
    results = run(a.root, a.deep, a.pi, a.log_dir)
    for r in results:
        print(f"{r.status:5s} {r.name:32s} {r.detail}")
        if r.fix and r.status in ("WARN", "FAIL"):
            print(f"      fix: {r.fix}")
    s = summarize(results)
    print(f"\n{s['PASS']} passed, {s['WARN']} warnings, {s['FAIL']} FAILED, {s['SKIP']} skipped")
    os.makedirs(a.log_dir, exist_ok=True)
    p = os.path.join(a.log_dir, time.strftime("selfcheck_%Y%m%d_%H%M%S.json"))
    json.dump({"results": [asdict(r) for r in results], "summary": s, "python": sys.version.split()[0]}, open(p, "w", encoding="utf-8"), indent=2)
    print("saved", p)
    sys.exit(exit_code(results))


if __name__ == "__main__":
    main()
