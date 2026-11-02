"""What this machine and these calibration files are, as plain data for the startup log, the self-check and the diagnostic package.
Nothing here raises: a missing library or file is reported as missing."""
import datetime
import json
import os
import platform
import sys


def file_info(path):
    if not path or not os.path.exists(path):
        return {"path": path, "exists": False}
    st = os.stat(path)
    return {"path": path, "exists": True, "bytes": st.st_size, "modified": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "age_days": round((datetime.datetime.now().timestamp() - st.st_mtime) / 86400, 1)}


def _version(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "installed")
    except Exception as e:
        return f"MISSING ({type(e).__name__})"


def environment_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "libs": {m: _version(m) for m in ("numpy", "cv2", "scipy", "torch", "psutil", "fastapi", "uvicorn", "matplotlib", "pandas")}}
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": p.name, "vram_gb": round(p.total_memory / 1e9, 1), "capability": f"{p.major}.{p.minor}", "torch_cuda": torch.version.cuda}
    except Exception as e:
        info["cuda_available"] = False
        info["gpu_error"] = f"{type(e).__name__}: {e}"
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    return info


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def calibration_summary(root="."):
    """The calibration files that decide whether distances and tracking are right, with their key numbers and any inconsistency."""
    j = lambda *p: os.path.join(root, *p)
    out = {}
    name = "stereo_calib_new" if os.path.exists(j("data", "stereo_calib_new.npz")) else "stereo_calib"     # the file the live pipeline uses by default, else the old name
    s = file_info(j("data", name + ".npz"))
    meta = _load_json(j("data", name + ".json")) if os.path.exists(j("data", name + ".json")) else None
    if meta and "_error" not in meta:
        s.update({k: meta.get(k) for k in ("baseline", "focal", "cx", "cy", "size", "rms_stereo", "epipolar_err_px", "n_views", "swapped", "verdict", "date",
                                           "rotate_left", "rotate_right")})
        s["warnings"] = [w for w in (
            "cameras were SWAPPED when this was calibrated" if meta.get("swapped") else None,
            f"stereo reprojection error {meta.get('rms_stereo'):.2f} px is high" if isinstance(meta.get("rms_stereo"), (int, float)) and meta["rms_stereo"] > 1.0 else None,
            f"verdict {meta.get('verdict')}" if str(meta.get("verdict", "")).startswith("POOR") else None) if w]
    out["stereo_calib"] = s
    sync = file_info(j("data", "imu_sync.json"))
    if sync["exists"]:
        d = _load_json(j("data", "imu_sync.json"))
        sync.update(delay_s=d.get("delay_s"), quality=d.get("quality"), source=d.get("source"), delay_range_ms=d.get("delay_range_ms"))
        try:
            import numpy as np
            R = np.array(d["R_ci"], float)
            sync["rotation_orthonormal_error"] = round(float(np.abs(R @ R.T - np.eye(3)).max()), 6)
            sync["rotation_det"] = round(float(np.linalg.det(R)), 4)
            sync["rotation_is_exact_axis_swap"] = bool(np.allclose(R, np.round(R), atol=1e-6))
        except Exception as e:
            sync["rotation_error"] = f"{type(e).__name__}: {e}"
    out["imu_sync"] = sync
    acc = file_info(j("data", "accel_calib.json"))
    if acc["exists"]:
        d = _load_json(j("data", "accel_calib.json"))
        acc.update(bias=d.get("bias"), scale=d.get("scale"), rms_before=d.get("rms_before"), rms_after=d.get("rms_after"),
                   n_orientations=d.get("n_orientations"), faces_missing=d.get("faces_missing"))
    out["accel_calib"] = acc
    out["mag_cal"] = file_info(j("data", "mag_cal.json"))
    out["mag_sync"] = file_info(j("data", "mag_sync.json"))
    return out
