"""Persistent, validated calibration files, one per sensor.

    data/calibration/hmc5883l.json   compass: hard-iron offset, soft-iron matrix, axis maps, quality, timestamps
    data/calibration/bmi160.json     IMU: gyro noise and residual (NOT a bias to subtract: the Pi measures and removes the bias at every start), accelerometer
                                     bias/scale only when the measurements support them, axis maps, timestamps

Every value in them refers to RAW sensor readings (counts for the compass, g and deg/s as the Pi sender delivers them). The correction is applied in one place only: the
running system (gabai.fusion) reads the same numbers from the legacy files that write_legacy_* produce (data/mag_cal.json, data/mag_sync.json, data/accel_calib.json,
data/device_frame.json), and gabai.mag.apply_cal / the accelerometer model there are the only functions that apply them. This module never applies a calibration.

A file is checked before it is used (validate_*): schema tag, sensor identity, finite numbers, matrices that are rotations or sane stretches, quality inside its limits.
A file that fails is reported and ignored, never half-used. Writes are atomic (temp file, then replace) and the previous file is kept as .bak.
"""
import json
import os
import time

import numpy as np

SCHEMA = "gabai.calibration/1"
MAG_ID = {"kind": "HMC5883L", "bus": 3, "address": "0x1E", "units": "raw counts, 0.92 mG per count at gain 1.3 Ga"}
IMU_ID = {"kind": "BMI160", "bus": 1, "address": "0x68", "units": "gyro deg/s (bias already removed by the Pi at start), accel g (raw)"}
MAG_FILE, IMU_FILE = "hmc5883l.json", "bmi160.json"
MAG_QUALITY_MAX = 0.12                  # relative spread of the corrected field strength above which a calibration is not accepted at all
IMU_ACCEL_BIAS_MAX = 1.5                # m/s^2
IMU_ACCEL_SCALE_MAX = 0.1


def _finite(a):
    return bool(np.all(np.isfinite(np.asarray(a, float))))


def is_rotation(M, tol=1e-3):
    M = np.asarray(M, float)
    return M.shape == (3, 3) and _finite(M) and np.allclose(M @ M.T, np.eye(3), atol=tol) and abs(np.linalg.det(M) - 1) < tol


def _check_header(d, kind):
    p = []
    if not isinstance(d, dict):
        return ["not a calibration object"]
    if d.get("schema") != SCHEMA:
        p.append(f"schema is {d.get('schema')!r}, expected {SCHEMA!r}")
    if (d.get("sensor") or {}).get("kind") != kind:
        p.append(f"file is for sensor {(d.get('sensor') or {}).get('kind')!r}, not {kind}")
    if not d.get("date"):
        p.append("no timestamp")
    if d.get("applies_to") != "raw":
        p.append("applies_to is not 'raw': the numbers would be applied to already-corrected readings")
    return p


def validate_mag(d):
    """List of problems (empty = usable)."""
    p = _check_header(d, "HMC5883L")
    if p and not isinstance(d, dict):
        return p
    try:
        par = d["parameters"]
        off, W = np.asarray(par["offset"], float), np.asarray(par["W"], float)
        if off.shape != (3,) or not _finite(off):
            p.append("offset is not three finite numbers")
        if W.shape != (3, 3) or not _finite(W):
            p.append("W is not a finite 3x3 matrix")
        else:
            sv = np.linalg.svd(W, compute_uv=False)
            if np.linalg.det(W) <= 0 or sv.max() / max(sv.min(), 1e-12) > 3.0:
                p.append("the soft-iron matrix is mirrored or stretches the field by more than 3 times: not a plausible compass correction")
        if not (np.isfinite(par["radius_counts"]) and par["radius_counts"] > 0):
            p.append("radius_counts is not positive")
        if not (np.isfinite(par["quality"]) and 0 <= par["quality"] <= MAG_QUALITY_MAX):
            p.append(f"quality {par.get('quality')} is outside 0..{MAG_QUALITY_MAX}")
        am = d["axis_mapping"]
        if am.get("matrix_to_imu") is not None and not is_rotation(am["matrix_to_imu"]):
            p.append("axis_mapping.matrix_to_imu is not a rotation")
    except (KeyError, TypeError, ValueError) as e:
        p.append(f"missing or malformed field: {e!r}")
    return p


def validate_imu(d):
    p = _check_header(d, "BMI160")
    if p and not isinstance(d, dict):
        return p
    try:
        g = d["gyro"]
        if not (_finite(g["noise_dps"]) and _finite(g["residual_bias_dps"])):
            p.append("gyro statistics are not finite")
        if g.get("bias_policy") != "not_applied":
            p.append("gyro.bias_policy must be 'not_applied': the Pi already removes the gyro bias, applying a saved one would remove it twice")
        a = d["accel"]
        if a.get("calibrated"):
            b, s = np.asarray(a["bias_ms2"], float), np.asarray(a["scale"], float)
            if b.shape != (3,) or s.shape != (3,) or not (_finite(b) and _finite(s)):
                p.append("accelerometer bias/scale are not three finite numbers each")
            elif np.abs(b).max() > IMU_ACCEL_BIAS_MAX or np.abs(s).max() > IMU_ACCEL_SCALE_MAX:
                p.append("accelerometer bias or scale is implausibly large for a BMI160")
        am = d["axis_mapping"]
        if am.get("device_from_imu") is not None and not is_rotation(am["device_from_imu"], 5e-3):
            p.append("axis_mapping.device_from_imu is not a rotation")
    except (KeyError, TypeError, ValueError) as e:
        p.append(f"missing or malformed field: {e!r}")
    return p


def _atomic_write(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        try:
            os.replace(path, path + ".bak")
        except OSError:
            pass
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)


def save(path, obj, validator):
    problems = validator(json.loads(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))))
    if problems:
        raise ValueError("refusing to save an invalid calibration: " + "; ".join(problems))
    _atomic_write(path, obj)


def load(path, validator):
    """(data or None, problems). A missing file is (None, []); a corrupt or invalid one is (None, [reasons])."""
    if not os.path.exists(path):
        return None, []
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, [f"cannot read {path}: {e}"]
    problems = validator(d)
    return (None, problems) if problems else (d, [])


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def mag_document(offset, W, radius_counts, quality, coverage, method, matrix_to_imu, reference, matches_reference, notes, validation, device_from_mag=None):
    return {"schema": SCHEMA, "sensor": MAG_ID, "date": now(), "tool": "gabai.tools.calibrate_sensors", "applies_to": "raw",
            "parameters": {"offset": np.asarray(offset, float).tolist(), "W": np.asarray(W, float).tolist(), "radius_counts": float(radius_counts), "quality": float(quality),
                           "coverage": float(coverage), "method": method, "formula": "corrected = W @ (raw - offset)"},
            "axis_mapping": {"reference": reference, "matrix_to_imu": None if matrix_to_imu is None else np.asarray(matrix_to_imu, float).tolist(),
                             "matches_reference_permutation": matches_reference, "notes": notes},
            "transforms": {"device_from_mag": None if device_from_mag is None else np.asarray(device_from_mag, float).tolist(),
                           "device_frame": "left camera, x right, y down, z forward (the frame the SLAM uses); see docs/sensor-calibration.md for the chest mount"},
            "validation": validation}


def imu_document(gyro, accel, device_from_imu, reference, matches_reference, notes, validation):
    return {"schema": SCHEMA, "sensor": IMU_ID, "date": now(), "tool": "gabai.tools.calibrate_sensors", "applies_to": "raw",
            "gyro": dict(gyro, bias_policy="not_applied"), "accel": accel,
            "axis_mapping": {"reference": reference, "device_from_imu": None if device_from_imu is None else np.asarray(device_from_imu, float).tolist(),
                             "matches_reference_permutation": matches_reference, "notes": notes},
            "validation": validation}


def write_legacy_mag(doc, data_dir="data"):
    """The files the running system reads, from a validated compass document: data/mag_cal.json and data/mag_sync.json (the compass delay already there is kept)."""
    par = doc["parameters"]
    _atomic_write(os.path.join(data_dir, "mag_cal.json"), {"offset": par["offset"], "scale": [1.0, 1.0, 1.0], "W": par["W"], "quality": par["quality"],
                                                          "radius_counts": par["radius_counts"], "coverage": par["coverage"], "method": par["method"], "date": doc["date"],
                                                          "source": "calibrate_sensors"})
    R = doc["axis_mapping"]["matrix_to_imu"]
    if R is not None:
        delay = 0.0
        old = os.path.join(data_dir, "mag_sync.json")
        if os.path.exists(old):
            try:
                delay = float(json.load(open(old)).get("delay_s", 0.0))
            except (OSError, ValueError):
                pass
        _atomic_write(old, {"delay_s": delay, "R_im": R, "score": None, "date": doc["date"], "source": "calibrate_sensors"})


def write_legacy_imu(doc, data_dir="data"):
    """data/accel_calib.json (only when the accelerometer fit was supported) and data/device_frame.json (when the device axes were measured)."""
    a = doc["accel"]
    if a.get("calibrated"):
        from gabai.slam.accel_cal import to_basalt_params
        _atomic_write(os.path.join(data_dir, "accel_calib.json"), {"bias": a["bias_ms2"], "scale": a["scale"], "basalt_params": to_basalt_params(a["bias_ms2"], a["scale"]),
                                                                  "rms_before": a.get("rms_before_ms2"), "rms_after": a.get("rms_after_ms2"), "n_orientations": a.get("n_orientations"),
                                                                  "faces_missing": a.get("faces_missing", []), "date": doc["date"]})
    R = doc["axis_mapping"]["device_from_imu"]
    if R is not None:
        _atomic_write(os.path.join(data_dir, "device_frame.json"), {"R_dev_imu": R, "date": doc["date"], "source": "calibrate_sensors: gravity face-down and upright"})
