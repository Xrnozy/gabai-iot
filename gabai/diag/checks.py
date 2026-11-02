"""Validation checks. Each one looks at real numbers and returns a Check, never a guess.

status:  pass     the number is inside its limit
         fail     the number is outside its limit: confirmed from the data
         suspect  between "fine" and "failed", or the number only hints at a problem: a possible cause, not a confirmed one
         unknown  the data needed was not available (the check says so instead of passing)
`hint` says where to look next when a check does not pass. Limits are starting values (module constants), not measured truths.
"""
from dataclasses import dataclass

import numpy as np

STALE_FRAME_S = 0.300            # same as alerts.STALE_AGE_S
SYNC_PX_SUSPECT, SYNC_PX_FAIL = 0.5, 1.0
MIN_VALID_FRACTION = 0.40        # same as zones.MIN_VALID_FRACTION
DISPARITY_CAP = 192.0            # LAS2 cannot match more than this (0.23 m on this rig)
GRAVITY = 9.80665
LATENCY_BUDGET_MS = 400.0        # the spec's target, not a measured result
MAX_STEP_M, MAX_SPEED_MPS, SUSPECT_SPEED_MPS = 0.5, 4.0, 2.5


@dataclass
class Check:
    name: str
    status: str
    component: str
    value: object = None
    limit: object = None
    detail: str = ""
    hint: str = ""

    def as_dict(self):
        v = self.value
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        return {"name": self.name, "status": self.status, "component": self.component, "value": v, "limit": self.limit,
                "detail": self.detail, "hint": self.hint if self.status in ("fail", "suspect") else ""}


def _unknown(name, component, what):
    return Check(name, "unknown", component, detail=f"{what} was not available, so this was not checked")


def freshness(tagged, age_s, limit_s):
    """Copy of a tagged value (gabai.diag.trace.q) marked stale if it is older than the limit."""
    out = dict(tagged)
    out["age_s"] = age_s
    if age_s is not None and age_s > limit_s and out.get("v") is not None:
        out.update(kind="stale", was=tagged.get("kind"), limit_s=limit_s)
    return out


def check_frame_age(age_s, limit=STALE_FRAME_S):
    if age_s is None:
        return _unknown("frame_age", "link", "the frame age")
    st = "pass" if age_s <= limit else "fail"
    return Check("frame_age", st, "link", round(age_s, 3), limit, f"newest frame is {age_s * 1000:.0f} ms old",
                 "stream is late: check Wi-Fi, Pi load/under-voltage (health.throttled), or the PC loop time" if st == "fail" else "")


def check_pair_sync(dt_ms, omega_dps, f_px):
    """Left/right capture offset turned into the false disparity it causes while the camera turns: f * omega * dt pixels."""
    if dt_ms is None or omega_dps is None:
        return _unknown("pair_sync", "cameras", "the left/right offset or the turn rate")
    px = float(f_px) * np.radians(abs(omega_dps)) * dt_ms / 1000.0
    st = "pass" if px <= SYNC_PX_SUSPECT else "suspect" if px <= SYNC_PX_FAIL else "fail"
    return Check("pair_sync", st, "cameras", round(px, 3), SYNC_PX_FAIL, f"{dt_ms:.1f} ms offset at {omega_dps:.0f} deg/s = {px:.2f} px false disparity",
                 "depth is biased while turning: slow down, or use a synchronised camera pair" if st != "pass" else "")


def check_image(img):
    """Is this a usable picture: not missing, not black, not saturated, not flat, not blurred or textureless."""
    if img is None or getattr(img, "size", 0) == 0:
        return Check("image", "fail", "cameras", detail="no image", hint="camera did not deliver a frame or decoding failed")
    import cv2
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, std = float(g.mean()), float(g.std())
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    sat = float((g >= 250).mean())
    issues, hard = [], False
    if mean < 8 or mean > 247 or std < 3:
        issues.append(f"black, white or flat frame (mean {mean:.0f}, std {std:.1f})")
        hard = True
    else:
        if mean < 35:
            issues.append(f"dark (mean {mean:.0f})")
        if std < 14:
            issues.append(f"low contrast (std {std:.1f})")
    if lap < 25:
        issues.append(f"blurred or textureless (laplacian variance {lap:.1f})")
    if sat > 0.25:
        issues.append(f"{sat * 100:.0f}% of pixels saturated")
    st = "fail" if hard else "suspect" if issues else "pass"
    return Check("image", st, "cameras", {"mean": round(mean, 1), "std": round(std, 1), "lap_var": round(lap, 1), "saturated": round(sat, 3)},
                 None, "; ".join(issues) or "exposure, contrast and sharpness look usable",
                 "stereo matching needs light and texture: raise gain/brightness, add light, check for motion blur" if issues else "")


def check_disparity_cap(disp, cap=DISPARITY_CAP):
    """Share of valid pixels sitting at the model's maximum disparity: they are probably wrong (object closer than the model can match)."""
    d = np.asarray(disp, float)
    v = np.isfinite(d) & (d > 0.5)
    if not v.any():
        return Check("disparity_cap", "unknown", "depth", detail="no valid disparity pixels")
    frac = float((d[v] >= 0.92 * cap).mean())
    st = "pass" if frac <= 0.02 else "suspect" if frac <= 0.10 else "fail"
    return Check("disparity_cap", st, "depth", round(frac, 4), 0.10, f"{frac * 100:.1f}% of valid pixels are at the disparity limit ({cap:.0f} px)",
                 "an object closer than f*B/192 (0.23 m on this rig) cannot be measured: those depths are wrong, not near" if st != "pass" else "")


def check_depth_coverage(valid_frac, limit=MIN_VALID_FRACTION):
    if valid_frac is None:
        return _unknown("depth_coverage", "depth", "the valid-depth fraction")
    st = "pass" if valid_frac >= limit else "fail"
    return Check("depth_coverage", st, "depth", round(float(valid_frac), 3), limit, f"{valid_frac * 100:.0f}% of the floor view has trusted depth",
                 "below this the system must say 'can't see ahead': check light, texture, and the left-right rejection rate" if st == "fail" else "")


def check_trusted_fraction(frac):
    """Share of pixels that survived the left-right consistency check."""
    if frac is None:
        return _unknown("trusted_fraction", "depth", "the left-right consistency result")
    st = "pass" if frac >= 0.25 else "suspect" if frac >= 0.10 else "fail"
    return Check("trusted_fraction", st, "depth", round(float(frac), 3), 0.25, f"{frac * 100:.0f}% of pixels pass the left-right check",
                 "most pixels rejected: occlusions, glare, dark/blank surfaces, or the left/right images are mis-rectified or swapped" if st != "pass" else "")


def check_floor_model(cam_h, imu_calibrated, have_up_vector):
    if cam_h is None:
        return Check("floor_model", "suspect", "zones", None, None, "no camera height given: no floor model, no drop-off check, obstacle speech off",
                     "start the viewer with --cam-h <metres>")
    if not imu_calibrated:
        return Check("floor_model", "suspect", "zones", cam_h, None, "camera height known but the IMU axes are not calibrated: floor assumes a level camera",
                     "run sync_imu_camera / check_extrinsics; a tilted camera makes the floor and every distance wrong")
    if not have_up_vector:
        return Check("floor_model", "suspect", "zones", cam_h, None, "IMU orientation not available yet", "wait for IMU data")
    return Check("floor_model", "pass", "zones", cam_h, None, "floor follows the IMU up-vector")


def check_imu_rate(rate_hz, max_gap_ms):
    if rate_hz is None:
        return _unknown("imu_rate", "imu", "the IMU rate")
    gap = 0.0 if max_gap_ms is None else max_gap_ms
    st = "fail" if (rate_hz < 100 or gap > 100) else "suspect" if (rate_hz < 150 or gap > 30) else "pass"
    return Check("imu_rate", st, "imu", {"rate_hz": round(float(rate_hz), 1), "max_gap_ms": round(float(gap), 1)}, {"rate_hz": 150, "max_gap_ms": 30},
                 f"{rate_hz:.0f} Hz, longest gap {gap:.0f} ms", "IMU samples are being lost: I2C errors, Pi overload, or Wi-Fi bursts" if st != "pass" else "")


def check_gravity(still_accel_norm):
    """A still accelerometer reads 9.807 m/s^2. Anything else is a calibration error that tracking will integrate into drift."""
    if still_accel_norm is None:
        return _unknown("gravity", "imu", "a still accelerometer reading")
    err = abs(float(still_accel_norm) - GRAVITY)
    st = "pass" if err <= 0.15 else "suspect" if err <= 0.6 else "fail"
    return Check("gravity", st, "imu", round(float(still_accel_norm), 3), GRAVITY, f"still |a| = {still_accel_norm:.3f}, off by {err:.2f} m/s^2",
                 "accelerometer bias or scale error: run gabai.tools.calibrate_accel (Basalt drifts metres otherwise)" if st != "pass" else "")


def check_pose_step(step_m, dt_s):
    """One step of the tracked trajectory: is it a believable thing for a walking person to do."""
    if step_m is None or not dt_s:
        return _unknown("pose_step", "slam", "two consecutive poses")
    speed = step_m / dt_s
    if step_m > MAX_STEP_M or speed > MAX_SPEED_MPS:
        return Check("pose_step", "fail", "slam", {"step_m": round(step_m, 3), "speed_mps": round(speed, 2)}, {"step_m": MAX_STEP_M, "speed_mps": MAX_SPEED_MPS},
                     f"jump of {step_m:.2f} m in {dt_s * 1000:.0f} ms ({speed:.1f} m/s)", "tracking jumped or diverged: check image blur, feature count, camera-IMU rotation and accelerometer calibration")
    st = "suspect" if speed > SUSPECT_SPEED_MPS else "pass"
    return Check("pose_step", st, "slam", {"step_m": round(step_m, 3), "speed_mps": round(speed, 2)}, SUSPECT_SPEED_MPS, f"{speed:.2f} m/s",
                 "faster than a walk: possible tracking glitch" if st != "pass" else "")


def check_latency(ms, budget=LATENCY_BUDGET_MS):
    if ms is None:
        return _unknown("latency", "pipeline", "the capture-to-speech time")
    st = "pass" if ms <= budget else "fail"
    return Check("latency", st, "pipeline", round(float(ms), 1), budget, f"{ms:.0f} ms against a {budget:.0f} ms target",
                 "see the per-stage times in the trace: the largest one is the bottleneck" if st == "fail" else "")


def check_queue(size, cap):
    if size is None or not cap:
        return _unknown("queue", "pipeline", "the queue size")
    r = size / cap
    st = "fail" if r >= 0.9 else "suspect" if r >= 0.6 else "pass"
    return Check("queue", st, "pipeline", int(size), int(cap), f"{size}/{cap} used", "consumer is slower than the producer: data will be dropped or delayed" if st != "pass" else "")


def check_resources(cpu_pct, ram_pct, gpu_temp_c):
    if cpu_pct is None and ram_pct is None and gpu_temp_c is None:
        return _unknown("resources", "system", "any CPU, memory or GPU reading")
    bad = []
    if cpu_pct is not None and cpu_pct >= 95:
        bad.append(f"CPU {cpu_pct:.0f}%")
    if ram_pct is not None and ram_pct >= 92:
        bad.append(f"RAM {ram_pct:.0f}%")
    hot = gpu_temp_c is not None and gpu_temp_c >= 85
    if gpu_temp_c is not None and gpu_temp_c >= 75:
        bad.append(f"GPU {gpu_temp_c:.0f} C")
    st = "fail" if hot else "suspect" if bad else "pass"
    return Check("resources", st, "system", {"cpu_pct": cpu_pct, "ram_pct": ram_pct, "gpu_temp_c": gpu_temp_c}, None,
                 ", ".join(bad) or "within limits", "the machine is saturated or hot: expect dropped frames and slower depth" if bad else "")


def check_pi_power(throttled):
    """The Pi's own power and thermal flags (`vcgencmd get_throttled`). Under-voltage right now is a confirmed power fault."""
    from gabai.diag.pi_status import decode_throttled
    d = decode_throttled(throttled)
    if d is None:
        return _unknown("pi_power", "pi", "the Pi throttle flags")
    now = [k for k in ("under_voltage_now", "freq_capped_now", "throttled_now", "soft_temp_limit_now") if d[k]]
    ever = [k for k in ("under_voltage_since_boot", "throttled_since_boot") if d[k]]
    if d["under_voltage_now"]:
        return Check("pi_power", "fail", "pi", str(throttled), "0x0", "under-voltage RIGHT NOW (supply below 4.63 V)",
                     "weak supply or cable: expect camera/USB glitches, dropped frames and CPU throttling; see docs/pi-power-guide.md")
    if now:
        return Check("pi_power", "fail", "pi", str(throttled), "0x0", "active: " + ", ".join(now), "heat or power limits are reducing the Pi's speed right now")
    if ever:
        return Check("pi_power", "suspect", "pi", str(throttled), "0x0", "happened since boot, clear now: " + ", ".join(ever),
                     "a past power or heat problem; reboot to clear the flags and watch whether it comes back")
    return Check("pi_power", "pass", "pi", str(throttled), "0x0", "no power or thermal flags")


def check_grid_agreement(zone_alert, corridor, grid_frames):
    """The zone check drives the warnings; the rolling grid is a second, independent view of the same frame. When they disagree one of
    them is wrong. This only compares, it changes nothing. `grid_frames` is how many frames the grid has seen: a young grid is not judged."""
    if corridor is None or grid_frames is None:
        return _unknown("grid_agreement", "grid", "the grid corridor")
    if grid_frames < 3:
        return Check("grid_agreement", "unknown", "grid", grid_frames, 3, "the grid has seen fewer than 3 frames", "")
    zone_ahead = zone_alert[1] if (zone_alert and zone_alert[0] == "ahead") else None
    near = corridor.get("nearest_occupied_m")
    if zone_ahead is not None and near is None:
        st = "suspect" if corridor.get("unknown_fraction", 1.0) > 0.5 else "fail"
        return Check("grid_agreement", st, "grid", {"zone_m": zone_ahead, "grid_m": None, "grid_unknown": corridor.get("unknown_fraction")}, None,
                     f"zones warn of an obstacle {zone_ahead:.1f} m ahead; the grid has nothing occupied in the corridor (unknown {corridor.get('unknown_fraction', 0) * 100:.0f}%)",
                     "either a false alarm (floor model wrong: camera height or tilt) or the grid height limits hide it; look at the saved depth picture")
    if zone_ahead is None and near is not None and near < 2.5:
        return Check("grid_agreement", "fail", "grid", {"zone_m": None, "grid_m": round(near, 2)}, None,
                     f"the grid has an obstacle {near:.1f} m ahead and the zone check says nothing",
                     "a possible MISSED obstacle: the zone check's floor model may be treating it as floor (camera height or tilt wrong)")
    if zone_ahead is not None and near is not None and abs(zone_ahead - near) > 0.5:
        return Check("grid_agreement", "suspect", "grid", {"zone_m": zone_ahead, "grid_m": round(near, 2)}, 0.5,
                     f"zones say {zone_ahead:.1f} m, the grid says {near:.1f} m", "distance estimates disagree: pose/tilt used by the grid vs the zone floor model")
    return Check("grid_agreement", "pass", "grid", {"zone_m": zone_ahead, "grid_m": None if near is None else round(near, 2)}, 0.5, "zone check and grid agree")


def check_range_vs_depth(depth_m, horizontal_m, limit=0.25):
    """Alerts quote the depth along the optical axis. On a tilted camera the distance you would walk is the horizontal range. A large
    difference means the spoken distance is optimistic by that much."""
    if depth_m is None or horizontal_m is None:
        return _unknown("range_vs_depth", "zones", "an obstacle depth and its horizontal range")
    diff = depth_m - horizontal_m
    st = "pass" if diff <= limit else "suspect"
    return Check("range_vs_depth", st, "zones", {"depth_m": round(depth_m, 2), "horizontal_m": round(horizontal_m, 2), "optimistic_by_m": round(diff, 2)}, limit,
                 f"spoken distance (optical depth) {depth_m:.2f} m, horizontal range {horizontal_m:.2f} m: optimistic by {diff * 100:.0f} cm",
                 "the camera is tilted so far that the spoken distance overstates how far the obstacle is to walk" if st != "pass" else "")
