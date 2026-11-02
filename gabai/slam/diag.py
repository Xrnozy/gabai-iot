"""Summaries and findings for a tracked trajectory. Everything here is computed from the poses (and the IMU, if given): there is
NO ground truth, so a clean summary means "nothing implausible happened", never "the position is correct". A finding is
`confirmed` when a number measured from these data breaks a stated limit, and `possible` when it is only a symptom that points
at a cause.

poses: array N x 8 = t (s), x, y, z (m), qx, qy, qz, qw       imu: array M x 7 = t (s), gx, gy, gz (rad/s), ax, ay, az (m/s^2 or g)
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

JUMP_M, JUMP_SPEED_MPS = 0.5, 4.0         # a step this big, or this fast, is not a person walking
GAP_S = 0.3                               # no pose for this long
STILL_GYRO_DPS = 3.0
STILL_MIN_S = 3.0
SPEC_STILL_DRIFT_CM_PER_MIN = 5.0         # spec: still for 60 s drifts under 5 cm
DIVERGED_M = 30.0                         # farther than this from the start inside a one-minute test is not credible
CAVEAT = "No ground truth: these numbers show only whether the trajectory is plausible, not whether it is correct."


def _stats(a):
    a = np.asarray(a, float)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3), "max": round(float(a.max()), 3)} if a.size else None


def still_intervals(imu, gyro_dps=STILL_GYRO_DPS, min_s=STILL_MIN_S):
    """(t0, t1) spans in which the gyro stayed under `gyro_dps`."""
    if imu is None or len(imu) < 10:
        return []
    t, w = imu[:, 0], np.linalg.norm(imu[:, 1:4], axis=1)
    quiet = w < np.radians(gyro_dps)
    out, i = [], 0
    while i < len(t):
        if quiet[i]:
            j = i
            while j + 1 < len(t) and quiet[j + 1]:
                j += 1
            if t[j] - t[i] >= min_s:
                out.append((float(t[i]), float(t[j])))
            i = j + 1
        else:
            i += 1
    return out


def rotation_consistency(poses, imu, stride=5):
    """Rotation between poses (from the tracker) against the gyro integrated over the same intervals. Basalt integrates the gyro, so a
    ratio far from 1 means the camera measurements and the gyro disagree: look at the camera-IMU rotation and time offset."""
    p = np.asarray(poses)
    if len(p) < stride + 3 or imu is None or len(imu) < 10:
        return {"n": 0, "ratio_median": None}
    q = R.from_quat(p[:, 4:8])
    ti, w = imu[:, 0], imu[:, 1:4]
    dts = np.gradient(ti)
    ratios = []
    for i in range(0, len(p) - stride, stride):
        a = (q[i + stride] * q[i].inv()).magnitude()
        m = (ti >= p[i, 0]) & (ti < p[i + stride, 0])
        if m.sum() < 3:
            continue
        g = np.linalg.norm((w[m] * dts[m, None]).sum(0))
        if g > np.radians(0.5) and a > np.radians(0.5):
            ratios.append(a / g)
    return {"n": len(ratios), "ratio_median": round(float(np.median(ratios)), 3) if ratios else None,
            "ratio_p10_p90": [round(float(x), 3) for x in np.percentile(ratios, [10, 90])] if ratios else None}


def summarize_trajectory(poses, imu=None):
    p = np.asarray(poses, float)
    n = len(p)
    s = {"n_poses": n, "caveat": CAVEAT}
    if n < 2:
        return s
    t, xyz = p[:, 0], p[:, 1:4]
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    speed = step / np.maximum(dt, 1e-6)
    ok = dt <= GAP_S                                   # speeds across a gap are not speeds
    jump = (step > JUMP_M) | ((speed > JUMP_SPEED_MPS) & ok)
    gap = dt > GAP_S
    s.update(duration_s=round(float(t[-1] - t[0]), 2), pose_rate_hz=round(float((n - 1) / max(t[-1] - t[0], 1e-6)), 2), path_m=round(float(step[~jump].sum()), 2),
             bbox_m=[round(float(x), 2) for x in xyz.max(0) - xyz.min(0)], start_to_end_m=round(float(np.linalg.norm(xyz[-1] - xyz[0])), 2),
             max_from_start_m=round(float(np.linalg.norm(xyz - xyz[0], axis=1).max()), 2),
             step_m=_stats(step), speed_mps=_stats(speed[ok]) if ok.any() else None,
             n_jumps=int(jump.sum()), jumps=[{"t": round(float(t[i + 1]), 2), "step_m": round(float(step[i]), 3), "speed_mps": round(float(speed[i]), 2)} for i in np.where(jump)[0][:20]],
             n_gaps=int(gap.sum()), gaps=[{"t": round(float(t[i]), 2), "gap_s": round(float(dt[i]), 2)} for i in np.where(gap)[0][:20]])
    if imu is not None and len(imu) > 10:
        spans = still_intervals(imu)
        if spans:
            t0, t1 = max(spans, key=lambda x: x[1] - x[0])
            m = (t >= t0) & (t <= t1)
            if m.sum() >= 5:
                span = float(t[m][-1] - t[m][0])
                drift = float(np.linalg.norm(xyz[m][-1] - xyz[m][0]))
                s["still"] = {"t0": round(t0, 2), "t1": round(t1, 2), "seconds": round(span, 2), "drift_cm": round(drift * 100, 2),
                              "drift_cm_per_min": round(drift * 100 / max(span, 1e-6) * 60, 2),
                              "note": "straight-line displacement while the gyro read under 3 deg/s; scaled to one minute"}
        s["rotation_consistency"] = rotation_consistency(p, imu)
    return s


def findings_for(s):
    """Ranked list of {level, code, component, message, evidence, possible_causes}."""
    out = []
    add = lambda level, code, msg, ev, causes=(): out.append({"level": level, "code": code, "component": "slam", "message": msg, "evidence": ev,
                                                                 "possible_causes": list(causes)})
    if s.get("n_poses", 0) < 5:
        add("confirmed", "too_few_poses", f"only {s.get('n_poses', 0)} poses were produced", {"n_poses": s.get("n_poses", 0)},
            ["Basalt never started (the rig was not still, see the HOLD STILL bar)", "frames or IMU were not being pushed (see pi.stats / link.stats)", "wrong calibration file"])
        return out
    if s["max_from_start_m"] > DIVERGED_M or (s["speed_mps"] and s["speed_mps"]["p50"] > 5.0):
        add("confirmed", "diverged", f"the estimate ran {s['max_from_start_m']:.0f} m from the start (median speed {s['speed_mps']['p50'] if s['speed_mps'] else 0:.1f} m/s): tracking diverged",
            {"max_from_start_m": s["max_from_start_m"], "speed_mps": s["speed_mps"]},
            ["accelerometer bias or scale not calibrated (run calibrate_accel)", "camera-IMU rotation wrong (run check_extrinsics)", "started while moving",
             "motion blur or too few features"])
    if s["n_jumps"]:
        add("confirmed", "pose_jump", f"{s['n_jumps']} pose jump(s) larger than {JUMP_M} m between consecutive poses (first at t={s['jumps'][0]['t']} s)",
            {"jumps": s["jumps"]}, ["motion blur or a featureless view", "accelerometer calibration", "camera-IMU time offset or rotation", "a frame gap that Basalt bridged badly"])
    if s["n_gaps"]:
        add("confirmed", "pose_gap", f"{s['n_gaps']} stretch(es) with no pose for more than {GAP_S} s (longest {max(g['gap_s'] for g in s['gaps']):.1f} s)",
            {"gaps": s["gaps"]}, ["the stream stalled (see link.gap and pi.stats)", "the still gate was waiting", "Basalt dropped frames while overloaded"])
    st = s.get("still")
    if st and st["seconds"] >= STILL_MIN_S and st["drift_cm_per_min"] > SPEC_STILL_DRIFT_CM_PER_MIN:
        extrap = "" if st["seconds"] >= 55 else (f" (the {st['seconds']:.0f} s window is scaled to a minute: an extrapolation, drift need not grow in a straight line; "
                                                  "record 60 s still for the real test)")
        add("confirmed", "still_drift", f"drifted {st['drift_cm']:.1f} cm in {st['seconds']:.0f} s while still, {st['drift_cm_per_min']:.0f} cm/min; the spec allows 5 cm in 60 s{extrap}",
            st,["accelerometer bias (calibrate_accel)", "gyro bias: the rig moved when the Pi measured it", "IMU noise values are datasheet numbers, not measured"])
    rc = s.get("rotation_consistency") or {}
    if rc.get("ratio_median") and not (0.8 <= rc["ratio_median"] <= 1.25) and rc.get("n", 0) >= 5:
        add("possible", "rotation_mismatch", f"rotation from the tracked poses is {rc['ratio_median']:.2f}x the integrated gyro (should be near 1)", rc,
            ["camera-IMU rotation wrong", "gyro scale or bias wrong", "camera-IMU time offset wrong"])
    return out
