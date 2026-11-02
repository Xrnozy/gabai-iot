"""Compass calibration maths for gabai.tools.calibrate_sensors: coverage of directions, the fit, the checks, and the compass-to-IMU axis map.
Everything here takes RAW compass counts and returns numbers; nothing is applied to the readings (applying is gabai.mag.apply_cal, in one place).
"""
import math

import numpy as np

from gabai.mag import MG_PER_COUNT, apply_cal, calibrate_ellipsoid, fit_sphere
from gabai.mag_sync import rotations_24

# Reference mapping, from the physical mounting of this device: IMU X front/back (the camera axis, so gravity is on X when it lies face-down), Y left/right, Z up/down;
# compass X left/right, Y front/back, Z up/down. Device frame (the camera): x right (left/right), y down (up/down), z forward (front/back). The permutations
# (signs are measured, never assumed):
REF_PERM_DEV_FROM_MAG = np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]])
REF_PERM_DEV_FROM_IMU = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])
REF_PERM_IMU_FROM_MAG = REF_PERM_DEV_FROM_IMU.T @ REF_PERM_DEV_FROM_MAG
REFERENCE_TEXT = {"compass": {"x": "left/right", "y": "front/back", "z": "up/down"}, "imu": {"x": "front/back", "y": "left/right", "z": "up/down"},
                  "device": {"x": "left/right (right)", "y": "up/down (down)", "z": "front/back (forward)"}}
MIN_PER_BIN = 8
BIN_NAMES = {}
for _k in ((a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)):
    BIN_NAMES[_k] = "".join(("+" if v > 0 else "-") + "XYZ"[i] for i, v in enumerate(_k) if v)


def _bin_keys(vecs, thr=0.38):
    v = np.asarray(vecs, float)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return [tuple(int(x) for x in ((r > thr).astype(int) - (r < -thr).astype(int))) for r in v]


def provisional_centre(points):
    p = np.asarray(points, float)
    if len(p) < 40:
        return p.mean(axis=0) if len(p) else np.zeros(3)
    return fit_sphere(p)[0]


def direction_coverage(points, centre=None, min_per_bin=MIN_PER_BIN):
    """Fraction of the 26 directions (6 faces, 12 edges, 8 corners of a cube) around the centre that have at least `min_per_bin` readings, with the missing ones named
    in the compass's own axes (e.g. '+X-Z'). 1.0 means the field has been seen from every side."""
    p = np.asarray(points, float)
    if len(p) < 10:
        return 0.0, list(BIN_NAMES.values())
    c = provisional_centre(p) if centre is None else centre
    d = p - c
    keep = np.linalg.norm(d, axis=1) > 1e-9
    counts = {k: 0 for k in BIN_NAMES}
    for k in _bin_keys(d[keep]):
        counts[k] += 1
    have = [k for k, n in counts.items() if n >= min_per_bin]
    return len(have) / 26.0, [BIN_NAMES[k] for k in BIN_NAMES if counts[k] < min_per_bin]


def pair_up(t_mag, t_imu, accel, smooth_s=0.4):
    """Unit 'up' (smoothed accelerometer direction) at each compass time, for the dip-constancy test. accel (n,3) in g at times t_imu."""
    t_imu, a = np.asarray(t_imu, float), np.asarray(accel, float)
    k = max(int(smooth_s / max(np.median(np.diff(t_imu)), 1e-3)), 1)
    ker = np.ones(k) / k
    sm = np.stack([np.convolve(a[:, i], ker, mode="same") for i in range(3)], axis=1)
    u = np.stack([np.interp(np.asarray(t_mag, float), t_imu, sm[:, i]) for i in range(3)], axis=1)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def estimate_axis_map(m_corrected, up, require_margin=2.0):
    """Which of the 24 axis-aligned rotations P (v_imu = P @ v_compass) makes the angle between the field and 'up' constant over all the orientations visited
    (the Earth's field keeps a fixed angle to gravity whatever way the sensor is turned). Returns (P, score, next_score, ok); score = variance of that angle in deg^2.
    Needs readings from many orientations: a single turn about one axis cannot tell axes apart."""
    m, u = np.asarray(m_corrected, float), np.asarray(up, float)
    sel = np.linalg.norm(m, axis=1) > 1e-6
    m, u = m[sel], u[sel]
    scores = []
    for P in rotations_24():
        v = m @ P.T
        ang = np.degrees(np.arccos(np.clip(np.sum(v * u, axis=1) / np.linalg.norm(v, axis=1), -1, 1)))
        scores.append((float(np.var(ang)), P, float(np.mean(ang))))
    scores.sort(key=lambda x: x[0])
    best, second = scores[0], scores[1]
    return best[1], best[0], second[0], bool(second[0] > require_margin * best[0] + 2.0), best[2]


def same_permutation(R, ref):
    return bool(np.array_equal(np.abs(np.round(np.asarray(R, float))), np.abs(np.round(np.asarray(ref, float)))))


def fit(points):
    """The calibration (offset, W) from raw readings, via gabai.mag.calibrate_ellipsoid; adds the 26-direction coverage."""
    p = np.asarray(points, float)
    cal = calibrate_ellipsoid(p)
    cov26, missing = direction_coverage(p, cal["offset"])
    cal = dict(cal, coverage26=cov26, missing_directions=missing)
    return cal


def checks(cal, points, up=None, P=None):
    """List of (name, status, text), status in pass / warn / fail. A 'fail' means the result must not be saved."""
    out = []
    p = np.asarray(points, float)
    c26 = cal["coverage26"]
    out.append(("coverage of directions", "pass" if c26 >= 0.8 else ("warn" if c26 >= 0.6 else "fail"),
                f"{c26 * 100:.0f}% of 26 directions covered" + ("" if c26 >= 0.8 else "; missing " + " ".join(cal["missing_directions"][:8]) + ": turn the device through those orientations")))
    q = cal["quality"]
    out.append(("sphere quality", "pass" if q < 0.03 else ("warn" if q < 0.06 else "fail"),
                f"corrected field strength varies by {q * 100:.1f}% ({'good' if q < 0.03 else 'usable' if q < 0.06 else 'too poor: more rotation, or something magnetic moves relative to the sensor'})"))
    ut = cal["radius_counts"] * MG_PER_COUNT / 10.0
    out.append(("field strength", "pass" if 20 <= ut <= 70 else "warn",
                f"{ut:.0f} uT after calibration (Earth: 25-65 uT)" + ("" if 20 <= ut <= 70 else "; check soft iron, magnets or current near the sensor")))
    sv = np.linalg.svd(np.asarray(cal["W"], float), compute_uv=False)
    ratio = float(sv.max() / sv.min())
    out.append(("soft iron", "pass" if ratio < 1.3 else "warn", f"the correction stretches the field by up to {ratio:.2f}x" + ("" if ratio < 1.3 else ": strong soft iron near the sensor")))
    off_ut = float(np.linalg.norm(cal["offset"]) * MG_PER_COUNT / 10.0)
    out.append(("hard iron", "pass" if off_ut < 2 * ut else "warn", f"offset {off_ut:.0f} uT" + ("" if off_ut < 2 * ut else ": a magnet or iron close to the sensor dominates the Earth's field")))
    corr = np.linalg.norm(apply_cal(cal, p), axis=1)
    spikes = float(np.mean(np.abs(corr / cal["radius_counts"] - 1) > 0.3))
    out.append(("abnormal readings", "pass" if spikes < 0.02 else "warn", f"{spikes * 100:.1f}% of readings are more than 30% off the corrected field strength"))
    out.append(("fit method", "pass" if cal["method"] == "ellipsoid" else "warn",
                "full soft-iron fit" if cal["method"] == "ellipsoid" else "sphere only (hard iron): the directions covered were not enough for the soft-iron fit"))
    if up is not None and P is not None:
        v = apply_cal(cal, p) @ np.asarray(P, float).T
        ang = np.degrees(np.arccos(np.clip(np.sum(v * up, axis=1) / np.linalg.norm(v, axis=1), -1, 1)))
        sd = float(np.std(ang))
        out.append(("angle to gravity stays constant", "pass" if sd < 6 else ("warn" if sd < 12 else "fail"),
                    f"spread {sd:.1f} deg (dip about {np.mean(ang) - 90:+.0f} deg); {'consistent' if sd < 6 else 'the compass-to-IMU axis map or the offset is wrong'}"))
    return out


def can_save(check_list):
    return not any(s == "fail" for _, s, _ in check_list)
