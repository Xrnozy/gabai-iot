"""Magnetometer maths: field strength, heading, and hard/soft-iron calibration. Works on raw counts from the
HMC5883L (default gain 1.3 Ga: 0.92 milligauss per count, 1 microtesla = 10 milligauss).

Why calibrate: nearby metal and magnets (the Pi, the power bank, screws) shift and stretch the readings. If you
rotate the sensor through every direction, an ideal sensor traces a perfect sphere around zero. A shifted centre is
"hard iron", an egg-shaped sphere is "soft iron". Fixing both makes heading trustworthy.
"""
import math

import numpy as np

MG_PER_COUNT = 0.92  # gain code 0x20 (1.3 Ga)
EARTH_RANGE_UT = (25.0, 65.0)


def field_ut(v):
    """Field strength in microtesla from raw counts (x, y, z)."""
    return float(np.linalg.norm(np.asarray(v, float)) * MG_PER_COUNT / 10.0)


def heading_deg(x, y):
    """Compass heading 0..360 for a board lying flat: 0 when +x points at magnetic north, 90 when +y does."""
    return math.degrees(math.atan2(y, x)) % 360.0


def sweep_deg(headings):
    """Total angle swept by a sequence of headings, following the shortest step each time (handles 360 -> 0)."""
    total = 0.0
    for a, b in zip(headings[:-1], headings[1:]):
        total += abs((b - a + 180.0) % 360.0 - 180.0)
    return total


def fit_sphere(points):
    """Least-squares sphere through points (n, 3): returns (centre, radius)."""
    p = np.asarray(points, float)
    A = np.column_stack([2 * p, np.ones(len(p))])
    b = (p ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    return c, float(math.sqrt(max(sol[3] + c @ c, 0.0)))


def coverage(points, centre):
    """Fraction of the 8 octants around the centre that have at least a few samples (1.0 = every direction visited)."""
    d = np.asarray(points, float) - np.asarray(centre, float)
    octant = (d[:, 0] > 0).astype(int) * 4 + (d[:, 1] > 0).astype(int) * 2 + (d[:, 2] > 0).astype(int)
    return float(sum(np.sum(octant == k) >= 5 for k in range(8)) / 8)


def calibrate(points):
    """Hard-iron offset (sphere centre), soft-iron per-axis scale, and quality of the corrected data.

    corrected = (raw - offset) * scale.  quality = relative spread of the corrected field strength (0 = perfect
    sphere). Rule of thumb: under 0.03 good, under 0.06 usable, above that recapture with more rotation."""
    p = np.asarray(points, float)
    c, _ = fit_sphere(p)
    lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    half = (hi - lo) / 2.0
    scale = half.mean() / np.maximum(half, 1e-9)
    fixed = (p - c) * scale
    # refit the centre on the rescaled data for a second, tighter pass
    c2, _ = fit_sphere(fixed)
    c = c + c2 / scale
    fixed = (p - c) * scale
    mag = np.linalg.norm(fixed, axis=1)
    return {"offset": c, "scale": scale, "quality": float(np.std(mag) / np.mean(mag)),
            "radius_counts": float(np.mean(mag)), "coverage": coverage(p, c)}


# ---- full soft-iron calibration, applying a calibration, tilt-compensated heading, and a gate for disturbed readings --------------------------------------
def calibrate_ellipsoid(points):
    """Hard-iron offset AND a full 3x3 soft-iron matrix W from readings taken while turning the sensor through every direction.

    The readings of a perfect compass lie on a sphere. Nearby iron stretches and TILTS that sphere into an ellipsoid; the per-axis scale of `calibrate`
    can undo a stretch along the sensor axes but not a tilt. This fits the general ellipsoid (x - c)^T N (x - c) = 1 by least squares and returns
    corrected = W @ (raw - offset) with W = r * sqrt(N), r chosen so the corrected field has the mean strength of the ellipsoid. Falls back to the sphere
    fit when the data do not give a valid ellipsoid (too few directions): then `method` says 'sphere' and W is diagonal.

    Not done: it does not tell a distant disturbance from the sensor's own iron, and it assumes the disturbance is fixed to the sensor (hard and soft iron
    that move with the board). Anything that moves relative to the sensor (a cable, a magnet that is not fixed) is not calibratable."""
    p = np.asarray(points, float)
    base = calibrate(p)
    cov = coverage(p, base["offset"])
    out = dict(base, W=np.diag(base["scale"]), method="sphere", coverage=cov)
    sv = np.linalg.svd(p - p.mean(0), compute_uv=False)
    spread = float(sv[-1] / sv[0])                                 # 1 for a full sphere, near 0 for readings that lie in a plane or on a line (turned flat only)
    if spread < 0.4:
        out["coverage"] = min(cov, 0.5)                            # the octant count is fooled by noise around a flat circle: say plainly that the directions are missing
        return out
    if len(p) < 100 or cov < 0.75:
        return out
    x, y, z = p.T
    D = np.column_stack([x * x, y * y, z * z, 2 * y * z, 2 * x * z, 2 * x * y, 2 * x, 2 * y, 2 * z])
    v, *_ = np.linalg.lstsq(D, np.ones(len(p)), rcond=None)
    M = np.array([[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]])
    try:
        c = -np.linalg.solve(M, v[6:9])
    except np.linalg.LinAlgError:
        return out
    if coverage(p, c) < 0.75:
        return out                                                 # the fitted centre is not surrounded by readings: the data do not pin the ellipsoid down
    k = 1.0 + c @ M @ c                                            # negative when the origin lies outside the ellipsoid (a strong hard-iron offset): M is then negative too
    if abs(k) < 1e-12:
        return out
    N = M / k
    w, V = np.linalg.eigh(N)
    if np.any(w <= 1e-12):
        return out
    r = float(np.prod(w) ** (-1.0 / 6.0))
    W = r * (V * np.sqrt(w)) @ V.T
    fixed = (p - c) @ W.T
    mag = np.linalg.norm(fixed, axis=1)
    quality = float(np.std(mag) / np.mean(mag))
    if quality >= base["quality"]:
        return out                                                 # the plain fit was as good or better: keep it
    return {"offset": c, "scale": np.ones(3), "W": W, "quality": quality, "radius_counts": float(np.mean(mag)), "coverage": cov, "method": "ellipsoid"}


def apply_cal(cal, raw):
    """Corrected field(s) for raw counts (3,) or (n, 3), from a calibration dict (old files with only offset and scale work too)."""
    r = np.asarray(raw, float) - np.asarray(cal["offset"], float)
    W = cal.get("W")
    return r @ np.asarray(W, float).T if W is not None else r * np.asarray(cal["scale"], float)


def tilt_heading_deg(m, up, forward):
    """Magnetic heading of the body axis `forward`, in the same convention as gabai.orient.HeadingTracker (world x = magnetic north, z = up, so the angle
    grows towards west): 0 when `forward` points north. m = corrected field and up = accelerometer direction (up), both in the sensor's axes. Returns
    (degrees in [-180, 180), horizontal fraction of the axis) or (None, fraction) when the axis is within about 20 degrees of vertical, or the field is
    along the vertical (at a magnetic pole)."""
    u = np.asarray(up, float)
    u = u / np.linalg.norm(u)
    m = np.asarray(m, float)
    f = np.asarray(forward, float)
    h = m - (m @ u) * u
    nh = np.linalg.norm(h)
    fh = f - (f @ u) * u
    frac = float(np.linalg.norm(fh))
    if nh < 1e-9 or frac < math.sin(math.radians(20)):
        return None, frac
    n = h / nh
    west = np.cross(u, n)
    return math.degrees(math.atan2(fh @ west, fh @ n)), frac


class MagGate:
    """Decides whether a compass reading can be trusted for the heading.

    A reading is rejected when its strength is far from the calibrated one (iron or a magnet came near, or the calibration no longer fits) or when the angle
    between the field and gravity (the dip) is far from what this place has been giving. Both are properties of the Earth's field that do not change when you
    turn the board, so a change in them means the reading is disturbed, not that you turned. If readings stay rejected for `reset_s` seconds the new place
    is accepted as normal (you carried the rig somewhere else), so a lasting change cannot lock the compass out for good."""

    def __init__(self, radius, strength_tol=0.25, dip_tol_deg=15.0, learn=60, reset_s=5.0, recover=5):
        self.radius, self.strength_tol, self.dip_tol, self.learn, self.reset_s, self.recover = float(radius), strength_tol, dip_tol_deg, learn, reset_s, recover
        self.dips = []
        self.rejected_since = None
        self.accepted = self.rejected = 0
        self.reason = ""
        self.ok, self.good_run = True, 0

    def check(self, t, m, up, trusted=True):
        """True when the reading may pull the heading. `trusted` says the accelerometer is measuring gravity alone right now (length 1 g, not turning hard):
        the angle to gravity is then compared with this place's usual one and added to it. When it is not (the hand is shaking the board), that angle says
        nothing about the compass (measured on your calibration run: 14 to 16 degrees of spread while it was waved about, against 1 when still), so only the
        field strength is tested and the angle is neither judged nor learned. After a reject, `recover` good readings in a row are needed before the
        compass counts again, so a reading that hovers at the limit does not switch the compass on and off several times a second."""
        m = np.asarray(m, float)
        n = float(np.linalg.norm(m))
        u = np.asarray(up, float)
        u = u / np.linalg.norm(u)
        dip = math.degrees(math.acos(max(-1.0, min(1.0, float(m @ u) / max(n, 1e-9)))))
        good, self.reason = True, ""
        if abs(n / self.radius - 1.0) > self.strength_tol:
            good, self.reason = False, f"field strength {n / self.radius:.2f} of the calibrated value"
        elif trusted and len(self.dips) >= self.learn and abs(dip - float(np.median(self.dips))) > self.dip_tol:
            good, self.reason = False, f"angle to gravity {dip:.0f} deg, this place has been giving {np.median(self.dips):.0f}"
        if good:
            self.good_run += 1
            self.rejected_since = None
            if trusted:
                self.dips.append(dip)
                del self.dips[:-300]
            if not self.ok and self.good_run >= self.recover:
                self.ok = True
            elif not self.ok:
                self.reason = f"{self.good_run} of {self.recover} good readings since the last reject"
        else:
            self.ok, self.good_run = False, 0
            self.rejected_since = t if self.rejected_since is None else self.rejected_since
            if t - self.rejected_since > self.reset_s:
                self.dips, self.rejected_since = [], None           # a lasting change: take it as the new normal
        if self.ok:
            self.accepted += 1
        else:
            self.rejected += 1
        return self.ok


def load_mag_cfg(cal_path="data/mag_cal.json", sync_path="data/mag_sync.json"):
    """Compass calibration + delay + axis map for gabai.fusion.SensorFusion, or None when the compass has not been calibrated and synced
    (python -m gabai.tools.sync_sensors makes both files)."""
    import json
    import os
    if not (os.path.exists(cal_path) and os.path.exists(sync_path)):
        return None
    c, d = json.load(open(cal_path)), json.load(open(sync_path))
    cfg = {"offset": np.array(c["offset"], float), "scale": np.array(c["scale"], float), "R_im": np.array(d["R_im"], float), "delay_s": float(d["delay_s"]),
           "radius_counts": float(c.get("radius_counts") or 0) or None}
    if c.get("W") is not None:
        cfg["W"] = np.array(c["W"], float)
    return cfg


def load_accel_cal(path="data/accel_calib.json"):
    """{bias, scale} (m/s^2, per axis) from gabai.tools.calibrate_accel, or None."""
    import json
    import os
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return {"bias": np.array(d["bias"], float), "scale": np.array(d["scale"], float)}
