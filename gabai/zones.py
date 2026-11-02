"""Front-zone obstacle, drop-off and unknown-ahead check on a disparity map (pixels, NaN = no trusted depth).

Assumes a level camera at height `cam_h` metres with principal row `cy`; floor disparity at row v is
B*(v-cy)/cam_h. Unknown depth is never treated as free: if the lower-centre view is mostly invalid the
result is "cant_see_ahead". Thresholds are starting values to tune on real footage.

analyze(..., evidence=True) additionally fills ZoneResult.evidence: per band how many pixels counted as obstacle, how many were
removed as floor or beyond range, where the nearest one is, how uncertain its depth is, and a written decision path. The alert
itself is computed by the same code either way.
"""
from dataclasses import dataclass, field

import numpy as np

MAX_WARN_DEPTH = 3.0
MIN_PIXEL_FRACTION = 0.015
DROP_RATIO = 0.5
DROP_FRACTION = 0.15
MIN_VALID_FRACTION = 0.4
DISPARITY_ERR_PX = 0.5          # the depth uncertainty reported in the evidence assumes this much disparity error
BANDS = (("ahead", 1 / 3, 2 / 3), ("left", 0.0, 1 / 3), ("right", 2 / 3, 1.0))  # order = tie-break


@dataclass
class ZoneResult:
    alert: tuple | None  # (key, rounded distance in m or None)
    distances: dict
    valid_fraction: float
    evidence: dict | None = field(default=None, compare=False)


def _round_half_m(z):
    return round(z * 2) / 2


def floor_disparity(shape, f, baseline, cam_h, cx, cy, up_cam=None):
    """Disparity a flat floor would have at every pixel for a camera `cam_h` metres above it. up_cam is the world's
    up direction written in camera coordinates (x right, y down, z forward); None means a level camera, up = (0, -1, 0).
    d = -f * B * (up . ray) / cam_h, defined only for rays that point below the horizon."""
    h, w = shape
    n = np.array([0.0, -1.0, 0.0]) if up_cam is None else np.asarray(up_cam, float)
    u = (np.arange(w, dtype=np.float32)[None, :] - cx) / f
    v = (np.arange(h, dtype=np.float32)[:, None] - cy) / f
    along = n[0] * u + n[1] * v + n[2]                       # up . ray, negative when the ray points down
    d = -f * baseline * along / cam_h
    return np.where(along < -4.0 / f, d, np.nan).astype(np.float32)


def analyze(disp, f, baseline, cam_h, cy, up_cam=None, cx=None, evidence=False):
    h, w = disp.shape
    cx = w / 2 if cx is None else cx
    rows = np.arange(h, dtype=np.float32)[:, None]
    if cam_h is None:  # no floor model (e.g. drone footage): obstacles by depth only, no drop detection
        floor = np.full((h, 1), np.nan, np.float32)
        below = np.broadcast_to(rows > cy + 20, disp.shape)
    else:
        floor = floor_disparity(disp.shape, f, baseline, cam_h, cx, cy, up_cam)
        below = np.isfinite(floor) & (floor > 1.5)
    valid = np.isfinite(disp) & (disp > 0.5)
    above_floor = np.where(np.isfinite(floor), disp > floor * 1.3 + 1.0, True)
    depth = np.where(valid, f * baseline / np.where(valid, disp, 1.0), np.inf)
    obstacle = valid & above_floor & (depth <= MAX_WARN_DEPTH)
    drop_px = valid & np.isfinite(floor) & below & (disp < floor * DROP_RATIO)

    found, dists = [], {}
    valid_frac = 0.0
    drop = False
    ev = None
    if evidence:
        nv = np.array([0.0, -1.0, 0.0]) if up_cam is None else np.asarray(up_cam, float)
        ev = {"shape": [h, w], "f_px": float(f), "baseline_m": float(baseline), "cam_h_m": cam_h, "cx": float(cx), "cy": float(cy),
              "pitch_down_deg": round(float(np.degrees(np.arcsin(np.clip(-nv[2] / max(np.linalg.norm(nv), 1e-9), -1, 1)))), 1),
              "floor_model": ("none (obstacles by depth only)" if cam_h is None
                              else "level camera" if up_cam is None else "tilted by IMU up-vector"),
              "valid_fraction_total": float(valid.mean()), "bands": {}, "decision_path": []}
    for order, (name, a, b) in enumerate(BANDS):
        x0, x1 = int(a * w), int(b * w)
        n = (h - 1) * (x1 - x0)
        sel = obstacle[:, x0:x1]
        n_sel = int(sel.sum())
        reached = n_sel > MIN_PIXEL_FRACTION * n
        z = None
        if reached:
            z = float(np.percentile(depth[:, x0:x1][sel], 5))
            dists[name] = z
            found.append((z, order, name))
        if ev is not None:
            vb = valid[:, x0:x1]
            ab = above_floor[:, x0:x1]
            db = depth[:, x0:x1]
            band = {"x_range": [x0, x1], "n_band_px": int(n), "n_valid_px": int(vb.sum()), "n_obstacle_px": n_sel,
                    "threshold_px": float(MIN_PIXEL_FRACTION * n), "reached_threshold": bool(reached),
                    "n_removed_as_floor": int((vb & ~ab & (db <= MAX_WARN_DEPTH)).sum()),
                    "n_beyond_range": int((vb & ab & (db > MAX_WARN_DEPTH)).sum()), "min_depth_m": None, "nearest_px": None,
                    "p5_depth_m": z, "depth_uncertainty_m": None}
            if n_sel:
                masked = np.where(sel, db, np.inf)
                r, c = np.unravel_index(int(np.argmin(masked)), masked.shape)
                band["min_depth_m"] = float(masked[r, c])
                band["nearest_px"] = [int(x0 + c), int(r)]
            if z is not None:
                band["depth_uncertainty_m"] = float(z * z / (f * baseline) * DISPARITY_ERR_PX)
                # The alert distance is the depth ALONG THE OPTICAL AXIS. On a tilted camera the distance to walk is the horizontal range, which is
                # a little shorter: record both so a difference shows up in the evidence (the alert itself still uses the optical depth).
                rr, cc = np.nonzero(sel)
                Zs = depth[:, x0:x1][rr, cc]
                P = np.stack([(cc + x0 - cx) / f * Zs, (rr - cy) / f * Zs, Zs], axis=1)
                nvec = np.array([0.0, -1.0, 0.0]) if up_cam is None else np.asarray(up_cam, float)
                band["p5_horizontal_range_m"] = float(np.percentile(np.sqrt(np.maximum((P ** 2).sum(1) - (P @ nvec) ** 2, 0.0)), 5))
            ev["bands"][name] = band
            line = (f"{name}: {n_sel} obstacle px of {n} ({100.0 * n_sel / max(n, 1):.1f}%), needs more than "
                    f"{band['threshold_px']:.0f} px ({100 * MIN_PIXEL_FRACTION:.1f}%) -> ")
            line += (f"yes, 5th percentile depth {z:.2f} m" if reached else "no")
            ev["decision_path"].append(line)
        if name == "ahead":
            band_below = below[:, x0:x1]
            valid_frac = float(valid[:, x0:x1][band_below].mean())
            drop = drop_px[:, x0:x1].sum() > DROP_FRACTION * band_below.sum()
            if ev is not None:
                ev["valid_fraction_below_horizon"] = valid_frac
                ev["drop"] = {"n_drop_px": int(drop_px[:, x0:x1].sum()), "n_floor_px": int(band_below.sum()),
                              "needs_more_than_px": float(DROP_FRACTION * band_below.sum()), "triggered": bool(drop)}
    if found:
        z, _, name = min(found)
        res = ZoneResult((name, _round_half_m(z)), dists, valid_frac)
        if ev is not None:
            ev["decision_path"].append(f"nearest band is {name} at {z:.2f} m, spoken as {_round_half_m(z):.1f} m")
    elif drop:
        res = ZoneResult(("drop_ahead", None), dists, valid_frac)
        if ev is not None:
            d = ev["drop"]
            ev["decision_path"].append(f"drop_ahead: {d['n_drop_px']} floor px read under half the expected disparity, needs more than "
                                       f"{d['needs_more_than_px']:.0f}")
    elif valid_frac < MIN_VALID_FRACTION:
        res = ZoneResult(("cant_see_ahead", None), dists, valid_frac)
        if ev is not None:
            ev["decision_path"].append(f"cant_see_ahead: valid depth below the horizon in the ahead band is {valid_frac:.2f}, "
                                       f"under the required {MIN_VALID_FRACTION:.2f}")
    else:
        res = ZoneResult(None, dists, valid_frac)
        if ev is not None:
            ev["decision_path"].append(f"no alert: no band reached the obstacle threshold, no drop, valid fraction {valid_frac:.2f}")
    res.evidence = ev
    return res
