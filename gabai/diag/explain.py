"""Follow one pixel from the model's disparity to a position in the world, naming every step and what kind of number it is.

Use this when a distance looks wrong: it shows which step could be responsible (disparity near the model limit, a floor model that
treats the pixel as ground, a missing or stale pose) instead of leaving that to guesswork.
"""
import numpy as np

from gabai.diag.trace import ESTIMATED, MODEL, q
from gabai.zones import DISPARITY_ERR_PX, floor_disparity

DISPARITY_CAP = 192.0


def explain_point(u, v, disp, f, baseline, cx, cy, cam_h, up_cam, R_wc, t_wc, pose_source=None):
    """u, v pixel; disp the model's disparity at that pixel (NaN/None = none). R_wc: camera-to-world rotation, t_wc: camera position
    in the world (both None when no pose). Returns a dict; each number is tagged (gabai.diag.trace.q)."""
    out = {"pixel": [float(u), float(v)], "warnings": []}
    ok = disp is not None and np.isfinite(disp) and disp > 0.5
    out["disparity_px"] = q(float(disp) if ok else None, MODEL, unit="px")
    if not ok:
        out.update(depth_m=q(None, ESTIMATED), camera_xyz_m=q(None, ESTIMATED), world_xyz_m=q(None, ESTIMATED), depth_uncertainty_m=None,
                   floor_test={"floor": "no disparity here", "above_floor": None})
        out["warnings"].append("no trusted disparity at this pixel (rejected, occluded or outside the model)")
        return out
    z = f * baseline / disp
    xyz = np.array([(u - cx) / f * z, (v - cy) / f * z, z])
    out["depth_m"] = q(float(z), ESTIMATED, formula="Z = f*B/d", f_px=float(f), baseline_m=float(baseline))
    out["camera_xyz_m"] = q([float(x) for x in xyz], ESTIMATED, frame="camera (x right, y down, z forward)")
    out["depth_uncertainty_m"] = float(z * z / (f * baseline) * DISPARITY_ERR_PX)
    out["depth_uncertainty_assumes"] = f"{DISPARITY_ERR_PX} px disparity error"
    if disp >= 0.92 * DISPARITY_CAP:
        out["warnings"].append(f"disparity {disp:.0f} px is at the model limit ({DISPARITY_CAP:.0f} px): the real object may be closer than "
                               f"{f * baseline / DISPARITY_CAP:.2f} m and this depth is not trustworthy")
    if R_wc is None or t_wc is None:
        out["world_xyz_m"] = q(None, ESTIMATED, wanted_source=pose_source)
        out["warnings"].append("no camera pose available: position in the world cannot be given")
    else:
        pw = np.asarray(R_wc, float) @ xyz + np.asarray(t_wc, float)
        out["world_xyz_m"] = q([float(x) for x in pw], ESTIMATED, source=pose_source or "unknown", frame="world (z up)")
    if cam_h is None:
        out["floor_test"] = {"floor": "no floor model (no camera height given)", "above_floor": None}
    else:
        fd = float(floor_disparity((int(cy * 2) + 2, int(cx * 2) + 2), f, baseline, cam_h, cx, cy, up_cam)[min(int(round(v)), int(cy * 2) + 1),
                                                                                                           min(int(round(u)), int(cx * 2) + 1)])
        if not np.isfinite(fd):
            out["floor_test"] = {"floor": "ray points above the horizon: no floor here", "above_floor": True, "floor_disparity_px": None}
        else:
            thr = fd * 1.3 + 1.0
            out["floor_test"] = {"floor": f"expected floor disparity {fd:.1f} px", "floor_disparity_px": fd, "obstacle_if_above_px": thr,
                                 "above_floor": bool(disp > thr), "rule": "obstacle only if disparity > 1.3 * floor + 1.0"}
    return out
