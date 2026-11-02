"""Diagnostics that make tracking and mapping problems visible. Each returns a Check (gabai.diag.checks) or a plain dict; none invents a number.

  map_alignment      the map seen from the current pose, against the depth the camera measures now. If the pose and the map agree, a point on a wall in
                     the map projects to the same depth the stereo sees. A constant offset means the pose is wrong now; a growing offset over a run means drift.
                     Occluded or moved points (more than OCCLUSION_M apart) are not counted as disagreement, but how many were skipped is reported.
  check_pose_stability   step sizes and rotation steps of the last poses
  check_tracking_quality inlier count of the visual odometry
  check_rectification    vertical residual of matched features between left and right (should be about 0 for a rectified pair)
  check_stereo_sync      left/right capture offset against the motion it implies; unknown when the offset was never measured
  stereo_geometry_report f, B, whether depth is metric, minimum range, depth error from half a pixel of disparity error
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.diag import checks as C
from gabai.diag.checks import Check

OCCLUSION_M = 0.5
ALIGN_PASS_M, ALIGN_FAIL_M = 0.05, 0.12
MIN_ALIGN_POINTS = 30
MAX_ROT_STEP_DEG = 45.0
RECT_PASS_PX, RECT_FAIL_PX = 1.0, 2.0


def map_alignment(map_pts, R_wc, t_wc, disp, f, B, cx, cy, min_z=0.4, max_z=6.0, max_pts=6000):
    pts = np.asarray(map_pts, float)
    if len(pts) == 0:
        return None
    pc = (pts - np.asarray(t_wc, float)) @ np.asarray(R_wc, float)                   # world -> camera
    z = pc[:, 2]
    ok = (z > min_z) & (z < max_z)
    pc, z = pc[ok], z[ok]
    if len(pc) == 0:
        return None
    u = np.round(pc[:, 0] / z * f + cx).astype(int)
    v = np.round(pc[:, 1] / z * f + cy).astype(int)
    h, w = disp.shape
    ins = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[ins], v[ins], z[ins]
    if len(u) > max_pts:
        sel = np.random.default_rng(0).choice(len(u), max_pts, replace=False)
        u, v, z = u[sel], v[sel], z[sel]
    d = disp[v, u]
    valid = np.isfinite(d) & (d > 0.5)
    if valid.sum() < MIN_ALIGN_POINTS:
        return None
    zd = f * B / d[valid]
    dz = z[valid] - zd
    close = np.abs(dz) < OCCLUSION_M
    if close.sum() < MIN_ALIGN_POINTS:
        return {"n": int(close.sum()), "n_skipped_occluded_or_moved": int((~close).sum()), "median_abs_m": float(np.median(np.abs(dz))),
                "median_signed_m": float(np.median(dz)), "within_10cm": 0.0, "note": "almost every point is more than 0.5 m from the measured depth"}
    dz = dz[close]
    return {"n": int(close.sum()), "n_skipped_occluded_or_moved": int((~close).sum()), "median_abs_m": float(np.median(np.abs(dz))),
            "median_signed_m": float(np.median(dz)), "within_10cm": float((np.abs(dz) < 0.10).mean()),
            "note": "signed: map depth minus measured depth; positive = the map is farther than what the camera sees now"}


def check_alignment(r):
    if r is None:
        return Check("map_alignment", "unknown", "map", detail="too few map points in view with measured depth to compare, so alignment was not checked")
    m = r["median_abs_m"]
    st = "pass" if m <= ALIGN_PASS_M else ("fail" if m >= ALIGN_FAIL_M else "suspect")
    return Check("map_alignment", st, "map", {"median_abs_m": round(m, 3), "median_signed_m": round(r["median_signed_m"], 3), "n": r["n"]}, {"pass_m": ALIGN_PASS_M, "fail_m": ALIGN_FAIL_M},
                 f"map and current depth differ by {m * 100:.1f} cm (median, {r['n']} points; signed {r['median_signed_m'] * 100:+.1f} cm)",
                 "the pose and the map disagree: either the pose has drifted/jumped, the camera-IMU or stereo calibration is wrong, or the scene moved. "
                 "Compare with check_pose_stability and tracking quality; a constant offset points at the pose now, a growing one at drift" if st != "pass" else "")


def check_pose_stability(history):
    """history: [(t_s, position (3), Rotation)] oldest first. Looks at the newest step."""
    if len(history) < 2:
        return Check("pose_stability", "unknown", "pose", detail="fewer than two poses so far")
    (t0, p0, r0), (t1, p1, r1) = history[-2], history[-1]
    step = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
    dt = max(t1 - t0, 1e-6)
    rot = float(np.degrees((r0.inv() * r1).magnitude()))
    pos = C.check_pose_step(step, dt)
    if pos.status == "fail":
        pos.name, pos.detail = "pose_stability", pos.detail + f"; rotation {rot:.1f} deg"
        return pos
    if rot > MAX_ROT_STEP_DEG:
        return Check("pose_stability", "fail", "pose", {"rotation_deg": round(rot, 1), "step_m": round(step, 3)}, MAX_ROT_STEP_DEG,
                     f"rotation of {rot:.0f} deg in {dt * 1000:.0f} ms", "the orientation jumped: lost tracking, or a wrong camera-IMU rotation")
    st = "suspect" if pos.status == "suspect" else "pass"
    return Check("pose_stability", st, "pose", {"rotation_deg": round(rot, 1), "step_m": round(step, 3)}, None, f"step {step:.3f} m, rotation {rot:.1f} deg in {dt * 1000:.0f} ms",
                 pos.hint)


def check_tracking_quality(info):
    n, tr = info.get("inliers"), info.get("tracked")
    if n is None:
        return Check("tracking_quality", "unknown", "pose", detail="this pose provider reports no inlier count")
    st = "pass" if n >= 40 else ("suspect" if n >= 15 else "fail")
    return Check("tracking_quality", st, "pose", {"inliers": int(n), "tracked": tr}, {"suspect_below": 40, "fail_below": 15},
                 f"{n} features agree on the camera motion ({tr} tracked)",
                 "few features: low texture, motion blur, or too little overlap between frames; the pose is unreliable" if st != "pass" else "")


def check_rectification(vertical_px):
    if vertical_px is None or not np.isfinite(vertical_px):
        return Check("rectification", "unknown", "stereo", detail="no vertical residual (too few matched features)")
    st = "pass" if vertical_px <= RECT_PASS_PX else ("fail" if vertical_px >= RECT_FAIL_PX else "suspect")
    return Check("rectification", st, "stereo", round(float(vertical_px), 2), {"pass_px": RECT_PASS_PX, "fail_px": RECT_FAIL_PX},
                 f"matched points differ by {vertical_px:.2f} px between rows of the left and right pictures",
                 "rows are not aligned: stereo matching assumes the same row. Recalibrate (gabai.tools.calibrate_stereo), or for plain video check the row-alignment fix" if st != "pass" else "")


def check_stereo_sync(sync, turn_dps, f_px=881.0):
    dt = sync.get("pair_dt_ms")
    if dt is None:
        return Check("stereo_sync", "unknown", "stereo", detail=f"left/right capture offset was never measured ({sync.get('note', 'no note')})")
    return C.check_pair_sync(dt, turn_dps, f_px)


def stereo_geometry_report(f, B, metric, source, max_disp=192.0, half_px=0.5):
    zs = (1.0, 2.0, 3.0, 5.0)
    return {"f_px": float(f), "baseline_m": float(B), "source": source, "metric": bool(metric), "min_range_m": float(f * B / max_disp),
            "depth_error_m_at": {z: z * z / (f * B) * half_px for z in zs},
            "note": ("calibrated: distances are metric if the calibration is right (check with target_test against a tape measure)" if metric else
                     "f and B are ASSUMED: depth is not metric, only proportional; the scale of any path or map is unknown") +
                    f". Depth error from {half_px} px of disparity error: " + ", ".join(f"{z * z / (f * B) * half_px:.2f} m at {z:g} m" for z in zs)}
