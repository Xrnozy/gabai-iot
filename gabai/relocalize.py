"""Saved map and relocaliser (spec phase 4: "fix drift with a saved map").

Basalt's pose drifts slowly. A cleaned-up saved map of a place is used to pull the pose back: the points the rig sees now are matched
to the saved map, and the rigid transform that lines them up is the correction to apply to Basalt's poses.

Frames: both clouds are gravity-aligned (z up, from the IMU), so only a yaw angle and a shift are unknown. The search therefore tries
many yaw seeds, aligns centroids for each, refines with ICP, and keeps the best. A result is accepted only when ALL of these hold:
  fitness >= MIN_FITNESS   share of the current points that land within MAX_CORR_M of the saved map
  rmse <= MAX_RMSE_M       average distance of those matches
  not ambiguous            no other seed gives a nearly as good fit at a different pose (a symmetric room does this)
Otherwise ok is False with the reason, and the caller must keep using the uncorrected pose. The numbers are thresholds chosen for a
5 cm map; they are checked against simulated rooms, not against a real saved map of a home.
"""
import json
import time

import numpy as np

from gabai.cloud import VoxelMap

MIN_POINTS = 200
MIN_FITNESS = 0.55
MAX_RMSE_M = 0.06
AMBIGUITY_FITNESS = 0.05
AMBIGUITY_M, AMBIGUITY_DEG = 0.5, 10.0
YAW_SEEDS = 24


def save_map(path, vm, meta=None):
    meta = dict(meta or {}, voxel=vm.voxel, window_m=vm.window_m, n_voxels=len(vm))
    np.savez_compressed(path, keys=vm.keys, sum=vm.sum, w=vm.w, n=vm.n, meta=json.dumps(meta))


def load_map(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    vm = VoxelMap(meta["voxel"], meta.get("window_m"))
    vm.keys, vm.sum, vm.n = z["keys"], z["sum"], z["n"]
    vm.w = z["w"] if "w" in z.files else vm.n.astype(float)             # maps saved before weights existed have unit weights
    return vm, meta


def apply_correction(T_corr, pose):
    """pose: 4x4 camera-to-live-world. Returns the pose in the saved map's frame."""
    return np.asarray(T_corr, float) @ np.asarray(pose, float)


def _rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def _yaw_deg(T):
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def _hist(p, cell, floor_margin=0.15):
    """2D count image of everything above the floor (walls, furniture): the floor itself tells nothing about where in the room you are."""
    h = p[p[:, 2] > np.percentile(p[:, 2], 2) + floor_margin]
    if len(h) < 20:
        h = p
    ij = np.floor(h[:, :2] / cell).astype(int)
    lo = ij.min(0)
    ij -= lo
    img = np.zeros(ij.max(0) + 1)
    np.add.at(img, (ij[:, 0], ij[:, 1]), 1)
    return np.log1p(img), lo


def _seeds(map_pts, live_pts, n_yaw, cell=0.1, keep=6):
    """Start poses from a 2D correlation over yaw and shift, so ICP starts near a plausible place instead of at a centroid."""
    from scipy.signal import fftconvolve
    M, m0 = _hist(map_pts, cell)
    ones_energy = {}
    out = []
    for k in range(n_yaw):
        yaw = 2 * np.pi * k / n_yaw
        Rz = _rotz(yaw)[:3, :3]
        L, l0 = _hist(live_pts @ Rz.T, cell)
        if L.shape[0] > M.shape[0] + 40 or L.shape[1] > M.shape[1] + 40:
            continue
        corr = fftconvolve(M, L[::-1, ::-1], mode="full")
        key = L.shape
        if key not in ones_energy:
            ones_energy[key] = fftconvolve(M ** 2, np.ones(L.shape), mode="full")
        score = corr / np.sqrt(np.maximum(ones_energy[key], 1e-9) * (L ** 2).sum())
        idx = np.unravel_index(int(np.argmax(score)), score.shape)
        dx, dy = idx[0] - (L.shape[0] - 1), idx[1] - (L.shape[1] - 1)
        out.append((float(score[idx]), yaw, np.array([(m0[0] + dx - l0[0]) * cell, (m0[1] + dy - l0[1]) * cell, 0.0])))
    out.sort(key=lambda x: -x[0])
    seeds = []
    for sc, yaw, sh in out[:keep]:
        T = _rotz(yaw)
        T[:3, 3] = sh
        # z: align the floors (2nd percentile of z) so a vertical offset between the two frames does not spoil the match
        T[2, 3] = np.percentile(map_pts[:, 2], 2) - np.percentile(live_pts[:, 2], 2)
        seeds.append(T)
    return seeds


def relocalize(map_pts, live_pts, voxel=0.1, seeds=YAW_SEEDS):
    """Transform T (4x4) such that T @ live ~ map, plus the evidence for it."""
    import open3d as o3d
    t0 = time.time()
    out = {"ok": False, "T": np.eye(4), "fitness": 0.0, "rmse": None, "ambiguous": False, "reason": "", "seconds": 0.0, "n_live": len(live_pts), "n_map": len(map_pts)}
    if len(live_pts) < MIN_POINTS or len(map_pts) < MIN_POINTS:
        out["reason"] = f"too few points to match (live {len(live_pts)}, map {len(map_pts)}; need {MIN_POINTS} each)"
        return out

    def cloud(p):
        c = o3d.geometry.PointCloud()
        c.points = o3d.utility.Vector3dVector(np.asarray(p, float))
        return c.voxel_down_sample(voxel)

    src, dst = cloud(live_pts), cloud(map_pts)
    reg = o3d.pipelines.registration
    results = []
    for T0 in _seeds(np.asarray(map_pts, float), np.asarray(live_pts, float), seeds):
        results.append(reg.registration_icp(src, dst, 0.4, T0, reg.TransformationEstimationPointToPoint(), reg.ICPConvergenceCriteria(max_iteration=40)))
    results.sort(key=lambda r: (-r.fitness, r.inlier_rmse))
    est = reg.TransformationEstimationPointToPoint()
    fines = [reg.registration_icp(src, dst, 0.12, r.transformation, est, reg.ICPConvergenceCriteria(max_iteration=60)) for r in results[:6]]
    fines.sort(key=lambda r: (-r.fitness, r.inlier_rmse))
    fine = fines[0]
    T, fit, rmse = fine.transformation, fine.fitness, fine.inlier_rmse
    for r in fines[1:]:                                           # compared AFTER refinement: coarse seeds slide along floors and walls
        if r.fitness >= fit - AMBIGUITY_FITNESS:
            d = np.linalg.inv(T) @ r.transformation
            dy = abs((_yaw_deg(d) + 180) % 360 - 180)
            if np.linalg.norm(d[:3, 3]) > AMBIGUITY_M or dy > AMBIGUITY_DEG:
                out["ambiguous"] = True
                break
    out.update(T=np.asarray(T), fitness=float(fit), rmse=float(rmse), seconds=round(time.time() - t0, 2))
    if fit < MIN_FITNESS:
        out["reason"] = f"only {fit * 100:.0f}% of the current points match the saved map (need {MIN_FITNESS * 100:.0f}%): probably a different place or too little overlap"
    elif rmse > MAX_RMSE_M:
        out["reason"] = f"matches are {rmse * 100:.1f} cm off on average (limit {MAX_RMSE_M * 100:.0f} cm)"
    elif out["ambiguous"]:
        out["reason"] = "ambiguous: another position fits almost as well (a symmetric room or too little structure); not accepted"
    else:
        out["ok"] = True
        out["reason"] = "matched"
    return out
