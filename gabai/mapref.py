"""Register the current frame to the persistent map and correct the camera pose. Stereo-only drift control.

Visual odometry only knows how the camera moved since the last keyframe, so small errors add up and a surface seen again lands beside its first copy: a
second, shifted layer in the map. The map already holds where the surface is. This module takes the points of the current frame, finds their nearest map
points and solves the rigid motion that lines them up (iterative closest point, with rejection of far matches). The corrected pose is what the frame is
placed with, and the visual odometry is told (KeyframeVO.correct) so the correction is kept instead of drifting back.

This is the scan-to-map step of the Aria idea ("closed-loop" trajectory = poses made consistent with what was seen before), without a pose graph and
without place recognition: it only corrects against the part of the map that is in view NOW, and only a little. It is not loop closure. Drift larger than
max_shift_m between two looks at the same place is not fixed.

Safety: a correction is applied only when it is small (max_shift_m, max_rot_deg), enough points matched, and the fit really improved. Otherwise the pose
is left exactly as the odometry gave it and the reason is returned. A flat or linear scene cannot fix the sliding along the surface (nothing in the data says
where along it you are): such a frame is refused (spread check), and the corrections applied in total are limited to a fraction of the distance travelled.
Without these two limits a close flat surface made the corrections feed on themselves and the map grew without bound (found 2026-10-28).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _kabsch(P, Q):
    """Rigid motion (R, t) that moves P onto Q (least squares)."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


class MapRefiner:
    def __init__(self, max_shift_m=0.06, max_rot_deg=1.0, n_points=800, iters=6, inlier_m=0.08, min_inliers=200, min_inlier_frac=0.4, every=2,
                 min_map_points=600, rebuild_frac=0.10, rebuild_every=20, min_gain=0.9, gain=0.5, min_spread=0.07, budget_frac=0.05, budget_rot_deg_per_m=3.0, seed=0):
        self.max_shift_m, self.max_rot_deg, self.n_points, self.iters, self.inlier_m = max_shift_m, max_rot_deg, n_points, iters, inlier_m
        self.min_inliers, self.min_inlier_frac, self.every = min_inliers, min_inlier_frac, every
        self.min_map_points, self.rebuild_frac, self.rebuild_every, self.min_gain, self.gain = min_map_points, rebuild_frac, rebuild_every, min_gain, gain
        self.min_spread, self.budget_frac, self.budget_rot_deg_per_m = min_spread, budget_frac, budget_rot_deg_per_m
        self.path_m, self.used_m, self.used_deg, self._last_t = 0.0, 0.0, 0.0, None
        self.rng = np.random.default_rng(seed)
        self.tree = None
        self.map_pts = np.zeros((0, 3))
        self._built_voxels, self._built_at = -1, -10 ** 9
        self.stats = {"map_solid": 0, "tried": 0, "applied": 0, "rejected": 0, "why": {}, "shift_cm": [], "rot_deg": []}

    def reset(self):
        self.tree, self.map_pts, self._built_voxels, self._built_at = None, np.zeros((0, 3)), -1, -10 ** 9
        self.used_m = self.used_deg = 0.0

    def _ensure_tree(self, vm, n):
        grown = len(vm) > self._built_voxels * (1 + self.rebuild_frac)
        if self.tree is None or grown or n - self._built_at >= self.rebuild_every:
            pts = vm.points(min_obs=3)                      # no core filter here: matches are trimmed by distance, and the filter removed 93% of a real map
            self._built_voxels, self._built_at = len(vm), n
            self.stats["map_solid"] = len(pts)
            if len(pts) < self.min_map_points:
                self.tree, self.map_pts = None, pts
                return False
            self.tree, self.map_pts = cKDTree(pts), pts
        return self.tree is not None

    def refine(self, vm, R_wc, t_wc, pts_cam, n):
        """pts_cam: this frame's points in the camera frame (before they are added to the map). Returns (R_wc, t_wc, info); info["applied"] says
        whether the pose was changed."""
        info = {"applied": False, "reason": ""}
        if self._last_t is not None:
            self.path_m += float(np.linalg.norm(np.asarray(t_wc, float) - self._last_t))      # distance travelled, the base of the correction budget
        self._last_t = np.asarray(t_wc, float).copy()
        if n % self.every:
            info["reason"] = "skipped (every %d frames)" % self.every
            return R_wc, t_wc, info
        if len(pts_cam) < self.min_inliers:
            info["reason"] = f"only {len(pts_cam)} points in this frame"
            return R_wc, t_wc, info
        if not self._ensure_tree(vm, n):
            info["reason"] = f"map too small to register against ({len(self.map_pts)} solid points, need {self.min_map_points})"
            return R_wc, t_wc, info
        self.stats["tried"] += 1
        sel = pts_cam if len(pts_cam) <= self.n_points else pts_cam[self.rng.choice(len(pts_cam), self.n_points, replace=False)]
        R0, t0 = np.asarray(R_wc, float), np.asarray(t_wc, float)
        Rd, td = np.eye(3), np.zeros(3)                                       # world-frame correction accumulated over the iterations
        rms0 = None
        for it in range(self.iters):
            P = (sel @ R0.T + t0) @ Rd.T + td
            dist, j = self.tree.query(P, distance_upper_bound=self.inlier_m)
            m = np.isfinite(dist)
            if m.sum() < self.min_inliers:
                break
            if rms0 is None:
                rms0 = float(np.sqrt((dist[m] ** 2).mean()))
            R_i, t_i = _kabsch(P[m], self.map_pts[j[m]])
            Rd, td = R_i @ Rd, R_i @ td + t_i
            if np.linalg.norm(t_i) < 5e-4 and np.arccos(np.clip((np.trace(R_i) - 1) / 2, -1, 1)) < 2e-4:
                break                                                         # converged: the last step moved less than half a millimetre
        P = (sel @ R0.T + t0) @ Rd.T + td
        dist, _ = self.tree.query(P, distance_upper_bound=self.inlier_m)
        m = np.isfinite(dist)
        n_in, frac = int(m.sum()), float(m.mean())
        info.update(inliers=n_in, inlier_frac=frac)
        if rms0 is None or n_in < self.min_inliers or frac < self.min_inlier_frac:
            return self._reject(R_wc, t_wc, info, "few_matches", f"too few points match the map ({n_in}, {frac:.0%})")
        ev = np.linalg.eigvalsh(np.cov(P[m].T))                                     # spread of the matched points along their weakest and strongest direction
        info["spread_ratio"] = float(np.sqrt(max(ev[0], 0.0) / max(ev[2], 1e-12)))
        if info["spread_ratio"] < self.min_spread:
            return self._reject(R_wc, t_wc, info, "degenerate", f"the matched points are almost flat or in a line (spread ratio {info['spread_ratio']:.3f}): "
                                "a registration cannot tell the camera's position along that surface, so it is not used")
        rms1 = float(np.sqrt((dist[m] ** 2).mean()))
        R_new, t_new = Rd @ R0, Rd @ t0 + td
        shift = float(np.linalg.norm(t_new - t0))
        rot = float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1))))
        info.update(shift_cm=shift * 100, rot_deg=rot, rms_before_cm=rms0 * 100, rms_after_cm=rms1 * 100)
        if shift > self.max_shift_m or rot > self.max_rot_deg:
            return self._reject(R_wc, t_wc, info, "too_large", f"correction too large ({shift * 100:.1f} cm, {rot:.1f} deg): not trusted")
        if rms1 > self.min_gain * rms0:
            return self._reject(R_wc, t_wc, info, "no_improvement", f"the fit did not improve ({rms0 * 100:.2f} -> {rms1 * 100:.2f} cm)")
        if self.gain < 1.0:                                                       # move part of the way: one noisy registration must not make the path jump
            R_new = (Rotation.from_rotvec(self.gain * Rotation.from_matrix(R_new @ R0.T).as_rotvec()) * Rotation.from_matrix(R0)).as_matrix()
            t_new = t0 + self.gain * (t_new - t0)
            shift = float(np.linalg.norm(t_new - t0))
        if (self.used_m + shift > 0.03 + self.budget_frac * self.path_m) or (self.used_deg + rot * self.gain > 2.0 + self.budget_rot_deg_per_m * self.path_m):
            return self._reject(R_wc, t_wc, info, "budget", "the corrections already applied add up to more than a plausible drift for the distance travelled: "
                                "the registration is probably following noise, not drift")
        self.used_m += shift
        self.used_deg += rot * self.gain
        info["applied"] = True
        self.stats["applied"] += 1
        info["shift_cm"] = shift * 100
        self.stats["shift_cm"].append(shift * 100)
        self.stats["rot_deg"].append(rot)
        return R_new, t_new, info

    def recover(self, vm, R_wc, t_wc, pts_cam, n, radii=(0.6, 0.35, 0.2, 0.12, 0.08, 0.08), max_shift_m=1.0, max_rot_deg=35.0, max_rms_m=0.04, min_frac=0.5):
        """The pose and the map disagree. Register this frame to the map from far away (match radius shrinks from 0.6 m to 0.08 m) and, only if the result is
        convincing - at least min_frac of the points land within 8 cm of the map, their rms is under max_rms_m, they are not a flat sheet - hand back the pose that
        fits. Unlike refine() it may move the pose by up to max_shift_m / max_rot_deg and it has no per-step budget: it is for the rare frame after a long
        disagreement, not for every frame. Returns (R_wc, t_wc, info); info["applied"] says whether the pose was changed."""
        info = {"applied": False, "reason": ""}
        if len(pts_cam) < self.min_inliers or not self._ensure_tree(vm, n):
            info["reason"] = "too few points in this frame, or a map too small to register against"
            return R_wc, t_wc, info
        sel = pts_cam if len(pts_cam) <= self.n_points else pts_cam[self.rng.choice(len(pts_cam), self.n_points, replace=False)]
        R0, t0 = np.asarray(R_wc, float), np.asarray(t_wc, float)
        Rd, td = np.eye(3), np.zeros(3)
        for r in radii:
            for _ in range(3):
                P = (sel @ R0.T + t0) @ Rd.T + td
                dist, j = self.tree.query(P, distance_upper_bound=r)
                m = np.isfinite(dist)
                if m.sum() < self.min_inliers:
                    info["reason"] = f"only {int(m.sum())} points within {r} m of the map"
                    return R_wc, t_wc, info
                R_i, t_i = _kabsch(P[m], self.map_pts[j[m]])
                Rd, td = R_i @ Rd, R_i @ td + t_i
        P = (sel @ R0.T + t0) @ Rd.T + td
        dist, _ = self.tree.query(P, distance_upper_bound=self.inlier_m)
        m = np.isfinite(dist)
        frac = float(m.mean())
        rms = float(np.sqrt((dist[m] ** 2).mean())) if m.any() else 1.0
        shift, rot = float(np.linalg.norm(Rd @ t0 + td - t0)), float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1))))
        info.update(inlier_frac=frac, rms_cm=rms * 100, shift_cm=shift * 100, rot_deg=rot)
        if frac < min_frac or rms > max_rms_m:
            info["reason"] = f"no convincing fit ({frac:.0%} within 8 cm, rms {rms * 100:.1f} cm)"
        elif m.sum() < self.min_inliers or np.sqrt(max(np.linalg.eigvalsh(np.cov(P[m].T))[0], 0.0) / max(np.linalg.eigvalsh(np.cov(P[m].T))[2], 1e-12)) < self.min_spread:
            info["reason"] = "the matched points are almost flat or in a line: the position along that surface is not known"
        elif shift > max_shift_m or rot > max_rot_deg:
            info["reason"] = f"correction too large ({shift * 100:.0f} cm, {rot:.0f} deg)"
        else:
            info["applied"] = True
            return Rd @ R0, Rd @ t0 + td, info
        return R_wc, t_wc, info

    def _reject(self, R, t, info, kind, why):
        info["reason"] = why
        self.stats["rejected"] += 1
        self.stats["why"][kind] = self.stats["why"].get(kind, 0) + 1
        return R, t, info
