"""Compare the two stereo visual odometries on a synthetic scene with a KNOWN camera path.

    py -m gabai.tools.vo_bench [--frames 120] [--seeds 5]

The scene is a textured wall seen by a moving stereo camera: the picture and the disparity are rendered exactly from the true pose, then noise is added
to the disparity (a stereo network is not exact) and to the picture. The estimate is compared with the truth in the first camera's frame, so the world
convention of the estimators does not matter. It measures the drift of the method on a flat textured scene with a good stereo depth. It does not measure
real-world accuracy: real scenes have depth that is wrong in places, moving things, blur and changes of light.
"""
import argparse
import time

import cv2
import numpy as np

from gabai.kfvo import KeyframeVO
from gabai.sim import look_pose
from gabai.tools.video_cloud_view import StereoVO

F, B, W, H = 600.0, 0.065, 640, 480
PLANE_Z, TEX_PX_PER_M = 2.5, 300.0


def texture(seed=0, w=2400, h=1800):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for s, a in ((3, 1.0), (7, 1.2), (15, 1.0), (40, 0.8)):
        img += a * cv2.resize(rng.standard_normal((h // s + 2, w // s + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def render(tex, R, t, noise_disp=0.3, noise_img=3.0, rng=None, persistent=0.7):
    """Picture and disparity of the wall z = PLANE_Z (first camera frame) as seen from the camera pose R, t (camera in the first camera's frame)."""
    rng = rng or np.random.default_rng(0)
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    ray = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], -1)
    d = ray @ R.T
    lam = (PLANE_Z - t[2]) / d[..., 2]
    hit = t + lam[..., None] * d
    mx = (hit[..., 0] * TEX_PX_PER_M + tex.shape[1] / 2).astype(np.float32)
    my = (hit[..., 1] * TEX_PX_PER_M + tex.shape[0] / 2).astype(np.float32)
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img = np.clip(img.astype(np.float32) + rng.normal(0, noise_img, img.shape), 0, 255).astype(np.uint8)
    n = cv2.GaussianBlur(rng.normal(0, 1, lam.shape).astype(np.float32), (0, 0), 2.0)             # stereo errors are correlated between neighbouring pixels
    n /= max(float(n.std()), 1e-6)
    fixed = cv2.remap(_fixed_noise(tex.shape), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)   # ...and a part of the error sticks to the surface: the same wrong depth every frame
    fixed /= max(float(fixed.std()), 1e-6)
    disp = (F * B / lam).astype(np.float32) + noise_disp * (persistent * fixed + np.sqrt(1 - persistent ** 2) * n)
    return img, disp


_FIXED = {}


def _fixed_noise(shape):
    if shape not in _FIXED:
        _FIXED[shape] = cv2.GaussianBlur(np.random.default_rng(99).normal(0, 1, shape).astype(np.float32), (0, 0), 2.0)
    return _FIXED[shape]


def truth_path(n):
    """A walk past the wall: sideways 0.9 m, towards it 0.5 m, a little yaw and pitch, with small bumps."""
    ts, Rs = [], []
    for i in range(n):
        s = i / max(n - 1, 1)
        yaw, pitch = np.radians(18 * np.sin(np.pi * s)), np.radians(4 * np.sin(2 * np.pi * s))
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Rs.append(Ry @ Rx)
        ts.append(np.array([0.9 * s, 0.03 * np.sin(8 * np.pi * s), 0.5 * s]))
    return Rs, ts


def truth_path_fast(n, seed=0):
    """Rough handheld motion: a swinging yaw (up to about 6 degrees per frame), a pitch wobble, random shake every frame, and one sudden 8 degree jerk."""
    r = np.random.default_rng(seed + 1000)
    ts, Rs = [], []
    for i in range(n):
        s = i / max(n - 1, 1)
        yaw = np.radians(22 * np.sin(2 * np.pi * 3 * s) + (8 if i >= n // 2 else 0) + r.normal(0, 0.4))
        pitch = np.radians(6 * np.sin(2 * np.pi * 5 * s) + r.normal(0, 0.4))
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        Rs.append(np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]))
        ts.append(np.array([0.5 * s, 0.0, 0.3 * s]) + r.normal(0, 0.004, 3))
    return Rs, ts


def motion_blur(img, flow_px):
    """Blur along the horizontal image motion with a box of half the per-frame motion (an exposure of half the frame time), capped at 25 px."""
    L = int(min(25, round(0.5 * abs(flow_px))))
    return img if L < 2 else cv2.filter2D(img, -1, np.full((1, L), 1.0 / L, np.float32))


def run_vo(vo, tex, n, seed=0, path=None, blur=False, prior_fn=None, dropout=None):
    """prior_fn(R_prev_truth, R_cur_truth) -> rotation between the two frames handed to the odometry as dR_prior (a stand-in for a gyro, tests only).
    dropout=(first_frame, count): those pictures are replaced by a flat grey one."""
    rng = np.random.default_rng(seed)
    Rs, ts = (path or truth_path)(n) if path is None or path is truth_path else path(n, seed)
    R0w, t0w = look_pose(np.zeros(3))
    est, rot_err, ms, lost = [], [], [], 0
    trusted = np.ones((H, W), bool)
    for i, (R, t) in enumerate(zip(Rs, ts)):
        img, disp = render(tex, R, t, rng=rng)
        if blur and i > 0:
            ang = np.arccos(np.clip((np.trace(Rs[i - 1].T @ R) - 1) / 2, -1, 1))
            img = motion_blur(img, F * ang)
        if dropout and dropout[0] <= i < dropout[0] + dropout[1]:
            img = np.full_like(img, 90)                                    # the camera is covered or the picture is lost for a few frames while it keeps moving
        kw = {} if prior_fn is None or i == 0 else {"dR_prior": prior_fn(Rs[i - 1], R)}
        t0 = time.perf_counter()
        Rw, tw, info = vo.step(img, disp, trusted, **kw)
        ms.append((time.perf_counter() - t0) * 1000)
        lost += not info["ok"]
        est.append(R0w.T @ (tw - t0w))
        rot_err.append(np.degrees(np.arccos(np.clip((np.trace((R0w.T @ Rw).T @ R) - 1) / 2, -1, 1))))
    err = np.linalg.norm(np.array(est) - np.array(ts), axis=1)
    return {"final_err_cm": 100 * float(err[-1]), "rms_err_cm": 100 * float(np.sqrt((err ** 2).mean())), "lost": int(lost), "ms_median": float(np.median(ms)),
            "rot_rms_deg": float(np.sqrt(np.mean(np.square(rot_err)))), "rot_final_deg": float(rot_err[-1]),
            "path_m": float(np.linalg.norm(np.diff(np.array(ts), axis=0), axis=1).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    tex = texture()
    for name, make in (("frame-to-frame (StereoVO)", lambda: _vo(StereoVO)), ("keyframe (KeyframeVO)", lambda: _vo(KeyframeVO))):
        rs = [run_vo(make(), tex, a.frames, s) for s in range(a.seeds)]
        print(f"{name:28s} final error {np.mean([r['final_err_cm'] for r in rs]):6.2f} cm   rms {np.mean([r['rms_err_cm'] for r in rs]):6.2f} cm   "
              f"lost {sum(r['lost'] for r in rs)}   {np.mean([r['ms_median'] for r in rs]):5.1f} ms/frame   (true path {rs[0]['path_m']:.2f} m, {a.seeds} seeds)")


def _vo(cls):
    vo = cls(F, F, W / 2, H / 2)
    vo.set_baseline(B)
    return vo



# ---- a room, to test the map: walk in, then back to where we started, so the same surfaces are seen twice ----------------------------------------------
PLANES = [(2, 3.0), (0, -1.5), (0, 1.5), (1, 1.2)]            # (axis, position) in the first camera frame: back wall, left wall, right wall, floor (y is down)
EXTENT = 2.4


def render_room(tex, R, t, noise_disp=0.3, noise_img=3.0, rng=None, persistent=0.7):
    rng = rng or np.random.default_rng(0)
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], -1) @ R.T
    best, mx, my = np.full(u.shape, np.inf), np.zeros(u.shape, np.float32), np.zeros(u.shape, np.float32)
    for k, (axis, c) in enumerate(PLANES):
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (c - t[axis]) / d[..., axis]
        hit = t + lam[..., None] * d
        a, b = [i for i in range(3) if i != axis]
        ok = (lam > 0.2) & (lam < best) & (np.abs(hit[..., a]) < EXTENT + 1) & (np.abs(hit[..., b]) < EXTENT + 1)
        best = np.where(ok, lam, best)
        mx = np.where(ok, (hit[..., a] * TEX_PX_PER_M + tex.shape[1] / 2 + 200 * k) % tex.shape[1], mx).astype(np.float32)
        my = np.where(ok, (hit[..., b] * TEX_PX_PER_M + tex.shape[0] / 2) % tex.shape[0], my).astype(np.float32)
    seen = np.isfinite(best)
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img = np.clip(img.astype(np.float32) + rng.normal(0, noise_img, img.shape), 0, 255).astype(np.uint8)
    n = cv2.GaussianBlur(rng.normal(0, 1, u.shape).astype(np.float32), (0, 0), 2.0)
    n /= max(float(n.std()), 1e-6)
    fixed = cv2.remap(_fixed_noise(tex.shape), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fixed /= max(float(fixed.std()), 1e-6)
    disp = np.where(seen, F * B / np.where(seen, best, 1), np.nan).astype(np.float32) + noise_disp * (persistent * fixed + np.sqrt(1 - persistent ** 2) * n)
    return img, disp


def room_path(n):
    """Forward 1.3 m with a yaw sweep, then back to the start: the second half looks at what the first half mapped."""
    Rs, ts = [], []
    for i in range(n):
        s = i / max(n - 1, 1)
        leg = 1 - abs(2 * s - 1)                                   # 0 -> 1 -> 0
        yaw, pitch = np.radians(25 * np.sin(2 * np.pi * s)), np.radians(5 * np.sin(4 * np.pi * s))
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        Rs.append(np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]))
        ts.append(np.array([0.25 * np.sin(2 * np.pi * s), 0.0, 1.3 * leg]))
    return Rs, ts


def map_thickness_cm(points_c0):
    """Distance of each map point to the nearest true wall/floor plane, centimetres. A map without drift lies on the planes; copies of the same wall that
    drifted apart show up as larger distances."""
    p = np.asarray(points_c0)
    return 100 * np.min(np.stack([np.abs(p[:, a] - c) for a, c in PLANES], 1), axis=1)


class DriftingVO:
    """Wraps an odometry and adds a steady drift to its pose (a stand-in for the error a real run builds up), so the map refiner has something to correct.
    correct() absorbs a correction into the offset, as the real odometry does into its keyframe."""

    def __init__(self, vo, step_t=(0.0003, 0.0, 0.0), step_yaw_deg=0.01):
        self.vo, self.off = vo, np.eye(4)
        self.D = np.eye(4)
        a = np.radians(step_yaw_deg)
        self.D[:3, :3] = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        self.D[:3, 3] = step_t
        self.T_vo = np.eye(4)

    def step(self, *a, **k):
        R, t, info = self.vo.step(*a, **k)
        self.T_vo = np.eye(4)
        self.T_vo[:3, :3], self.T_vo[:3, 3] = R, t
        self.off = self.D @ self.off
        T = self.off @ self.T_vo
        return T[:3, :3].copy(), T[:3, 3].copy(), info

    def correct(self, R, t):
        Tc = np.eye(4)
        Tc[:3, :3], Tc[:3, 3] = R, t
        self.off = Tc @ np.linalg.inv(self.T_vo)


def run_room(make_vo, refine, n=200, seed=0, noise_disp=0.3, weighted=True):
    from gabai.mapref import MapRefiner
    from gabai.tools.slam_cloud_view import SlamCloud
    tex = texture()
    rng = np.random.default_rng(seed)
    Rs, ts = room_path(n)
    R0w, t0w = look_pose(np.zeros(3))
    vo = make_vo()
    cloud = SlamCloud(F, B, W / 2, H / 2, voxel=0.05, window_m=None, semi_dense=True, stride=3, max_range=4.0, weighted=weighted, refiner=MapRefiner() if refine else None)
    trusted = np.ones((H, W), bool)
    est, applied, t_ms = [], 0, []
    for R, t in zip(Rs, ts):
        img, disp = render_room(tex, R, t, noise_disp=noise_disp, rng=rng)
        trusted_f = np.isfinite(disp)
        disp = np.where(trusted_f, disp, np.nan).astype(np.float32)
        Rw, tw, info = vo.step(img, disp, trusted_f)
        if info["ok"]:
            t0 = time.perf_counter()
            cloud.on_pose(0, Rw, tw, img, disp, trusted_f)
            t_ms.append((time.perf_counter() - t0) * 1000)
            if cloud.refine_info and cloud.refine_info.get("applied"):
                applied += 1
                if hasattr(vo, "correct"):
                    vo.correct(*cloud.pose)
                tw = cloud.pose[1]
        est.append(R0w.T @ (tw - t0w))
    err = np.linalg.norm(np.array(est) - np.array(ts), axis=1)
    pts = cloud.map.points(min_obs=3)
    th = map_thickness_cm((pts - t0w) @ R0w)
    return {"final_err_cm": 100 * float(err[-1]), "rms_err_cm": 100 * float(np.sqrt((err ** 2).mean())), "thick_median_cm": float(np.median(th)), "thick_p90_cm": float(np.percentile(th, 90)),
            "voxels": len(cloud.map), "solid_points": len(pts), "corrections": applied, "map_ms_median": float(np.median(t_ms)) if t_ms else None}


def main_room():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.3, help="disparity noise in pixels")
    ap.add_argument("--drift", action="store_true", help="add a steady drift to the odometry (about 6 cm and 2 degrees over the walk)")
    a = ap.parse_args()
    w = (lambda f: (lambda: DriftingVO(f()))) if a.drift else (lambda f: f)
    cases = [("frame VO, no refine", w(lambda: _vo(StereoVO)), False, False), ("keyframe VO, no refine", w(lambda: _vo(KeyframeVO)), False, True),
             ("keyframe VO + map refine", w(lambda: _vo(KeyframeVO)), True, True)]
    for name, mk, ref, wt in cases:
        rs = [run_room(mk, ref, a.frames, s, a.noise, wt) for s in range(a.seeds)]
        m = lambda k: np.mean([r[k] for r in rs])
        print(f"{name:26s} path err final {m('final_err_cm'):5.2f} rms {m('rms_err_cm'):5.2f} cm | map-to-wall median {m('thick_median_cm'):4.2f} p90 {m('thick_p90_cm'):5.2f} cm | "
              f"voxels {m('voxels'):6.0f} corrections {m('corrections'):5.1f} map step {np.mean([r['map_ms_median'] for r in rs]):5.1f} ms", flush=True)


if __name__ == "__main__":
    import sys
    main_room() if "--room" in sys.argv and not sys.argv.remove("--room") else main()
