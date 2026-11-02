"""Loop closure end to end on a textured room the camera walks around once: detect the revisit, correct the trajectory, and re-lay the map.

The room is rendered by ray casting (textured walls, floor, ceiling), so ORB features, tracking and depth are all real image processing. The odometry is given a baseline
that grows 30% over the lap, so its steps get longer as it goes: a drift that opens the loop by about 0.35 m (one constant scale error would only make the circle bigger and
leave it closed), with the true answer known (the camera ends where it started). The map is built from the true depth, as in life: the poses drift, the pictures do not."""
import cv2
import numpy as np

from gabai.kfvo import KeyframeVO
from gabai.sim import look_pose
from gabai.tools.slam_cloud_view import SlamCloud

W, H, F, B = 320, 240, 260.0, 0.065
ROOM = np.array([[-3.0, 3.0], [-3.0, 3.0], [0.0, 2.5]])          # x, y, z extent of the room, metres
ALL = np.ones((H, W), bool)
_TEX = {}


def texture(key):
    if key not in _TEX:
        rng = np.random.default_rng(key)
        _TEX[key] = cv2.GaussianBlur(rng.integers(0, 255, (1024, 1024), dtype=np.uint8), (0, 0), 2.0)
    return _TEX[key]


def render(R_wc, t_wc):
    """(left gray image, disparity) of the room from the camera pose. Perfect depth, textured surfaces (about 120 texture pixels per metre)."""
    u = (np.arange(W)[None, :] - W / 2) / F
    v = (np.arange(H)[:, None] - H / 2) / F
    d = np.stack([np.broadcast_to(u, (H, W)), np.broadcast_to(v, (H, W)), np.ones((H, W))], -1) @ np.asarray(R_wc).T
    o = np.asarray(t_wc, float)
    best, gray = np.full((H, W), np.inf), np.zeros((H, W), np.float32)
    for axis in range(3):
        for side in range(2):
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (ROOM[axis, side] - o[axis]) / d[..., axis]
                s = np.where(np.isfinite(s), s, -1.0)
            hit = o + s[..., None] * d
            a1, a2 = [k for k in range(3) if k != axis]
            ok = (s > 0) & (s < best) & (hit[..., a1] >= ROOM[a1, 0] - 1e-6) & (hit[..., a1] <= ROOM[a1, 1] + 1e-6) & (hit[..., a2] >= ROOM[a2, 0] - 1e-6) & (hit[..., a2] <= ROOM[a2, 1] + 1e-6)
            tex = texture(axis * 2 + side)
            mx = ((hit[..., a1] - ROOM[a1, 0]) * 120).astype(np.float32) % 1023
            my = ((hit[..., a2] - ROOM[a2, 0]) * 120).astype(np.float32) % 1023
            g = cv2.remap(tex, mx, my, cv2.INTER_LINEAR)
            gray = np.where(ok, g, gray)
            best = np.where(ok, s, best)
    disp = (F * B / best).astype(np.float32)
    return np.clip(gray, 0, 255).astype(np.uint8), disp


def walk(vo_kwargs, cloud=False, step_m=0.04, radius=1.2, drift=0.3):
    """One lap of a circle of `radius` m, looking along the direction of travel; the baseline the odometry assumes grows by `drift` over the lap. Returns the VO, the SlamCloud
    (or None), the frame count."""
    vo = KeyframeVO(F, F, W / 2, H / 2, **vo_kwargs)
    sc = SlamCloud(F, B, W / 2, H / 2, voxel=0.05, window_m=None, semi_dense=False, stride=4, max_range=6.0, frame_points=2500) if cloud else None
    n = int(2 * np.pi * radius / step_m) + 6
    seen = None
    for k in range(n):
        a = min(k * step_m / radius, 2 * np.pi)
        pos = np.array([radius * np.cos(a), radius * np.sin(a), 1.2])
        R_wc, t_wc = look_pose(pos, yaw_deg=np.degrees(a) + 90.0)
        gray, disp = render(R_wc, t_wc)
        vo.set_baseline(B * (1.0 + drift * k / n))
        Rm, tm, info = vo.step(gray, disp, ALL)
        if cloud and info["ok"]:
            if seen is not None and info["loop_version"] != seen:
                sc.rebuild(vo.node_pose)
            seen = info["loop_version"]
            sc.on_pose(k, Rm, tm, gray, disp, ALL, anchor=info.get("anchor"))
    vo.flush()
    if cloud:
        sc.rebuild(vo.node_pose)
    return vo, sc, n


LOOP = dict(loop_closure=True, loop_apply=True, loop_min_path_m=3.0, loop_radius_m=1.0, window_ba=False, graph_async=False, max_keyframes=300)


def test_without_loop_closure_the_scale_error_leaves_the_circle_open():
    vo, _, _ = walk(dict(loop_closure=False, window_ba=False))
    R, t = vo.trajectory()
    assert np.linalg.norm(t[-1] - t[0]) > 0.25, np.linalg.norm(t[-1] - t[0])


def test_a_confirmed_loop_closes_the_circle_and_is_reported():
    vo, _, _ = walk(LOOP)
    R, t = vo.trajectory()
    assert vo.stats["loops_applied"] >= 1 and vo.loop_version >= 1, vo.stats
    gap = float(np.linalg.norm(t[-1] - t[0]))
    assert gap < 0.12, f"circle still open by {gap * 100:.0f} cm"


def test_a_look_alike_place_is_not_a_loop():
    """The same room but the camera has only walked half way: the first and last views are different places, and the check must not close anything."""
    vo = KeyframeVO(F, F, W / 2, H / 2, **LOOP)
    vo.set_baseline(B)
    for k in range(60):
        a = np.pi * k / 59
        R_wc, t_wc = look_pose([1.2 * np.cos(a), 1.2 * np.sin(a), 1.2], yaw_deg=np.degrees(a) + 90.0)
        gray, disp = render(R_wc, t_wc)
        vo.step(gray, disp, ALL)
    assert vo.stats["loops_applied"] == 0


def test_after_the_loop_the_map_of_the_first_wall_is_one_layer_not_two():
    def wall_spread(sc):
        p = sc.map.points(min_obs=2)
        wall = p[(np.abs(p[:, 0] - 3.0) < 0.6) & (np.abs(p[:, 1]) < 1.5)]              # the wall the camera faced at the start and again at the end (the odometry's world: x ahead of the first view) 
        return float(np.percentile(wall[:, 0], 75) - np.percentile(wall[:, 0], 25)), len(wall)             # the middle half: a few stray points do not decide it
    _, open_map, _ = walk(dict(loop_closure=False, window_ba=False), cloud=True)
    vo, closed_map, _ = walk(LOOP, cloud=True)
    s_open, n_open = wall_spread(open_map)
    s_closed, n_closed = wall_spread(closed_map)
    assert closed_map.rebuilds >= 1 and vo.stats["loops_applied"] >= 1
    assert n_open > 50 and n_closed > 50
    assert s_closed < 0.6 * s_open, f"first-wall thickness (middle half) {s_open * 100:.0f} cm without loop closure, {s_closed * 100:.0f} cm with it"
