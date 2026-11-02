import os
import struct
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from rtabmap_lab import orientation
from rtabmap_lab.bridge_client import EXE, Bridge, decode
from rtabmap_lab.dashboard import OPTICAL, Dashboard

REC = os.path.join("data", "recordings", "vio_20261101_014854")


def test_orientation_judge_tells_upright_from_upside_down_and_refuses_when_unsure():
    R_ic = np.eye(3)                                             # camera axes = IMU axes: the camera looks along z, its y axis points DOWN, so 'up' is -y
    still = lambda v, n=100: np.tile(np.asarray(v, float), (n, 1)) + np.random.default_rng(0).normal(0, 0.01, (n, 3))
    assert orientation.judge(still([0, -1, 0]), R_ic)["decision"] == "as-is"
    up = orientation.judge(still([0, 1, 0]), R_ic)
    assert up["decision"] == "flip" and up["angle_as_is"] > 170
    assert "tilted" in orientation.judge(still([1, 0, 0]), R_ic)["reason"]                    # sideways: cannot tell, leaves it
    shaken = np.random.default_rng(1).normal(0, 1.0, (100, 3)) + [0, 1, 0]
    assert orientation.judge(shaken, R_ic)["decision"] == "as-is" and "moving" in orientation.judge(shaken, R_ic)["reason"]
    assert orientation.judge(still([0, 1, 0], 5), R_ic)["decision"] == "as-is"                 # too few samples
    assert orientation.judge(still([0, 1, 0]), None)["decision"] == "as-is"


def test_a_status_message_decodes_with_the_layout_the_bridge_writes():
    pose = struct.pack("<12f", 1, 0, 0, 0.5, 0, 1, 0, 0.6, 0, 0, 1, 0.7)
    p = (struct.pack("<idi", 7, 12.5, 0) + pose + struct.pack("<12f", *np.eye(4)[:3].ravel()) + struct.pack("<4i", 300, 80, 120, 900) + struct.pack("<3f", 11.0, 22.0, 33.0)
         + struct.pack("<3i", 1, 5, 0) + struct.pack("<3i", 42, 3, 1) + struct.pack("<f", 2.5))
    m = decode(101, p)
    assert (m["id"], m["stamp"], m["lost"], m["features"], m["inliers"], m["matches"], m["local_map"]) == (7, 12.5, False, 300, 80, 120, 900)
    assert np.allclose(m["odom"][:3, 3], [0.5, 0.6, 0.7]) and (m["odom_ms"], m["rtabmap_ms"], m["disparity_ms"]) == (11.0, 22.0, 33.0)
    assert (m["added"], m["loop_id"], m["proximity_id"], m["nodes"], m["loops"], m["restarts"], m["distance"]) == (True, 5, 0, 42, 3, 1, 2.5)
    assert decode(101, p[:16] + b"\0" * 48 + p[64:])["odom"] is None                               # an all-zero pose is RTAB-Map's null


def test_the_map_is_laid_z_up_a_point_above_the_camera_is_above_in_the_world(monkeypatch):
    logged = {}
    import rtabmap_lab.dashboard as D
    monkeypatch.setattr(D.rr, "log", lambda path, arch, **kw: logged.__setitem__(path, arch))
    d = Dashboard({"f": 400, "cx": 320, "cy": 180}, (640, 360), viewer=False)
    d.raw_points = True
    pts = np.array([[0, 0, 2.0],            # 2 m straight ahead (optical z)
                    [0, -1.0, 0.0],         # 1 m UP (optical y points down)
                    [1.0, 0, 0.0]], np.float32)  # 1 m to the camera's right
    d.node({"id": 1, "pose": np.eye(4), "points": pts, "bgr": np.zeros((3, 3), np.uint8)})
    d.log_map(force=True)
    w = np.array(logged["world/map"].positions.as_arrow_array().to_pylist(), np.float64)
    ahead, up, right = (w[np.argmax(w[:, 0])], w[np.argmax(w[:, 2])], w[np.argmin(w[:, 1])])
    assert np.allclose(ahead, [2, 0, 0], atol=1e-5) and np.allclose(up, [0, 0, 1], atol=1e-5) and np.allclose(right, [0, -1, 0], atol=1e-5)    # x forward, z up, right = -y (y is left)
    assert np.allclose(OPTICAL[:3, :3] @ OPTICAL[:3, :3].T, np.eye(3)) and np.linalg.det(OPTICAL[:3, :3]) > 0                  # a proper rotation, not a mirror


@pytest.mark.skipif(not (os.path.exists(EXE) and os.path.exists(REC)), reason="needs the built bridge (py -m rtabmap_lab setup) and the recording")
def test_the_bridge_runs_rtabmaps_own_stereo_odometry_depth_and_mapping_on_a_recording():
    from rtabmap_lab.live import frames
    a = SimpleNamespace(source="replay", folder=REC, port=5600, pipeline_config="config/pipeline.toml")
    cfg, source = frames(a)
    b = Bridge(db="", port=5731, config=os.path.join("rtabmap_lab", "config", "rtabmap_stereo.ini"))
    status, nodes, graphs, disp, depths = [], [], [], [], []
    try:
        for i, fr in enumerate(source):
            if i == 0:
                h, w = fr.left.shape[:2]
                c = fr.cal
                b.init(c["f"], c["f"], c["cx"], c["cy"], c["B"], w, h)
            b.wait_idle()
            b.send_frame(fr.idx, fr.t, fr.left, cv2.cvtColor(fr.right, cv2.COLOR_BGR2GRAY))
            b.wait_idle()
            for m in b.poll():
                {"status": status, "node": nodes, "graph": graphs, "disparity": disp, "depth": depths}.get(m["type"], []).append(m)
            if i >= 70:
                break
    finally:
        b.close(save=False)
    assert len(status) >= 60 and sum(s["lost"] for s in status) < 0.3 * len(status)
    assert nodes and len(nodes[0]["points"]) > 1000 and nodes[0]["points"][:, 2].min() > 0.2          # RTAB-Map's own disparity gave metric points in front of the camera
    assert graphs and disp and status[-1]["nodes"] >= 3 and status[-1]["features"] > 50
    d = depths[0]                                                                                    # the dense depth + colour of a node, what the TSDF integrates
    assert d["depth_mm"].shape == d["bgr"].shape[:2] == (360, 640) and (d["depth_mm"] > 0).mean() > 0.03 and d["depth_mm"][d["depth_mm"] > 0].min() >= 300


# -- the TSDF stage ----------------------------------------------------------------------------------------------------------------------------------------
FX, CX, CY, W, H = 400.0, 320.0, 180.0, 640, 360


def _render_plane(T_base, normal, d, color=(90, 160, 220)):
    """Depth (mm) and colour of the world plane  normal . p = d  as the camera at base pose T_base sees it (the camera is the base turned by OPTICAL)."""
    from rtabmap_lab.tsdf import OPTICAL as OPT
    Twc = T_base @ OPT
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u - CX) / FX, (v - CY) / FX, np.ones_like(u, float)], -1) @ Twc[:3, :3].T
    denom = rays @ np.asarray(normal, float)
    z = (d - np.asarray(normal, float) @ Twc[:3, 3]) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    z = np.where((z > 0.3) & (z < 6.0), z, 0.0)
    tex = (np.indices((H, W)).sum(0) // 8 % 2 * 60 + np.array(color)[:, None, None]).transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    return (z * 1000).astype(np.uint16), tex[:, :, ::-1].copy()


def _pose(x=0.0, y=0.0, z=0.0, pitch_up_deg=0.0):
    a = np.radians(pitch_up_deg)
    T = np.eye(4)
    T[:3, :3] = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])         # base x (forward) tilts toward +z for a positive pitch up
    T[:3, 3] = [x, y, z]
    return T


def test_the_tsdf_puts_a_ceiling_above_and_a_wall_ahead_and_rebuilds_when_a_loop_closure_moves_the_poses():
    from rtabmap_lab.tsdf import TsdfMap
    t = TsdfMap((FX, FX, CX, CY), (W, H), voxel=0.02, trunc=0.08, threaded=False)
    for i, (x, y) in enumerate([(0, 0), (0.3, 0.1), (0.6, -0.1)]):
        T = _pose(x, y, 0.0, pitch_up_deg=40)                                          # looking forward and up at a ceiling 1.5 m above the camera
        d, c = _render_plane(T, (0, 0, 1), 1.5)
        t.add_node(i + 1, d, c)
        t.set_poses({i + 1: T})
        t.step(force_mesh=True)
    v = np.asarray(t.mesh.vertices)
    assert len(v) > 2000 and abs(np.median(v[:, 2]) - 1.5) < 0.03 and np.percentile(np.abs(v[:, 2] - 1.5), 95) < 0.06      # the surface is ABOVE (z = +1.5), not below
    assert t.stats["rebuilds"] == 0 and t.stats["integrated"] == 3
    t.set_poses({3: _pose(0.6, -0.1, 0.0, pitch_up_deg=40) @ np.array([[1, 0, 0, 0.15], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])})      # the optimiser moved node 3 by 15 cm
    t.step(force_mesh=True)
    assert t.stats["rebuilds"] == 1 and t.stats["integrated"] == 3
    t.set_poses({3: t.poses[3] @ _pose(0.004)})                                        # 4 mm: noise, not a correction
    t.step(force_mesh=True)
    assert t.stats["rebuilds"] == 1
    path = os.path.join(os.environ.get("TEMP", "."), "tsdf_test.ply")
    assert t.save_mesh(path) and os.path.getsize(path) > 10000
    t.close()


def test_a_wall_ahead_is_at_positive_x():
    from rtabmap_lab.tsdf import TsdfMap
    t = TsdfMap((FX, FX, CX, CY), (W, H), voxel=0.02, trunc=0.08, threaded=False)
    for i, x in enumerate([0.0, 0.3]):
        T = _pose(x)
        d, c = _render_plane(T, (1, 0, 0), 3.0)
        t.add_node(i + 1, d, c)
        t.set_poses({i + 1: T})
    v = np.asarray(t.flush().vertices)
    assert abs(np.median(v[:, 0]) - 3.0) < 0.03 and v[:, 1].std() > 0.5                # the wall is 3 m ahead (+x) and spreads sideways
    t.close()


def test_the_pipelines_flip_flags_are_accepted_as_orientation_choices():
    import argparse
    from rtabmap_lab.live import add_args
    parse = lambda *x: add_args(argparse.ArgumentParser()).parse_args(list(x)).orientation
    assert parse() == "auto" and parse("--flip180") == "flip" and parse("--no-flip180") == "as-is" and parse("--orientation", "flip") == "flip"


# -- the GabAi depth option -------------------------------------------------------------------------------------------------------------------------------------
def _fake_depth_fn(disp, fill=None, fail=False):
    def fn(left, right):
        if fail:
            raise RuntimeError("gpu out of memory")
        fn.filled = fill
        return disp.copy(), np.isfinite(disp), 12.0
    fn.filled = None
    return fn


def _cfg_for_depth(reach=0, surface=False):
    return SimpleNamespace(stereo=SimpleNamespace(), map=SimpleNamespace(fill_reach_px=reach, surface_filter=surface))


def test_gabai_depth_turns_the_fused_disparity_into_millimetres_and_keeps_stereo_pixels_untouched():
    from rtabmap_lab.gabai_depth import GabaiDepth
    disp = np.full((36, 64), 20.0, np.float32)
    disp[10:20, 10:20] = np.nan                                         # stereo found nothing here
    aligned = np.full((36, 64), 10.0, np.float32)                       # the monocular maps' fill, on the stereo scale
    mask = np.zeros((36, 64), bool)
    mask[10:20, 10:20] = True
    mask[0:3, 0:3] = True                                               # a fill that would sit on a trusted stereo pixel: never used
    g = GabaiDepth(_cfg_for_depth(), depth_fn=_fake_depth_fn(disp, (aligned, mask & np.isnan(disp), {})))
    d, ms = g.depth_mm(np.zeros((36, 64, 3), np.uint8), np.zeros((36, 64, 3), np.uint8), {"f": 600.0, "B": 0.04})
    assert ms == 12.0 and d.dtype == np.uint16
    assert d[30, 30] == 1200 and d[1, 1] == 1200                        # 600 * 0.04 / 20 = 1.2 m, and the trusted pixel kept its stereo value
    assert d[15, 15] == 2400                                            # the hole took the monocular fill: 24 / 10 = 2.4 m
    assert (g.depth_mm(np.zeros((36, 64, 3), np.uint8), np.zeros((36, 64, 3), np.uint8), {"f": 600.0, "B": 0.04}, 0.3, 2.0)[0][15, 15]) == 0         # beyond the range: none


def test_points_from_depth_are_in_the_optical_frame():
    from rtabmap_lab.gabai_depth import points_from_depth
    d = np.zeros((360, 640), np.uint16)
    d[180, 320], d[100, 500] = 2000, 2000
    p, c = points_from_depth(d, np.full((360, 640, 3), 7, np.uint8), 400.0, 400.0, 320.0, 180.0, stride=1)
    on_axis, off = p[np.argmin(np.abs(p[:, 0]))], p[np.argmax(np.abs(p[:, 0]))]
    assert np.allclose(on_axis, [0, 0, 2.0]) and off[0] > 0 and off[1] < 0 and np.isclose(off[2], 2.0) and c.shape == (2, 3)   # right of the axis is +x, above it is -y (y points down)


def test_the_upgrader_gives_a_node_the_gabai_depth_or_the_bridges_never_nothing():
    import time
    from rtabmap_lab.gabai_depth import DepthUpgrader, GabaiDepth
    got, pts = {}, {}
    cal = {"f": 600.0, "B": 0.04, "cx": 32.0, "cy": 18.0}
    disp = np.full((36, 64), 20.0, np.float32)
    msg = lambda nid, fid: {"id": nid, "frame_id": fid, "depth_mm": np.full((36, 64), 111, np.uint16), "bgr": np.zeros((36, 64, 3), np.uint8)}
    up = DepthUpgrader(GabaiDepth(_cfg_for_depth(), depth_fn=_fake_depth_fn(disp)), cal, (64, 36), lambda i, d, b, extra=None: got.__setitem__(i, d), lambda i, p, c: pts.__setitem__(i, p), stride=4)
    try:
        up.remember(5, np.zeros((36, 64, 3), np.uint8), np.zeros((36, 64, 3), np.uint8))
        up.node(msg(1, 5))                                              # its pair is kept: the GabAi depth, a moment later
        up.node(msg(2, 99))                                             # its pair is gone: the bridge's depth, at once
        assert got[2][0, 0] == 111
        end = time.time() + 5
        while 1 not in got and time.time() < end:
            time.sleep(0.01)
        assert got[1][0, 0] == 1200 and len(pts[1]) > 0 and up.stats["gabai"] == 1 and up.stats["fallback"] == 1
    finally:
        up.close()
    bad = DepthUpgrader(GabaiDepth(_cfg_for_depth(), depth_fn=_fake_depth_fn(disp, fail=True)), cal, (64, 36), lambda i, d, b, extra=None: got.__setitem__(i, d), lambda i, p, c: None)
    try:
        bad.remember(7, np.zeros((36, 64, 3), np.uint8), np.zeros((36, 64, 3), np.uint8))
        bad.node(msg(3, 7))
        end = time.time() + 5
        while 3 not in got and time.time() < end:
            time.sleep(0.01)
        assert got[3][0, 0] == 111 and "gpu" in bad.stats["error"]      # the chain failed: the node still got a depth
    finally:
        bad.close()


# -- surfels, plane snapping, the consistency gate --------------------------------------------------------------------------------------------------------------
def _noisy_wall(T_base, seed, sigma=0.02, outliers=0.05):
    d, c = _render_plane(T_base, (1, 0, 0), 3.0)
    rng = np.random.default_rng(seed)
    z = d.astype(np.float32) / 1000.0
    ok = z > 0
    z = np.where(ok, z + rng.normal(0, sigma, z.shape), 0)
    bad = ok & (rng.random(z.shape) < outliers)
    z = np.where(bad, rng.uniform(0.6, 3.8, z.shape), z)                                 # single-view outliers: wrong, and different in every picture
    return (np.clip(z, 0, 6) * 1000).astype(np.uint16), c, ok & ~bad


def _wall_poses():
    return [_pose(x, y) for x, y in [(0, 0), (0.25, 0.1), (0.5, -0.1), (0.75, 0.05), (0.4, 0.0)]]


def test_surfels_keep_a_wall_thin_and_throw_away_what_only_one_picture_saw():
    from rtabmap_lab import flatness
    from rtabmap_lab.surfel import SurfelFusion
    from rtabmap_lab.tsdf import OPTICAL as OPT
    f = SurfelFusion((FX, FX, CX, CY), FX * 0.0417, threaded=False, min_views=2)
    raw = []
    for i, T in enumerate(_wall_poses()):
        d, c, _ = _noisy_wall(T, i)
        f.add_node(i + 1, d, c)
        f.set_poses({i + 1: T})
        from rtabmap_lab.gabai_depth import points_from_depth
        p, _ = points_from_depth(d, c, FX, FX, CX, CY, 2)
        raw.append(p @ (T @ OPT)[:3, :3].T + (T @ OPT)[:3, 3])
    P, N, C = f.flush()
    raw = np.concatenate(raw)
    off_wall = lambda x: float(np.mean(np.abs(x[:, 0] - 3.0) > 0.12))                      # more than 12 cm from the wall
    assert len(P) > 5000 and off_wall(P) < 0.2 * off_wall(raw) and off_wall(raw) > 0.03
    assert abs(np.median(P[:, 0]) - 3.0) < 0.02 and np.std(P[np.abs(P[:, 0] - 3.0) < 0.12, 0]) < 0.02
    assert np.mean(N[:, 0] < -0.6) > 0.7                                                  # normals face the cameras (toward -x)
    f.close()


def test_a_surfel_in_front_of_what_a_picture_sees_through_loses_confidence():
    from rtabmap_lab.surfel import SurfelMap
    from rtabmap_lab.tsdf import OPTICAL as OPT
    m = SurfelMap((FX, FX, CX, CY), FX * 0.0417, min_views=2)
    T = _pose(0.0)
    d, c = _render_plane(T, (1, 0, 0), 2.0)                                               # a wall at 2 m, seen once: probationary surfels
    m.integrate(d, c, T @ OPT)
    n0 = len(m)
    d2, c2 = _render_plane(T, (1, 0, 0), 3.0)                                             # the next pictures see a wall at 3 m through the places where those surfels are
    m.integrate(d2, c2, T @ OPT)
    m.integrate(d2, c2, T @ OPT)
    assert n0 > 5000 and (m.W > 0).all() and np.mean(np.abs(m.P[:, 0] - 2.0) < 0.05) < 0.2     # the 2 m surfels were seen through and dropped


def test_planarize_flattens_a_wall_and_moves_nothing_that_is_not_one():
    from rtabmap_lab.planes import planarize
    rng = np.random.default_rng(0)
    wall = np.c_[np.full(4000, 3.0) + rng.normal(0, 0.01, 4000), rng.uniform(-2, 2, 4000), rng.uniform(0, 2.5, 4000)]
    ball = rng.normal(0, 0.15, (900, 3)) + [1.5, 0.0, 1.0]                                  # a roundish object in front of the wall
    pts = np.vstack([wall, ball]).astype(np.float32)
    nrm = np.vstack([np.tile([-1, 0, 0], (4000, 1)), (pts[4000:] - [1.5, 0, 1.0]) / 0.15]).astype(np.float32)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    out, used, planes = planarize(pts, nrm)
    assert used[:4000].mean() > 0.9 and abs(planes[0][0][0]) > 0.99
    mem = out[:4000][used[:4000]].astype(np.float64)
    thick = np.linalg.svd(mem - mem.mean(axis=0), compute_uv=False)[-1] / np.sqrt(len(mem))
    assert thick < 1e-3 and np.std(pts[:4000, 0]) > 0.009                                  # the wall is a sheet now (a millimetre, from a centimetre)
    assert np.allclose(out[4000:][~used[4000:]], pts[4000:][~used[4000:]])                   # what was not on a plane did not move
    assert np.abs(out - pts).max() <= 0.026


def test_the_gate_removes_depth_no_other_picture_confirms_and_keeps_the_wall():
    from rtabmap_lab.surfel import SurfelFusion
    from rtabmap_lab.tsdf import OPTICAL as OPT
    f = SurfelFusion((FX, FX, CX, CY), FX * 0.0417, threaded=False, min_views=2)
    for i, T in enumerate(_wall_poses()[:4]):
        d, c, _ = _noisy_wall(T, i)
        f.add_node(i + 1, d, c)
        f.set_poses({i + 1: T})
    f.flush()
    T = _wall_poses()[4]
    d, c, good = _noisy_wall(T, 9)
    g = f.map.gate_depth(d, T @ OPT)
    kept = g > 0
    assert (kept & good).sum() > 0.5 * good.sum()                                           # most of the wall stays
    assert (kept & ~good & (d > 0)).sum() < 0.15 * ((~good) & (d > 0)).sum()                # most of the outliers go
    f.close()


def test_the_upgrader_runs_the_models_only_for_posed_nodes():
    import time
    from rtabmap_lab.gabai_depth import DepthUpgrader, GabaiDepth
    got = {}
    cal = {"f": 600.0, "B": 0.04, "cx": 32.0, "cy": 18.0}
    disp = np.full((36, 64), 20.0, np.float32)
    msg = lambda nid, fid: {"id": nid, "frame_id": fid, "depth_mm": np.full((36, 64), 111, np.uint16), "bgr": np.zeros((36, 64, 3), np.uint8)}
    up = DepthUpgrader(GabaiDepth(_cfg_for_depth(), depth_fn=_fake_depth_fn(disp)), cal, (64, 36), lambda i, d, b, extra=None: got.__setitem__(i, d), lambda i, p, c: None, stride=4, wait_pose_s=0.3)
    try:
        for fid in (5, 6):
            up.remember(fid, np.zeros((36, 64, 3), np.uint8), np.zeros((36, 64, 3), np.uint8))
        up.node(msg(1, 5))
        up.node(msg(2, 6))
        time.sleep(0.1)
        assert not got and up.pending == 2                          # held: no pose yet, no model run
        up.posed({1: np.eye(4), 2: None})                           # node 1 posed
        end = time.time() + 5
        while 1 not in got and time.time() < end:
            time.sleep(0.01)
        assert got[1][0, 0] == 1200 and 2 not in got
        time.sleep(0.4)
        up.posed({2: None})                                         # node 2 never got a pose: dropped
        assert up.stats["skipped"] == 1 and up.pending == 0 and up.stats["gabai"] == 1
    finally:
        up.close()


def test_sgbm_depth_recovers_a_known_plane_depth():
    from rtabmap_lab.sgbm_depth import SgbmDepth
    rng = np.random.default_rng(0)
    h, w, f, B, z = 240, 320, 400.0, 0.05, 1.0
    d = f * B / z                                                   # 20 px
    tex = cv2.GaussianBlur((rng.random((h, w + 40)) * 255).astype(np.uint8), (0, 0), 1.2)
    left = tex[:, 20:20 + w]
    right = tex[:, 20 + int(d):20 + int(d) + w]                     # the right picture sees the same texture shifted by d
    mm, ms, extra = SgbmDepth().depth_mm(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR), {"f": f, "B": B}, 0.3, 4.0)
    ok = mm[40:200, 120:300] > 0
    assert ok.mean() > 0.8 and abs(np.median(mm[40:200, 120:300][ok]) - 1000) < 30 and extra is None
