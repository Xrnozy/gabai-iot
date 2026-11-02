"""NavMap on a synthetic scene: floor, a box 2 m ahead, a wall; the world frame is tilted against gravity, the camera looks 15 deg down."""
import numpy as np

from rtabmap_lab.navmap import NavMap

W, H, F, CX, CY = 640, 360, 500.0, 320.0, 180.0


def scene_depth(T_gc):
    """Depth (z along the camera axis) of: floor z=0, box x in [1.8,2.2] y in [-0.2,0.2] z in [0,0.6], wall x=4. Camera pose T_gc in the gravity frame G."""
    v, u = np.mgrid[0:H, 0:W]
    ray_c = np.stack([(u - CX) / F, (v - CY) / F, np.ones((H, W))], -1)
    R, t = T_gc[:3, :3], T_gc[:3, 3]
    r = ray_c @ R.T                                                        # direction (scaled so that its camera z is 1)
    best = np.full((H, W), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -t[2] / r[..., 2]                                            # floor
        best = np.where(lam > 0, lam, best)
        lam = (4.0 - t[0]) / r[..., 0]                                     # wall
        best = np.where((lam > 0) & (lam < best), lam, best)
        lam = (1.8 - t[0]) / r[..., 0]                                     # front face of the box
        hit = t + r * lam[..., None]
        m = (lam > 0) & (lam < best) & (np.abs(hit[..., 1]) < 0.2) & (hit[..., 2] > 0) & (hit[..., 2] < 0.6)
        best = np.where(m, lam, best)
        lam = (0.6 - t[2]) / r[..., 2]                                     # top of the box
        hit = t + r * lam[..., None]
        m = (lam > 0) & (lam < best) & (np.abs(hit[..., 1]) < 0.2) & (hit[..., 0] > 1.8) & (hit[..., 0] < 2.2)
        best = np.where(m, lam, best)
    best[~np.isfinite(best)] = 0
    return best.astype(np.float32)


def pose(x, pitch_deg):
    """Camera at (x, 0, 1.3) in G looking along +x pitched down; camera axes x right (-y of G), y down, z forward."""
    p = np.radians(pitch_deg)
    fwd = np.array([np.cos(p), 0, -np.sin(p)])
    down = np.array([-np.sin(p), 0, -np.cos(p)])
    right = np.cross(down, fwd)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, fwd], 1)
    T[:3, 3] = [x, 0, 1.3]
    return T


def test_levelled_map_finds_the_box_and_routes_around_it():
    tilt = np.eye(4)
    a = np.radians(25)
    tilt[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]       # the odometry world is turned 25 deg against gravity
    nm = NavMap(half_size_m=3.0, cell_m=0.05, min_hits=3)
    res = {"ready": False}
    for k, x in enumerate(np.linspace(0.0, 0.3, 14)):
        T_gc = pose(x, 15)
        res = nm.update(k * 0.1, scene_depth(T_gc), F, CX, CY, tilt @ T_gc, goal_w=None)
    assert res["ready"]
    up_true = tilt[:3, :3] @ np.array([0, 0, 1.0])
    assert abs(nm.up_w @ up_true) > np.cos(np.radians(4)), "gravity from the floor plane"
    st = res["state"]
    g = nm.grid
    t_g = res["cam_cell"]
    assert (st == 2).sum() > 5, "the box is occupied"
    assert (st == 1).sum() > 200, "the floor is free"
    ox, oy = g.index(t_g_x := g.centre(*t_g)[0] + (2.0 - 0.3), g.centre(*t_g)[1])
    assert st[ox, oy] == 2 or st[ox - 1:ox + 2, oy - 1:oy + 2].max() == 2, "the cell where the box is"
    assert len(res["path_cells"]) > 10, "a path exists"
    ys = np.array([g.centre(i, j)[1] for i, j in res["path_cells"]])
    assert not any(st[i, j] == 2 for i, j in res["path_cells"]) and res["nearest_obstacle_ahead_m"] is not None
    assert res["ms"] < 100
    assert nm.image(res).shape[2] == 3


def test_nav_sink_takes_each_node_once_when_it_has_a_pose():
    from rtabmap_lab.navmap import NavMap, NavSink
    got = []
    sink = NavSink(NavMap(), F, CX, CY, on_update=lambda r, nm: got.append(r), optical=np.eye(4))
    for k in range(8):
        T = pose(0.05 * k, 15)
        sink.add_node(k, (scene_depth(T) * 1000).astype(np.uint16), None)
    sink.set_poses({k: pose(0.05 * k, 15) for k in range(4)})            # only some nodes have a pose yet
    n1 = len(got)
    sink.set_poses({k: pose(0.05 * k, 15) for k in range(8)})
    sink.set_poses({k: pose(0.05 * k, 15) for k in range(8)})            # nothing is processed twice
    assert n1 <= 4 and len(sink.done) == 8 and len(got) <= 8
