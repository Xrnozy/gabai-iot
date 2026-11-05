"""Regressions for "it always says path is blocked" (replay of data/mapper/pi: 249 of 608 nodes)."""
import numpy as np

from mapper.navigate import Navigator

F, CX, CY, W, H = 400.0, 320.0, 180.0, 640, 360


def floor_depth(cam_h=1.3, pitch_deg=0.0, wall_m=None):
    """Depth of a flat floor seen by a camera cam_h above it, pitched down by pitch_deg; optional wall wall_m ahead."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    ray_y, ray_z = (v - CY) / F, np.ones_like(v)                        # optical: y down, z forward
    p = np.radians(pitch_deg)
    down = ray_y * np.cos(p) + ray_z * np.sin(p)                          # how fast the ray goes down, per unit of z
    fwd = ray_z * np.cos(p) - ray_y * np.sin(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(down > 1e-3, cam_h / down, np.inf)
        if wall_m is not None:
            t = np.minimum(t, np.where(fwd > 1e-3, wall_m / fwd, np.inf))
    return np.where(np.isfinite(t) & (t < 6), t, 0).astype(np.float32)


def cue(depth, pitch_deg=0.0, up_error_deg=0.0, n=40):
    """Run n frames with an identity pose; the 'up' handed over makes the camera pitched by pitch_deg, wrong by up_error_deg (the mean-camera-up prior of a tilted rig)."""
    nav, res = Navigator(F, CX, CY), None
    a = np.radians(up_error_deg - pitch_deg)
    up = np.array([np.sin(a), 0.0, np.cos(a)])
    for _ in range(n):
        res = nav.update(depth, np.ones_like(depth), np.eye(4), up)
    return res["cue"]


def test_open_floor_with_a_wrong_up_is_clear():
    for err in (8.0, -8.0):                                               # the replay measured 7 deg between the map's floor and the up the navigator used
        assert cue(floor_depth(pitch_deg=20.0), 20.0, err) == "path is clear", err


def test_wall_2_5_m_ahead_is_walkable_not_blocked():
    assert cue(floor_depth(pitch_deg=10.0, wall_m=2.5), 10.0) not in ("path is blocked", "stop")


def test_wall_right_in_front_is_still_a_stop():
    assert cue(floor_depth(pitch_deg=10.0, wall_m=0.5), 10.0) == "stop"


def test_step_down_ahead_stops_the_path_and_is_a_dropoff_hazard():
    from gabai.relay.results import SlamGuide
    d = floor_depth(pitch_deg=20.0)
    v, u = np.mgrid[0:H, 0:W]
    far = d * np.cos(np.radians(20.0)) - (v - CY) / F * d * np.sin(np.radians(20.0))      # horizontal distance of each floor pixel
    d = np.where(far > 1.6, d * (1.3 + 0.5) / 1.3, d).astype(np.float32)                 # beyond 1.6 m the floor is 0.5 m lower: a step down
    nav = Navigator(F, CX, CY)
    a = np.radians(-20.0)
    for _ in range(40):
        r = nav.update(d, np.ones_like(d), np.eye(4), np.array([np.sin(a), 0.0, np.cos(a)]))
    i0 = len(r["free"]) // 2                                                            # the lower floor shows from about 2.2 m: the edge hides the part right after the step
    assert len(r["drops"]) > 8 and r["free"][i0] < 2.5 and (r["grid"]["kind"] == 4).any()
    assert any(h["kind"] == "dropoff" for h in SlamGuide(hazard_m=2.5).result(1, 0.0, r)["hazards"])


def test_a_wall_is_an_obstacle_in_the_grid_not_overhead():
    d = floor_depth(pitch_deg=20.0, wall_m=2.0)
    nav, a = Navigator(F, CX, CY), np.radians(-20.0)
    for _ in range(20):
        r = nav.update(d, np.ones_like(d), np.eye(4), np.array([np.sin(a), 0.0, np.cos(a)]))
    k = r["grid"]["kind"]
    assert (k == 2).sum() > 0 and (k == 3).sum() == 0 and len(r["overhead"]) == 0
