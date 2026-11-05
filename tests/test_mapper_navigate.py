import numpy as np

from mapper.navigate import Navigator
from mapper.semantics import split_valleys
from mapper.align import OPTICAL

F, W, H = 400.0, 640, 480


def render(obstacles=(), wall=6.0, cam_h=1.3):
    """Depth image of a flat floor, a far wall and boxes (x0, y0, y1, height) seen from a camera at the origin looking along +x (body frame x forward, y left, z up)."""
    v, u = np.mgrid[0:H, 0:W]
    d = (OPTICAL[:3, :3] @ np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u, float)]).reshape(3, -1)).T     # body-frame ray per pixel, optical z = 1
    t = np.full(len(d), np.inf)
    dn = d[:, 2] < -1e-6
    t[dn] = -cam_h / d[dn, 2]
    t = np.minimum(t, wall / d[:, 0])
    for x0, y0, y1, hgt in obstacles:
        tx = x0 / d[:, 0]
        y, z = tx * d[:, 1], tx * d[:, 2]
        hit = (y > y0) & (y < y1) & (z > -cam_h) & (z < -cam_h + hgt)
        t = np.where(hit, np.minimum(t, tx), t)
    return t.reshape(H, W).astype(np.float32)


def run(depth):
    nav = Navigator(F, W / 2, H / 2)
    res = None
    for _ in range(3):                                                          # a change of cue must repeat once
        res = nav.update(depth, np.ones_like(depth), np.eye(4), [0, 0, 1])
    return res


def test_open_floor_is_clear():
    assert run(render())["cue"] == "path is clear"


def test_box_ahead_gives_a_veer_or_stop():
    r = run(render([(1.2, -0.4, 0.4, 1.0)]))
    assert r["cue"] in ("move left", "move right", "turn left", "turn right", "slow down")
    assert run(render([(0.5, -2, 2, 1.5)]))["cue"] == "stop"


def test_box_on_the_right_steers_left():
    r = run(render([(1.5, -0.6, 0.1, 1.0)]))                                     # y negative = right
    assert r["cue"] in ("move left", "turn left") and r["heading_deg"] > 0


def test_floor_is_not_an_obstacle_and_fast():
    r = run(render())
    assert len(r["obstacles"]) < 5 and r["ms"] < 15 and abs(r["floor_h"] + 1.3) < 0.2


def test_two_monitors_split_at_the_bezel():
    rng = np.random.default_rng(0)
    a = np.c_[rng.uniform(0, 0.6, 3000), np.zeros(3000), rng.uniform(0, 0.35, 3000)]
    b = a + [0.64, 0, 0]
    P = np.vstack([a, b])
    assert len(split_valleys(P)) == 2 and len(split_valleys(a)) == 1
