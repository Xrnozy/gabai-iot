import numpy as np

from mapper.navigate import Navigator, STOP_M

F, CX, CY, W, H = 600.0, 320.0, 180.0, 640, 360


def nav():
    return Navigator(F, CX, CY, floor_h=-1.3)


def run(depth, n=3):
    conf = np.ones_like(depth)
    nv, res = nav(), None
    for _ in range(n):
        res = nv.update(depth, conf, np.eye(4), np.array([0.0, 0.0, 1.0]))
    return res


def test_open_room_is_not_a_stop():
    d = np.full((H, W), 4.0, np.float32)
    assert run(d)["cue"] != "stop"


def test_flying_pixel_line_in_front_is_not_an_obstacle():
    d = np.full((H, W), 4.0, np.float32)
    d[100:260, 318:321] = 0.5                              # a 3 px wide line of mixed-depth pixels, the kind an object outline produces
    flying = Navigator.flying(d)
    assert flying[180, 319] and not flying[180, 100]


def test_wide_real_object_keeps_its_interior():
    d = np.full((H, W), 4.0, np.float32)
    d[100:260, 280:360] = 0.5                              # an 80 px wide object 0.5 m away
    flying = Navigator.flying(d)
    assert not flying[180, 320] and flying[180, 280]
