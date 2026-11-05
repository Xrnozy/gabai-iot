import numpy as np

from gabai.detect import Det
from mapper.dynamic import PersonTracker, static_conf

F, CX, CY, W, H = 600.0, 320.0, 180.0, 640, 360


def person(u, z=2.0):
    """A person 80 px wide, 200 px tall, centred at column u, with depth z under the whole box."""
    box = (u - 40, 80, u + 40, 280)
    mask = np.zeros((H, W), bool)
    mask[80:280, u - 40:u + 40] = True
    return Det("person", 0.9, tuple(map(float, box)), mask)


def scene(z=2.0):
    return np.full((H, W), z, np.float32), np.ones((H, W), np.float32)


def test_walking_person_keeps_one_id_and_gets_a_velocity():
    tr = PersonTracker(F, CX, CY)
    depth, conf = scene()
    u = 150
    for k in range(10):                                   # 1 m/s sideways at 10 fps: 0.1 m = 30 px at 2 m
        out = tr.update([person(u + 30 * k)], depth, conf, np.eye(4), k * 0.1)
    assert len(out) == 1 and out[0].id == 1 and out[0].visible
    assert 0.8 < out[0].speed < 1.2 and tr.moving(out[0])


def test_standing_person_is_not_moving():
    tr = PersonTracker(F, CX, CY)
    depth, conf = scene()
    for k in range(10):
        out = tr.update([person(320)], depth, conf, np.eye(4), k * 0.1)
    assert out[0].speed < 0.05 and not tr.moving(out[0])


def test_person_who_left_is_a_ghost_then_gone_and_never_a_hazard():
    tr = PersonTracker(F, CX, CY, ghost_s=1.0)
    depth, conf = scene()
    for k in range(5):
        tr.update([person(320)], depth, conf, np.eye(4), k * 0.1)
    out = tr.update([], depth, conf, np.eye(4), 0.6)
    assert len(out) == 1 and not out[0].visible and tr.hazards(out) == []
    assert tr.update([], depth, conf, np.eye(4), 2.0) == []


def test_person_pixels_are_kept_out_of_the_static_maps():
    depth, conf = scene()
    c = static_conf(conf, [person(320)])
    assert c[180, 320] == 0 and c[180, 20] == 1 and conf[180, 320] == 1


def test_person_coming_towards_the_camera_is_approaching():
    tr = PersonTracker(F, CX, CY)
    for k in range(10):                                   # 0.5 m/s towards the camera
        depth, conf = scene(3.0 - 0.05 * k)
        out = tr.update([person(320)], depth, conf, np.eye(4), k * 0.1)
    h = tr.hazards(out)
    assert h and h[0]["kind"] == "person" and h[0]["approaching"] and h[0]["zone"] == "forward"
