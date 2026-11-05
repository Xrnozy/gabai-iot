import numpy as np

from gabai.net.lexicon import say
from mapper.egocentric import Egocentric


def _obj(i, label, p, dims=(0.4, 0.4, 0.4)):
    return {"id": i, "label": label, "p_w": np.array(p, float), "dims": np.array(dims, float)}


def test_person_on_the_right_near():
    ego = Egocentric()
    m = ego.update(0.0, np.eye(4), [_obj(1, "person", [0.1, -0.9, 0.0], (0.4, 0.4, 1.7))])      # y negative = right
    assert say(m["d"][0]) == "Human near you on your right."


def test_pose_rotation_changes_direction_and_behind():
    T = np.eye(4); T[:3, :3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]                              # user turned around
    m = Egocentric().update(0.0, T, [_obj(1, "chair", [2.0, 0.0, 0.0])])
    assert say(m["d"][0]).startswith("Chair") and "behind you" in say(m["d"][0])


def test_approaching_relation_and_delta():
    ego = Egocentric(delta=True)
    objs = [_obj(1, "person", [4.0, 0.0, 0.0], (0.4, 0.4, 1.7)), _obj(2, "dining table", [6.0, 0.0, 0.0], (1.6, 0.8, 0.0))]
    ego.update(0.0, np.eye(4), objs)
    objs[0]["p_w"] = np.array([3.0, 0.0, 0.0])                                                  # 1 m closer in 1 s
    m = ego.update(1.0, np.eye(4), objs)
    assert "approaching" in say(m["d"][0])
    ego.update(2.0, np.eye(4), objs)                                                            # standing still: state clears, once
    assert ego.update(3.0, np.eye(4), objs) == {}                                               # nothing changed -> nothing sent
    desk = [_obj(1, "cup", [1.0, 0, 1.1]), _obj(2, "dining table", [1.0, 0, 0.45], (1.0, 1.0, 0.9))]
    out = [say(p) for p in Egocentric().update(0.0, np.eye(4), desk)["d"]]
    assert any("on top of dining table" in s for s in out if s.startswith("Cup"))
