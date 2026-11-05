import numpy as np

from gabai.relay.publish import RelayPublisher
from gabai.relay.results import SlamGuide
from mapper.navigate import HEADINGS


def _res(cue, free_ahead=3.0, head=None):
    free = np.full(len(HEADINGS), 3.0)
    free[len(HEADINGS) // 2] = free_ahead
    return {"cue": cue, "heading_deg": head, "free": free}


def test_cue_is_said_once_danger_repeats_and_hazards_use_the_phone_zones():
    g = SlamGuide()
    a = g.result(1, 100.0, _res("path is clear"))
    assert a["speech"]["text"] == "Path is clear" and a["speech"]["kind"] == "clear" and a["hazards"] == [] and a["mode"] == "slam"
    assert "speech" not in g.result(2, 100.2, _res("path is clear"))                      # same cue: silent
    s = g.result(3, 101.0, _res("stop", 0.5))
    assert s["speech"]["interrupt"] is True and s["hazards"] == [{"zone": "forward", "distance_m": 0.5, "kind": "obstacle"}]
    assert "speech" not in g.result(4, 102.0, _res("stop", 0.5))                          # repeats only every 3 s
    assert "speech" in g.result(5, 104.5, _res("stop", 0.5))
    assert g.result(6, 105.0, _res("turn left", 2.0, head=40.0))["speech"]["kind"] == "steer"


def test_pose_and_lost_tracking_without_a_free_fan():
    g = SlamGuide()
    T = np.eye(4); T[:3, 3] = [1.234, 0, 2]
    r = g.result(9, 1.0, {"cue": "lost tracking", "heading_deg": None, "free": None}, T)
    assert r["pose"] == [1.23, 0.0, 2.0] and r["hazards"] == [] and r["speech"]["kind"] == "caution"


def test_publisher_drops_quietly_when_the_relay_is_down(capsys):
    p = RelayPublisher("http://127.0.0.1:9", token="x")                                   # nothing listens on port 9
    p.publish({"a": 1})
    import time
    time.sleep(2.5)
    assert "cannot reach" in capsys.readouterr().out
