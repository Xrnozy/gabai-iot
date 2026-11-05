import json

from gabai.net.lexicon import LEX, decode, message, packet, say


def test_the_example_packet():
    assert packet("human", "near you", "on your right") == [1, 1, 2]
    assert say([1, 1, 2]) == "Human near you on your right."
    assert LEX["dist"]["2"] == "far from you" and LEX["dir"]["3"] == "on your left" and LEX["dir"]["4"] == "in front of you" and LEX["dir"]["5"] == "behind you"


def test_relation_state_and_message():
    p = packet("chair", "a few steps away", "in front of you", rel="next to", obj2="table", state="blocked")
    assert say(p) == "Chair a few steps away in front of you, next to table, blocked."
    assert say(packet("dog", "near you", "on your left", state="running")) == "Dog near you on your left, running."
    m = message([("human", "near you", "on your right")], env=["street"], cue=["stop"])
    assert decode(m) == ["Human near you on your right.", "Street.", "Stop."] and len(json.dumps(m, separators=(",", ":"))) < 45


def test_ids_are_unique_text_per_table():
    for t in ("obj", "dist", "dir", "rel", "state", "env", "cue"):
        assert len(set(LEX[t].values())) == len(LEX[t])
