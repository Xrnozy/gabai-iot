import numpy as np

from gabai.detect import Det
from gabai.pipeline.held import HeldAnalyzer
from tests.test_hands import make_hand

F, B, CX, CY = 881.0, 0.0507, 319.5, 239.5
I3 = np.eye(3)


class FakeHands:
    def __init__(self, seq):
        self.seq, self.i = seq, 0

    def detect(self, bgr, t_ms):
        h = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return h


class FakeDet:
    def __init__(self, seq):
        self.seq, self.i = seq, 0

    def detect(self, bgr):
        d = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return d


def scene(z=0.9):
    return np.full((480, 640), F * B / z, np.float32), np.ones((480, 640), bool), np.zeros((480, 640, 3), np.uint8)


def det(cx, cy, label="cup", half=30, conf=0.88):
    return Det(label, conf, (cx - half, cy - half, cx + half, cy + half), None)


def analyzer(hands, dets):
    return HeldAnalyzer(F, B, CX, CY, detector=FakeDet(dets), hand_tracker=FakeHands(hands))


def run(a, n, disp_tr_img=None, hand_fn=None):
    disp, tr, img = disp_tr_img or scene()
    out = None
    for k in range(n):
        out = a.frame(k / 15, img, disp, tr, I3, np.zeros(3))
    return out


def rel(out):
    return [r["relation"] for r in out["relations"]]


def test_far_apart_hand_and_object_are_apart_and_say_nothing_about_holding():
    a = analyzer([[make_hand(False, (150, 240), 60)]], [[det(500, 240)]])
    out = run(a, 3)
    assert rel(out) == ["apart"] and out["events"] == [] and out["hands"][0]["state"] == "closed"


def test_hand_resting_on_the_object_is_contact_not_held():
    a = analyzer([[make_hand(True, (320, 240), 60)]], [[det(320, 240)]])
    out = run(a, 8)
    assert rel(out) == ["contact"]


def test_a_hand_in_front_of_the_object_in_the_picture_but_far_in_depth_is_not_contact():
    disp, tr, img = scene(0.9)
    obj_far = np.full((480, 640), F * B / 0.9, np.float32)
    obj_far[210:270, 290:350] = F * B / 2.0                            # the object region is at 2.0 m, the hand at 0.9 m
    a = analyzer([[make_hand(True, (270, 240), 60)]], [[det(320, 240)]])          # the hand overlaps the object's box at its left edge, but most of the hand is at 0.9 m
    out = run(a, 4, (obj_far, tr, img))
    assert rel(out) == ["overlap_not_touching"] and out["hands"][0]["solve"]["ok"]


def test_overlap_without_trusted_depth_is_reported_uncertain_not_guessed():
    disp, tr, img = scene()
    disp[:] = np.nan
    a = analyzer([[make_hand(True, (320, 240), 60)]], [[det(320, 240)]])
    out = run(a, 4, (disp, np.zeros_like(tr), img))
    assert rel(out) == [] or rel(out)[0] == "uncertain"
    if out["relations"]:
        assert "depth" in out["relations"][0]["why"]


def pickup(follow=True, n_hold=18):
    xs = [320] * 8 + [320 + 12 * i for i in range(1, n_hold)]
    hands = [[make_hand(False, (x, 240), 60)] for x in xs]
    dets = [[det(320, 240)] for _ in range(8)] + [[det(x if follow else 320, 240)] for x in xs[8:]]
    return xs, hands, dets


def test_a_real_pickup_becomes_held_with_the_evidence_that_the_object_followed():
    xs, hands, dets = pickup(True)
    a = analyzer(hands, dets)
    disp, tr, img = scene()
    events, last = [], None
    for k in range(len(xs)):
        last = a.frame(k / 15, img, disp, tr, I3, np.zeros(3))
        events += last["events"]
    assert [e["type"] for e in events][:1] == ["grab"] and events[0]["object_observed_following"] is True
    assert "held" in rel(last) and last["relations"][0]["why"]


def test_a_fist_moving_away_while_the_object_stays_is_never_held():
    xs, hands, dets = pickup(False)
    a = analyzer(hands, dets)
    disp, tr, img = scene()
    events = []
    for k in range(len(xs)):
        out = a.frame(k / 15, img, disp, tr, I3, np.zeros(3))
        events += out["events"]
        assert "held" not in rel(out)
    assert "grab" not in [e["type"] for e in events]


def test_object_hidden_by_the_hand_is_held_unconfirmed_and_flagged_as_such():
    xs, hands, dets = pickup(True)
    dets = dets[:8] + [[] for _ in dets[8:]]                           # after the grab the object cannot be detected
    a = analyzer(hands, dets)
    disp, tr, img = scene()
    last = None
    for k in range(len(xs)):
        last = a.frame(k / 15, img, disp, tr, I3, np.zeros(3))
    held = [r for r in last["relations"] if r["relation"].startswith("held")]
    assert held and held[0]["relation"] == "held_unconfirmed" and "not seen" in held[0]["why"]


def test_positions_are_given_in_the_world_frame_with_a_pose():
    a = analyzer([[make_hand(True, (320, 240), 60)]], [[det(320, 240)]])
    disp, tr, img = scene(1.0)
    T = np.array([5.0, 0.0, 0.0])
    out = a.frame(0.0, img, disp, tr, I3, T)
    h = out["hands"][0]
    assert abs(h["palm_world"][2] - 1.0) < 0.05 and abs(h["palm_world"][0] - 5.0) < 0.2


def test_the_detector_or_hand_model_being_off_is_reported():
    a = HeldAnalyzer(F, B, CX, CY, detector=None, hand_tracker=None)
    disp, tr, img = scene()
    out = a.frame(0.0, img, disp, tr, I3, np.zeros(3))
    assert out["status"] == {"hands": "off", "objects": "off"} and out["hands"] == [] and out["objects"] == []
