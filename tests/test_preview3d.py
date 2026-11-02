import numpy as np

from gabai.detect import Det
from gabai.hands import Hand
from gabai.tools.preview_3d import ListSink, Preview3D

F, B, CX, CY = 881.0, 0.0507, 319.5, 239.5


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


def fist(cx, cy, closed=True, size=60):
    from tests.test_hands import make_hand
    return make_hand(not closed, (cx, cy), size)


def scene(z=0.9):
    return np.full((480, 640), F * B / z, np.float32), np.ones((480, 640), bool), np.zeros((480, 640, 3), np.uint8)


def box_det(cx, cy, label="cup", half=30):
    return Det(label, 0.9, (cx - half, cy - half, cx + half, cy + half), None)


def test_a_pickup_in_the_pictures_gives_a_grab_event_and_the_phrase_to_speak():
    xs = [320] * 8 + list(range(320, 320 + 15 * 12, 12))
    hands = FakeHands([[fist(x, 240)] for x in xs])
    # the cup sits at the start place, then follows the hand
    dets = FakeDet([[box_det(320, 240)] for _ in range(8)] + [[box_det(x, 240)] for x in xs[8:]])
    sink = ListSink()
    p = Preview3D(F, B, CX, CY, detector=dets, hand_tracker=hands, sink=sink)
    disp, tr, img = scene()
    ev = []
    for k in range(len(xs)):
        ev += p.frame(k / 15, img, disp, tr)["events"]
    assert [e["type"] for e in ev][:1] == ["grab"] and ev[0]["label"] == "cup" and ev[0]["phrase"] == "holding: cup"
    assert sink.n_frames == len(xs) and sink.last["hands"] and len(p.map) > 0


def test_people_and_hands_are_cut_out_of_the_map():
    person = Det("person", 0.9, (200, 100, 440, 460), None)
    sink = ListSink()
    p = Preview3D(F, B, CX, CY, detector=FakeDet([[person]]), hand_tracker=FakeHands([[]]), sink=sink)
    disp, tr, img = scene(2.0)
    out = p.frame(0.0, img, disp, tr)
    assert out["cloud_info"]["dropped_excluded"] > 1000
    pts = sink.last["live_points"]
    u = pts[:, 0] / pts[:, 2] * F + CX
    v = pts[:, 1] / pts[:, 2] * F + CY
    assert not ((u > 215) & (u < 425) & (v > 115) & (v < 445)).any()      # nothing inside the person's box (rows outside it are legitimately kept)


def test_no_detector_and_no_hands_still_runs_and_says_what_was_off():
    p = Preview3D(F, B, CX, CY, detector=None, hand_tracker=None, sink=ListSink())
    disp, tr, img = scene()
    out = p.frame(0.0, img, disp, tr)
    assert out["detector"] == "off" and out["hands_tracker"] == "off" and out["events"] == []


def test_a_hand_with_no_trusted_depth_gives_no_palm_and_no_grab():
    hands = FakeHands([[fist(320, 240)]] * 20)
    p = Preview3D(F, B, CX, CY, detector=FakeDet([[box_det(320, 240)]]), hand_tracker=hands, sink=ListSink())
    disp, tr, img = scene()
    disp[:] = np.nan
    ev = []
    for k in range(20):
        ev += p.frame(k / 15, img, disp, np.zeros_like(tr))["events"]
    assert ev == []
