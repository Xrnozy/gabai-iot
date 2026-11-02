"""One 3D hand per real hand: stable identity across frames, and no copies left behind when the hand moves, flips MediaPipe's left/right label, or is lost."""
import numpy as np

from gabai.detect import Det
from gabai.pipeline.hand3d import HandIdentity
from gabai.pipeline.held import HeldAnalyzer
from gabai.pipeline.viz import Viz
from tests.test_held3d import Seq, frame_data, pose_at
from gabai.hands import Hand

F, B, CX, CY, W, H = 881.0, 0.0507, 319.5, 239.5, 640, 480


def d(palm=None, px=(320, 240)):
    return {"palm": None if palm is None else np.asarray(palm, float), "px": px}


def test_a_moving_hand_keeps_its_id():
    idn = HandIdentity()
    ids = []
    for k in range(30):
        i, _ = idn.assign([d([0.01 * k, 0, 0.8], (300 + 3 * k, 240))])
        ids.append(i[0])
    assert len(set(ids)) == 1


def test_two_hands_keep_their_own_ids_whatever_order_they_are_listed_in():
    idn = HandIdentity()
    a0, b0 = [-0.2, 0, 0.8], [0.2, 0, 0.8]
    first, _ = idn.assign([d(a0, (150, 240)), d(b0, (500, 240))])
    for k in range(1, 20):
        a, b = [-0.2 + 0.005 * k, 0, 0.8], [0.2 - 0.005 * k, 0, 0.8]
        pair = [d(b, (500 - 4 * k, 240)), d(a, (150 + 4 * k, 240))] if k % 2 else [d(a, (150 + 4 * k, 240)), d(b, (500 - 4 * k, 240))]
        got, _ = idn.assign(pair)
        by_pos = {"a": got[1] if k % 2 else got[0], "b": got[0] if k % 2 else got[1]}
        assert by_pos["a"] == first[0] and by_pos["b"] == first[1], k
    assert first[0] != first[1]


def test_a_short_loss_keeps_the_id_and_a_long_loss_retires_it():
    idn = HandIdentity(lost_frames=4)
    i0, _ = idn.assign([d([0, 0, 0.8], (300, 240))])
    for _ in range(3):
        idn.assign([])
    i1, _ = idn.assign([d([0.02, 0, 0.8], (305, 240))])
    assert i1 == i0
    retired_all = []
    for _ in range(6):
        _, r = idn.assign([])
        retired_all += r
    assert i0[0] in retired_all
    i2, _ = idn.assign([d([0.02, 0, 0.8], (305, 240))])
    assert i2 != i0


def test_a_hand_that_jumps_far_is_a_new_hand_not_the_old_one_teleporting():
    idn = HandIdentity()
    i0, _ = idn.assign([d([0, 0, 0.8], (300, 240))])
    i1, _ = idn.assign([d([1.5, 0, 0.8], (30, 20))])
    assert i1 != i0


def test_without_depth_the_picture_position_still_keeps_the_id():
    idn = HandIdentity()
    i0, _ = idn.assign([d(None, (300, 240))])
    i1, _ = idn.assign([d(None, (310, 244))])
    assert i0 == i1


# -- what Rerun is told ---------------------------------------------------------------------------------------------------------
class Live(Viz):
    """Records the state Rerun would show: an entity is live from the moment it is logged until it is cleared (a recursive clear also removes its children)."""

    def __init__(self):
        self.cal, self.size, self.t0, self.n_logged, self._held_paths = {"f": F}, (W, H), None, 0, set()
        self._hand_paths, self._obj_paths = set(), set()
        self.live = set()

    def _log(self, path, arch, **kw):
        self.live.add(path)

    def _clear(self, path, recursive=True):
        self.live = {p for p in self.live if not (p == path or (recursive and p.startswith(path + "/")))}

    def hand_roots(self):
        return {p.split("/")[2] for p in self.live if p.startswith("world/hands/")}


def analyzer_with(hands_seq):
    class FakeObj:
        def detect(self, bgr):
            return []
    return HeldAnalyzer(F, B, CX, CY, detector=FakeObj(), hand_tracker=Seq(hands_seq))


def moving_hand_frames(n=30, flip_labels=True):
    hands, disps = [], []
    for k in range(n):
        h, d_, _, _, X, _ = frame_data(k + 6, True, False)
        h = Hand("Left" if (flip_labels and k % 2) else "Right", h.kp2d, h.score, h.z_rel)
        hands.append([h])
        disps.append(d_)
    return hands, disps


def test_a_hand_that_moves_and_changes_its_label_stays_one_hand_and_leaves_no_copies():
    hands, disps = moving_hand_frames()
    a = analyzer_with(hands)
    v = Live()
    ids = set()
    for k, d_ in enumerate(disps):
        out = a.frame(k / 15, np.zeros((H, W, 3), np.uint8), d_, np.isfinite(d_), np.eye(3), np.zeros(3))
        assert len(out["hands"]) == 1
        ids.add(out["hands"][0]["id"])
        v.hands3d(out, {})
        assert len(v.hand_roots()) == 1, (k, v.hand_roots())            # exactly one hand in the scene after every frame
    assert len(ids) == 1, ids


def test_when_the_hand_is_lost_its_3d_model_is_removed_and_when_it_returns_there_is_still_one():
    hands, disps = moving_hand_frames(30, flip_labels=False)
    seq = hands[:10] + [[] for _ in range(6)] + hands[10:20]
    dseq = disps[:10] + [disps[9]] * 6 + disps[10:20]
    a = analyzer_with(seq)
    v = Live()
    counts, ids = [], []
    for k, d_ in enumerate(dseq):
        out = a.frame(k / 15, np.zeros((H, W, 3), np.uint8), d_, np.isfinite(d_), np.eye(3), np.zeros(3))
        v.hands3d(out, {})
        counts.append(len(v.hand_roots()))
        ids += [h["id"] for h in out["hands"]]
    assert counts[:10] == [1] * 10 and counts[10:16] == [0] * 6 and counts[16:] == [1] * 10, counts
    assert len(set(ids)) == 1, "the hand coming back after a short loss must keep its id"


def test_a_hand_whose_3d_could_not_be_made_this_frame_does_not_leave_the_last_one_standing():
    hands, disps = moving_hand_frames(12, flip_labels=False)
    a = analyzer_with(hands)
    v = Live()
    for k in range(6):
        out = a.frame(k / 15, np.zeros((H, W, 3), np.uint8), disps[k], np.isfinite(disps[k]), np.eye(3), np.zeros(3))
        v.hands3d(out, {})
    assert len(v.hand_roots()) == 1
    nodepth = np.full((H, W), np.nan, np.float32)
    out = a.frame(6 / 15, np.zeros((H, W, 3), np.uint8), nodepth, np.zeros((H, W), bool), np.eye(3), np.zeros(3))
    assert not out["hands"][0]["solve"]["ok"]
    v.hands3d(out, {})
    assert v.hand_roots() == set(), v.live


def test_the_frame_and_the_held_object_under_it_go_when_the_hand_goes():
    hands, disps = moving_hand_frames(8, flip_labels=False)
    a = analyzer_with(hands + [[]])
    v = Live()
    for k in range(8):
        out = a.frame(k / 15, np.zeros((H, W, 3), np.uint8), disps[k], np.isfinite(disps[k]), np.eye(3), np.zeros(3))
        v.hands3d(out, {})
    assert any(p.endswith("/frame") for p in v.live)
    out = a.frame(9 / 15, np.zeros((H, W, 3), np.uint8), disps[7], np.isfinite(disps[7]), np.eye(3), np.zeros(3))
    v.hands3d(out, {})
    assert not any(p.startswith("world/hands/") for p in v.live), v.live


def test_2d_overlays_are_cleared_when_nothing_is_detected_and_clear_tracked_removes_everything():
    hands, disps = moving_hand_frames(6, flip_labels=False)
    a = analyzer_with(hands + [[]])
    v = Live()
    img = np.zeros((H, W, 3), np.uint8)
    for k in range(6):
        out = a.frame(k / 15, img, disps[k], np.isfinite(disps[k]), np.eye(3), np.zeros(3))
        v.annotate(img, out)
    assert {"annotated/left/hands", "annotated/left/keypoints", "annotated/left/bones"} <= v.live
    out = a.frame(7 / 15, img, disps[5], np.isfinite(disps[5]), np.eye(3), np.zeros(3))
    v.annotate(img, out)
    assert not ({"annotated/left/hands", "annotated/left/keypoints", "annotated/left/bones"} & v.live)
    for k in range(3):
        out = a.frame(k / 15, img, disps[k], np.isfinite(disps[k]), np.eye(3), np.zeros(3))
        v.annotate(img, out)
    v.clear_tracked()
    assert not any(p.startswith(("world/hands", "annotated/left/hands", "annotated/left/keypoints", "world/objects/")) for p in v.live), v.live


def test_object_center_points_are_removed_when_the_object_disappears_and_while_it_is_held():
    v = Live()
    obj = {"key": "cup0", "label": "cup", "box": [0, 0, 10, 10], "conf": 0.9, "center_cam": np.array([0, 0, 1.0]), "center_world": np.array([0, 0, 1.0])}
    base = {"hands": [], "relations": [], "models": [], "released": [], "events": []}
    v.hands3d(dict(base, objects=[obj]), {})
    assert "world/objects/cup0" in v.live
    v.hands3d(dict(base, objects=[]), {})
    assert "world/objects/cup0" not in v.live
    v.hands3d(dict(base, objects=[obj]), {})
    v.hands3d(dict(base, objects=[obj], models=[{"key": "cup0", "frame": None, "n_points": 0, "hand": "hand1"}]), {})
    assert "world/objects/cup0" not in v.live, "the old position of a held object must not stay in the scene"
    v.hands3d(dict(base, objects=[], released=[{"key": "cup0", "label": "cup", "points_world": np.zeros((3, 3)), "unobserved": False, "box_world": None}]), {})
    assert "world/objects/cup0/model" in v.live, "the frozen object after release stays on purpose"
    v.hands3d(dict(base, objects=[]), {})
    assert "world/objects/cup0/model" in v.live
