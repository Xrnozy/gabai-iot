"""The 3D hand and the held object, end to end through HeldAnalyzer, on a synthetic pick-up with known geometry: a block rigid in a moving, turning hand."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.detect import Det
from gabai.hands import Hand
from gabai.pipeline.held import HeldAnalyzer
from tests.test_hand3d import LOCAL

F, B, CX, CY, W, H = 881.0, 0.0507, 319.5, 239.5, 640, 480
FINGERS = {5: (6, 7, 8), 9: (10, 11, 12), 13: (14, 15, 16), 17: (18, 19, 20)}
rng = np.random.default_rng(3)
BLOCK = rng.uniform([-0.035, 0.03, 0.02], [0.035, 0.11, 0.07], (4000, 3))         # in the hand's own frame (x across the palm, y toward the fingers, z out of the palm)


def hand_shape(closed):
    loc = LOCAL.copy()
    if closed:
        for mcp, js in FINGERS.items():
            for j in js:
                loc[j] = loc[mcp] + (loc[j] - loc[mcp]) * 0.22
                loc[j, 2] += 0.03                                              # curl toward the palm side
    return loc


def project(X):
    return np.stack([X[:, 0] / X[:, 2] * F + CX, X[:, 1] / X[:, 2] * F + CY], 1)


def pose_at(k, n_still=8, n_move=24):
    a = 0.0 if k < n_still else min((k - n_still) / n_move, 1.0)
    Rh = R.from_euler("z", 40 * a, degrees=True).as_matrix() @ R.from_euler("x", 25, degrees=True).as_matrix()
    return Rh, np.array([-0.05 + 0.20 * a, 0.0, 0.75 + 0.05 * a])


def frame_data(k, closed, with_object, hand_noise=0.0):
    Rh, th = pose_at(k)
    X = hand_shape(closed) @ Rh.T + th
    zp = X[[0, 5, 9, 13, 17], 2].mean()
    hand = Hand("Right", project(X), 0.95, (X[:, 2] - X[0, 2]) * F / (W * zp))
    d = np.full((H, W), F * B / 2.5, np.float32)
    for (u, v), z in zip(project(X), X[:, 2]):
        d[int(v) - 7:int(v) + 8, int(u) - 7:int(u) + 8] = F * B / z
    mask = None
    obj_world = BLOCK @ Rh.T + th
    if with_object:
        uv = project(obj_world)
        mask = np.zeros((H, W), bool)
        order = np.argsort(-obj_world[:, 2])                                   # far first, so nearer points win
        for (u, v), z in zip(uv[order], obj_world[order, 2]):
            ui, vi = int(u), int(v)
            d[vi - 1:vi + 2, ui - 1:ui + 2] = F * B / z
            mask[vi - 1:vi + 2, ui - 1:ui + 2] = True
        ys, xs = np.nonzero(mask)
    return hand, d, mask, obj_world, X, (Rh, th)


class Seq:
    def __init__(self, items):
        self.items, self.i = items, 0

    def detect(self, *a):
        v = self.items[min(self.i, len(self.items) - 1)]
        self.i += 1
        return v


def run_sequence(n=44, hide_from=30, open_from=36):
    frames, hands, dets = [], [], []
    for k in range(n):
        closed = k < open_from
        hand, d, mask, obj_w, X, pose = frame_data(k, closed, with_object=k < hide_from)
        frames.append((d, obj_w, X, pose))
        hands.append([hand])
        if mask is not None:
            ys, xs = np.nonzero(mask)
            dets.append([Det("cup", 0.9, (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())), mask)])
        else:
            dets.append([])
    a = HeldAnalyzer(F, B, CX, CY, detector=Seq(dets), hand_tracker=Seq(hands))
    outs = []
    for k in range(n):
        d = frames[k][0]
        outs.append(a.frame(k / 15, np.zeros((H, W, 3), np.uint8), d, np.isfinite(d), np.eye(3), np.zeros(3)))
    return outs, frames


def test_the_hand_is_placed_in_3d_close_to_the_truth_in_every_frame():
    outs, frames = run_sequence()
    errs = []
    for o, (d, obj_w, X, pose) in zip(outs, frames):
        h = o["hands"][0]
        assert h["solve"]["ok"] and h["frame"] is not None
        errs.append(float(np.sqrt(np.nanmean(np.sum((h["xyz_world"] - X) ** 2, axis=1)))))
    assert np.median(errs) < 0.02 and max(errs) < 0.04, (np.median(errs), max(errs))


def test_a_grab_makes_a_model_that_is_filled_from_the_pictures():
    outs, _ = run_sequence()
    grabs = [k for k, o in enumerate(outs) if any(e["type"] == "grab" for e in o["events"])]
    assert grabs, "no grab detected"
    seen = [m for o in outs for m in o["models"] if m["state"].startswith("observed")]
    assert seen and max(m["n_points"] for m in seen) > 50 and max(m["n_views"] for m in seen) >= 3


def test_the_object_stays_in_the_hand_after_the_detector_loses_it_and_moves_with_the_hand():
    outs, frames = run_sequence()
    hidden = [(k, m) for k, o in enumerate(outs) for m in o["models"] if m["state"].startswith("hidden")]
    assert len(hidden) >= 3, [m["state"] for o in outs for m in o["models"]][-8:]
    errs = []
    for k, m in hidden:
        Rh, th = m["frame"]
        world = m["points_local"] @ Rh.T + th
        truth = frames[k][1]
        dists = np.linalg.norm(world[:, None, :] - truth[None, ::4, :], axis=2).min(axis=1)
        errs.append(float(np.median(dists)))
    assert max(errs) < 0.02, errs
    # and it really moved with the hand: the model's world centroid travelled as far as the palm did
    first, last = hidden[0][1], hidden[-1][1]
    c = lambda m: (m["points_local"] @ m["frame"][0].T + m["frame"][1]).mean(0)
    p = lambda m: m["frame"][1]
    assert abs(np.linalg.norm(c(last) - c(first)) - np.linalg.norm(p(last) - p(first))) < 0.02


def test_the_model_never_grows_outliers_from_the_background():
    outs, _ = run_sequence()
    for o in outs:
        for m in o["models"]:
            if m["n_points"]:
                assert np.abs(m["points_local"]).max() < 0.25, "a point far from the hand got into the object model"


def test_release_freezes_the_object_in_the_world_where_the_hand_put_it():
    outs, frames = run_sequence()
    rel = [(k, r) for k, o in enumerate(outs) for r in o["released"]]
    assert rel, [e["type"] for o in outs for e in o["events"]]
    k, r = rel[0]
    truth = frames[k][1]
    d = np.linalg.norm(r["points_world"][:, None, :] - truth[None, ::4, :], axis=2).min(axis=1)
    assert np.median(d) < 0.02 and r["label"] == "cup" and r["box_world"] is not None


def test_a_hand_that_cannot_be_placed_in_3d_keeps_the_object_unattached_and_says_why():
    outs, frames = run_sequence()
    hand, d, mask, obj_w, X, pose = frame_data(10, True, True)
    d[:] = np.nan                                                              # no trusted depth at all
    a = HeldAnalyzer(F, B, CX, CY, detector=Seq([[]]), hand_tracker=Seq([[hand]]))
    o = a.frame(0.0, np.zeros((H, W, 3), np.uint8), d, np.zeros((H, W), bool), np.eye(3), np.zeros(3))
    h = o["hands"][0]
    assert not h["solve"]["ok"] and h["frame"] is None and "depth" in h["solve"]["reason"]


def test_rerun_gets_the_hand_its_frame_and_the_object_inside_that_frame_and_the_frozen_object_after_release():
    import os
    import tempfile
    from gabai.pipeline.viz import Viz

    class Capture(Viz):
        def __init__(self):
            self.cal, self.size, self.t0, self.n_logged, self._held_paths = {"f": F}, (W, H), None, 0, set()
            self._hand_paths, self._obj_paths = set(), set()
            self.log_calls = []

        def _log(self, path, arch, **kw):
            self.log_calls.append((path, type(arch).__name__))

    v = Capture()
    outs, _ = run_sequence()
    held_during, after = set(), None
    for k, o in enumerate(outs):
        v.log_calls.clear()
        v.hands3d(o, {})
        paths = {p for p, _ in v.log_calls}
        if any(m["n_points"] for m in o["models"]):
            held_during |= {p for p in paths if "/frame/held/" in p}
            assert any(p.startswith("world/hands/hand") and p.endswith("/frame") for p in paths)      # the object is logged under the hand-frame transform: it cannot detach
        if o["released"]:
            after = [(p, t) for p, t in v.log_calls]
    assert any(p.endswith("/frame/held/cup0") for p in held_during) and any(p.endswith("/box") for p in held_during)
    assert after and ("world/objects/cup0/model", "Points3D") in after
    cleared = [c for c in after if c[1] == "Clear"]
    assert any("/frame/held/cup0" in p for p, _ in cleared), "the hand-attached copy must be cleared when the object is put down"

    # the same data goes into a real Rerun recording without errors
    path = os.path.join(tempfile.mkdtemp(), "t.rrd")
    real = Viz({"f": F, "cx": CX, "cy": CY}, (W, H), save=path, viewer=False)
    for k, o in enumerate(outs):
        real.at(k, k / 15)
        real.hands3d(o, {})
    import rerun as rr
    rr.disconnect()
    assert os.path.getsize(path) > 1000
