import numpy as np

from gabai.grasp import GraspTracker

DT = 1 / 15


def hand(p, closed=True, hid="R"):
    return {"id": hid, "palm": None if p is None else np.asarray(p, float), "closed": closed, "open": not closed}


def obj(p, label="mug", r=0.05):
    return {"label": label, "center": None if p is None else np.asarray(p, float), "radius": r}


def run(tr, frames):
    ev = []
    for k, (hands, objs) in enumerate(frames):
        ev += tr.update(k * DT, hands, objs)
    return ev


def types(ev):
    return [e["type"] for e in ev]


def pickup_and_place(n_near=6, n_move=15, n_rest=10, follow=True):
    """Hand closes on a mug at the origin, lifts it 30 cm and puts it down there, opens."""
    frames = []
    o0 = np.array([0, 0, 0.0])
    for _ in range(n_near):
        frames.append(([hand(o0)], {"m": obj(o0)}))
    for i in range(1, n_move + 1):
        p = o0 + np.array([0, 0, 0.3 * i / n_move])
        frames.append(([hand(p)], {"m": obj(p if follow else o0)}))
    end = o0 + np.array([0, 0, 0.3])
    for _ in range(n_rest):
        frames.append(([hand(end, closed=False)], {"m": obj(end)}))
    return frames


def test_pickup_and_place_gives_one_grab_then_one_release_at_the_new_place():
    ev = run(GraspTracker(), pickup_and_place())
    assert types(ev) == ["grab", "release"]
    assert ev[0]["label"] == "mug" and ev[1]["label"] == "mug"
    assert abs(ev[1]["pos"][2] - 0.3) < 0.05


def test_a_fist_passing_by_without_stopping_is_not_a_grab():
    frames = []
    for i in range(30):
        frames.append(([hand([0.04 * (i - 15) * 0.5, 0, 0])], {"m": obj([0, 0, 0])}))      # sweeps through at 0.3 m/s
    assert "grab" not in types(run(GraspTracker(), frames))


def test_an_open_hand_resting_on_the_object_is_not_a_grab():
    frames = [([hand([0, 0, 0], closed=False)], {"m": obj([0, 0, 0])}) for _ in range(30)]
    assert run(GraspTracker(), frames) == []


def test_a_closed_hand_that_leaves_while_the_object_stays_cancels_and_says_why():
    ev = run(GraspTracker(), pickup_and_place(follow=False))
    assert "grab" not in types(ev) and ev and ev[0]["type"] == "cancel" and "stayed" in ev[0]["reason"]


def test_when_the_object_cannot_be_seen_during_the_move_the_grab_is_confirmed_and_marked_as_unobserved():
    frames = pickup_and_place()
    frames = [(h, {"m": obj(None)} if i >= 6 else o) for i, (h, o) in enumerate(frames)]
    ev = run(GraspTracker(), frames)
    assert types(ev)[0] == "grab" and ev[0]["object_observed_following"] is False


def test_the_held_object_follows_the_palm_with_the_offset_it_had_at_the_grab():
    tr = GraspTracker()
    frames = pickup_and_place(n_rest=0)
    run(tr, frames)
    held = tr.held()
    assert len(held) == 1 and abs(held["R"]["center"][2] - 0.3) < 0.03


def test_the_nearest_object_is_picked_when_two_are_close():
    frames = []
    for _ in range(6):
        frames.append(([hand([0, 0, 0])], {"a": obj([0.02, 0, 0], "cup"), "b": obj([0.12, 0, 0], "keys")}))
    for i in range(1, 16):
        p = np.array([0, 0, 0.3 * i / 15])
        frames.append(([hand(p)], {"a": obj(p + [0.02, 0, 0], "cup"), "b": obj([0.12, 0, 0], "keys")}))
    ev = run(GraspTracker(), frames)
    assert ev[0]["type"] == "grab" and ev[0]["label"] == "cup"


def test_losing_the_hand_while_holding_reports_lost_after_the_timeout_and_keeps_the_last_place():
    frames = pickup_and_place(n_rest=0) + [([hand(None)], {"m": obj(None)}) for _ in range(30)]
    ev = run(GraspTracker(), frames)
    assert types(ev) == ["grab", "lost"] and ev[1]["pos"] is not None


def test_phrase_for_the_alert_queue():
    ev = run(GraspTracker(), pickup_and_place())
    from gabai.grasp import phrase
    assert phrase(ev[0]) == "holding: mug" and phrase(ev[1]) == "put down: mug"


def test_the_object_disappearing_from_the_detections_entirely_still_confirms_the_grab_with_its_label():
    frames = pickup_and_place()
    frames = [(h, {} if i >= 6 else o) for i, (h, o) in enumerate(frames)]            # the key is gone from the dict, not just centre=None
    ev = run(GraspTracker(), frames)
    assert ev and ev[0]["type"] == "grab" and ev[0]["label"] == "mug" and ev[0]["object_observed_following"] is False
