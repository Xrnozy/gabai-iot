import os
import tempfile

import numpy as np

from gabai.memory import ObjectMemory, describe


HX = np.array([1.0, 0.0])


def mem():
    return ObjectMemory(os.path.join(tempfile.mkdtemp(), "m.db"))


def test_a_new_sighting_is_stored_and_found_by_name():
    m = mem()
    m.observe("keys", [2.0, 1.0, 0.8], t=100.0)
    r = m.where("keys", now=160.0)
    assert len(r) == 1 and np.allclose(r[0]["pos"], [2.0, 1.0, 0.8]) and r[0]["age_s"] == 60.0 and r[0]["n_obs"] == 1


def test_seeing_it_again_nearby_updates_the_same_object_and_far_away_makes_a_second_place():
    m = mem()
    m.observe("keys", [2.0, 1.0, 0.8], 10.0)
    m.observe("keys", [2.1, 1.0, 0.8], 20.0)
    assert len(m.where("keys", now=30)) == 1 and m.where("keys", now=30)[0]["n_obs"] == 2
    m.observe("keys", [6.0, 4.0, 0.8], 40.0)
    r = m.where("keys", now=50)
    assert len(r) == 2 and np.allclose(r[0]["pos"], [6.0, 4.0, 0.8])          # newest first


def test_it_survives_closing_and_reopening_the_file():
    d = os.path.join(tempfile.mkdtemp(), "m.db")
    m = ObjectMemory(d)
    m.observe("cup", [1, 1, 1], 5.0)
    m.close()
    assert len(ObjectMemory(d).where("cup", now=6.0)) == 1


def test_a_place_event_is_stored_as_placed_and_beats_an_old_sighting_in_the_answer():
    m = mem()
    m.observe("keys", [2.0, 1.0, 0.8], 10.0)
    m.apply_event({"type": "grab", "label": "keys", "pos": np.array([2.0, 1.0, 0.8]), "t": 50.0})
    assert m.where("keys", now=60.0)[0]["state"] == "held"
    m.apply_event({"type": "release", "label": "keys", "pos": np.array([5.0, 2.0, 0.9]), "t": 80.0})
    r = m.where("keys", now=90.0)
    assert r[0]["state"] == "placed" and np.allclose(r[0]["pos"], [5.0, 2.0, 0.9]) and r[0]["source"] == "placed by hand"


def test_a_lost_hand_marks_the_last_position_unobserved_and_says_so():
    m = mem()
    m.apply_event({"type": "grab", "label": "keys", "pos": np.array([1.0, 1.0, 1.0]), "t": 1.0})
    m.apply_event({"type": "lost", "label": "keys", "pos": np.array([1.5, 1.0, 1.0]), "t": 5.0})
    r = m.where("keys", now=6.0)[0]
    assert r["state"] == "unobserved" and "unobserved" in describe(r, np.array([0, 0, 1.0]), HX)


def test_marking_the_spot_empty_when_in_view_lowers_trust_instead_of_deleting():
    m = mem()
    m.observe("keys", [2.0, 1.0, 0.8], 10.0)
    m.mark_absent("keys", [2.0, 1.0, 0.8], 30.0)
    r = m.where("keys", now=40.0)[0]
    assert r["absent_checks"] == 1 and r["last_absent_t"] == 30.0 and "not seen there since" in describe(r, np.zeros(3), HX).lower()


def test_unknown_name_gives_an_honest_not_found():
    m = mem()
    assert m.where("passport", now=1.0) == []
    assert m.answer("passport", np.zeros(3), HX, now=1.0) == "I have not seen passport."


def test_the_spoken_answer_gives_direction_distance_and_age_from_the_current_pose():
    m = mem()
    m.observe("keys", [3.0, 1.0, 0.8], 100.0)                                   # 3 m ahead, 1 m to the left (x forward, y left)
    s = m.answer("keys", np.array([0, 0, 1.4]), HX, now=100.0 + 300)
    assert "3" in s and "ahead" in s and "left" in s and "5 minutes" in s


def test_distance_and_bearing_use_the_facing_direction():
    from gabai.memory import bearing, heading
    b = bearing(np.array([0.0, 2.0, 0.8]), np.zeros(3), np.array([0.0, 1.0]))      # facing +y: a point on +y is straight ahead
    assert b["dist_m"] > 1.9 and b["side"] == "ahead"
    cam = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])                              # camera z (forward) = world +y, y down = world -z
    assert np.allclose(heading(cam), [0, 1])


def test_a_camera_looking_straight_down_has_no_heading_instead_of_a_made_up_one():
    from gabai.memory import heading
    try:
        heading(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1.0]]))              # forward axis (third column) points at the floor
    except ValueError:
        return
    assert False


def test_several_items_are_kept_apart():
    m = mem()
    for i, name in enumerate(["keys", "cup", "phone"]):
        m.observe(name, [i, 0, 0.8], 10.0 + i)
    assert {r["label"] for r in m.all(now=20.0)} == {"keys", "cup", "phone"}
