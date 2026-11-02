import time

import numpy as np

from gabai.objects3d import ObjectMap, Objects3D

R, T0 = np.eye(3), np.zeros(3)
CAL = dict(f=100.0, cx=2.0, cy=2.0, B=0.05)


def det(label, p):
    return dict(label=label, conf=0.9, p_w=np.array(p, float), dims=np.array([.2, .2, .2]), R_obj=np.eye(3))


def wait(o):
    for _ in range(200):
        s = o.poll()
        if s is not None:
            return s
        time.sleep(0.01)
    raise AssertionError("no result")


def made(lift, **kw):
    o = Objects3D(lambda: lift, warm=False, **kw)
    for _ in range(200):                                                     # the worker loads on its own thread
        if o.lift is not None:
            break
        time.sleep(0.01)
    return o


def frame():
    return np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), np.ones((4, 4), bool)


def test_submit_never_waits_and_drops_while_busy():
    o = made(lambda *a: (time.sleep(0.3), [])[1], every_s=0.0)
    t0 = time.perf_counter()
    assert o.submit(0.0, *frame(), CAL, R, T0)
    assert not o.submit(0.1, *frame(), CAL, R, T0)                            # worker busy: dropped, not queued
    assert time.perf_counter() - t0 < 0.1
    wait(o)
    assert o.submit(0.5, *frame(), CAL, R, T0)


def test_interval():
    o = made(lambda *a: [], every_s=1.0)
    assert o.submit(0.0, *frame(), CAL, R, T0)
    wait(o)
    assert not o.submit(0.5, *frame(), CAL, R, T0) and o.submit(1.1, *frame(), CAL, R, T0)


def test_looks_become_one_landmark_alias_and_range():
    seq = iter([[det("tv", [0, 0, 1])], [det("monitor", [0.05, 0, 1])], [det("tv", [0.9, 0, 1])], [det("chair", [0, 0, 5])]])
    o = made(lambda *a: next(seq), every_s=0.0, min_looks=2)
    for i in range(4):
        o.submit(float(i), *frame(), CAL, R, T0)
        s = wait(o)
    assert len(o.tracks) == 2                                               # tv/monitor 5 cm apart is one object; the one 0.9 m away is another; the chair at 5 m is out of range
    assert [m["looks"] for m in s["landmarks"]] == [2]


def test_lift_error_keeps_thread_alive():
    calls = iter([RuntimeError("boom"), [det("sofa", [0, 0, 1])]])

    def lift(*a):
        r = next(calls)
        if isinstance(r, Exception):
            raise r
        return r
    o = made(lift, every_s=0.0, min_looks=1)
    o.submit(0.0, *frame(), CAL, R, T0)
    wait(o)
    o.submit(1.0, *frame(), CAL, R, T0)
    assert len(wait(o)["landmarks"]) == 1 and isinstance(o.error, RuntimeError)


def test_failed_load_stays_off_and_reset_forgets():
    def bad():
        raise FileNotFoundError("no weights")
    o = Objects3D(bad)
    time.sleep(0.2)
    assert not o.submit(0.0, *frame(), CAL, R, T0) and isinstance(o.error, FileNotFoundError)
    o = made(lambda *a: [det("sofa", [0, 0, 1])], every_s=0.0, min_looks=1)
    o.submit(0.0, *frame(), CAL, R, T0)
    wait(o)
    o.reset()
    o.submit(1.0, *frame(), CAL, R, T0)
    assert wait(o)["landmarks"][0]["looks"] == 1                            # the first look was forgotten


def d2(label, p, conf=0.8, dims=(.5, .1, .4)):
    return dict(label=label, conf=conf, p_w=np.array(p, float), dims=np.array(dims, float), R_obj=np.eye(3))


def feed(m, labels_at, n=1, t0=0.0, view=lambda p: False):
    for i, (label, p, conf) in enumerate(labels_at * n):
        m.update(t0 + i, [d2(label, p, conf)], view)


def test_same_object_new_label_keeps_id_and_label_needs_evidence():
    m = ObjectMap(min_looks=1)
    feed(m, [("tv", [0, 0, 1], 0.8)], 4)
    feed(m, [("monitor", [0.03, 0, 1], 0.4)], 1, t0=10)                      # one odd, weaker detection: same object, same label
    assert len(m.objs) == 1 and m.objs[0]["label"] == "tv"
    feed(m, [("monitor", [0.02, 0, 1], 0.9)], 6, t0=20)                      # repeated, stronger evidence: renamed, same id, no second object
    assert len(m.objs) == 1 and m.objs[0]["label"] == "monitor" and m.objs[0]["id"] == 0


def test_different_label_overlapping_box_is_one_object_and_far_one_is_another():
    m = ObjectMap(min_looks=1)
    m.update(0, [d2("tv", [0, 0, 1])], lambda p: False)
    m.update(1, [d2("laptop", [0.05, 0, 1])], lambda p: False)               # boxes overlap a lot: same physical thing whatever it is called
    m.update(2, [d2("tv", [1.5, 0, 1])], lambda p: False)
    assert len(m.objs) == 2


def test_small_object_on_big_one_is_not_merged():
    m = ObjectMap(min_looks=1)
    m.update(0, [d2("table", [0, 0, 1], dims=(1.2, 0.05, 0.7))], lambda p: False)
    m.update(1, [d2("laptop", [0, -0.05, 1], dims=(0.3, 0.02, 0.2))], lambda p: False)
    assert len(m.objs) == 2


def test_late_duplicate_is_merged_into_the_older_object():
    m = ObjectMap(min_looks=1, gate=0.05)
    m.update(0, [d2("tv", [0, 0, 1])], lambda p: False)
    m.update(1, [d2("chair", [0.9, 0, 1])], lambda p: False)
    m.objs[1]["P"][:] = [np.array([0.0, 0.0, 1.0])]                          # the second one drifts onto the first: two copies of one thing
    m.objs[1]["ev"] = {"monitor": 1.0}
    m.objs[1]["n_label"] = {"monitor": 1}
    m._merge()
    assert len(m.objs) == 1 and m.objs[0]["id"] == 0 and "monitor" in m.objs[0]["ev"]


def test_unseen_object_is_kept_unless_in_view_and_missed_repeatedly():
    m = ObjectMap(min_looks=1, miss_drop=5)
    feed(m, [("sofa", [0, 0, 1], 0.9)], 3)
    for i in range(50):
        m.update(10 + i, [], lambda p: False)                                  # out of view or occluded for a long time: kept
    assert len(m.objs) == 1
    for i in range(4):
        m.update(100 + i, [], lambda p: True)
    assert len(m.objs) == 1                                                  # not yet enough misses
    m.update(110, [], lambda p: True)
    assert len(m.objs) == 0                                                  # five frames in a row in view and not found: gone


def test_one_off_never_seen_again_is_dropped_established_stays():
    m = ObjectMap(min_looks=3, stale_s=20)
    m.update(0, [d2("cup", [0, 0, 1])], lambda p: False)
    feed(m, [("sofa", [2, 0, 1], 0.9)], 4)
    m.update(100, [], lambda p: False)
    assert [T["label"] for T in m.objs] == ["sofa"]
