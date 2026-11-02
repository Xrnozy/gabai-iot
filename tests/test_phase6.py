import threading
import time

import numpy as np

from gabai.background import BackgroundRunner
from gabai.rooms import RoomUnavailable, layout_from_planes, scenescript_layout, shaper_mesh


def box_room(rng, n=4000, noise=0.01, L=6.0, W=4.0, H=2.6):
    f = np.c_[rng.uniform(0, L, n), rng.uniform(0, W, n), np.zeros(n)]
    c = np.c_[rng.uniform(0, L, n // 2), rng.uniform(0, W, n // 2), np.full(n // 2, H)]
    w = [np.c_[np.zeros(n), rng.uniform(0, W, n), rng.uniform(0, H, n)], np.c_[np.full(n, L), rng.uniform(0, W, n), rng.uniform(0, H, n)],
         np.c_[rng.uniform(0, L, n), np.zeros(n), rng.uniform(0, H, n)], np.c_[rng.uniform(0, L, n), np.full(n, W), rng.uniform(0, H, n)]]
    p = np.vstack([f, c] + w)
    return p + rng.normal(0, noise, p.shape)


# -- rooms (RANSAC planes, the spec's backup for SceneScript) ---------------------------------------------------------------------
def test_ransac_finds_floor_ceiling_and_four_walls_and_the_room_size():
    r = layout_from_planes(box_room(np.random.default_rng(0)))
    kinds = sorted(p["kind"] for p in r["planes"])
    assert kinds.count("floor") == 1 and kinds.count("ceiling") == 1 and kinds.count("wall") == 4
    assert abs(r["height_m"] - 2.6) < 0.05 and abs(r["floor_z"]) < 0.03
    assert abs(r["size_xy_m"][0] - 6.0) < 0.15 and abs(r["size_xy_m"][1] - 4.0) < 0.15
    assert r["source"] == "RANSAC planes" and r["scenescript"] == "not used"


def test_furniture_does_not_become_a_wall_and_the_room_is_still_found():
    rng = np.random.default_rng(1)
    sofa = np.c_[rng.uniform(1, 2.5, 1500), rng.uniform(1, 2, 1500), rng.uniform(0, 0.8, 1500)]
    r = layout_from_planes(np.vstack([box_room(rng), sofa]))
    assert sum(p["kind"] == "wall" for p in r["planes"]) == 4


def test_a_cloud_with_no_planes_gives_an_empty_layout_with_the_reason():
    r = layout_from_planes(np.random.default_rng(2).uniform(0, 3, (600, 3)))
    assert r["planes"] == [] and r["note"]
    r2 = layout_from_planes(np.zeros((10, 3)))
    assert r2["planes"] == [] and "few points" in r2["note"]


def test_a_tilted_cloud_is_not_called_floor_and_wall_without_gravity_alignment():
    p = box_room(np.random.default_rng(3))
    a = np.radians(25)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    r = layout_from_planes(p @ Rx.T, tolerance_deg=10)
    assert sum(q["kind"] == "other" for q in r["planes"]) >= 1                  # leaning planes are reported as 'other', not forced into floor/wall


def test_scenescript_and_shaper_say_they_are_not_available_instead_of_pretending():
    for fn in (scenescript_layout, shaper_mesh):
        try:
            fn(np.zeros((10, 3)))
            assert False
        except RoomUnavailable as e:
            assert "not" in str(e).lower() and "available" in str(e).lower()


# -- background jobs only when idle -----------------------------------------------------------------------------------------------
IDLE = {"background_ok": True, "tier": "normal"}
BUSY = {"background_ok": False, "tier": "normal"}


def spin(stop, n=2000):
    for i in range(n):
        if stop():
            return "stopped"
        time.sleep(0.002)
    return "done"


def wait(pred, t=3.0):
    t0 = time.time()
    while time.time() - t0 < t:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_a_job_does_not_start_while_the_rig_is_busy():
    b = BackgroundRunner()
    b.submit("room", lambda stop: "ok")
    for _ in range(5):
        b.tick(BUSY, lane_ms={})
    assert b.status()["running"] is None and b.status()["queued"] == ["room"]


def test_a_job_runs_when_idle_and_its_result_is_kept():
    b = BackgroundRunner()
    b.submit("room", lambda stop: "layout")
    b.tick(IDLE, {})
    assert wait(lambda: (b.tick(IDLE, {}), b.status()["done"])[1])
    assert b.result("room") == {"ok": True, "value": "layout", "stopped": False}


def test_busy_signal_stops_a_running_job_quickly_and_it_stays_queued_for_later():
    b = BackgroundRunner()
    b.submit("shape", spin)
    b.tick(IDLE, {})
    assert wait(lambda: b.status()["running"] == "shape")
    t0 = time.time()
    b.tick(BUSY, {})
    assert wait(lambda: b.status()["running"] is None, 1.0) and time.time() - t0 < 1.0
    assert b.status()["queued"] == ["shape"]


def test_a_slow_lane_stops_background_work_even_if_the_governor_said_ok():
    b = BackgroundRunner()
    b.submit("shape", spin)
    b.tick(IDLE, {})
    assert wait(lambda: b.status()["running"] == "shape")
    b.tick(IDLE, {"depth": 140.0})
    assert wait(lambda: b.status()["running"] is None, 1.0)


def test_a_job_that_raises_is_recorded_and_does_not_break_the_loop():
    b = BackgroundRunner()

    def bad(stop):
        raise ValueError("boom")

    b.submit("bad", bad)
    b.tick(IDLE, {})
    assert wait(lambda: (b.tick(IDLE, {}), b.status()["done"])[1])
    r = b.result("bad")
    assert not r["ok"] and "boom" in r["error"]


def test_a_job_that_ignores_stop_is_reported_not_silently_waited_for():
    b = BackgroundRunner(stop_grace_s=0.2)
    release = threading.Event()
    b.submit("deaf", lambda stop: release.wait(3.0))
    b.tick(IDLE, {})
    assert wait(lambda: b.status()["running"] == "deaf")
    b.tick(BUSY, {})
    time.sleep(0.4)
    b.tick(BUSY, {})
    assert b.status()["ignoring_stop"] == "deaf"
    release.set()


def test_the_tick_itself_never_blocks_the_caller():
    b = BackgroundRunner()
    b.submit("slow", lambda stop: time.sleep(0.5))
    t0 = time.time()
    b.tick(IDLE, {})
    b.tick(BUSY, {})
    assert time.time() - t0 < 0.1
    time.sleep(0.7)
