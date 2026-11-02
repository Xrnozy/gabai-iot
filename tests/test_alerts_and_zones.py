import numpy as np

from gabai.alerts import AlertQueue, LinkMonitor
from gabai.zones import analyze

F, B, H, CY, W, HT = 310.0, 0.12, 1.5, 180.0, 640, 360  # focal px, baseline m, camera height m


def floor_disp():
    """Disparity of a flat floor seen by a level camera (rows below the horizon)."""
    d = np.full((HT, W), np.nan, np.float32)
    for v in range(int(CY) + 8, HT):
        d[v, :] = B * (v - CY) / H
    return d


def add_wall(d, z, x0=0, x1=W):
    d[:, x0:x1] = F * B / z
    return d


def test_floor_only_is_clear():
    r = analyze(floor_disp(), F, B, H, CY)
    assert r.alert is None


def test_wall_ahead_reports_rounded_distance():
    r = analyze(add_wall(floor_disp(), 1.0, 200, 440), F, B, H, CY)
    assert r.alert == ("ahead", 1.0)


def test_wall_on_left_only():
    r = analyze(add_wall(floor_disp(), 1.5, 0, 150), F, B, H, CY)
    assert r.alert[0] == "left"


def test_mostly_unknown_is_cant_see_not_clear():
    d = np.full((HT, W), np.nan, np.float32)
    r = analyze(d, F, B, H, CY)
    assert r.alert == ("cant_see_ahead", None)


def test_no_floor_model_still_finds_wall_and_never_reports_drop():
    d = add_wall(floor_disp(), 1.0, 200, 440)
    assert analyze(d, F, B, None, CY).alert == ("ahead", 1.0)
    d2 = floor_disp()
    d2[250:, 200:440] = 1.0
    assert analyze(d2, F, B, None, CY).alert != ("drop_ahead", None)


def test_missing_floor_is_drop_ahead():
    d = floor_disp()
    d[250:, 200:440] = B * (250 - CY) / H * 0.2  # floor far lower than expected: step down
    assert analyze(d, F, B, H, CY).alert[0] == "drop_ahead"


def test_priority_cuts_in_and_dedup():
    q = AlertQueue()
    assert q.push("obstacle_far", priority=4, key="ahead", distance=2.0, now=0.0) == "ahead"  # spoken
    assert q.push("obstacle_far", priority=4, key="ahead", distance=2.0, now=1.0) is None      # dedup within 3 s
    assert q.push("obstacle_far", priority=4, key="ahead", distance=1.0, now=1.5) == "ahead"   # closer: repeats
    assert q.push("system", priority=1, key="stopped", distance=None, now=1.6) == "stopped"    # higher cuts in
    assert q.push("obstacle_far", priority=4, key="left", distance=2.0, now=1.7) is None       # lower while system active


def test_link_monitor_stale_and_dead():
    m = LinkMonitor()
    assert m.check(now=0.0, last_frame_t=0.0, frame_age=0.05) is None
    assert m.check(now=1.0, last_frame_t=0.9, frame_age=0.35) == "tracking_delayed"
    assert m.check(now=2.5, last_frame_t=1.0, frame_age=0.0) == "gabai_stopped"
    assert m.check(now=5.0, last_frame_t=1.0, frame_age=0.0) is None      # repeats only every 5 s
    assert m.check(now=7.6, last_frame_t=1.0, frame_age=0.0) == "gabai_stopped"


def tilted_floor(theta_deg, cam_h=H):
    """Disparity of a flat floor seen by a camera pitched down by theta (v measured from the principal row)."""
    th = np.radians(theta_deg)
    v = np.arange(HT, dtype=np.float32)[:, None]
    d = B * (np.cos(th) * (v - CY) + F * np.sin(th)) / cam_h
    d = np.where(d > 0.5, d, np.nan).astype(np.float32)
    return np.repeat(d, W, axis=1), np.array([0.0, -np.cos(th), -np.sin(th)])


def test_tilted_camera_floor_is_clear_when_the_up_direction_is_known():
    d, up = tilted_floor(20)
    assert analyze(d, F, B, H, CY, up_cam=up).alert is None


def test_assuming_a_level_camera_on_a_tilted_one_gives_a_false_alert():
    d, up = tilted_floor(20)
    assert analyze(d, F, B, H, CY).alert is not None


def test_wall_is_still_found_with_a_tilted_camera():
    d, up = tilted_floor(15)
    d[:, 200:440] = np.maximum(d[:, 200:440], F * B / 1.0)
    assert analyze(d, F, B, H, CY, up_cam=up).alert == ("ahead", 1.0)
