import numpy as np

from gabai.zones import analyze, MAX_WARN_DEPTH

F, B = 881.0, 0.0507
H, W = 480, 640
CX, CY = 319.5, 239.5


def floor_scene(cam_h=1.4):
    from gabai.zones import floor_disparity
    d = floor_disparity((H, W), F, B, cam_h, CX, CY)
    return np.where(np.isfinite(d), d, np.nan)


def with_box(d, z, x0=260, x1=380, y0=250, y1=420):
    d = d.copy()
    d[y0:y1, x0:x1] = F * B / z
    return d


def test_evidence_is_off_by_default_and_the_decision_is_unchanged_when_on():
    d = with_box(floor_scene(), 1.2)
    plain = analyze(d, F, B, 1.4, CY, cx=CX)
    rich = analyze(d, F, B, 1.4, CY, cx=CX, evidence=True)
    assert plain.evidence is None and rich.evidence is not None
    assert plain.alert == rich.alert and plain.distances == rich.distances and plain.valid_fraction == rich.valid_fraction
    assert plain.alert[0] == "ahead"


def test_evidence_names_the_nearest_pixel_the_depth_and_how_uncertain_it_is():
    d = with_box(floor_scene(), 1.2)
    ev = analyze(d, F, B, 1.4, CY, cx=CX, evidence=True).evidence
    a = ev["bands"]["ahead"]
    assert 260 <= a["nearest_px"][0] < 380 and 250 <= a["nearest_px"][1] < 420
    assert abs(a["min_depth_m"] - 1.2) < 0.01 and abs(a["p5_depth_m"] - 1.2) < 0.01
    # half a pixel of disparity error at 1.2 m: dZ = Z^2 / (f B) * 0.5
    assert abs(a["depth_uncertainty_m"] - 1.2 ** 2 / (F * B) * 0.5) < 0.002
    assert a["reached_threshold"] and ev["floor_model"] == "level camera"
    assert any("ahead" in s and "1.2" in s for s in ev["decision_path"])


def test_floor_pixels_are_counted_as_removed_not_as_obstacle():
    # at 0.5 m camera height the floor reaches within 3 m in the lower rows (at 1.4 m it never does in a 480-row image)
    ev = analyze(floor_scene(0.5), F, B, 0.5, CY, cx=CX, evidence=True).evidence
    assert ev["bands"]["ahead"]["n_obstacle_px"] == 0 and ev["bands"]["ahead"]["n_removed_as_floor"] > 1000


def test_cant_see_ahead_says_why():
    d = np.full((H, W), np.nan, np.float32)
    r = analyze(d, F, B, 1.4, CY, cx=CX, evidence=True)
    assert r.alert[0] == "cant_see_ahead"
    ev = r.evidence
    assert ev["valid_fraction_below_horizon"] < 0.01
    assert any("cant_see_ahead" in s and "0.40" in s for s in ev["decision_path"])


def test_a_tilted_camera_is_reported_as_using_the_imu_up_vector():
    up = np.array([0.0, -np.cos(0.3), -np.sin(0.3)])
    ev = analyze(floor_scene(), F, B, 1.4, CY, up_cam=up, cx=CX, evidence=True).evidence
    assert ev["floor_model"] == "tilted by IMU up-vector"


def test_no_camera_height_means_no_floor_model_and_no_drop_check():
    ev = analyze(floor_scene(), F, B, None, CY, cx=CX, evidence=True).evidence
    assert ev["floor_model"] == "none (obstacles by depth only)"


def test_far_objects_are_counted_as_out_of_range():
    d = with_box(floor_scene(), MAX_WARN_DEPTH + 1.5)
    ev = analyze(d, F, B, 1.4, CY, cx=CX, evidence=True).evidence
    assert ev["bands"]["ahead"]["n_beyond_range"] > 1000


def test_evidence_records_both_the_optical_depth_and_the_horizontal_range_on_a_tilted_camera():
    from gabai.sim import Box, Camera, look_pose, render_disparity
    cam = Camera(F, B, CX, CY, (640, 480))
    R, t = look_pose(np.array([0.0, 0.0, 1.2]), pitch_deg=35.0)
    disp = render_disparity(cam, R, t, boxes=[Box(lo=(1.5, -0.4, 0.0), hi=(1.9, 0.4, 0.4))])
    ev = analyze(disp, F, B, 1.2, CY, up_cam=R[2, :], cx=CX, evidence=True).evidence
    b = ev["bands"]["ahead"]
    assert abs(ev["pitch_down_deg"] - 35.0) < 0.5
    assert 1.6 < b["p5_depth_m"] < 1.8                                   # what the alert quotes: depth along the optical axis
    assert 1.4 < b["p5_horizontal_range_m"] < 1.6                        # how far the box really is along the ground (true 1.5 m)
    assert b["p5_depth_m"] - b["p5_horizontal_range_m"] > 0.1            # the alert is optimistic here: documented, not hidden


def test_a_level_camera_has_no_difference_between_depth_and_range_straight_ahead():
    d = with_box(floor_scene(0.5), 1.2)
    ev = analyze(d, F, B, 0.5, CY, cx=CX, evidence=True).evidence
    b = ev["bands"]["ahead"]
    assert ev["pitch_down_deg"] == 0.0 and abs(b["p5_depth_m"] - b["p5_horizontal_range_m"]) < 0.08
