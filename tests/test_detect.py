import numpy as np

from gabai.detect import Det, DetectorUnavailable, UltralyticsDetector, det_depth, person_mask
from gabai.sim import Box, Camera, look_pose, render_disparity

CAM = Camera(f=881.0, B=0.0507, cx=319.5, cy=239.5, size=(640, 480))


def scene(z_front=2.0):
    R, t = look_pose(np.array([0.0, 0.0, 1.2]), pitch_deg=20.0)
    return render_disparity(CAM, R, t, boxes=[Box(lo=(z_front, -0.4, 0.0), hi=(z_front + 0.4, 0.4, 1.7))], noise_px=0.0), R, t


def mask_of(disp, xs, ys):
    m = np.zeros(disp.shape, bool)
    m[ys[0]:ys[1], xs[0]:xs[1]] = True
    return m


def test_missing_weights_say_where_to_put_them_and_never_download_on_their_own():
    try:
        UltralyticsDetector(weights="definitely/not/here.pt")
        raise AssertionError("must refuse")
    except DetectorUnavailable as e:
        msg = str(e)
    assert "definitely/not/here.pt" in msg and "yolo11n-seg" in msg and "download" in msg.lower()


def test_the_person_mask_is_the_union_of_person_masks_grown_a_little():
    shape = (480, 640)
    a = np.zeros(shape, bool)
    a[100:200, 100:150] = True
    b = np.zeros(shape, bool)
    b[300:400, 400:450] = True
    dets = [Det("person", 0.9, (100, 100, 150, 200), a), Det("chair", 0.9, (0, 0, 10, 10), np.ones(shape, bool)), Det("person", 0.8, (400, 300, 450, 400), b)]
    m = person_mask(dets, shape, grow_px=6)
    assert m[150, 125] and m[350, 425] and m[150, 97] and not m[250, 300]
    assert not m[5, 5]                                                  # the chair is not a person, so it is not masked out
    assert person_mask([], shape).sum() == 0


def test_a_box_without_a_mask_masks_its_rectangle():
    shape = (480, 640)
    m = person_mask([Det("person", 0.9, (10, 20, 60, 120), None)], shape, grow_px=0)
    assert m[50, 30] and not m[50, 5]


def test_depth_of_a_detection_comes_from_the_trusted_pixels_inside_it():
    disp, R, t = scene(2.0)
    ok = np.isfinite(disp)
    ys, xs = np.where(ok & (disp > 0.5))
    box_pixels = ok & (disp > (CAM.f * CAM.B / 2.4))                     # pixels nearer than 2.4 m: the box
    yy, xx = np.where(box_pixels)
    det = Det("person", 0.9, (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())), box_pixels)
    r = det_depth(det, disp, CAM.f, CAM.B)
    assert r["depth_m"]["kind"] == "estimated" and r["depth_m"]["v"] is not None
    assert 1.9 < r["depth_m"]["v"] < 2.6                                  # the person box's front is at about 2.0 m along x
    assert r["valid_fraction"] > 0.9 and r["n_pixels"] > 500


def test_a_detection_over_untrusted_depth_has_no_depth_not_a_guess():
    disp = np.full((480, 640), np.nan, np.float32)
    r = det_depth(Det("person", 0.9, (100, 100, 200, 300), None), disp, CAM.f, CAM.B)
    assert r["depth_m"]["v"] is None and r["depth_m"]["kind"] == "unavailable" and r["valid_fraction"] == 0.0


def test_a_detection_mostly_over_missing_depth_is_flagged_as_weak_evidence():
    disp = np.full((480, 640), np.nan, np.float32)
    disp[100:140, 100:200] = 30.0                                         # only 40 of 200 rows have depth
    r = det_depth(Det("person", 0.9, (100, 100, 200, 300), None), disp, CAM.f, CAM.B)
    assert r["depth_m"]["v"] is not None and r["valid_fraction"] < 0.3 and r["weak"] is True


def test_a_wide_spread_of_depths_inside_one_detection_is_reported():
    disp = np.full((480, 640), 40.0, np.float32)
    disp[:, 150:200] = 10.0                                               # a quarter of the box is far background
    r = det_depth(Det("person", 0.9, (100, 100, 200, 300), None), disp, CAM.f, CAM.B)
    assert r["spread_m"] > 3.0 and "background" in r["note"]
