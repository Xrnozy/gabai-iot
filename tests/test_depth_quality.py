import json
import os
import tempfile
import time

import cv2
import numpy as np
import pytest

from gabai import depthq as Q
from gabai.net import protocol as P
from gabai.pipeline.config import PipelineConfig, validate
from gabai.skewcomp import SkewCompensator
from tests.test_pipeline_run import H, W, ListViz, cfg_for, make_recording, plane_depth


def test_the_left_right_offset_travels_with_the_frame_and_old_senders_still_parse():
    assert P.unpack_frame_dt(P.pack_frame(b"LL", b"RRR", -12.5)) == (b"LL", b"RRR", -12.5)
    assert P.unpack_frame_dt(P.pack_frame(b"LL", b"RRR")) == (b"LL", b"RRR", None)
    assert P.unpack_frame(P.pack_frame(b"LL", b"RRR", 3.0)) == (b"LL", b"RRR")
    with pytest.raises(P.ProtocolError):
        P.unpack_frame_dt(P.pack_frame(b"LL", b"RRR", 1.0)[:-1])                        # a torn trailer is a damaged frame, not an offset


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 2.0)


def _at(img, t, vx=150.0, vy=60.0):
    return cv2.warpAffine(img, np.float32([[1, 0, vx * t], [0, 1, vy * t]]), (W, H), borderMode=cv2.BORDER_REFLECT)


def _err(a, b):
    return float(np.abs(a[40:-40, 40:-40].astype(np.float32) - b[40:-40, 40:-40].astype(np.float32)).mean())


def test_a_right_picture_taken_later_is_wound_back_to_the_left_cameras_instant():
    s, dt = _scene(), 0.0167
    sc = SkewCompensator()
    sc.apply(_at(s, -1 / 30), _at(s, -1 / 30), -1 / 30, None)                          # first pair: only remembers the picture
    left, right_late, truth = _at(s, 0.0), _at(s, dt), _at(s, 0.0)
    out, info = sc.apply(left, right_late, 0.0, dt * 1000)
    assert info["applied"] and info["shift_px"] > 2.0
    assert _err(out, truth) < 0.35 * _err(right_late, truth), (_err(out, truth), _err(right_late, truth))


def test_an_earlier_right_picture_is_corrected_the_other_way():
    s, dt = _scene(1), -0.012
    sc = SkewCompensator()
    sc.apply(_at(s, -1 / 30), _at(s, -1 / 30), -1 / 30, None)
    left, right_early, truth = _at(s, 0.0), _at(s, dt), _at(s, 0.0)
    out, info = sc.apply(left, right_early, 0.0, dt * 1000)
    assert info["applied"] and _err(out, truth) < 0.4 * _err(right_early, truth)


def test_nothing_is_changed_when_the_offset_is_unknown_tiny_or_the_motion_cannot_be_measured():
    s = _scene(2)
    left, right = _at(s, 0.0), _at(s, 0.01)
    sc = SkewCompensator()
    out, info = sc.apply(left, right, 0.0, None)
    assert out is right and not info["applied"] and "did not send" in info["why"]
    out, info = sc.apply(left, right, 1 / 30, 0.4)                                      # 0.4 ms: below the threshold
    assert out is right and not info["applied"]
    out, info = sc.apply(left, right, 5.0, 10.0)                                        # the last left picture is 5 s old
    assert out is right and not info["applied"] and "close enough" in info["why"]
    sc2 = SkewCompensator()
    sc2.apply(left, left, 0.0, None)
    out, info = sc2.apply(left, right, 1 / 30, 10.0)                                    # a rig standing still: nothing moved during 10 ms
    assert out is right and not info["applied"] and "too small" in info["why"]


def test_every_dropped_pixel_gets_one_named_cause_and_kept_pixels_stay_kept():
    h, w = 60, 120
    disp = np.full((h, w), 10.0, np.float32)
    disp[:, :4] = 30.0                                                                  # x < d: outside the right picture
    gray = np.tile(np.arange(w, dtype=np.float32) * 7 % 255, (h, 1)).astype(np.uint8)  # textured everywhere ...
    gray[:, 60:80] = 90                                                                 # ... except one flat band
    lr_ok = np.ones((h, w), bool)
    lr_ok[:, 50:90] = False                                                             # left/right disagree across the band and the texture beside it
    lr_ok[:, :4] = False
    after_mono = lr_ok.copy()
    after_mono[20:25, 10:20] = False                                                    # the monocular model vetoed this block
    final = after_mono.copy()
    final[40:42, 30:33] = False                                                         # a speckle was removed here
    lab_in = Q.reject_reasons(disp, lr_ok, after_mono, final, gray)
    assert (lab_in[:, :4] == Q.OUTSIDE).all()
    assert (lab_in[:, 65:75] == Q.FLAT).all() and (lab_in[:, 52:56] == Q.DISAGREE).all()
    assert (lab_in[40:42, 30:33] == Q.SPECKLE).all()
    assert (lab_in[final] == Q.TRUSTED).all() and not (lab_in[~final] == Q.TRUSTED).any()
    assert abs(sum(Q.reason_fractions(lab_in).values()) - 1.0) < 1e-9
    assert Q.paint(lab_in).shape == (h, w, 3)


def test_the_run_reports_why_depth_was_missing_and_draws_it():
    def fn(left, right):
        d, t, ms = plane_depth(left, right)
        t = t.copy()
        t[:, : W // 4] = False
        fn.reasons = np.where(t, Q.TRUSTED, Q.FLAT).astype(np.uint8)
        return np.where(t, d, np.nan).astype(np.float32), t, ms
    fn.reasons = None
    from gabai.pipeline.run import run
    ListViz.instances.clear()
    s = run(cfg_for(make_recording(n=6)), depth_fn=fn, viz_factory=ListViz, log_dir=tempfile.mkdtemp(), quiet=True)
    v = ListViz.instances[0]
    assert abs(s["depth_pixels_by_cause"]["trusted"] - 0.75) < 1e-3 and abs(s["depth_pixels_by_cause"]["no texture"] - 0.25) < 1e-3
    assert "depth/frac_no_texture" in v.paths and "stereo/valid" in v.paths


def test_a_calibration_off_the_designed_lens_spacing_is_reported_and_the_right_one_is_not():
    c = PipelineConfig()
    c.pipeline.mode, c.source.kind = "stereo_only", "live"
    z = dict(np.load("data/stereo_calib_new.npz"))
    for baseline, expect in ((0.0421, False), (0.0506, True)):
        path = os.path.join(tempfile.mkdtemp(), "c.npz")
        z["baseline"], z["square_mm"] = np.array(baseline), np.array(25.0)
        np.savez(path, **z)
        c.source.calib = path
        warns = validate(c)[1]
        assert any("DOES NOT AGREE" in w for w in warns) == expect, (baseline, warns)
    c.stereo.rig_baseline_mm = 0.0                                                      # the check can be switched off
    assert not any("DOES NOT AGREE" in w for w in validate(c)[1])


def test_camera_imu_rotation_measured_before_the_cameras_were_recalibrated_is_flagged():
    d = tempfile.mkdtemp()
    c = PipelineConfig()
    c.pipeline.mode, c.source.kind = "stereo_imu", "live"
    c.source.calib, c.source.imu_sync = os.path.join(d, "cal.npz"), os.path.join(d, "sync.json")
    np.savez(c.source.calib, **dict(np.load("data/stereo_calib_new.npz")))
    json.dump({"R_ci": np.eye(3).tolist(), "delay_s": 0.02}, open(c.source.imu_sync, "w"))
    old = time.time() - 3600
    os.utime(c.source.imu_sync, (old, old))
    assert any("older than" in w for w in validate(c)[1])
    os.utime(c.source.imu_sync, None)
    assert not any("older than" in w for w in validate(c)[1])


def _calib_with(devices):
    z = dict(np.load("data/stereo_calib_new.npz"))
    for k in ("left_dev", "right_dev"):
        z.pop(k, None)
    if devices:
        z.update(left_dev=np.array(devices[0]), right_dev=np.array(devices[1]))
    path = os.path.join(tempfile.mkdtemp(), "c.npz")
    np.savez(path, **z)
    return path


def test_the_sender_must_read_the_cameras_in_the_order_the_calibration_was_made_with():
    from gabai.pipeline.sources import check_devices
    p = _calib_with(("/dev/video2", "/dev/video0"))
    ok, text = check_devices(p, {"left": "/dev/video2", "right": "/dev/video0"})
    assert ok and "confirmed" in text
    ok, text = check_devices(p, {"left": "/dev/video0", "right": "/dev/video2"})
    assert not ok and "WRONG CAMERA ORDER" in text and "--left /dev/video2 --right /dev/video0" in text
    ok, text = check_devices(p, {})                                                      # an older sender: allowed, but said out loud
    assert ok and "older" in text
    ok, text = check_devices(_calib_with(None), {"left": "/dev/video2", "right": "/dev/video0"})
    assert ok and "does not record" in text


def test_the_banner_names_the_file_baseline_and_camera_order():
    from gabai.pipeline.sources import calibration_banner
    text = "\n".join(calibration_banner(_calib_with(("/dev/video2", "/dev/video0"))))
    assert "baseline 41.7" in text and "left /dev/video2, right /dev/video0" in text and "turned 180 / 0" in text


def test_the_new_calibration_is_valid_for_the_pipeline_and_close_to_the_designed_spacing():
    c = PipelineConfig()
    c.pipeline.mode, c.source.kind = "stereo_only", "live"
    assert c.source.calib == "data/stereo_calib_new.npz"
    errs, warns = validate(c)
    assert not errs and not any("DOES NOT AGREE" in w for w in warns), (errs, warns)
    from gabai.tools.calibrate_stereo import Rectifier
    r = Rectifier(c.source.calib)
    assert r.size == (640, 360) and abs(r.B - 0.042) < 0.0015 and r.rotation == (180, 0)


def test_the_row_check_tells_a_calibration_that_fits_from_one_used_with_the_other_camera_order():
    from gabai.pipeline.sources import row_alignment_verdict
    from gabai.tools.calibrate_stereo import Rectifier
    l, r = (cv2.imread(os.path.join("data", "calib_pairs_15", f"pair_03_{s}.png")) for s in "LR")
    up_l, up_r = cv2.rotate(r, cv2.ROTATE_180), cv2.rotate(l, cv2.ROTATE_180)             # the same cameras read in the other order
    ok, text = row_alignment_verdict(*Rectifier("data/stereo_calib_new.npz").apply(up_l, up_r))
    assert ok and "fits" in text, text
    ok, text = row_alignment_verdict(*Rectifier("data/stereo_calib_new.npz").apply(l, r))
    assert not ok and "other order" in text, text


def test_a_calibration_made_from_the_pairs_has_the_designed_baseline_and_rows_that_line_up():
    z = np.load("data/stereo_calib_new.npz")
    assert abs(float(z["baseline"]) - 0.042) < 0.0015 and float(z["epipolar_err_px"]) < 0.25 and float(z["rms_stereo"]) < 0.8
    assert float(z["T"].ravel()[0]) < 0 and str(z["left_dev"]) == "/dev/video2" and str(z["right_dev"]) == "/dev/video0" and int(z["n_views"]) >= 12


def _first_live_frame(flip, dt_ms=6.0):
    import json as _json
    import socket
    import threading
    from gabai.pipeline import sources
    c = PipelineConfig()
    c.pipeline.mode, c.source.kind, c.source.folder = "stereo_only", "live", tempfile.mkdtemp()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    c.source.port = s.getsockname()[1]
    s.close()
    c.source.flip180, c.source.skew_comp = flip, False
    rect, delay, k = sources.prepare_live(c)
    gen = sources.live_source(c, rect, delay, k)
    rng = np.random.default_rng(3)
    left, right = (cv2.imencode(".jpg", cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (0, 0), 2.0))[1].tobytes() for _ in range(2))

    def pi():
        for _ in range(100):
            try:
                sk = socket.create_connection(("127.0.0.1", c.source.port), timeout=1)
                break
            except OSError:
                time.sleep(0.05)
        sk.sendall(P.pack(P.HELLO, time.monotonic(), _json.dumps({"token": c.source.token, "name": "fake-pi", "left": str(np.load(c.source.calib)["left_dev"]),
                                                                  "right": str(np.load(c.source.calib)["right_dev"])}).encode()))
        for _ in range(3):
            sk.sendall(P.pack(P.FRAME, time.monotonic(), P.pack_frame(left, right, dt_ms)))
            time.sleep(0.1)
        time.sleep(0.5)
    threading.Thread(target=pi, daemon=True).start()
    fr = next(gen)
    gen.close()
    return fr, c, rect, k


def test_flip180_turns_both_pictures_and_swaps_them_and_moves_the_principal_point():
    from gabai.pipeline.sources import flipped_principal_point
    a, c, rect, k = _first_live_frame(False)
    b, _, _, _ = _first_live_frame(True)
    assert np.array_equal(b.left, cv2.rotate(a.right, cv2.ROTATE_180)) and np.array_equal(b.right, cv2.rotate(a.left, cv2.ROTATE_180))
    assert b.cal["cx"] == flipped_principal_point(a.cal["cx"], 640) and b.cal["cy"] == flipped_principal_point(a.cal["cy"], 360)
    assert b.cal["f"] == a.cal["f"] and b.cal["B"] == a.cal["B"] and "turned 180" in b.cal["source"]
    assert "+6.0 ms" in a.sync["note"] and "-6.0 ms" in b.sync["note"]                  # the right camera is now the one that was left


def test_flip180_gives_basalt_the_turned_camera():
    from scipy.spatial.transform import Rotation as Rot
    out = {}
    for flip in (False, True):
        c = PipelineConfig()
        c.pipeline.mode, c.source.kind, c.source.folder, c.source.flip180 = "stereo_imu", "live", tempfile.mkdtemp(), flip
        from gabai.pipeline import sources
        sources.prepare_live(c)
        v = json.load(open(os.path.join(c.source.folder, "calib.json")))["value0"]
        q = v["T_imu_cam"][0]
        out[flip] = (v["intrinsics"][0]["intrinsics"], Rot.from_quat([q["qx"], q["qy"], q["qz"], q["qw"]]).as_matrix())
    (ka, Ra), (kb, Rb) = out[False], out[True]
    assert abs(kb["cx"] - (639 - ka["cx"])) < 1e-9 and abs(kb["cy"] - (359 - ka["cy"])) < 1e-9 and kb["fx"] == ka["fx"]
    assert np.allclose(Rb, Ra @ np.diag([-1.0, -1.0, 1.0]), atol=1e-9)


def test_scale_runs_the_model_on_enlarged_pictures_and_returns_native_disparity():
    import gabai.spike.las2_runner as LR
    from gabai.pipeline.run import default_depth
    seen = []

    class Fake:
        def __init__(self, size):
            pass

        def disparity_lr_batched(self, l, r):
            seen.append(l.shape[1])
            d = np.full(l.shape[:2], 10.0 * l.shape[1] / 640.0, np.float32)      # a scene whose disparity is 10 px at 640 wide
            ok = np.ones(l.shape[:2], bool)
            ok[:, : l.shape[1] // 4] = False                                       # left quarter rejected by the left-right check
            return d, ok, 0.01
    real = LR.Las2Depth
    LR.Las2Depth = Fake
    try:
        img = np.zeros((360, 640, 3), np.uint8)
        d1, t1, _ = default_depth("h", 0)(img, img)
        d2, t2, _ = default_depth("h", 0, scale=1.5)(img, img)
    finally:
        LR.Las2Depth = real
    assert seen == [640, 960]
    assert d2.shape == d1.shape == (360, 640) and t2.shape == t1.shape == (360, 640)
    assert abs(np.nanmedian(d2) - 10.0) < 1e-4 and abs(np.nanmedian(d1) - 10.0) < 1e-4       # disparity is back in native pixels
    assert abs(t2.mean() - t1.mean()) < 0.01 and not t2[:, :150].any() and t2[:, 170:].all()
    from gabai.pipeline.config import PipelineConfig, validate
    c = PipelineConfig()
    c.stereo.scale = 3.0
    assert any("stereo.scale" in e for e in validate(c, check_files=False)[0])
