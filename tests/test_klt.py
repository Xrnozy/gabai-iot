import argparse
import tempfile
import threading
import time

import cv2
import numpy as np

from gabai.diag import log as dlog
from gabai.net import protocol as P
from gabai.pc.receiver import LiveReceiver
from gabai.pi import sender as S
from gabai.pi.klt import KltTracker, _synthetic


def test_klt_message_round_trips_ids_positions_status_and_unseen_right_points():
    ids, xy = np.array([5, 9, 70000]), np.array([[10.25, 20.5], [639.0, 359.0], [0.0, 3.0]], np.float32)
    right = np.array([[8.0, 20.5], [np.nan, np.nan], [1.0, 3.0]], np.float32)
    fs, i, p, st, q, r, ms, mo = P.unpack_klt(P.pack_klt(77, ids, xy, np.array([0, 1, 2]), np.array([0, 40, 255], np.uint8), right, 4.5))
    assert fs == 77 and list(i) == [5, 9, 70000] and list(st) == [0, 1, 2] and list(q) == [0, 40, 255] and abs(ms - 4.5) < 1e-6
    assert np.abs(p - xy).max() <= 1 / 32 and np.isnan(r[1]).all() and np.abs(r[[0, 2]] - right[[0, 2]]).max() <= 1 / 32
    assert P.unpack_klt(P.pack_klt(1, ids, xy, np.array([0, 0, 0])))[5] is None and mo is None
    m = {"prev_seq": 9, "dt": 0.066, "rvec": [0.01, -0.02, 0.03], "t": [0.1, 0.0, -0.2], "n_inl": 80, "rms": 0.7}
    got = P.unpack_klt(P.pack_klt(10, ids, xy, np.array([0, 0, 0]), None, right, 1.0, m))
    assert got[0] == 10 and list(got[1]) == [5, 9, 70000] and got[5] is not None and got[7]["prev_seq"] == 9 and got[7]["n_inl"] == 80
    assert np.allclose(got[7]["rvec"], m["rvec"], atol=1e-6) and np.allclose(got[7]["t"], m["t"], atol=1e-6) and abs(got[7]["dt"] - 0.066) < 1e-6


def test_points_follow_a_known_shift_in_full_picture_pixels_and_keep_their_ids():
    tr = KltTracker(scale=0.5, max_features=100)
    a = tr.update(_synthetic(640, 360, 0))
    b = tr.update(_synthetic(640, 360, 6))
    assert a.n_new > 30 and b.n_lost < 0.2 * a.n_new
    keep = b.status == P.KLT_TRACKED
    first = dict(zip(a.ids, a.xy))
    moved = np.array([first[i] - xy for i, xy in zip(b.ids[keep], b.xy[keep])])      # the picture content moves left, so x falls by 6
    assert keep.sum() > 30 and np.abs(np.median(moved[:, 0]) - 6) < 0.6 and np.abs(np.median(moved[:, 1])) < 0.6


def test_reduced_tracking_puts_corners_where_they_are_in_the_full_picture():
    img = np.zeros((360, 640, 3), np.uint8)
    cv2.rectangle(img, (200, 100), (301, 201), (255, 255, 255), -1)           # pixels 200..301 x 100..201 are white
    jp = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    corners = np.array([[200, 100], [301, 100], [200, 201], [301, 201]], np.float32)
    for scale in (1.0, 0.5):                                                  # a quarter-size picture cannot place a corner within 2 px
        r = KltTracker(scale=scale, max_features=8, min_dist=3).update(jp)
        d = np.linalg.norm(r.xy[:, None] - corners[None], axis=2).min(axis=1)
        assert len(r.xy) >= 4 and d.max() < 2.0, (scale, d)


def test_features_are_only_replenished_when_too_few_are_left():
    tr = KltTracker(max_features=60, replenish_frac=0.5, detect_gap=1)
    tr.update(_synthetic(640, 360, 0))
    assert tr.n > 30
    steady = [tr.update(_synthetic(640, 360, 1 + i % 2)) for i in range(3)]
    assert all(r.n_new == 0 for r in steady)
    tr.n = 5                                                                 # most tracks gone
    assert tr.update(_synthetic(640, 360, 1)).n_new > 20


def test_lost_tracks_are_reported_once_with_their_last_position_then_dropped():
    tr = KltTracker(max_features=50, detect_gap=100)
    a = tr.update(_synthetic(640, 360, 0))
    other = _synthetic(640, 360, 0, seed=2)                                   # a different scene: nothing can be followed into it
    b = tr.update(other)
    lost = b.status == P.KLT_LOST
    assert b.n_lost >= 0.8 * a.n_new and tr.n == b.n_tracked
    first = dict(zip(a.ids, a.xy))
    assert all(np.allclose(first[i], xy) for i, xy in zip(b.ids[lost], b.xy[lost]))
    again = tr.update(other)
    assert not set(b.ids[lost]) & set(again.ids[again.status == P.KLT_LOST])


def test_stereo_points_land_at_the_right_pictures_disparity():
    tr = KltTracker(max_features=60, stereo=True)
    r = tr.update(_synthetic(640, 360, 0), _synthetic(640, 360, 8))
    seen = np.isfinite(r.xy_right[:, 0])
    assert seen.sum() > 20 and abs(np.median(r.xy[seen, 0] - r.xy_right[seen, 0]) - 8) < 0.8


def test_forward_backward_check_only_removes_tracks():
    plain, strict = KltTracker(max_features=80), KltTracker(max_features=80, fb_px=0.5)
    for t in (plain, strict):
        t.update(_synthetic(640, 360, 0))
    assert strict.update(_synthetic(640, 360, 5)).n_tracked <= plain.update(_synthetic(640, 360, 5)).n_tracked


def _session(klt):
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="pi", level="INFO")
    rx = LiveReceiver(port=0, token="gabai")
    rx.start()
    a = argparse.Namespace(host="127.0.0.1", port=rx.port, token="gabai", width=640, height=480, fps=15, sndbuf=65536, health_every=1.0,
                           stats_every=1.0, imu_obj=None, mag_obj=None, quality=80, klt_obj=KltTracker(640, 480) if klt else None)
    th = threading.Thread(target=lambda: _quiet(S.run_session, a, S.TestPattern(a)), daemon=True)
    th.start()
    end = time.time() + 8
    while time.time() < end and rx.stats()["frames"] < 20:
        time.sleep(0.05)
    time.sleep(0.2)
    pair = rx.latest()
    out = (rx.stats(), pair, rx.klt_for(pair.t_capture))
    rx.stop()
    th.join(timeout=8)
    dlog.shutdown()
    return out


def _quiet(fn, *a):
    try:
        fn(*a)
    except Exception:
        pass


def test_sender_with_klt_sends_points_stamped_with_the_pair_and_the_video_is_unchanged():
    st, pair, k = _session(True)
    assert st["frames"] >= 20 and st["msg_counts"].get(str(P.KLT), 0) >= 15
    assert k is not None and k.t_capture == pair.t_capture and len(k.ids) > 0 and k.klt_ms > 0


def test_sender_without_klt_sends_no_klt_message():
    st, pair, k = _session(False)
    assert st["frames"] >= 20 and str(P.KLT) not in st["msg_counts"] and k is None
