import csv
import json
import os
import tempfile
import threading

import cv2
import numpy as np

from gabai.pipeline import sources
from gabai.pipeline.config import PipelineConfig, validate
from gabai.tools.record_pairs import PairRecorder


def _pairs(n=6):
    rng = np.random.default_rng(5)
    base = cv2.GaussianBlur(rng.integers(0, 255, (360, 700, 3), dtype=np.uint8), (0, 0), 2.0)
    return [(base[:, i * 4:i * 4 + 640].copy(), base[:, i * 4 + 12:i * 4 + 652].copy()) for i in range(n)]


def _record(dts=(None, 3.5, -2.0, 8.0, 0.5, 12.0), codec="ffv1"):
    folder = os.path.join(tempfile.mkdtemp(), "rec")
    rec = PairRecorder(folder, (640, 360), "data/stereo_calib_new.npz", codec=codec)
    pairs = _pairs(len(dts))
    for i, (l, r) in enumerate(pairs):
        rec.add(l, r, 500.0 + i / 15.0, 500.01 + i / 15.0, dts[i], i + 1 + (2 if i >= 4 else 0), {"exposure_ms": 16.0, "gain": 60.0, "picture_mean": 88.0})
    rec.close({"left": "/dev/video2", "right": "/dev/video0"}, {"throttled": 0}, (180, 0))
    return folder, pairs


def test_a_recording_keeps_both_videos_in_step_with_the_times_offsets_and_camera_settings():
    folder, pairs = _record()
    meta = json.load(open(os.path.join(folder, "meta.json")))
    rows = list(csv.DictReader(open(os.path.join(folder, "pairs.csv"))))
    assert meta["pairs_recorded"] == len(rows) == 6 and meta["video_codec"] == "ffv1" and meta["picture_size"] == [640, 360]
    assert meta["pairs_missed_by_the_receiver"] == 2                                   # sequence 4 jumped to 7: two pairs were overwritten before the recorder saw them
    assert meta["left_right_offset_ms"]["max"] == 12.0 and meta["pictures_turned_deg_left_right"] == [180, 0]
    assert meta["calibration"]["left_dev"] == "/dev/video2" and abs(meta["calibration"]["baseline_mm"] - 41.7) < 0.2
    assert rows[0]["dt_ms"] == "" and rows[1]["dt_ms"] == "3.50" and rows[2]["dt_ms"] == "-2.00" and rows[0]["exposure_ms"] == "16.0"
    cl, cr = cv2.VideoCapture(os.path.join(folder, "left.avi")), cv2.VideoCapture(os.path.join(folder, "right.avi"))
    for l, r in pairs:                                                                 # frame n of both videos is pair n, bit for bit
        ok1, a = cl.read()
        ok2, b = cr.read()
        assert ok1 and ok2 and np.array_equal(a, l) and np.array_equal(b, r)


def test_a_full_queue_drops_and_counts_instead_of_blocking():
    folder = os.path.join(tempfile.mkdtemp(), "rec")
    rec = PairRecorder(folder, (640, 360), "data/stereo_calib_new.npz", queue_len=1)
    gate = threading.Event()
    real = rec.vl

    class SlowDisk:
        def write(self, f):
            gate.wait(5)
            real.write(f)

        def release(self):
            real.release()
    rec.vl = SlowDisk()
    l, r = _pairs(1)[0]
    results = [rec.add(l, r, 1.0 + i, 1.0 + i, 1.0, i + 1, {}) for i in range(6)]
    gate.set()
    meta = rec.close()
    assert results.count(False) >= 3 and meta["pairs_dropped_by_the_recorder"] == results.count(False)


def _replay_cfg(folder, **over):
    c = PipelineConfig()
    c.pipeline.mode, c.source.kind, c.source.folder, c.source.max_frames = "stereo_only", "replay", folder, 0
    for k, v in over.items():
        setattr(c.source, k, v)
    return c


def test_a_replay_gives_the_pipeline_the_recorded_times_and_offsets():
    folder, pairs = _record()
    c = _replay_cfg(folder, skew_comp=False)
    assert not validate(c)[0]
    frames = list(sources.open_source(c))
    assert len(frames) == 6 and frames[0].sync["kind"] == "replay" and frames[0].left.shape == (360, 640, 3)
    assert [round(f.t, 3) for f in frames[:2]] == [500.0, round(500.0 + 1 / 15.0, 3)]
    assert frames[0].sync["pair_dt_ms"] is None and frames[1].sync["pair_dt_ms"] == 3.5 and frames[2].sync["pair_dt_ms"] == 2.0
    assert frames[1].cal["B"] > 0.04 and frames[1].sync["cam"]["exposure_ms"] == 16.0


def test_replay_applies_the_offset_correction_like_the_live_run_when_asked():
    folder, _ = _record(dts=(None, 12.0, 12.0, 12.0, 12.0, 12.0))
    off = list(sources.open_source(_replay_cfg(folder, skew_comp=False)))
    on = list(sources.open_source(_replay_cfg(folder, skew_comp=True)))
    assert np.array_equal(off[3].left, on[3].left) and any(f.sync["skew_comp"]["applied"] for f in on[1:])
    assert not np.array_equal(off[3].right, on[3].right) and all(not f.sync["skew_comp"]["applied"] for f in off)


def test_a_folder_that_is_not_a_recording_or_asks_for_the_imu_is_refused_with_the_reason():
    errs = validate(_replay_cfg(tempfile.mkdtemp()))[0]
    assert any("meta.json" in e and "record_pairs" in e for e in errs)
    folder, _ = _record()
    c = _replay_cfg(folder)
    c.pipeline.mode = "stereo_imu"
    assert any("no IMU" in e for e in validate(c)[0])


def _prep(auto=True):
    c = PipelineConfig()
    c.pipeline.mode, c.source.kind, c.source.auto_layout, c.source.skew_comp = "stereo_only", "live", auto, False
    rect, delay, k = sources.prepare_replay(c)
    return sources.PairPrep(c, rect, k, delay, kind="replay")


def test_cameras_that_came_out_in_the_other_order_are_detected_from_the_rows_and_put_right():
    l, r = (cv2.imread(os.path.join("data", "calib_pairs_15", f"pair_12_{s}.png")) for s in "LR")
    right_way = (cv2.rotate(r, cv2.ROTATE_180), cv2.rotate(l, cv2.ROTATE_180))         # the layout stereo_calib_new.npz was made for
    other_way = (l, r)                                                                   # the same cameras with the two /dev/video numbers traded
    good, bad = _prep(), _prep()
    f_good = good.frame(0, *right_way, 1.0, 6.0, 1.0, [], {})
    f_bad = bad.frame(0, *other_way, 1.0, 6.0, 1.0, [], {})
    assert not good.swapped and bad.swapped and bad.layout_decided
    assert f_bad.sync["layout"] == "swapped" and f_good.sync["layout"] == "as calibrated"
    assert np.array_equal(f_bad.left, f_good.left) and np.array_equal(f_bad.right, f_good.right)    # identical pictures once the layout is put right
    assert f_good.sync["pair_dt_ms"] == 6.0 == f_bad.sync["pair_dt_ms"] and "+6.0 ms" in f_good.sync["note"] and "-6.0 ms" in f_bad.sync["note"]


def test_the_layout_check_can_be_switched_off_and_leaves_a_textureless_start_undecided_until_it_can_tell():
    l, r = (cv2.imread(os.path.join("data", "calib_pairs_15", f"pair_12_{s}.png")) for s in "LR")
    off = _prep(auto=False)
    off.frame(0, l, r, 1.0, None, 1.0, [], {})
    assert not off.swapped and off.layout_decided
    blank = np.full((360, 640, 3), 90, np.uint8)
    p = _prep()
    p.frame(0, blank, blank, 1.0, None, 1.0, [], {})
    assert not p.layout_decided and not p.swapped                                        # nothing to measure yet
    p.frame(1, l, r, 1.1, None, 1.1, [], {})
    assert p.swapped and p.layout_decided
