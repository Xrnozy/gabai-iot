import csv
import os
import tempfile
import threading
import time

import numpy as np

from gabai.pi.cams import SyncedCameras
from gabai.pi.pacing import RateGovernor, nearest_offsets_ms, parse_power_save, phase_delay_ms
from gabai.tools.record_pairs import PairRecorder


def test_the_wait_before_restarting_the_right_stream_moves_the_offset_to_zero():
    period = 1000 / 30
    for dt in (-14.3, 14.3, 10.46, -3.0, 0.0):
        off = (dt + phase_delay_ms(dt, period) + period / 2) % period - period / 2         # offset after the restart, brought into +-half a period
        assert abs(off) < 1e-9, (dt, off)
    assert 0 <= phase_delay_ms(-14.3, period) < period and abs(phase_delay_ms(-14.3, period) - 14.3) < 1e-9


def test_the_nearest_right_frame_gives_the_signed_offset():
    left = [1.000, 1.0333, 1.0667]
    right = [t + 0.0143 for t in left]                                  # right frames 14.3 ms after the left ones
    assert np.allclose(nearest_offsets_ms(left, right), [14.3, 14.3, 14.3], atol=0.1)
    assert nearest_offsets_ms([], right) == [] and nearest_offsets_ms(left, []) == []


class FakeCameras(SyncedCameras):
    """Two cameras whose offset is set by when the right stream was last started, like the real ones."""

    def __init__(self, offset_ms, noise_ms=0.0):
        self.capture_fps, self._reopen, self._stop, self.off, self.waits = 30, {}, False, offset_ms, []
        self.noise = noise_ms
        self.th = threading.Thread(target=self._reader_sim, daemon=True)
        self.th.start()

    def _reader_sim(self):
        period = 1000 / 30
        while not self._stop:
            req = self._reopen.pop("R", None)
            if req is not None:
                self.waits.append(req)
                self.off = (self.off + req + period / 2) % period - period / 2
            time.sleep(0.005)

    def _offset_samples(self, seconds):
        return [self.off + self.noise * ((i % 3) - 1) for i in range(20)]


def test_align_phase_restarts_the_right_stream_until_the_offset_is_small():
    cams = FakeCameras(-14.3)
    hist = cams.align_phase(target_ms=4.0)
    cams._stop = True
    assert abs(hist[-1]["offset_ms"]) <= 4.0 and len(hist) == 2 and abs(cams.waits[0] - 14.3) < 1e-6
    assert abs(hist[0]["offset_ms"] + 14.3) < 1e-6


def test_align_phase_leaves_a_small_offset_alone_and_gives_up_after_the_tries():
    cams = FakeCameras(2.0)
    hist = cams.align_phase(target_ms=4.0)
    cams._stop = True
    assert len(hist) == 1 and cams.waits == []
    stuck = FakeCameras(10.0)
    stuck._reader_sim = lambda: None                                      # restarts change nothing
    stuck._stop = True
    stuck.th.join()
    stuck._stop = False
    t = threading.Thread(target=lambda: [stuck._reopen.pop("R", None) or time.sleep(0.005) for _ in range(2000)], daemon=True)
    t.start()
    hist = stuck.align_phase(target_ms=4.0, tries=2)
    assert len(hist) == 3 and all(abs(h["offset_ms"] - 10.0) < 1e-6 for h in hist)


def test_the_send_rate_drops_when_sends_are_slow_and_recovers_slowly_when_they_are_quick():
    g = RateGovernor(15, min_fps=5)
    now = 0.0
    for _ in range(5):
        now += 1.0
        g.update(400.0, now)
    assert g.fps < 15
    for _ in range(10):                                                    # the link stays slow: the rate keeps falling, never below the floor
        now += 1.0
        g.update(400.0, now)
    low = g.fps
    assert low < 15 and low >= 5
    for _ in range(80):                                                    # then a long run of quick sends
        now += 1.0
        g.update(30.0, now)
    assert g.fps > low and g.fps <= 15
    g2 = RateGovernor(15, min_fps=5)
    for i in range(200):
        g2.update(2000.0, i * 4.0)
    assert g2.fps == 5                                                     # never below the floor
    g3 = RateGovernor(15)
    for i in range(50):
        g3.update(30.0, i)
    assert g3.fps == 15                                                    # a good link keeps the full rate


def test_the_wifi_power_save_state_is_read_from_iw_output():
    assert parse_power_save("Power save: on") == "on" and parse_power_save("Power save: off\n") == "off" and parse_power_save("") is None


def test_the_recording_keeps_the_pi_power_and_link_state_per_pair_for_later_analysis():
    folder = os.path.join(tempfile.mkdtemp(), "rec")
    rec = PairRecorder(folder, (640, 360), "data/stereo_calib_new.npz")
    img = np.zeros((360, 640, 3), np.uint8)
    rec.add(img, img, 10.0, 10.1, 5.0, 1, {"throttled": "0x50005", "rtt_s": 0.2, "temp_c": 47.2, "load1": 0.3, "cam_fps_left": 29.9, "cam_fps_right": 30.0})
    rec.add(img, img, 10.07, 10.2, 5.0, 2, {})
    rec.close()
    rows = list(csv.DictReader(open(os.path.join(folder, "pairs.csv"))))
    assert rows[0]["throttled"] == str(0x50005) and rows[0]["rtt_ms"] == "200.0" and rows[0]["cam_fps_left"] == "29.9" and rows[1]["throttled"] == ""
    from importlib import import_module
    import sys
    sys.path.insert(0, "scripts")
    out = import_module("analyze_timing").analyze(folder)
    assert out["pairs"] == 2 and out["pi_power"]["pairs_with_under_voltage_or_throttle_now_pct"] == 50.0
