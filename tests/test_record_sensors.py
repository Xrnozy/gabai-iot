import os
import sys
import tempfile

import numpy as np

from gabai.tools.record_sensors import gap_report, record

sys.path.insert(0, "scripts")


class FakeRx:
    def __init__(self):
        self.n = 0

    def drain_imu(self):
        self.n += 1
        return [(self.n + i * 0.005, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0) for i in range(3)]

    def drain_mag(self):
        return [(self.n, 10.0, 20.0, 30.0)]

    def stats(self):
        return {"connected": True}


def test_the_recording_keeps_the_raw_samples_and_reports_rate_and_gaps():
    out = os.path.join(tempfile.mkdtemp(), "x.npz")
    clock = iter(np.arange(0, 100, 0.5))
    imu, mag = record(FakeRx(), 3.0, out, clock=lambda: next(clock), sleep=lambda s: None, say=lambda *_: None)
    d = np.load(out)
    assert d["imu"].shape == (len(imu), 7) and d["mag"].shape == (len(mag), 4) and len(imu) >= 6
    t = np.r_[np.arange(0, 1, 0.005), np.arange(1.1, 2, 0.005)]                    # one 100 ms hole
    rate, longest, holes = gap_report(t, 200.0)
    assert 170 < rate < 200 and 90 < longest < 110 and holes == 1


def test_the_evaluation_runs_on_a_sensor_recording_with_a_compass():
    from evaluate_orientation import evaluate
    rng = np.random.default_rng(0)
    t = np.arange(0, 12, 0.006)
    g = rng.normal(0, 0.1, (len(t), 3))
    a = np.array([0, 0, 1.0]) + rng.normal(0, 0.004, (len(t), 3))
    m = np.column_stack([np.arange(0, 12, 0.02), np.full(600, 18.0), np.zeros(600), np.full(600, -42.0)])
    mag = (m[:, 0], m[:, 1:4])
    rows = evaluate("synthetic", t, g, a, mag)
    assert rows["after"]["still"]["correction_rms"] < 1.0 and rows["before"]["still"]["correction_rms"] > 3.0
    assert rows["after"]["heading_step_rms_still"] < 0.1
