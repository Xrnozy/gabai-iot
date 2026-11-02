import numpy as np

from gabai.kltfuse import KltObs
from gabai.pipeline.viz import Viz


class Cap(Viz):
    def __init__(self):
        super().__init__({"f": 400, "cx": 320, "cy": 180}, (640, 360), viewer=False, scale=0.5)
        self.logged = {}

    def _log(self, path, arch, **kw):
        self.logged.setdefault(path, []).append(arch)


def _obs(shift, lost=False, new=False):
    ids = np.arange(5)
    uv = np.array([[100, 100], [200, 120], [300, 200], [400, 250], [500, 300]], np.float32) + shift
    q = np.array([0, 10, 30, 80, 200], np.uint8)
    return KltObs(ids, uv, q, np.array([new] + [False] * 4), np.array([9]) if lost else None, np.array([[50.0, 60.0]]) if lost else None, 3, 4.2)


def test_the_klt_panel_draws_points_ids_trails_motion_new_and_lost_over_the_picture():
    v, img = Cap(), np.zeros((360, 640, 3), np.uint8)
    v.klt(img, _obs(0))
    v.klt(img, _obs(4, lost=True, new=True))
    for path in ("klt/left", "klt/left/tracked", "klt/left/new", "klt/left/lost", "klt/left/trails", "klt/left/motion", "klt/features/left"):
        assert path in v.logged, path
    tracked = v.logged["klt/left/tracked"][-1]
    assert len(tracked.positions) == 4 and len(v.logged["klt/left/new"][-1].positions) == 1 and len(v.logged["klt/left/lost"][-1].positions) == 1
    assert len(v.logged["klt/left/trails"][-1].strips) == 4 and "klt/count/tracked" in v.logged
    v.klt(img, None)                                                                    # no points this frame: the overlay is cleared, nothing stale is left


def test_sync_scalars_show_the_numbers_that_prove_the_streams_line_up_and_warn_when_they_do_not():
    v = Cap()
    info = {"seq": 41, "klt_seq": 41, "imu_n": 13, "imu_margin_ms": 8.0, "imu_complete": True, "imu_gap_s": 0.0, "klt_missing": False, "skipped_before": 0, "backlog": 2,
            "imu_queue": 3, "wait_ms": 1.0, "latency_ms": 55.0, "totals": {"imu_missing": 0, "pi_skipped": 2}}
    v.sync(info, 1)
    assert v.logged["sync/klt_seq_minus_frame_seq"][-1].scalars.as_arrow_array().to_pylist()[0] == 0 and v.logged["sync/ok"][-1].scalars.as_arrow_array().to_pylist()[0] == 1.0
    assert "sync/total_pi_skipped" in v.logged and "sync/pose_lag_frames" in v.logged and "diag/log" not in v.logged
    v.sync(dict(info, imu_complete=False, imu_gap_s=0.2, skipped_before=7), 0)
    assert v.logged["sync/ok"][-1].scalars.as_arrow_array().to_pylist()[0] == 0.0 and "diag/log" in v.logged


def test_the_right_picture_shows_the_points_the_pi_followed_into_it():
    v, img = Cap(), np.zeros((360, 640, 3), np.uint8)
    o = _obs(0)
    o.right_uv = o.uv - [20.0, 0.0]
    o.right_uv[2] = np.nan                                                              # one point the Pi could not find in the right picture
    v.klt(img, o, img)
    assert "klt/right" in v.logged and len(v.logged["klt/right/tracked"][-1].positions) == 4
    v.klt(img, _obs(0), img)                                                            # no right points sent: nothing stale stays
    assert "klt/right/tracked" in v.logged
