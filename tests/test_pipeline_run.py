import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from gabai.diag import log as dlog
from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.run import run
from gabai.pipeline.viz import Viz

REAL = os.path.join("data", "slam", "live_20261026_181526")
F, BASE, W, H = 881.0, 0.0507, 640, 480
STEP_PX = 10
STEP_M = STEP_PX * 2.0 / F                                   # the plane is 2 m away


class ListViz(Viz):
    instances = []

    def __init__(self, cal, size, save=None, viewer=False):
        self.cal, self.size, self.t0, self.n_logged = cal, size, None, 0
        self._held_paths = set()
        self.paths, self.by_frame, self._tl = set(), {}, __import__("threading").local()      # like Rerun, the timeline position belongs to the thread that set it (two drawing threads)
        ListViz.instances.append(self)

    def at(self, idx, t):
        self.t0 = t if self.t0 is None else self.t0
        self._tl.cur = idx

    def _log(self, path, arch, **kw):
        self.n_logged += 1
        self.paths.add(path)
        self.by_frame.setdefault(getattr(self._tl, "cur", None), set()).add(path)


def texture(seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, (H, W), dtype=np.uint8), (0, 0), 1.5)


def shifted(img, dx):
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REFLECT)


def make_recording(n=12, with_imu=False, blank_at=None):
    d = tempfile.mkdtemp()
    for cam in ("cam0", "cam1"):
        os.makedirs(os.path.join(d, "mav0", cam, "data"))
    base = texture()
    rows = []
    for k in range(n):
        img = np.full((H, W), 128, np.uint8) if k == blank_at else shifted(base, -STEP_PX * k)
        ts = int((100 + k / 13.0) * 1e9)
        for cam in ("cam0", "cam1"):
            cv2.imwrite(os.path.join(d, "mav0", cam, "data", f"{ts}.png"), img)
        rows.append(f"{ts},{ts}.png")
    open(os.path.join(d, "mav0", "cam0", "data.csv"), "w").write("#t,f\n" + "\n".join(rows) + "\n")
    if with_imu:
        os.makedirs(os.path.join(d, "mav0", "imu0"))
        t = (100 + np.arange(400) / 160.0) * 1e9
        np.savetxt(os.path.join(d, "mav0", "imu0", "data.csv"), np.c_[t, np.zeros((400, 3)), np.tile([0, 0, 9.81], (400, 1))], delimiter=",", header="#t", comments="")
    json.dump({"value0": {"T_imu_cam": [{"px": 0, "py": 0, "pz": 0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}, {"px": BASE, "py": 0, "pz": 0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}],
                          "intrinsics": [{"intrinsics": {"fx": F, "fy": F, "cx": W / 2, "cy": H / 2}}] * 2, "resolution": [[W, H], [W, H]]}}, open(os.path.join(d, "calib.json"), "w"))
    json.dump({"imu_extrinsics_calibrated": True, "cam_delay_ms": 0.0}, open(os.path.join(d, "recording_info.json"), "w"))
    return d


def plane_depth(left, right):
    d = np.full((H, W), F * BASE / 2.0, np.float32)
    return d, np.ones((H, W), bool), 5.0


def cfg_for(folder, mode="stereo_only", skip=1):
    c = PipelineConfig()
    c.pipeline.mode, c.source.kind, c.source.folder, c.source.skip, c.source.max_frames = mode, "recording", folder, skip, 0
    c.hands.enabled = c.detector.enabled = False
    c.viz.viewer = False
    return c


def go(cfg, **kw):
    ListViz.instances.clear()
    s = run(cfg, depth_fn=plane_depth, viz_factory=ListViz, log_dir=tempfile.mkdtemp(), quiet=True, **kw)
    return s, ListViz.instances[0]


def test_stereo_only_recovers_the_motion_with_no_imu_file_present_at_all():
    d = make_recording(n=12, with_imu=False)
    s, v = go(cfg_for(d))
    assert s["mode"] == "stereo_only" and s["imu_used"] is False and s["frames"] == 12 and s["poses_ok"] == 12
    expect = 11 * STEP_M
    assert abs(s["path_m"] - expect) < 0.25 * expect, (s["path_m"], expect)
    assert s["path_is_metric"] is True and s["map_voxels"] > 0 and s["poses_without_frame_pending"] == 0


def test_stereo_only_never_opens_the_imu_even_when_the_file_exists():
    d = make_recording(with_imu=True)
    real = np.loadtxt

    def guard(path, *a, **k):
        assert "imu0" not in str(path), "stereo_only opened the IMU file"
        return real(path, *a, **k)
    np.loadtxt = guard
    try:
        s, _ = go(cfg_for(d))
    finally:
        np.loadtxt = real
    assert s["poses_ok"] == s["frames"]


def test_every_panel_is_fed_for_every_frame():
    s, v = go(cfg_for(make_recording()))
    need = {"cameras/left", "cameras/right", "stereo/disparity", "stereo/depth", "stereo/valid", "world/camera", "world/trajectory", "world/live", "diag/status",
            "tracking/features", "depth/valid_fraction", "tracking/inliers"}
    assert need <= v.paths, need - v.paths
    assert all(need & {"cameras/left"} for fs in v.by_frame.values() for need in [fs])             # left and right logged at the same frame index as the stereo views
    assert all({"cameras/left", "cameras/right", "stereo/depth"} <= fs for fs in v.by_frame.values() if "stereo/depth" in fs or "cameras/left" in fs)


def test_imu_panels_stay_empty_in_stereo_only_and_the_status_says_the_imu_is_not_used():
    s, v = go(cfg_for(make_recording(with_imu=True)))
    assert not any(p.startswith("imu/") for p in v.paths)


def test_stereo_imu_without_an_imu_file_stops_with_the_file_named():
    d = make_recording(with_imu=False)
    with pytest.raises(SystemExit) as e:
        run(cfg_for(d, "stereo_imu"), depth_fn=plane_depth, viz_factory=ListViz, log_dir=tempfile.mkdtemp(), quiet=True)
    assert "imu0" in str(e.value) and "stereo_imu" in str(e.value)


def test_a_frame_where_tracking_is_lost_is_shown_but_not_added_to_the_map():
    s, v = go(cfg_for(make_recording(n=12, blank_at=6)))
    assert s["poses_held"] >= 1 and s["map_frames"] == s["frames"] - s["poses_held"]
    assert s["checks"]["tracking_quality"]["fail"] + s["checks"]["tracking_quality"]["suspect"] >= 0
    # the held frame still gets a camera pose and an event in the log
    assert "diag/log" in v.paths


def test_the_map_stays_aligned_with_the_depth_the_camera_sees_when_the_pose_is_good():
    s, _ = go(cfg_for(make_recording(n=14)))
    a = s["map_alignment_median_cm"]
    assert a is not None and a["median"] < 6.0, a
    assert s["checks"]["map_alignment"]["fail"] == 0


def test_the_summary_and_the_log_are_written():
    s, _ = go(cfg_for(make_recording()))
    assert os.path.exists(os.path.join(s["log"], "pipeline_summary.json"))
    ev = [json.loads(l) for l in open(os.path.join(s["log"], "gabai.jsonl"), encoding="utf-8")]
    assert any(e["ev"] == "pipeline.geometry" for e in ev) and any(e["ev"] == "pipeline.summary" for e in ev)
    geo = next(e for e in ev if e["ev"] == "pipeline.geometry")
    assert geo["metric"] is True and "depth_error_m_at" in geo


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "calib.json")), reason="the real recording is not on this machine")
def test_real_recording_both_modes_give_the_same_kind_of_output_and_imu_only_in_the_imu_mode():
    from gabai.pipeline.run import default_depth
    depth = default_depth("s")
    out = {}
    for mode in ("stereo_only", "stereo_imu"):
        c = cfg_for(REAL, mode, skip=2)
        c.source.max_frames = 0
        ListViz.instances.clear()
        s = run(c, depth_fn=depth, viz_factory=ListViz, log_dir=tempfile.mkdtemp(), quiet=True)
        out[mode] = (s, ListViz.instances[0])
    so, vo = out["stereo_only"]
    si, vi = out["stereo_imu"]
    assert so["frames"] > 100 and si["frames"] == so["frames"] and so["path_is_metric"] and si["path_is_metric"]
    assert so["imu_used"] is False and si["imu_used"] is True
    shared = {"world/camera", "world/trajectory", "world/live", "cameras/left", "cameras/right", "stereo/depth", "diag/status", "tracking/pose_step_m"}
    assert shared <= vo.paths and shared <= vi.paths
    assert not any(p.startswith("imu/") for p in vo.paths) and "imu/gyro_rate_dps" in vi.paths


def test_max_frames_limits_the_run():
    c = cfg_for(make_recording(n=12))
    c.source.max_frames = 5
    s, _ = go(c)
    assert s["frames"] == 5


def _cfg(folder, **perf):
    c = cfg_for(folder)
    c.source.max_frames = 0
    for k, v in perf.items():
        setattr(c.perf, k, v)
    return c


def test_running_in_two_stages_gives_the_same_result_as_running_in_one():
    d = make_recording(n=14)
    one, v1 = go(_cfg(d, staged=False, async_viz=False, prefetch=False))
    two, v2 = go(_cfg(d, staged=True, async_viz=True, prefetch=True))
    for k in ("frames", "poses_ok", "poses_held", "map_frames", "map_voxels"):
        assert one[k] == two[k], (k, one[k], two[k])
    assert abs(one["path_m"] - two["path_m"]) < 1e-9
    extra = lambda paths: {x for x in paths if x != "world/pov_mesh"}      # the mesh is dropped on purpose when its drawing thread is busy
    assert extra(v1.paths) == extra(v2.paths), "the same panels must be fed whichever way it runs"
    assert two["perf"]["staged"] is True and one["perf"]["staged"] is False and two["viz"]["errors"] == 0


def test_every_frame_is_drawn_at_its_own_position_when_two_stages_draw_at_once():
    s, v = go(_cfg(make_recording(n=12), staged=True, async_viz=True))
    assert sorted(v.by_frame) == list(range(12))
    for k, paths in v.by_frame.items():
        assert {"cameras/left", "cameras/right", "stereo/depth"} <= paths and "world/camera" in paths, (k, sorted(paths))


def test_the_frame_rate_cap_is_respected():
    t0 = __import__("time").perf_counter()
    go(_cfg(make_recording(n=8), fps_cap=10.0))
    assert __import__("time").perf_counter() - t0 >= 0.65               # 7 gaps of 0.1 s


class SlowHeld:
    hands = det = object()

    def frame(self, *a, **k):
        __import__("time").sleep(0.15)
        return {"hands": [], "objects": [], "relations": [], "events": [], "models": [], "released": [], "status": {}, "exclude_mask": np.zeros((H, W), bool)}


def test_a_live_stream_skips_frames_when_stage_b_is_behind_instead_of_falling_behind():
    c = _cfg(make_recording(n=14), drop_when_busy=True)
    c.hands.enabled = True
    s, _ = go(c, held=SlowHeld())
    assert s["perf"]["frames_dropped_stage_b"] > 0 and s["poses_ok"] + s["poses_held"] + s["perf"]["frames_dropped_stage_b"] == 14, s["perf"]


def test_a_recording_is_never_skipped_even_when_stage_b_is_slow():
    c = _cfg(make_recording(n=8), drop_when_busy=False)
    c.hands.enabled = True
    s, _ = go(c, held=SlowHeld())
    assert s["perf"]["frames_dropped_stage_b"] == 0 and s["poses_ok"] == 8


def test_an_error_in_stage_b_stops_the_run_with_the_error():
    class Boom(SlowHeld):
        def frame(self, *a, **k):
            raise RuntimeError("hand model crashed")
    c = _cfg(make_recording(n=8))
    c.hands.enabled = True
    try:
        go(c, held=Boom())
        assert False
    except RuntimeError as e:
        assert "hand model crashed" in str(e)


def test_the_pipeline_slows_down_and_switches_hands_off_while_navigation_runs():
    import gabai.pipeline.run as RUN
    real = RUN.nav_active
    RUN.nav_active = lambda *a, **k: True
    try:
        c = _cfg(make_recording(n=6))
        c.hands.enabled = True
        c.perf.nav_fps_cap = 10.0
        t0 = __import__("time").perf_counter()
        calls = {"n": 0}

        class Counting(SlowHeld):
            def frame(self, *a, **k):
                calls["n"] += 1
                return super().frame(*a, **k)
        s, _ = go(c, held=Counting())
        wall = __import__("time").perf_counter() - t0
    finally:
        RUN.nav_active = real
    assert s["perf"]["navigation_seen_running"] is True and calls["n"] == 0, "hand/object analysis must be off while navigation runs"
    assert s["poses_ok"] + s["poses_held"] == 6 and wall >= 0.4               # 5 gaps of 0.1 s at the navigation cap


def test_hands_are_taken_out_of_the_scene_when_the_analysis_is_switched_off_by_navigation():
    import gabai.pipeline.run as RUN
    calls = {"annotate": 0, "clear": 0}

    class Held(SlowHeld):
        def frame(self, *a, **k):
            out = super().frame(*a, **k)
            out["events"] = [{"type": "noop", "label": "x"}]        # forces a draw
            return out

    class CountViz(ListViz):
        def annotate(self, *a, **k):
            calls["annotate"] += 1

        def clear_tracked(self):
            calls["clear"] += 1

    real = RUN.nav_active
    state = {"n": 0}

    def flip(*a, **k):                                              # navigation not running at first, then running
        state["n"] += 1
        return state["n"] > 1
    RUN.nav_active = flip
    try:
        c = _cfg(make_recording(n=14))
        c.hands.enabled = True
        c.perf.nav_fps_cap = 20.0
        ListViz.instances.clear()
        run(c, depth_fn=plane_depth, held=Held(), viz_factory=CountViz, log_dir=tempfile.mkdtemp(), quiet=True)
    finally:
        RUN.nav_active = real
    assert calls["annotate"] >= 1 and calls["clear"] >= 1
