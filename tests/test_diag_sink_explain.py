import json
import os
import tempfile

import numpy as np

from gabai.diag import checks as C
from gabai.diag import log as dlog
from gabai.diag.artifacts import ArtifactSaver
from gabai.diag.explain import explain_point
from gabai.diag.sink import TraceSink
from gabai.diag.sysmon import snapshot
from gabai.diag.trace import FrameTrace

F, B, CX, CY = 881.0, 0.0507, 319.5, 239.5


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def test_sink_writes_a_regular_sample_and_every_interesting_frame_but_not_the_rest():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="s", level="INFO")
    sink = TraceSink(every=10, max_interesting_per_s=1000)
    wrote = []
    for i in range(40):
        t = FrameTrace(fid=i)
        t.add_check(C.check_frame_age(0.9 if i == 13 else 0.1))
        wrote.append(sink.offer(t))
    dlog.shutdown()
    evs = [r for r in rows(os.path.join(run, "gabai.jsonl")) if r["ev"] == "frame.trace"]
    fids = [r["fid"] for r in evs]
    assert 13 in fids                                              # the failing frame is always kept
    assert fids.count(0) == 1 and 10 in fids and 20 in fids        # the regular 1-in-10 sample
    assert len(fids) < 10                                          # and the other ~30 frames were not written


def test_interesting_frames_are_rate_limited_so_a_bad_minute_cannot_flood_the_log():
    run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name="s2", level="INFO")
    sink = TraceSink(every=1000, max_interesting_per_s=5)
    for i in range(200):
        t = FrameTrace(fid=i)
        t.add_check(C.check_frame_age(0.9))
        sink.offer(t)
    dlog.shutdown()
    n = len([r for r in rows(os.path.join(run, "gabai.jsonl")) if r["ev"] == "frame.trace"])
    assert n <= 12                                                 # about 5 per second plus the first one
    assert sink.skipped_interesting > 150                          # and the count of what was skipped is known


def test_ring_buffer_dump_keeps_the_last_frames_for_post_mortems():
    sink = TraceSink(every=1000, ring=5)
    for i in range(20):
        sink.offer(FrameTrace(fid=i))
    p = os.path.join(tempfile.mkdtemp(), "ring.jsonl")
    sink.dump_ring(p)
    assert [r["fid"] for r in rows(p)] == [15, 16, 17, 18, 19]


def test_artifacts_are_off_unless_asked_and_stay_inside_their_storage_budget():
    d = tempfile.mkdtemp()
    gy, gx = np.mgrid[0:480, 0:640]
    img = np.stack([gx * 255 // 640, gy * 255 // 480, (gx + gy) * 255 // 1120], axis=2).astype(np.uint8)   # smooth, compresses like a real frame
    disp = (20.0 + gx / 640.0 * 60.0).astype(np.float32)
    off = ArtifactSaver(d, what=())
    assert off.maybe_save(1, frame_left=img, disp=disp, interesting=True) == []
    assert os.listdir(d) == []
    on = ArtifactSaver(d, what=("depth", "frame"), every=1, max_mb=0.05, min_interval_s=0)
    saved = []
    for i in range(30):
        saved += on.maybe_save(i, frame_left=img, disp=disp)
    total = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
    assert saved and total <= 0.05 * 1e6 + 60_000                  # budget respected (one file may overshoot)
    assert on.stopped_for_budget


def test_explain_point_shows_every_step_from_pixel_to_world_position():
    u, v, d = 400.0, 300.0, F * B / 1.5                            # 1.5 m away
    R_wc = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])      # camera looking along world +y, x right, down is world -z
    e = explain_point(u, v, d, F, B, CX, CY, cam_h=1.4, up_cam=None, R_wc=R_wc, t_wc=np.array([0.0, 0.0, 1.4]), pose_source="basalt")
    assert abs(e["depth_m"]["v"] - 1.5) < 1e-6 and e["depth_m"]["kind"] == "estimated" and e["disparity_px"]["kind"] == "model_prediction"
    assert abs(e["camera_xyz_m"]["v"][0] - (u - CX) / F * 1.5) < 1e-6
    assert abs(e["depth_uncertainty_m"] - 1.5 ** 2 / (F * B) * 0.5) < 1e-6
    assert e["world_xyz_m"]["kind"] == "estimated" and e["world_xyz_m"]["source"] == "basalt"
    assert abs(e["world_xyz_m"]["v"][1] - 1.5) < 1e-6              # forward along world +y
    assert "floor" in e["floor_test"] and e["floor_test"]["above_floor"] in (True, False)


def test_explain_point_marks_missing_pose_and_a_disparity_at_the_model_limit():
    e = explain_point(320, 240, 191.0, F, B, CX, CY, cam_h=None, up_cam=None, R_wc=None, t_wc=None)
    assert e["world_xyz_m"]["kind"] == "unavailable" and e["world_xyz_m"]["v"] is None
    assert any("limit" in w for w in e["warnings"])
    e2 = explain_point(320, 240, float("nan"), F, B, CX, CY, cam_h=None, up_cam=None, R_wc=None, t_wc=None)
    assert e2["depth_m"]["kind"] == "unavailable"


def test_system_snapshot_never_invents_a_reading():
    s = snapshot()
    for k in ("cpu_pct", "ram_pct", "proc_rss_mb", "gpu_temp_c", "gpu_util_pct", "gpu_mem_mb"):
        assert k in s and (s[k] is None or isinstance(s[k], (int, float)))
