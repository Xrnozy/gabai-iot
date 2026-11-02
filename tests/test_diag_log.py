import json
import os
import tempfile
import time

import numpy as np

from gabai.diag import log as dlog


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def fresh(**kw):
    d = tempfile.mkdtemp()
    run = dlog.configure(log_dir=d, run_name="t", **kw)
    return d, run


def test_each_line_is_json_with_the_fields_an_investigator_needs():
    d, run = fresh(level="INFO")
    lg = dlog.get("depth")
    lg.info("depth.done", fid=123456, dur_ms=12.34, valid_frac=0.5)
    dlog.shutdown()
    rows = read(os.path.join(run, "gabai.jsonl"))
    r = [x for x in rows if x["ev"] == "depth.done"][0]
    assert r["comp"] == "depth" and r["lvl"] == "INFO" and r["fid"] == 123456 and r["dur_ms"] == 12.34
    assert "ts" in r and "mono" in r and "run" in r


def test_debug_is_off_by_default_and_on_when_asked_per_component():
    d, run = fresh(level="INFO", component_levels={"slam": "DEBUG"})
    dlog.get("depth").debug("depth.detail", x=1)
    dlog.get("slam").debug("slam.detail", x=2)
    dlog.shutdown()
    evs = [r["ev"] for r in read(os.path.join(run, "gabai.jsonl"))]
    assert "slam.detail" in evs and "depth.detail" not in evs


def test_lazy_fields_are_only_computed_when_the_level_is_enabled():
    d, run = fresh(level="INFO")
    calls = []
    dlog.get("depth").debug("x", big=lambda: calls.append(1) or 5)
    assert calls == []
    dlog.get("depth").info("y", big=lambda: calls.append(1) or 5)
    dlog.shutdown()
    assert calls == [1]


def test_numpy_nan_bytes_and_exceptions_are_written_as_valid_json():
    d, run = fresh(level="INFO")
    lg = dlog.get("x")
    try:
        raise ValueError("boom")
    except ValueError:
        lg.error("x.fail", n=np.int32(3), f=np.float32(1.5), bad=float("nan"), arr=np.zeros((4, 5)), raw=b"abc", exc=True)
    dlog.shutdown()
    r = [x for x in read(os.path.join(run, "gabai.jsonl")) if x["ev"] == "x.fail"][0]
    assert r["n"] == 3 and r["f"] == 1.5 and r["bad"] is None
    assert r["arr"]["shape"] == [4, 5] and r["raw"]["bytes"] == 3
    assert "ValueError: boom" in r["exc"]


def test_logging_never_blocks_the_caller_even_when_the_writer_is_stuck():
    d, run = fresh(level="INFO", queue_size=50)
    dlog._pause_writer_for_test(True)
    t0 = time.perf_counter()
    for i in range(2000):
        dlog.get("x").info("flood", i=i)
    took = time.perf_counter() - t0
    dlog._pause_writer_for_test(False)
    dropped = dlog.dropped_count()
    dlog.shutdown()
    assert took < 1.0 and dropped > 1000                     # the caller was never held up; the loss is counted, not hidden


def test_drops_are_reported_in_the_log_after_the_writer_recovers():
    d, run = fresh(level="INFO", queue_size=20)
    dlog._pause_writer_for_test(True)
    for i in range(200):
        dlog.get("x").info("flood", i=i)
    dlog._pause_writer_for_test(False)
    dlog.get("x").info("after")
    dlog.shutdown()
    evs = [r["ev"] for r in read(os.path.join(run, "gabai.jsonl"))]
    assert "log.dropped" in evs


def test_files_rotate_and_old_runs_are_deleted():
    d = tempfile.mkdtemp()
    for i in range(5):
        r = dlog.configure(log_dir=d, run_name=f"r{i}", keep_runs=3)
        dlog.get("x").info("hello")
        dlog.shutdown()
        time.sleep(0.02)
    runs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
    assert len(runs) == 3
    d2, run = None, dlog.configure(log_dir=tempfile.mkdtemp(), run_name="rot", max_mb=0.002, backups=2)
    for i in range(400):
        dlog.get("x").info("fill", pad="y" * 50, i=i)
    dlog.shutdown()
    files = sorted(os.listdir(run))
    assert "gabai.jsonl" in files and "gabai.jsonl.1" in files and "gabai.jsonl.3" not in files


def test_rate_limit_keeps_one_line_and_counts_what_it_swallowed():
    d, run = fresh(level="INFO")
    lg = dlog.get("x")
    for _ in range(100):
        lg.warning_every("link.late", 60.0, "link.late", age=0.4)
    dlog.shutdown()
    rows = [r for r in read(os.path.join(run, "gabai.jsonl")) if r["ev"] == "link.late"]
    assert len(rows) == 1


def test_per_call_overhead_is_small():
    d, run = fresh(level="INFO")
    lg = dlog.get("x")
    t0 = time.perf_counter()
    for i in range(5000):
        lg.info("e", fid=i, dur_ms=1.5, a=1, b=2.5)
    per_call_us = (time.perf_counter() - t0) / 5000 * 1e6
    dlog.shutdown()
    assert per_call_us < 300, per_call_us


def test_rate_limits_do_not_carry_over_from_one_run_to_the_next():
    for name in ("a", "b"):
        run = dlog.configure(log_dir=tempfile.mkdtemp(), run_name=name, level="INFO")
        dlog.get("x").warning_every("k", 3600.0, "once.per.run")
        dlog.shutdown()
        assert any(r["ev"] == "once.per.run" for r in read(os.path.join(run, "gabai.jsonl")))
