import threading
import time

import numpy as np

from gabai.pipeline.perf import AsyncViz, Stopwatch, lower_priority


class Slow:
    """A visualisation sink that is very slow, like a viewer that cannot keep up."""

    def __init__(self, ms):
        self.ms, self.calls, self.order = ms, [], []

    def at(self, idx, t):
        self.order.append(("at", idx))

    def stereo(self, *a, **k):
        time.sleep(self.ms / 1000)
        self.order.append(("stereo", None))

    def scalar(self, path, v):
        self.order.append(("scalar", path))

    def event(self, text, level="INFO"):
        self.order.append(("event", text))


def test_a_slow_viewer_does_not_slow_the_caller():
    s = Slow(80)
    v = AsyncViz(s, max_queue=4)
    t0 = time.perf_counter()
    for k in range(40):
        v.at(k, k / 15)
        v.stereo(np.zeros((8, 8)))
        v.scalar("x", 1.0)
    caller_ms = (time.perf_counter() - t0) * 1000
    v.close()
    assert caller_ms < 100, caller_ms                      # 40 frames x 80 ms of drawing would be 3200 ms if the caller waited
    assert v.dropped["stereo"] > 20                        # the heavy picture calls were dropped, and counted
    assert [o for o in s.order if o[0] == "scalar"].__len__() == 40, "scalars and events are cheap and are never dropped"


def test_order_is_kept_so_the_timeline_position_matches_what_is_logged_after_it():
    s = Slow(0)
    v = AsyncViz(s)
    for k in range(5):
        v.at(k, k)
        v.scalar("a", k)
    v.close()
    assert [o[0] for o in s.order] == ["at", "scalar"] * 5 and [o[1] for o in s.order if o[0] == "at"] == list(range(5))


def test_an_error_in_the_viewer_thread_is_counted_and_does_not_kill_it():
    class Bad(Slow):
        def scalar(self, path, v):
            raise RuntimeError("viewer broke")

    s = Bad(0)
    v = AsyncViz(s)
    v.scalar("a", 1)
    v.event("still alive")
    v.close()
    assert v.errors == 1 and ("event", "still alive") in s.order


def test_lag_is_measured():
    v = AsyncViz(Slow(30), max_queue=100)
    for k in range(10):
        v.stereo(np.zeros(1))
    v.close()
    st = v.stats()
    assert st["done"] == 10 and st["lag_ms_max"] > 100 and st["dropped"] == {}


def test_stopwatch_reports_median_mean_and_p95_per_stage():
    sw = Stopwatch()
    for k in range(20):
        with sw("a"):
            time.sleep(0.002)
    with sw("b"):
        pass
    r = sw.report()
    assert r["a"]["n"] == 20 and r["a"]["median_ms"] >= 1.5 and r["a"]["p95_ms"] >= r["a"]["median_ms"] and "b" in r


def test_lowering_priority_reports_what_it_did_and_never_raises():
    r = lower_priority()
    assert isinstance(r, str) and r


def test_two_threads_each_keep_their_own_timeline_position_in_the_drawing_thread():
    from gabai.pipeline.perf import AsyncViz

    class Rec:
        def __init__(self):
            self.pos, self.log = None, []

        def at(self, idx, t):
            self.pos = idx

        def scalar(self, path, v):
            self.log.append((path, self.pos))

    r = Rec()
    v = AsyncViz(r)
    barrier = threading.Barrier(2)

    def worker(name, idx):
        v.at(idx, idx)
        barrier.wait()                                     # both threads have set their position before either logs
        for _ in range(20):
            v.scalar(name, 1.0)

    ts = [threading.Thread(target=worker, args=("a", 7)), threading.Thread(target=worker, args=("b", 3))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    v.close()
    assert all(pos == (7 if path == "a" else 3) for path, pos in r.log), "a call was drawn at another thread's frame"
    assert len(r.log) == 40


def test_prefetch_gives_every_item_in_order_and_passes_errors_on():
    from gabai.pipeline.perf import Prefetch
    assert list(Prefetch(range(50), depth=3)) == list(range(50))

    def gen():
        yield 1
        yield 2
        raise ValueError("decode failed")
    got = []
    try:
        for x in Prefetch(gen()):
            got.append(x)
        assert False
    except ValueError as e:
        assert "decode failed" in str(e) and got == [1, 2]


def test_prefetch_hides_a_slow_source_behind_the_work():
    from gabai.pipeline.perf import Prefetch

    def slow():
        for i in range(10):
            time.sleep(0.03)
            yield i
    t0 = time.perf_counter()
    for _ in Prefetch(slow(), depth=4):
        time.sleep(0.03)                                   # the work takes as long as reading
    wall = time.perf_counter() - t0
    assert wall < 0.30 + 0.12, wall                        # sequential would be 0.6 s


def test_stage_worker_runs_in_order_and_overlaps_with_the_caller():
    from gabai.pipeline.perf import StageWorker
    out = []

    def slow(x):
        time.sleep(0.03)
        out.append(x)
    w = StageWorker(slow, maxsize=2)
    t0 = time.perf_counter()
    for i in range(10):
        time.sleep(0.03)                                   # the caller's own work
        w.submit(i)
    w.close()
    assert out == list(range(10)) and time.perf_counter() - t0 < 0.30 + 0.15


def test_stage_worker_can_drop_instead_of_waiting_and_counts_it():
    from gabai.pipeline.perf import StageWorker
    w = StageWorker(lambda x: time.sleep(0.1), maxsize=1, drop_when_busy=True)
    accepted = [w.submit(i) for i in range(10)]
    w.close()
    assert sum(accepted) < 10 and w.dropped == 10 - sum(accepted) and w.dropped > 0


def test_stage_worker_error_reaches_the_caller():
    from gabai.pipeline.perf import StageWorker

    def boom(x):
        raise RuntimeError("stage b failed")
    w = StageWorker(boom)
    w.submit(1)
    try:
        for _ in range(50):
            time.sleep(0.01)
            w.submit(2)
        assert False
    except RuntimeError as e:
        assert "stage b failed" in str(e)


def test_priority_really_is_lowered_and_can_be_put_back():
    import os
    import psutil
    p = psutil.Process()
    before = p.nice()
    try:
        lower_priority()
        assert p.nice() != psutil.NORMAL_PRIORITY_CLASS if os.name == "nt" else p.nice() >= before
    finally:
        try:
            p.nice(before)
        except Exception:
            pass
