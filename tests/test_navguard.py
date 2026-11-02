import json
import os
import subprocess
import sys
import tempfile
import time

from gabai.navguard import NavHeartbeat, nav_active


def path():
    return os.path.join(tempfile.mkdtemp(), "nav.json")


def other_process():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])


def test_no_file_means_navigation_is_not_running():
    assert nav_active(path()) is False


def test_a_fresh_heartbeat_from_another_live_process_is_active():
    p = path()
    proc = other_process()
    try:
        json.dump({"pid": proc.pid, "t": time.time(), "what": "demo_viewer"}, open(p, "w"))
        assert nav_active(p) is True
    finally:
        proc.kill()


def test_a_stale_heartbeat_is_not_active():
    p = path()
    proc = other_process()
    try:
        json.dump({"pid": proc.pid, "t": time.time() - 10, "what": "demo_viewer"}, open(p, "w"))
        assert nav_active(p) is False
    finally:
        proc.kill()


def test_a_heartbeat_from_a_dead_process_is_not_active():
    p = path()
    proc = other_process()
    proc.kill()
    proc.wait()
    json.dump({"pid": proc.pid, "t": time.time(), "what": "demo_viewer"}, open(p, "w"))
    assert nav_active(p) is False


def test_our_own_heartbeat_does_not_count_as_somebody_else_navigating():
    p = path()
    json.dump({"pid": os.getpid(), "t": time.time(), "what": "x"}, open(p, "w"))
    assert nav_active(p) is False


def test_garbage_in_the_file_is_not_active_and_does_not_raise():
    p = path()
    open(p, "w").write("{not json")
    assert nav_active(p) is False
    open(p, "w").write(json.dumps({"pid": "abc"}))
    assert nav_active(p) is False


def test_the_heartbeat_writes_refreshes_and_removes_its_file():
    p = path()
    hb = NavHeartbeat(p, period_s=0.1).start()
    t1 = json.load(open(p))["t"]
    time.sleep(0.35)
    t2 = json.load(open(p))["t"]
    assert t2 > t1 and json.load(open(p))["pid"] == os.getpid()
    hb.stop()
    assert not os.path.exists(p)


def test_a_heartbeat_that_cannot_be_written_does_not_raise():
    hb = NavHeartbeat(os.path.join(tempfile.mkdtemp(), "no", "such", "\0bad.json"), period_s=0.1)
    hb._write()                                                           # must swallow the OSError / ValueError
