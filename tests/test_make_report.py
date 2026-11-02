import csv
import os
import subprocess
import sys
import tempfile

from tests.test_report_metrics import rows, write_run


def run_report(args):
    r = subprocess.run([sys.executable, "-m", "gabai.tools.make_report"] + args, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout


def test_report_is_written_with_graphs_and_honest_gaps():
    with tempfile.TemporaryDirectory() as d:
        run = os.path.join(d, "walk1.csv")
        write_run(rows(alert=lambda i: "ahead" if 20 <= i < 30 else "", speech_ms=lambda i: 190 if i == 20 else "",
                       throttled=lambda i: "0x50005" if i > 70 else "0x0"), run)
        with open(run + ".marks.csv", "w", newline="") as f:
            f.write("t,kind\n1000.5,false_alarm\n")
        with open(os.path.join(d, "targets.csv"), "w", newline="") as f:
            f.write("time,label,light,true_m,est_m,p10_m,p90_m,valid_pct,model\n"
                    "x,wall,lit,1.0,1.02,1.0,1.05,99,m\nx,glass,lit,1.5,,,,0,m\nx,glass,lit,2.0,4.1,3.0,6.0,12,m\n")
        out = os.path.join(d, "rep")
        run_report(["--run", run, "--targets", os.path.join(d, "targets.csv"), "--out", out])
        md = open(os.path.join(out, "report.md"), encoding="utf-8").read()
        html = open(os.path.join(out, "report.html"), encoding="utf-8").read()
        files = os.listdir(out)
    for must in ("walk1_latency.png", "walk1_pipeline.png", "walk1_zones.png", "walk1_health.png", "walk1_speech.png",
                 "accuracy.png", "report.md", "report.html"):
        assert must in files, must
    assert "ahead x1" in md and "1 false, 0 missed" in md
    assert "Pi under-voltage since boot | YES" in md                       # a power fault is stated, not hidden
    assert "not measured" in md and "Known limits" in md
    assert "a target, not a result" in md
    assert "| glass | 2 | 1 |" in md                                        # one of the two glass tries had no depth at all
    assert "data:image/png;base64," in html                                # one self-contained page


def test_a_log_without_latency_says_not_measured_instead_of_zero():
    with tempfile.TemporaryDirectory() as d:
        run = os.path.join(d, "old.csv")
        with open(run, "w", newline="") as f:
            f.write("t,fps_in,loop_ms,alert\n1,15,60,\n2,15,70,\n")
        out = os.path.join(d, "rep")
        run_report(["--run", run, "--out", out])
        md = open(os.path.join(out, "report.md"), encoding="utf-8").read()
        files = os.listdir(out)
    assert "Capture to PC receipt, ms (p50 / p95) | not measured" in md
    assert "old_latency.png" not in files and "old_pipeline.png" in files


def test_a_run_that_was_never_done_is_listed_as_not_measured_instead_of_crashing():
    with tempfile.TemporaryDirectory() as d:
        run = os.path.join(d, "done.csv")
        write_run(rows(), run)
        out = os.path.join(d, "rep")
        run_report(["--run", run, "--run", os.path.join(d, "never_done.csv"), "--out", out])
        md = open(os.path.join(out, "report.md"), encoding="utf-8").read()
    assert "never_done: run not done" in md and "Run: done" in md
