import json
import os
import tempfile
import zipfile

import numpy as np

from gabai.tools.collect_diag import collect, redact


def project():
    r = tempfile.mkdtemp()
    for d in ("data", "logs/viewer_a", "logs/viewer_b", "logs/viewer_a/artifacts", "data/runs", "data/slam/rec_x/mav0/cam0/data", "data/slam/rec_x/mav0/imu0"):
        os.makedirs(os.path.join(r, d), exist_ok=True)
    w = lambda p, t: open(os.path.join(r, p), "w", encoding="utf-8").write(t)
    for run in ("viewer_a", "viewer_b"):
        w(f"logs/{run}/gabai.jsonl", json.dumps({"ts": "x", "mono": 1.0, "run": run, "comp": "log", "lvl": "INFO", "ev": "log.start", "run_dir": run}) + "\n")
        w(f"logs/{run}/run.json", json.dumps({"run": run, "argv": ["x"], "env": {}}))
    w("logs/viewer_a/artifacts/1_left.jpg", "J" * 3000)
    w("data/stereo_calib.json", json.dumps({"baseline": 0.05, "token": "super-secret"}))
    w("data/imu_sync.json", "{}")
    w("data/runs/M2.csv", "t,fps_in\n1,15\n")
    w("data/telemetry_log.csv", "t,fps_in\n" + "\n".join(f"{i},15" for i in range(5000)))
    w("data/slam/rec_x/recording_info.json", json.dumps({"frames": 5}))
    w("data/slam/rec_x/trajectory.txt", "1 0 0 0 0 0 0 1\n")
    w("data/slam/rec_x/mav0/imu0/data.csv", "#h\n1,0,0,0,0,0,1\n")
    w("data/slam/rec_x/mav0/cam0/data/1.png", "P" * 5000)
    w("data/slam/rec_x/mav0/cam0/data.csv", "#h\n1,1.png\n")
    return r


def names(zpath):
    return zipfile.ZipFile(zpath).namelist()


def test_the_package_has_logs_calibration_summaries_and_a_manifest_but_no_pictures_by_default():
    r = project()
    z = collect(r, out=os.path.join(tempfile.mkdtemp(), "d.zip"), run_selfcheck=False)
    n = names(z)
    assert any(x.endswith("viewer_b/gabai.jsonl") for x in n) and any(x.endswith("stereo_calib.json") for x in n) and "MANIFEST.json" in n and "README.txt" in n
    assert any(x.endswith("rec_x/recording_info.json") for x in n) and any(x.endswith("rec_x/mav0/imu0/data.csv") for x in n)
    assert not any(x.endswith(".png") or x.endswith(".jpg") for x in n)                      # no camera pictures unless asked
    assert any("environment.json" in x for x in n)


def test_pictures_are_included_only_when_asked():
    r = project()
    z = collect(r, out=os.path.join(tempfile.mkdtemp(), "d.zip"), with_images=True, run_selfcheck=False)
    assert any(x.endswith("1_left.jpg") for x in names(z))


def test_secrets_are_redacted_in_everything_that_goes_into_the_zip():
    r = project()
    z = collect(r, out=os.path.join(tempfile.mkdtemp(), "d.zip"), run_selfcheck=False)
    zf = zipfile.ZipFile(z)
    cal = [x for x in zf.namelist() if x.endswith("stereo_calib.json")][0]
    text = zf.read(cal).decode()
    assert "super-secret" not in text and "<redacted>" in text
    assert redact('{"token": "abc", "x": 1}') == '{"token": "<redacted>", "x": 1}'
    assert redact('args token=hunter2 host=1.2.3.4') == 'args token=<redacted> host=1.2.3.4'


def test_the_manifest_lists_every_file_with_a_hash_and_says_what_was_left_out():
    r = project()
    z = collect(r, out=os.path.join(tempfile.mkdtemp(), "d.zip"), run_selfcheck=False)
    m = json.loads(zipfile.ZipFile(z).read("MANIFEST.json"))
    assert m["files"] and all(len(f["sha256"]) == 64 and f["bytes"] >= 0 for f in m["files"])
    assert any(s["why"] == "pictures excluded (use --with-images)" for s in m["left_out"])


def test_the_size_cap_skips_big_files_and_records_it():
    r = project()
    z = collect(r, out=os.path.join(tempfile.mkdtemp(), "d.zip"), max_mb=0.0005, run_selfcheck=False)
    m = json.loads(zipfile.ZipFile(z).read("MANIFEST.json"))
    assert any("size cap" in s["why"] for s in m["left_out"])


def test_only_the_newest_runs_are_taken_and_the_long_telemetry_csv_is_cut_to_its_tail():
    r = project()
    z = collect(r, out=os.path.join(tempfile.mkdtemp(), "d.zip"), runs=1, run_selfcheck=False)
    zf = zipfile.ZipFile(z)
    assert sum(x.endswith("gabai.jsonl") for x in zf.namelist()) == 1
    tel = [x for x in zf.namelist() if x.endswith("telemetry_log.csv")][0]
    assert len(zf.read(tel).decode().splitlines()) <= 2001
