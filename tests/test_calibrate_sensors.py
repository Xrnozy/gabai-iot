"""The guided calibration, end to end on a simulated device that moves the way each instruction asks, plus the maths and the files on their own.
Simulated truth: IMU axes = device (camera) axes by R_DEV_IMU, compass axes = IMU axes by P_TRUE, a hard-iron offset, a tilted soft-iron stretch, an accelerometer with a
known bias and scale error. The assertions compare what the script finds and saves with that truth."""
import json
import math
import os
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from gabai import calimu, calmag, calstore
from gabai.frames import nearest_rotation
from gabai.mag import apply_cal, calibrate_ellipsoid, load_accel_cal, load_mag_cfg
from gabai.mag_sync import rotations_24
from gabai.slam.accel_cal import G
from gabai.tools.calibrate_sensors import NoData, Session, UI

EARTH, DIP = 489.0, math.radians(55)
B_W = EARTH * np.array([math.cos(DIP), 0.0, -math.sin(DIP)])
HARD = np.array([-120.0, 900.0, 130.0])
_Q = Rotation.from_euler("xyz", [25, 30, 40], degrees=True).as_matrix()
SOFT = _Q @ np.diag([0.85, 1.1, 1.0]) @ _Q.T
R_DEV_IMU = np.array([[0.0, -1, 0], [0, 0, 1], [-1, 0, 0]])            # device x = -IMU y, y = IMU z, z = -IMU x: the camera axis is IMU X (gravity +1 g on X face-down)
P_TRUE = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])                # IMU <- compass: compass X and IMU Y, compass Y and IMU X, Z and Z, one sign to stay a rotation
A_BIAS, A_SCALE = np.array([0.25, -0.15, 0.10]), np.array([0.010, -0.020, 0.015])     # m/s^2 and per-axis scale: raw = (g + bias) / (1 + scale)


def up_from(up_dev):
    """Device -> world rotation that makes the device-frame 'up' vector the world z."""
    u = np.asarray(up_dev, float)
    u = u / np.linalg.norm(u)
    ax = np.cross(u, [0, 0, 1.0])
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return np.eye(3) if u[2] > 0 else np.diag([1.0, -1, -1])
    return Rotation.from_rotvec(ax / s * math.atan2(s, u[2])).as_matrix()


def facedown(yaw=0.0):
    return Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ up_from([0, 0, -1.0])      # camera side (device z) pointing down


def upright(pitch_down=8.0):
    return up_from([0, -math.cos(math.radians(pitch_down)), -math.sin(math.radians(pitch_down))])


class SimSource:
    """Stands in for the Pi link. Before each step the session says what it is about to ask (hint); the simulated hand then does that movement."""

    def __init__(self, seed=0, interrupt=None, dead=False, still=False):
        self.rng = np.random.default_rng(seed)
        self.t, self.R, self.activity, self.omega, self.calls = 0.0, facedown(30.0), "rest_facedown", np.zeros(3), 0
        self.interrupt, self.dead, self.still = interrupt, dead, still
        self.hold_i, self.hold_t, self.mag_buf = 0, 0.0, []
        self.faces = [(+1, 0, 0), (-1, 0, 0), (0, +1, 0), (0, -1, 0), (0, 0, +1), (0, 0, -1)]

    def hint(self, name):
        self.activity, self.calls = name, 0
        if name in ("rest_facedown", "imu_rest_facedown"):
            self.R = facedown(30.0)
        elif name == "imu_upright":
            self.R = upright()
        self.hold_i, self.hold_t = 0, 99.0

    def _omega_imu(self):
        a, w = self.activity, np.zeros(3)
        if self.still and a.startswith("mag_"):
            return w                                                               # the device is not moved at all
        if a == "mag_eights":
            self.omega = 0.97 * self.omega + 0.03 * self.rng.normal(0, 260, 3)             # smooth random turning in every direction
            w = self.omega
        elif a == "imu_gyro_clockwise":
            w = R_DEV_IMU.T @ np.array([0, 0, -40.0])                              # clockwise seen from the camera side: negative about the device's forward axis
        return w

    def imu(self):
        if self.dead:
            return []
        self.calls += 1
        if self.interrupt and self.activity == self.interrupt[0] and self.calls > self.interrupt[1]:
            raise KeyboardInterrupt
        out = []
        for _ in range(5):
            self.t += 0.01
            if self.activity == "imu_holds":
                self.hold_t += 0.01
                if self.hold_t > 4.2:                                                   # the next side
                    self.hold_i, self.hold_t = (self.hold_i + 1) % 6, 0.0
                    f = self.faces[self.hold_i]
                    self.R = up_from(R_DEV_IMU @ np.array(f, float))                    # IMU axis f pointing up
                w_imu = np.zeros(3) if self.hold_t > 0.7 else np.array([0, 0, 90.0])
            else:
                w_imu = self._omega_imu()
            self.R = self.R @ Rotation.from_rotvec(R_DEV_IMU @ np.radians(w_imu) * 0.01).as_matrix()
            up_imu = R_DEV_IMU.T @ (self.R.T @ [0, 0, 1.0])
            g = (up_imu * G * (1 + A_SCALE) - A_BIAS) * 0 + (up_imu * G + A_BIAS) / (1 + A_SCALE)       # raw accelerometer reading in m/s^2
            gyro = w_imu + self.rng.normal(0, 0.08, 3)
            acc = g / G + self.rng.normal(0, 0.003, 3)
            out.append((self.t, *gyro, *acc))
            if int(round(self.t * 100)) % 2 == 0:
                b_imu = R_DEV_IMU.T @ (self.R.T @ B_W)
                self.mag_buf.append((self.t, *(SOFT @ (P_TRUE.T @ b_imu) + HARD + self.rng.normal(0, 2.0, 3))))
        return out

    def mag(self):
        out, self.mag_buf = self.mag_buf, []
        return out

    def close(self):
        pass


def scripted(answers=None, record=None):
    answers = answers or {}

    def ask(prompt):
        if record is not None:
            record.append(prompt)
        for k, v in answers.items():
            if k in prompt:
                return v
        return ""
    return ask


def session(tmp, source=None, ask=None, **kw):
    ui = UI(enabled=False)
    return Session(source or SimSource(), ui, ask or scripted({"Save": "y", "Continue with": "n", "Continue with the BMI": "y"}), data_dir=tmp, no_data_s=5.0, **kw), ui


def test_the_whole_compass_phase_on_a_simulated_device_saves_a_correct_validated_calibration():
    tmp = tempfile.mkdtemp()
    s, ui = session(tmp)
    doc = s.run_mag()
    assert doc is not None and os.path.exists(s.mag_path)
    assert calstore.validate_mag(json.load(open(s.mag_path))) == []
    par = doc["parameters"]
    assert np.abs(np.array(par["offset"]) - HARD).max() < 12.0, par["offset"]
    corrected = apply_cal({"offset": par["offset"], "W": par["W"], "scale": [1, 1, 1]}, np.array([(SOFT @ (P_TRUE.T @ (R_DEV_IMU.T @ (Rotation.random(random_state=k).as_matrix().T @ B_W))) + HARD) for k in range(300)]))
    n = np.linalg.norm(corrected, axis=1)
    assert np.std(n) / np.mean(n) < 0.03                                       # the saved correction really makes the field a sphere
    assert np.array_equal(np.array(doc["axis_mapping"]["matrix_to_imu"]).round().astype(int), P_TRUE.astype(int)), doc["axis_mapping"]["matrix_to_imu"]
    assert doc["axis_mapping"]["matches_reference_permutation"] is True and doc["applies_to"] == "raw" and par["coverage"] >= 0.8
    legacy = json.load(open(os.path.join(tmp, "mag_cal.json")))
    assert np.allclose(legacy["W"], par["W"]) and np.allclose(legacy["offset"], par["offset"])
    assert np.allclose(json.load(open(os.path.join(tmp, "mag_sync.json")))["R_im"], P_TRUE)
    kinds = [c[0] for c in ui.calls]
    texts = " ".join(c[4] + c[5] for c in ui.calls if c[0] == "instruct")
    assert kinds.count("instruct") == 2 and "LIFT" in texts and "figure-eights" in texts
    assert any(c[0] == "progress" and "coverage" in c[1] for c in ui.calls) and any(c[0] == "cloud" for c in ui.calls) and any(c[0] == "saved" for c in ui.calls)
    assert not os.path.exists(os.path.join(tmp, "calibration", "mag_partial.npz"))


def test_nothing_is_saved_without_the_users_yes():
    tmp = tempfile.mkdtemp()
    s, _ = session(tmp, ask=scripted({"Save": "n"}))
    assert s.run_mag() is None and not os.path.exists(s.mag_path) and not os.path.exists(os.path.join(tmp, "mag_cal.json"))
    assert os.path.exists(os.path.join(tmp, "calibration", "mag_partial.npz"))       # the capture is kept to continue


def test_too_little_rotation_is_reported_and_the_result_is_not_even_offered_for_saving():
    tmp = tmp_ = tempfile.mkdtemp()
    asked = []
    s, ui = session(tmp, source=SimSource(still=True), ask=scripted({"Save": "y"}, asked), max_step_s=12.0)
    assert s.run_mag() is None and not os.path.exists(s.mag_path)
    assert not any("Save" in a for a in asked)
    assert s.recording is None
    assert sum(1 for c in ui.calls if c[0] == "instruct") == 2                  # stayed on the first movement step: it never advanced to the next one
    assert any(c[0] == "status" and "Step not completed" in " ".join(c[1]) for c in ui.calls)


def test_stopping_with_ctrl_c_saves_nothing_and_leaves_no_recording_and_a_rerun_finishes():
    tmp = tempfile.mkdtemp()
    s, _ = session(tmp, source=SimSource(interrupt=("mag_eights", 300)))
    try:
        s.run_mag()
        assert False, "should have been interrupted"
    except KeyboardInterrupt:
        pass
    assert not os.path.exists(s.mag_path) and s.recording is None                    # the one figure-eights step was not finished, so nothing is kept or saved
    s2, _ = session(tmp, source=SimSource(seed=5), ask=scripted({"Save": "y"}))
    assert s2.run_mag() is not None and os.path.exists(s2.mag_path)


def test_the_imu_phase_will_not_start_before_the_compass_is_saved_and_then_calibrates_the_accelerometer_and_the_device_axes():
    tmp = tempfile.mkdtemp()
    s, ui = session(tmp)
    assert s.run_imu() is None and any("must be completed and saved" in " ".join(c[1]) for c in ui.calls if c[0] == "status")
    assert s.run_mag() is not None
    s, ui = session(tmp, source=SimSource(seed=2))
    doc = s.run_imu()
    assert doc is not None and calstore.validate_imu(json.load(open(s.imu_path))) == []
    a = doc["accel"]
    assert a["calibrated"], a
    assert np.abs(np.array(a["bias_ms2"]) - A_BIAS).max() < 0.05 and np.abs(np.array(a["scale"]) - A_SCALE).max() < 0.004, (a["bias_ms2"], a["scale"])
    R = np.array(doc["axis_mapping"]["device_from_imu"])
    assert np.allclose(R, R_DEV_IMU, atol=0.05) and doc["axis_mapping"]["matches_reference_permutation"] is True
    assert doc["gyro"]["bias_policy"] == "not_applied" and max(doc["gyro"]["noise_dps"]) < 0.2
    cal = load_accel_cal(os.path.join(tmp, "accel_calib.json"))
    assert cal is not None and np.allclose(cal["bias"], a["bias_ms2"])
    assert np.allclose(json.load(open(os.path.join(tmp, "device_frame.json")))["R_dev_imu"], R)
    gyro_check = [t for c in ui.calls if c[0] == "report" for n, st, t in c[2] if n == "gyro direction"]
    assert gyro_check and "predicts" in gyro_check[-1] and all(st == "pass" for c in ui.calls if c[0] == "report" for n, st, t in c[2] if n == "gyro direction")


def test_the_device_still_reads_as_the_instructions_say_when_the_imu_holds_do_not_cover_the_faces():
    tmp = tempfile.mkdtemp()
    s, _ = session(tmp)
    s.run_mag()
    two = [np.array(R_DEV_IMU @ f) * 1.0 for f in ((0, 0, 1.0), (0, 0, -1.0))]
    r = calimu.fit_accel([((x + A_BIAS / G) / (1 + A_SCALE)) for x in [R_DEV_IMU.T @ v for v in two]])
    assert r["calibrated"] is False and "six faces" in r["reason"] and r["faces_missing"]


def test_no_data_from_the_pi_stops_with_a_clear_error_instead_of_hanging():
    tmp = tempfile.mkdtemp()
    now = {"t": 0.0}
    ui = UI(enabled=False)
    s = Session(SimSource(dead=True), ui, scripted(), data_dir=tmp, clock=lambda: now.__setitem__("t", now["t"] + 1.0) or now["t"], sleep=lambda x: None, no_data_s=5.0)
    try:
        s.run_mag()
        assert False
    except NoData as e:
        assert "Pi sender" in str(e)


def test_files_are_validated_before_use_and_a_bad_one_is_ignored_with_the_reason():
    tmp = tempfile.mkdtemp()
    s, _ = session(tmp)
    s.run_mag()
    good = json.load(open(s.mag_path))
    assert calstore.validate_mag(good) == []
    cases = {"nan offset": lambda d: d["parameters"].__setitem__("offset", [float("nan"), 0, 0]), "mirrored W": lambda d: d["parameters"].__setitem__("W", np.diag([1, 1, -1]).tolist()),
             "wild quality": lambda d: d["parameters"].__setitem__("quality", 0.5), "wrong sensor": lambda d: d["sensor"].__setitem__("kind", "BMI160"),
             "not raw": lambda d: d.__setitem__("applies_to", "corrected"), "no date": lambda d: d.pop("date"), "wrong schema": lambda d: d.__setitem__("schema", "x/9"),
             "bad rotation": lambda d: d["axis_mapping"].__setitem__("matrix_to_imu", (np.eye(3) * 2).tolist()), "missing section": lambda d: d.pop("parameters")}
    for name, damage in cases.items():
        d = json.loads(json.dumps(good))
        damage(d)
        assert calstore.validate_mag(d), name
        try:
            calstore.save(os.path.join(tmp, "x.json"), d, calstore.validate_mag)
            assert False, name
        except ValueError:
            pass
    open(s.mag_path, "w").write("{broken")
    assert any("INVALID" in l for l in session(tmp)[0].status_lines())
    doc, problems = calstore.load(s.mag_path, calstore.validate_mag)
    assert doc is None and problems


def test_a_saved_imu_file_that_would_subtract_the_gyro_bias_twice_is_rejected():
    tmp = tempfile.mkdtemp()
    s, _ = session(tmp)
    s.run_mag()
    s2, _ = session(tmp, source=SimSource(seed=2))
    doc = s2.run_imu()
    bad = json.loads(json.dumps(doc))
    bad["gyro"]["bias_policy"] = "applied"
    assert any("twice" in p for p in calstore.validate_imu(bad))
    bad = json.loads(json.dumps(doc))
    bad["accel"] = {"calibrated": True, "bias_ms2": [9, 0, 0], "scale": [0, 0, 0]}
    assert any("implausibly" in p for p in calstore.validate_imu(bad))


def test_valid_results_are_loaded_automatically_and_the_correction_is_applied_once():
    tmp = tempfile.mkdtemp()
    s, _ = session(tmp)
    doc = s.run_mag()
    s2, _ = session(tmp)
    assert all("calibrated 20" in l for l in s2.status_lines()[:1])
    cfg = load_mag_cfg(os.path.join(tmp, "mag_cal.json"), os.path.join(tmp, "mag_sync.json"))        # what the running system loads
    raw = np.array([-100.0, 1000.0, 90.0])
    once = np.asarray(doc["parameters"]["W"]) @ (raw - np.asarray(doc["parameters"]["offset"]))
    assert np.allclose(apply_cal(cfg, raw), once) and cfg["radius_counts"] == doc["parameters"]["radius_counts"]
    assert doc["applies_to"] == "raw" and np.allclose(cfg["R_im"], doc["axis_mapping"]["matrix_to_imu"])


# ---- the maths on its own -----------------------------------------------------------------------------------------------------------------------------------
def tumble(n=3000, seed=0, P=P_TRUE):
    rng = np.random.default_rng(seed)
    Rs = [Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix() for _ in range(n)]
    ups = np.array([R.T @ [0, 0, 1.0] for R in Rs])
    mags = np.array([SOFT @ (P.T @ (R.T @ B_W)) + HARD + rng.normal(0, 2.0, 3) for R in Rs])
    return mags, ups


def test_every_compass_axis_map_is_found_from_a_general_tumble():
    for k, P in enumerate(rotations_24()):
        mags, ups = tumble(1500, seed=k, P=P)
        cal = calmag.fit(mags)
        got, score, nxt, ok, ang = calmag.estimate_axis_map(apply_cal(cal, mags), ups)
        assert np.array_equal(got, P) and ok, (k, score, nxt)


def test_coverage_names_what_is_missing_and_reaches_one_for_a_full_tumble():
    mags, _ = tumble(1500)
    cov, missing = calmag.direction_coverage(mags)
    assert cov > 0.95 and len(missing) <= 1
    half = mags[mags[:, 0] > HARD[0]]
    cov2, missing2 = calmag.direction_coverage(half)
    assert cov2 < 0.7 and any(m.startswith("-X") or "-X" in m for m in missing2)
    assert calmag.direction_coverage(mags[:5])[0] == 0.0


def test_the_checks_flag_a_flat_turn_a_strong_magnet_and_spikes_and_pass_a_good_tumble():
    mags, ups = tumble(2500)
    cal = calmag.fit(mags)
    cl = calmag.checks(cal, mags, up=ups, P=P_TRUE)
    assert calmag.can_save(cl) and all(s != "fail" for _, s, _ in cl), cl
    rng = np.random.default_rng(1)
    flat = np.array([SOFT @ (P_TRUE.T @ (Rotation.from_euler("z", a, degrees=True).as_matrix().T @ B_W)) + HARD + rng.normal(0, 2, 3) for a in rng.uniform(0, 360, 800)])
    cl = calmag.checks(calmag.fit(flat), flat)
    assert not calmag.can_save(cl) and any(n == "coverage of directions" and s == "fail" for n, s, _ in cl)
    spiky = mags.copy()
    spiky[::20] += np.array([900.0, 0, 0])
    assert any(n == "abnormal readings" and s == "warn" for n, s, _ in calmag.checks(calmag.fit(spiky), spiky))
    wrong = calmag.checks(cal, mags, up=ups, P=np.eye(3))
    assert any(n.startswith("angle to gravity") and s != "pass" for n, s, _ in wrong)


def test_gyro_and_accelerometer_statistics_flag_motion_a_big_residual_and_a_bad_scale():
    rng = np.random.default_rng(0)
    t = np.arange(1000) / 100
    still = rng.normal(0, 0.08, (1000, 3))
    gs, ac = calimu.gyro_stats(t, still), calimu.accel_stats(np.tile([0, 0, 1.0], (1000, 1)) + rng.normal(0, 0.003, (1000, 3)))
    assert calimu.warnings_for(gs, ac) == []
    gs2 = calimu.gyro_stats(t, still + [1.5, 0, 0])
    assert any("residual" in w for w in calimu.warnings_for(gs2, ac))
    assert any("scale or bias" in w for w in calimu.warnings_for(gs, calimu.accel_stats(np.tile([0, 0, 0.8], (10, 1)))))


def test_device_axes_from_gravity_and_the_dominant_rotation_axis():
    up_down = R_DEV_IMU.T @ np.array([0, 0, -1.0])                   # face-down: 'up' is the device's -z
    up_worn = R_DEV_IMU.T @ np.array([0, -1.0, 0])
    assert np.allclose(calimu.device_from_imu(up_down, up_worn), R_DEV_IMU, atol=1e-9)
    t = np.arange(500) / 100
    g = np.zeros((500, 3))
    g[:, 1] = 60.0
    assert calimu.dominant_axis(t, g)[0] == 1 and abs(calimu.angle_about(t, g, 1)[0] - 300) < 1


def test_the_real_rerun_front_end_runs_every_call_and_records_a_file():
    d = tempfile.mkdtemp()
    out = os.path.join(d, "cal.rrd")
    ui = UI(enabled=True, spawn=False, save=out)
    mags, _ = tumble(300)
    ui.instruct("HMC5883L compass", 1, 6, "Roll", "**LIFT** it")
    ui.status(["a", "b"])
    ui.sample_imu(0.0, [0, 0, 1], [0, 0, 1])
    ui.sample_mag(0.0, mags[0])
    ui.cloud(0.1, mags, corrected=mags - mags.mean(0), centre=mags.mean(0))
    ui.progress(0.1, coverage=0.5, movement_done=0.2)
    ui.report("result", [("coverage", "pass", "ok"), ("quality", "warn", "meh"), ("map", "fail", "bad")], ["extra"])
    ui.saved("done")
    import rerun as rr
    rr.disconnect()
    assert os.path.exists(out) and os.path.getsize(out) > 500


def test_the_session_drives_the_real_rerun_front_end_through_a_whole_phase():
    d = tempfile.mkdtemp()
    out = os.path.join(d, "session.rrd")
    ui = UI(enabled=True, spawn=False, save=out)
    s = Session(SimSource(seed=3), ui, scripted({"Save": "y", "Continue with": "n"}), data_dir=d, no_data_s=5.0)
    assert s.run_mag() is not None
    import rerun as rr
    rr.disconnect()
    assert os.path.getsize(out) > 20000


# ---- the four reported problems, one group each --------------------------------------------------------------------------------------------------------------
import threading
import time

from gabai.orient import q_to_R
from gabai.tools.calibrate_sensors import GYRO_TURN_DONE_DEG, MIN_STEP_MAG


class Link:
    """A hardware stand-in with a real buffer like LiveReceiver: samples pile up while nobody reads; discard() empties it. Counts every read."""

    def __init__(self, gyro=(0.0, 0.0, 0.0), with_mag=True):
        self.t, self.gyro, self.with_mag = 0.0, np.array(gyro, float), with_mag
        self.pi, self.pm, self.reads, self.discards, self.hints = [], [], 0, 0, []

    def produce(self, n):
        for _ in range(n):
            self.t += 0.01
            self.pi.append((self.t, *self.gyro, 0.0, 0.0, 1.0))
            if self.with_mag and int(round(self.t * 100)) % 2 == 0:
                self.pm.append((self.t, 100.0 + self.t, 200.0, -300.0))

    def imu(self):
        self.reads += 1
        self.produce(5)
        out, self.pi = self.pi, []
        return out

    def mag(self):
        out, self.pm = self.pm, []
        return out

    def hint(self, name):
        self.hints.append(name)

    def discard(self):
        n = len(self.pi) + len(self.pm)
        self.pi, self.pm = [], []
        self.discards += 1
        return n

    def close(self):
        pass


def rows_turning(axis, dps, seconds, wobble=0.0, hz=100):
    """IMU rows of a body turning about one IMU axis at dps (+ an optional back-and-forth wobble of the same axis, degrees/second amplitude)."""
    out = []
    for i in range(int(seconds * hz)):
        t = i / hz
        g = np.zeros(3)
        g[axis] = dps + wobble * math.sin(2 * math.pi * 3 * t)
        out.append((t, *g, 0.0, 0.0, 1.0))
    return out


# 1. missing compass samples ------------------------------------------------------------------------------------------------------------------------------------
def _entities_with_rows(path):
    """Rows per entity actually stored in a saved .rrd, read back with Rerun's own `rrd print` (no extra packages)."""
    import re
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-m", "rerun", "rrd", "print", path], capture_output=True, text=True, timeout=120).stdout
    seen = {}
    for rows, ent in re.findall(r"Chunk\(\w+\) with (\d+) rows .*? - (/\S+) - data columns", out):
        seen[ent] = seen.get(ent, 0) + int(rows)
    return seen


def test_compass_samples_are_in_the_rerun_recording_graphs_and_3d_cloud():
    d = tempfile.mkdtemp()
    out = os.path.join(d, "compass.rrd")
    ui = UI(enabled=True, spawn=False, save=out)
    s = Session(SimSource(seed=5), ui, scripted({"Save": "y", "Continue with": "n"}), data_dir=d, no_data_s=5.0)
    assert s.run_mag() is not None
    import rerun as rr
    rr.disconnect()
    seen = _entities_with_rows(out)
    assert all(seen.get(f"/mag/raw/{k}", 0) > 100 for k in "xyz"), seen                                       # the compass graphs have real rows
    assert seen.get("/mag3d/raw_samples", 0) > 5, seen                                                        # and the 3D cloud grew
    assert seen.get("/model/imu", 0) > 100 and seen.get("/ui/recording", 0) >= 2 and seen.get("/imu/gyro_dps/x", 0) > 100, seen


def test_the_live_view_draws_compass_while_waiting_for_enter_and_keeps_none_of_it():
    link = Link()
    ui = UI(enabled=False)
    gate = threading.Event()

    def slow_ask(prompt):
        gate.wait(0.4)                                        # the user is reading the instructions
        return ""

    s = Session(link, ui, slow_ask, data_dir=tempfile.mkdtemp(), no_data_s=5.0, live=True)
    s.ask("Press Enter: ")
    kinds = [c[0] for c in ui.calls]
    assert kinds.count("mag") > 3 and "live_cloud" in kinds and "model" in kinds, {k: kinds.count(k) for k in set(kinds)}   # compass, cloud and model drawn while waiting
    assert [c for c in ui.calls if c[0] == "recording"][0][1] is False                                              # announced as live view, not recording
    t_end = link.t
    link.produce(300)                                        # samples that pile up unread, right after the answer
    imu, mag = s.collect("rest_facedown", seconds=0.3)
    assert min(r[0] for r in imu) > t_end + 2.9 and min(r[0] for r in mag) > t_end + 2.9, (min(r[0] for r in imu), t_end)   # none of the live or buffered samples are in the capture
    assert link.discards >= 2


# 2. BMI160 3D model --------------------------------------------------------------------------------------------------------------------------------------------
def _tilting_rows(axis, total_deg, seconds, hz=100):
    """Accelerometer and gyro rows of a board really rotating about one IMU axis starting level (z up): gravity in sensor axes comes from the true rotation."""
    rows, rate = [], np.radians(total_deg) / seconds
    unit = np.eye(3)[axis]
    for i in range(int(seconds * hz) + 1):
        t = i / hz
        R_ws = Rotation.from_rotvec(unit * rate * t).as_matrix()          # sensor -> world after turning
        up_sensor = R_ws.T @ np.array([0, 0, 1.0])
        rows.append((t, *np.degrees(unit * rate), *up_sensor))
    return rows


def test_the_3d_model_follows_the_real_rotation_of_the_sensor():
    for axis in (0, 1):
        s, ui = session(tempfile.mkdtemp(), source=Link())
        rows = _tilting_rows(axis, 70.0, 3.0)
        s._attitude(rows)
        want = np.array(rows[-1][4:7])                                    # true 'up' in sensor axes at the end
        got = q_to_R(s.att.q).T @ np.array([0, 0, 1.0])
        assert np.degrees(math.acos(np.clip(np.dot(want, got), -1, 1))) < 3.0, (axis, want, got)
        assert any(c[0] == "model" for c in ui.calls)
    # 90 degrees about IMU x: world up, seen in the sensor axes, moves from +z to -y or +y (not staying on z)
    s, _ = session(tempfile.mkdtemp(), source=Link())
    s._attitude(_tilting_rows(0, 90.0, 3.0))
    up = q_to_R(s.att.q).T @ np.array([0, 0, 1.0])
    assert abs(up[2]) < 0.1 and abs(up[1]) > 0.95, up


def test_the_model_axes_are_the_reference_ones_and_the_view_has_the_camera_axis_pointing_down_face_down():
    from scipy.spatial.transform import Rotation
    from gabai.orient import q_from_accel
    from gabai.tools.calibrate_sensors import VIEW_FLIP, quat_mul
    s, ui = session(tempfile.mkdtemp(), source=Link())
    ms = [c for c in ui.calls if c[0] == "model_static"]
    assert np.allclose(ms[-1][1][2], [1, 0, 0])                                     # forward is IMU X
    q = quat_mul(VIEW_FLIP, q_from_accel(np.array([1.0, 0, 0])))                    # face-down: gravity reading on IMU X
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    assert np.allclose(r.apply([1, 0, 0]), [0, 0, -1], atol=1e-6)                   # IMU X, the camera axis, is drawn pointing down


# 3. a step advances only when its requirements are met ----------------------------------------------------------------------------------------------------------
def _req(s, key, axis, imu, mag_n=300, cov=0.0, pts=300):
    step = max(1, len(imu) // max(1, mag_n))
    mag = [(r[0], 1.0, 2.0, 3.0) for r in imu[::step]][:mag_n]
    return s.step_requirements(key, axis, imu, mag, imu[-1][0] - imu[0][0], cov, ["+x"], pts)


def test_a_movement_step_needs_the_full_turn_not_just_a_lot_of_shaking():
    s, _ = session(tempfile.mkdtemp(), source=Link())
    ok, need, deg = _req(s, "nod", 1, rows_turning(1, 60.0, 8.0))               # 480 deg of real turning
    assert ok and not need and deg >= GYRO_TURN_DONE_DEG
    ok, need, deg = _req(s, "nod", 1, rows_turning(1, 0.0, 30.0, wobble=200.0))    # 30 s of violent wobble about the right axis: huge total path, no turn
    assert not ok and deg < 60 and any("more degrees" in n for n in need), (deg, need)
    ok, need, deg = _req(s, "nod", 1, rows_turning(1, 60.0, 3.0))               # a half turn, fast
    assert not ok and 150 < deg < 200 and any("more degrees" in n for n in need) and any("s more" in n for n in need)
    ok, need, deg = _req(s, "nod", 1, rows_turning(2, 60.0, 8.0))               # a full turn, but about the wrong axis
    assert not ok and deg < 10 and any("IMU Z" in n and "IMU Y" in n for n in need), need


def test_a_movement_step_also_needs_enough_valid_compass_readings():
    s, _ = session(tempfile.mkdtemp(), source=Link())
    ok, need, _ = _req(s, "nod", 1, rows_turning(1, 60.0, 8.0), mag_n=40)
    assert not ok and any(f"{MIN_STEP_MAG - 40} more compass readings" in n for n in need), need


def test_the_figure_eights_step_needs_coverage_and_readings():
    s, _ = session(tempfile.mkdtemp(), source=Link())
    imu = rows_turning(0, 0.0, 10.0)
    assert not _req(s, "eights", None, imu, cov=0.6, pts=1000)[0]
    assert not _req(s, "eights", None, imu, cov=0.95, pts=100)[0]
    assert _req(s, "eights", None, imu, cov=0.95, pts=1000)[0]


def test_a_half_finished_step_never_advances_on_a_timer():
    class HalfTurn(SimSource):
        """Turns about the step's axis for 200 degrees, then holds still for ever."""

        def __init__(self):
            super().__init__(seed=1)
            self.turned = 0.0

        def _omega_imu(self):
            w = super()._omega_imu()
            if self.activity == "mag_eights":
                if self.turned >= 200.0:
                    return np.zeros(3)
                self.turned += float(np.abs(w).max()) * 0.01
            return w

    asked = []
    s, ui = session(tempfile.mkdtemp(), source=HalfTurn(), ask=scripted({"Save": "y"}, asked), max_step_s=60.0)
    assert s.run_mag() is None
    assert sum(1 for c in ui.calls if c[0] == "instruct") == 2 and not any("Save" in a for a in asked)      # still on the first movement step, nothing saved
    status = " ".join(" ".join(c[1]) for c in ui.calls if c[0] == "status")
    assert "STILL NEEDED: cover more directions" in status, status[-300:]


def test_without_a_timeout_the_step_keeps_waiting_and_only_stops_when_the_requirement_is_met():
    link = Link(gyro=(0.0, 60.0, 0.0))
    s, ui = session(tempfile.mkdtemp(), source=link)

    def until(imu, mag):
        return s.step_requirements("nod", 1, imu, mag, imu[-1][0] - imu[0][0], 0.0, [], 300)[0]

    imu, mag = s.collect("mag_nod", until=until)
    deg = calimu.sweep([r[0] for r in imu], [r[1:4] for r in imu], 1)[1]
    assert GYRO_TURN_DONE_DEG <= deg < 60.0 * 7.0 and len(mag) >= MIN_STEP_MAG and imu[-1][0] - imu[0][0] >= 6.0


# 4. recording really stops ---------------------------------------------------------------------------------------------------------------------------------------
def test_a_capture_is_announced_and_stops_reading_the_sensors_when_it_ends():
    link = Link()
    s, ui = session(tempfile.mkdtemp(), source=link)
    assert s.recording is None
    imu, mag = s.collect("rest_facedown", seconds=1.0)
    assert s.recording is None
    reads = link.reads
    time.sleep(0.2)
    assert link.reads == reads                                  # nothing reads the sensor once the capture returned
    marks = [(c[1], c[2]) for c in ui.calls if c[0] == "recording"]
    assert marks[0][0] is True and marks[-1][0] is False and "stopped" in marks[-1][1] and f"{len(imu)} IMU" in marks[-1][1]
    assert link.discards >= 2                                   # buffer emptied at the start and at the end


def test_a_failing_capture_still_stops_recording():
    link = Link()
    s, ui = session(tempfile.mkdtemp(), source=link)

    def boom(imu, mag, el):
        raise ValueError("tick failed")

    try:
        s.collect("mag_roll", seconds=5.0, tick=boom)
        assert False
    except ValueError:
        pass
    assert s.recording is None
    assert [c[1] for c in ui.calls if c[0] == "recording"][-1] is False
    reads = link.reads
    time.sleep(0.1)
    assert link.reads == reads


def test_two_captures_cannot_run_at_once_and_a_prompt_cannot_open_inside_one():
    s, _ = session(tempfile.mkdtemp(), source=Link())
    seen = {}

    def tick(imu, mag, el):
        try:
            s.collect("second", seconds=0.1)
        except RuntimeError as e:
            seen["nested"] = str(e)
        try:
            s.ask("Press Enter: ")
        except AssertionError as e:
            seen["ask"] = str(e)

    s.collect("first", seconds=0.2, tick=tick)
    assert "still recording" in seen["nested"] and "while recording" in seen["ask"]


def test_leaving_a_step_leaves_no_recording_running_across_the_whole_phase():
    s, ui = session(tempfile.mkdtemp(), source=SimSource(seed=2))
    assert s.run_mag() is not None
    marks = [c[1] for c in ui.calls if c[0] == "recording"]
    assert marks.count(True) >= 2 and marks[-1] is False
    assert all(not (a is True and b is True) for a, b in zip(marks, marks[1:]))        # a start is always followed by its stop first
    assert s.recording is None


def test_the_reference_mapping_has_the_camera_axis_on_imu_x_and_the_compass_agrees():
    assert calmag.REF_PERM_DEV_FROM_IMU[2].tolist() == [1, 0, 0]                          # device forward = IMU X
    assert calmag.same_permutation(P_TRUE, calmag.REF_PERM_IMU_FROM_MAG)                  # the compass map found on the real unit is the reference
    assert calmag.same_permutation(R_DEV_IMU, calmag.REF_PERM_DEV_FROM_IMU)


def test_a_device_lying_still_cannot_pass_the_figure_eights_on_compass_noise_alone():
    s, _ = session(tempfile.mkdtemp(), source=Link())
    imu = rows_turning(0, 0.0, 8.0)
    ok, need, _ = s.step_requirements("eights", None, imu, [], 8.0, 0.95, [], 1000, spread=2.0)
    assert not ok and any("hardly changes" in n for n in need)
    assert s.step_requirements("eights", None, imu, [], 8.0, 0.95, [], 1000, spread=250.0)[0]


def test_the_imu_phase_draws_no_compass_samples_or_cloud_points():
    tmp = tempfile.mkdtemp()
    s, _ = session(tmp)
    assert s.run_mag() is not None
    s, ui = session(tmp, source=SimSource(seed=2))
    assert s.run_imu() is not None
    kinds = {c[0] for c in ui.calls}
    assert "imu" in kinds and not kinds & {"mag", "cloud", "live_cloud"}, kinds
