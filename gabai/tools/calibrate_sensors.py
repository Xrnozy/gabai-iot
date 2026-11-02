"""Guided, one-sensor-at-a-time calibration of the HMC5883L compass and the BMI160 IMU, shown live in Rerun.

    PC:  py -m gabai.tools.calibrate_sensors            (Pi: python3 -m gabai.pi.sender --host <PC_IP> --left /dev/video2 --right /dev/video0)

Start with the device resting FACE-DOWN on the table (camera side down), still. The script asks for every movement; the instructions are in the Rerun window
("What to do now") and printed here; you press Enter here to start each step and answer y/n to confirm what is saved.
  Phase 1  HMC5883L: face-down rest, then lift it and draw slow figure-eights in every orientation (a full sphere) until 26 directions are covered. Hard-iron offset, soft-iron matrix, quality
           checks and the compass-to-IMU axis map are computed; you review them and confirm before anything is written.
  Phase 2  BMI160 (only after phase 1 is saved and you choose to continue): face-down rest (gyro noise, residual), still holds on all six faces (accelerometer bias and
           scale, only if the holds support them), upright still hold (device axes from gravity), a gyro direction check; review and confirm.
Ctrl+C at any time: completed results stay saved, and an unfinished capture is kept and offered when you start again.

Files (see gabai/calstore.py): data/calibration/hmc5883l.json and bmi160.json (the record: parameters, axis maps, matrices, timestamps) and the legacy files the running system
reads (data/mag_cal.json, mag_sync.json, accel_calib.json, device_frame.json). They are loaded and checked at every start. Raw readings are never modified; a correction is
applied in one place only, in the running system.
Options: --redo-mag, --redo-imu, --phase mag|imu, --no-viewer, --save-rrd FILE, --log-level DEBUG.
"""
import argparse
import os
import threading
import time

import numpy as np

from gabai import calimu, calmag, calstore
from gabai.diag import log as dlog
from gabai.frames import classify_mount, load_frames
from gabai.mag import MG_PER_COUNT, apply_cal
from gabai.orient import Madgwick, q_from_accel
from gabai.slam.accel_cal import FACES, G, StillCapture, cluster_orientations, faces_covered

log = dlog.get("calibrate")
GYRO_TURN_DONE_DEG = 180.0   # half a turn is enough to tell clockwise from counter-clockwise; a full circle was too hard to measure with the slow link
COVERAGE_GOAL = 0.9
MIN_STEP_MAG = 150          # valid compass readings a movement step must collect before it can finish
MIN_FIELD_SPREAD = 100.0       # counts: a compass that is really being turned through a sphere spreads by hundreds; a device lying still only jitters by a few
# Turn the whole view 180 deg about the world Y axis so the camera axis (IMU X, vertical when the device lies on the table) is drawn pointing DOWN when the device is face-down (the table is drawn above).
VIEW_FLIP = (0.0, 0.0, 1.0, 0.0)
# Picture only: device right / down / forward (rows) in IMU axes when worn upright: forward = IMU X, down = -IMU Z, right = -IMU Y.
MODEL_DEV_FROM_IMU = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
STILL_GYRO_DPS = 5.0           # a still hold: gyro below this (deg/s) ...
STILL_SHAKE_MS2 = 0.15         # ... and accelerometer steadier than this (m/s2 std); hands tremble, so looser than StillCapture's defaults (3 deg/s, 0.08)
MIN_STEP_S = 6.0            # a slow full turn takes longer than this; faster means the readings are too sparse
AXIS_NAMES = {0: "IMU X", 1: "IMU Y", 2: "IMU Z"}


class NoData(Exception):
    pass


class StepIncomplete(Exception):
    """A step's requirement was not met within the optional step timeout. Nothing advances and nothing is saved."""


# ---- Rerun front end -----------------------------------------------------------------------------------------------------------------------------------------
def quat_mul(a, b):
    """Hamilton product of two (w,x,y,z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


class UI:
    """Everything shown in Rerun. With enabled=False (tests) every call is recorded in .calls and nothing is drawn."""

    def __init__(self, enabled=True, spawn=True, save=None):
        self.enabled, self.calls, self.t0, self.rr = enabled, [], None, None
        self._last = {}
        if enabled:
            import rerun as rr
            self.rr = rr
            rr.init("gabai_calibrate_sensors", spawn=spawn)
            if save:
                rr.save(save)
            self._blueprint()
            rr.log("mag3d", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr.log("model", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def _blueprint(self):
        try:
            import rerun.blueprint as rrb
            self.rr.send_blueprint(rrb.Blueprint(rrb.Vertical(
                rrb.Horizontal(rrb.TextDocumentView(origin="ui/instructions", name="What to do now"), rrb.TextDocumentView(origin="ui/status", name="Progress, checks, warnings"),
                               rrb.TextDocumentView(origin="ui/recording", name="Recording or live view?"), column_shares=[3, 2, 1.2]),
                rrb.Horizontal(rrb.Spatial3DView(origin="mag3d", name="Compass samples (raw counts, sensor axes)"),
                               rrb.Spatial3DView(origin="model", name="Device orientation from the BMI160 (gyro + gravity)")),
                rrb.Horizontal(rrb.TimeSeriesView(origin="mag/raw", name="Compass raw X Y Z"), rrb.TimeSeriesView(origin="imu", name="Gyro (deg/s) and accelerometer (g)")),
                rrb.TimeSeriesView(origin="progress", name="Progress (recording = 1 while a calibration capture is running)"), row_shares=[2, 3.2, 2.2, 1.2])))
        except Exception as e:                                           # an older Rerun without these classes still shows everything in its default layout
            log.warning("ui.blueprint_failed", error=repr(e))

    def _t(self, t):
        if self.enabled:
            self.t0 = t if self.t0 is None else self.t0
            self.rr.set_time("t", duration=float(t - self.t0))

    def _rec(self, kind, *a):
        self.calls.append((kind, *a))

    def instruct(self, sensor, step, total, title, text):
        self._rec("instruct", sensor, step, total, title, text)
        if self.enabled:
            self.rr.log("ui/instructions", self.rr.TextDocument(f"# {sensor}: step {step} of {total}\n## {title}\n\n{text}\n", media_type="text/markdown"))
        print(f"\n=== {sensor}, step {step}/{total}: {title} ===\n{text}", flush=True)

    def status(self, lines):
        self._rec("status", list(lines))
        if self.enabled:
            self.rr.log("ui/status", self.rr.TextDocument("\n".join(f"- {l}" for l in lines), media_type="text/markdown"))

    def scalar(self, path, t, v):
        self._rec("scalar", path)
        if self.enabled:
            self._t(t)
            self.rr.log(path, self.rr.Scalars(float(v)))

    def sample_imu(self, t, g, a):
        if self.enabled:
            self._t(t)
            for k, v in zip("xyz", g):
                self.rr.log(f"imu/gyro_dps/{k}", self.rr.Scalars(float(v)))
            for k, v in zip("xyz", a):
                self.rr.log(f"imu/accel_g/{k}", self.rr.Scalars(float(v)))
            self.rr.log("imu/accel_g/norm", self.rr.Scalars(float(np.linalg.norm(a))))
        self._rec("imu")

    def sample_mag(self, t, m):
        if self.enabled:
            self._t(t)
            for k, v in zip("xyz", m):
                self.rr.log(f"mag/raw/{k}", self.rr.Scalars(float(v)))
            self.rr.log("mag/field_uT", self.rr.Scalars(float(np.linalg.norm(m) * MG_PER_COUNT / 10.0)))
        self._rec("mag")

    def cloud(self, t, raw, corrected=None, centre=None):
        self._rec("cloud", len(raw))
        if not self.enabled or not len(raw):
            return
        self._t(t)
        p = np.asarray(raw, float)
        if len(p) > 4000:
            p = p[np.linspace(0, len(p) - 1, 4000).astype(int)]
        shade = np.linspace(80, 255, len(p)).astype(np.uint8)
        self.rr.log("mag3d/raw_samples", self.rr.Points3D(p, colors=np.stack([shade, 160 * np.ones_like(shade), 255 - shade // 2], 1), radii=2.0))
        if centre is not None:
            self.rr.log("mag3d/centre", self.rr.Points3D([centre], colors=[[255, 80, 80]], radii=8.0))
        if corrected is not None and len(corrected):
            c = np.asarray(corrected, float)
            if len(c) > 4000:
                c = c[np.linspace(0, len(c) - 1, 4000).astype(int)]
            self.rr.log("mag3d/corrected_samples", self.rr.Points3D(c, colors=[[80, 255, 120]], radii=2.0))

    def progress(self, t, **values):
        self._rec("progress", dict(values))
        for k, v in values.items():
            if self.enabled:
                self._t(t)
                self.rr.log(f"progress/{k}", self.rr.Scalars(float(v)))

    def report(self, title, check_list, extra=()):
        self._rec("report", title, list(check_list))
        icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
        lines = [f"**{title}**"] + [f"{icon[s]}  {n}: {txt}" for n, s, txt in check_list] + list(extra)
        if self.enabled:
            self.rr.log("ui/status", self.rr.TextDocument("\n\n".join(lines), media_type="text/markdown"))
        print("\n" + "\n".join(lines), flush=True)

    def recording(self, on, label, t=None):
        """Marks which process is active: a calibration capture (samples are kept) or the live view (samples are only drawn and thrown away)."""
        self._rec("recording", bool(on), label)
        text = (f"# ● RECORDING\n\nCalibration capture running: **{label}**.\n\nSamples are being kept for the calibration." if on else
                f"# ○ LIVE VIEW ONLY\n\n{label}\n\nSamples are drawn but **not kept** for the calibration.")
        if self.enabled:
            self.rr.log("ui/recording", self.rr.TextDocument(text, media_type="text/markdown"))
            if t is not None:
                self._t(t)
            self.rr.log("progress/recording", self.rr.Scalars(1.0 if on else 0.0))
            if on:
                self.rr.log("mag3d/live_samples", self.rr.Clear(recursive=False))
        print(("\n[RECORDING] " if on else "\n[live view only] ") + label, flush=True)

    def live_cloud(self, t, pts):
        """The last few compass samples seen while NOT recording (grey), so the sensor can be checked at any time."""
        self._rec("live_cloud", len(pts))
        if self.enabled and len(pts):
            self._t(t)
            self.rr.log("mag3d/live_samples", self.rr.Points3D(np.asarray(pts, float), colors=[[170, 170, 170]], radii=2.0))

    def model(self, t, q, mag_in_imu=None):
        """Turn the sensor board and the device it sits in (drawn once by model_static) to the estimated orientation q (w,x,y,z; world z up)."""
        self._rec("model", np.asarray(q, float).tolist())
        if not self.enabled:
            return
        rr = self.rr
        self._t(t)
        q = quat_mul(VIEW_FLIP, q)
        rr.log("model/imu", rr.Transform3D(rotation=rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]])))
        if mag_in_imu is not None:
            rr.log("model/imu/compass", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[np.asarray(mag_in_imu, float)], colors=[[255, 200, 0]], labels=["compass"]))

    def model_static(self, R_dev_imu, device_known):
        """Static parts of the 3D model, drawn once. R_dev_imu maps IMU axes to device (camera) axes; the device box uses it, its arrows are named right / down / forward."""
        self._rec("model_static", np.asarray(R_dev_imu, float).tolist(), bool(device_known))
        if not self.enabled:
            return
        rr = self.rr
        rr.log("model/imu/board", rr.Boxes3D(half_sizes=[[0.04, 0.025, 0.004]], colors=[[120, 170, 255]], fill_mode="solid", labels=["BMI160 board"]), static=True)
        for name, vec, col in (("x", [0.09, 0, 0], [255, 60, 60]), ("y", [0, 0.09, 0], [60, 220, 60]), ("z", [0, 0, 0.09], [80, 120, 255])):
            rr.log(f"model/imu/axis_{name}", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[vec], colors=[col], labels=[f"IMU {name}"]), static=True)
        R_imu_dev = np.asarray(R_dev_imu, float).T                       # device vector -> IMU vector
        rr.log("model/imu/device", rr.Transform3D(mat3x3=R_imu_dev), static=True)
        rr.log("model/imu/device/body", rr.Boxes3D(half_sizes=[[0.07, 0.035, 0.02]], colors=[[200, 200, 200]], fill_mode="MajorWireframe",
                                                   labels=["device (camera) " + ("saved axes" if device_known else "reference axes, not measured yet")]), static=True)
        for name, vec, col in (("right", [0.13, 0, 0], [255, 140, 140]), ("down", [0, 0.13, 0], [140, 255, 140]), ("forward", [0, 0, 0.13], [140, 170, 255])):
            rr.log(f"model/imu/device/{name}", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[vec], colors=[col], labels=[name]), static=True)

    def saved(self, text):
        self._rec("saved", text)
        if self.enabled:
            self.rr.log("ui/instructions", self.rr.TextDocument(f"# SAVED\n\n{text}\n", media_type="text/markdown"))
        print("\n" + text, flush=True)


# ---- data source ---------------------------------------------------------------------------------------------------------------------------------------------
class LiveSource:
    """The Pi's IMU and compass samples over the existing link (gabai.pc.receiver)."""

    def __init__(self, port=5600, token="gabai"):
        from gabai.pc.receiver import LiveReceiver
        self.rx = LiveReceiver(port, token)
        self.rx.start()

    def imu(self):
        return self.rx.drain_imu()

    def mag(self):
        return self.rx.drain_mag()

    def hint(self, name):
        pass

    def discard(self):
        """Throw away what the link buffered while nobody was capturing (the wait for Enter)."""
        return len(self.rx.drain_imu()) + len(self.rx.drain_mag())

    def close(self):
        self.rx.stop()


# ---- the guided session --------------------------------------------------------------------------------------------------------------------------------------
MAG_STEPS = [
    ("eights", "Figure-eights over the whole sphere", None,
     "**LIFT the device and draw big, slow figure-eights in the air while turning it, in every orientation**: as if painting the inside of a ball around you. Include **upside down and "
     "onto each side**. Keep going until the coverage reaches 90% (the coverage bar and the missing directions are in the progress panel)."),
]


class Session:
    """live=True (the real run): while the script waits for Enter the sensors are still READ so the Rerun graphs, the compass cloud and the 3D model stay alive, but
    those samples are only drawn and then thrown away; they are never part of a calibration capture. A capture exists only inside collect() and is announced as RECORDING.
    max_step_s=None: a movement step waits as long as it takes (Ctrl+C is the only way out). A number makes it raise StepIncomplete instead of advancing."""

    def __init__(self, source, ui, ask=input, data_dir="data", clock=time.monotonic, sleep=time.sleep, no_data_s=8.0, max_step_s=None, live=False):
        self.source, self.ui, self.data_dir = source, ui, data_dir
        self._ask = ask
        self.live = live
        self.cal_dir = os.path.join(data_dir, "calibration")
        self.clock, self.sleep, self.no_data_s, self.step_timeout_s = clock, sleep, no_data_s, max_step_s
        self.mag_path, self.imu_path = os.path.join(self.cal_dir, calstore.MAG_FILE), os.path.join(self.cal_dir, calstore.IMU_FILE)
        self.u_down = None
        self.recording = None                       # label of the capture that is running, else None
        self.cloud_base = []                        # compass rows kept by earlier steps, drawn together with the running one
        self.att, self._att_t, self._model_t, self._live_pts = None, None, -1.0, []
        self.show_mag = True                   # False in the IMU phase: the compass is not being calibrated, so it is not drawn
        self._live = {"imu": -1.0, "mag": -1.0, "cloud": -1.0, "last": self.clock()}
        self.refresh_model()

    # -- 3D model ------------------------------------------------------------------------------------------------------------------------------------------------
    def refresh_model(self):
        """Draw the device model with the reference axes: camera = IMU X, left/right = IMU Y, up/down = IMU Z. Only the picture uses them; no calibration reads them."""
        self._R_dev_imu, self._dev_known = MODEL_DEV_FROM_IMU, False
        self.ui.model_static(self._R_dev_imu, False)
        self.ui.model(0.0, q_from_accel(np.array([1.0, 0.0, 0.0])))           # until real samples arrive: the device lies face-down, IMU X down
        log.info("cal.model.axes", device_from_imu=np.round(self._R_dev_imu, 2).tolist(), source="reference")

    def _attitude(self, imu, mag=None):
        """Gyro + gravity orientation of the BMI160; the 3D model follows it. Heading has no reference here and can slowly drift: the tilt is what is shown."""
        for r in imu:
            if self.att is None:
                self.att = Madgwick(beta=0.1)
                self.att.q = q_from_accel(r[4:7])
                self._att_t = r[0]
                continue
            dt = r[0] - self._att_t
            self._att_t = r[0]
            if 0.0 < dt < 0.5:
                self.att.update(np.radians(r[1:4]), r[4:7], dt)
        if imu and self.att is not None and imu[-1][0] - self._model_t >= 1 / 30:
            self._model_t = imu[-1][0]
            m = None
            if mag and self.show_mag:
                m = calmag.REF_PERM_IMU_FROM_MAG @ (np.asarray(mag[-1][1:4], float) / 4000.0)       # compass drawn in IMU axes with the reference map
            self.ui.model(imu[-1][0], self.att.q, m)

    # -- live view while nothing is recorded -----------------------------------------------------------------------------------------------------------------
    def _pump_live(self):
        """Read the link, draw it, throw it away. Never touches a capture."""
        ni, nm = self.source.imu(), self.source.mag()
        if ni or nm:
            self._live["last"] = self.clock()
        for s in ni:
            if s[0] - self._live["imu"] >= 0.05:
                self._live["imu"] = s[0]
                self.ui.sample_imu(s[0], s[1:4], s[4:7])
        for s in nm if self.show_mag else ():
            if s[0] - self._live["mag"] >= 0.05:
                self._live["mag"] = s[0]
                self.ui.sample_mag(s[0], s[1:4])
        self._live_pts = (self._live_pts + [r[1:4] for r in nm if self.show_mag])[-400:]
        self._attitude(ni, nm)
        ref = ni or nm
        if ref and ref[-1][0] - self._live["cloud"] >= 0.5 and self._live_pts:
            self._live["cloud"] = ref[-1][0]
            self.ui.live_cloud(ref[-1][0], self._live_pts)
        log.debug("cal.live", imu=len(ni), mag=len(nm))
        if self.clock() - self._live["last"] > self.no_data_s:
            self._live["last"] = self.clock()
            self.ui.status(["No IMU or compass data from the Pi for %.0f s. Is the sender running, with the IMU and compass found? (docs/sensor-calibration.md, troubleshooting)" % self.no_data_s])
            log.warning("cal.live.no_data", seconds=self.no_data_s)
        if not (ni or nm):
            self.sleep(0.01)

    def ask(self, prompt):
        """Prompt the user. With live=True the sensors stay drawn while waiting (marked as live view, not recording). Whatever arrived during the wait is dropped at the end,
        so a capture only ever holds data from after the answer."""
        assert self.recording is None, "a prompt while recording: the capture would never stop"
        if self.live:
            self.ui.recording(False, "Waiting for your answer in the terminal. The sensors are drawn live; nothing is stored.")
            done, box = threading.Event(), {}

            def reader():
                try:
                    box["a"] = self._ask(prompt)
                except BaseException as e:             # also EOF in the reader
                    box["e"] = e
                finally:
                    done.set()

            threading.Thread(target=reader, daemon=True).start()
            while not done.is_set():
                self._pump_live()
            self._drop_stale("prompt")
            if "e" in box:
                raise box["e"]
            return box["a"]
        self.ui.recording(False, "Waiting for your answer in the terminal. Nothing is recorded and nothing is drawn.")
        try:
            return self._ask(prompt)
        finally:
            self._drop_stale("prompt")

    def _drop_stale(self, why):
        n = self.source.discard() if hasattr(self.source, "discard") else 0
        log.debug("cal.rec.discard", why=why, dropped=n)

    # -- stored results ------------------------------------------------------------------------------------------------------------------------------------------
    def stored(self):
        """{'mag': (doc|None, problems), 'imu': (doc|None, problems)} straight from the files, validated."""
        return {"mag": calstore.load(self.mag_path, calstore.validate_mag), "imu": calstore.load(self.imu_path, calstore.validate_imu)}

    def status_lines(self):
        out = []
        for key, name in (("mag", "HMC5883L compass"), ("imu", "BMI160 IMU")):
            doc, problems = self.stored()[key]
            out.append(f"{name}: " + (f"calibrated {doc['date']}" if doc else ("INVALID file ignored: " + "; ".join(problems) if problems else "not calibrated yet")))
        return out

    # -- capture -------------------------------------------------------------------------------------------------------------------------------------------------
    def collect(self, hint, seconds=None, until=None, tick=None):
        """One calibration capture. Reads the link until `seconds` of SAMPLE time passed, or until() is true (a requirement, so it waits as long as it takes; with a step
        timeout set it raises StepIncomplete instead of returning). Returns (imu rows (t,gx,gy,gz,ax,ay,az), mag rows (t,x,y,z)).
        The capture starts with a RECORDING marker and always ends with a stop marker, whatever way it ends; nothing is read from the link after it returns."""
        if self.recording is not None:
            raise RuntimeError(f"capture '{hint}' started while '{self.recording}' is still recording")
        if hasattr(self.source, "hint"):
            self.source.hint(hint)
        self._drop_stale("start of " + hint)
        self.recording = hint
        self.ui.recording(True, hint)
        log.info("cal.rec.start", capture=hint, base_compass_rows=len(self.cloud_base))
        imu, mag = [], []
        last = self.clock()
        shown = {"imu": -1.0, "mag": -1.0, "cloud": -1.0}
        outcome = "ended"
        try:
            while True:
                ni, nm = self.source.imu(), self.source.mag()
                if ni or nm:
                    last = self.clock()
                imu += ni
                mag += nm
                for s in ni:                                                 # decimated for Rerun: about 20 IMU and 20 compass points a second
                    if s[0] - shown["imu"] >= 0.05:
                        shown["imu"] = s[0]
                        self.ui.sample_imu(s[0], s[1:4], s[4:7])
                for s in nm if self.show_mag else ():
                    if s[0] - shown["mag"] >= 0.05:
                        shown["mag"] = s[0]
                        self.ui.sample_mag(s[0], s[1:4])
                self._attitude(ni, nm)
                ref = imu or mag
                elapsed = (ref[-1][0] - ref[0][0]) if ref else 0.0
                if mag and self.show_mag and ref[-1][0] - shown["cloud"] >= 0.5:               # the compass cloud grows from the first reading on, in every step
                    shown["cloud"] = ref[-1][0]
                    self.ui.cloud(ref[-1][0], np.array([r[1:4] for r in self.cloud_base + mag], float))
                if tick:
                    tick(imu, mag, elapsed)
                if (seconds is not None and elapsed >= seconds) or (until is not None and until(imu, mag)):
                    outcome = "done"
                    return imu, mag
                if self.step_timeout_s is not None and until is not None and elapsed >= self.step_timeout_s:
                    outcome = "timeout"
                    raise StepIncomplete(f"'{hint}': the requirement was not met after {elapsed:.0f} s of samples")
                if self.clock() - last > self.no_data_s:
                    outcome = "no_data"
                    raise NoData("no IMU or compass data for %.0f s: is the Pi sender running with IMU and compass found? (see the troubleshooting section of docs/sensor-calibration.md)" % self.no_data_s)
                if not (ni or nm):
                    self.sleep(0.01)
        except BaseException as e:
            if outcome == "ended":
                outcome = type(e).__name__
            raise
        finally:
            self.recording = None
            self.ui.recording(False, f"Capture '{hint}' stopped ({outcome}): {len(imu)} IMU and {len(mag)} compass samples kept. Nothing more is recorded until the next capture.")
            log.info("cal.rec.stop", capture=hint, outcome=outcome, imu=len(imu), compass=len(mag))
            self._drop_stale("end of " + hint)

    def rest(self, hint, seconds, label):
        """Still capture; returns (imu array, mag array, problems) and refuses a capture where the device moved."""
        while True:
            imu, mag = self.collect(hint, seconds=seconds, tick=lambda i, m, el: self.ui.progress(i[-1][0] if i else 0, **{label + "_done": min(1.0, el / seconds)}))
            A = np.array(imu, float)
            if A[:, 1:4].std(axis=0).max() < 3.0 and np.linalg.norm(A[:, 4:7].std(axis=0)) < 0.03 and np.linalg.norm(A[:, 1:4].mean(axis=0)) < 3.0:
                return A, np.array(mag, float)
            self.ui.status(["The device moved during the still capture. Put it down and hold it STILL; capturing again."])
            self.ask("It moved. Hold it still, then press Enter to capture again: ")

    # -- phase 1: compass ----------------------------------------------------------------------------------------------------------------------------------------
    def run_mag(self):
        self.show_mag = True
        n_steps = 1 + len(MAG_STEPS)
        rows, ups_t, ups = [], [], []                                          # compass rows (t, x, y, z); gravity record (t, ax, ay, az)
        part = os.path.join(self.cal_dir, "mag_partial.npz")
        if os.path.exists(part) and self.ask("An unfinished compass capture was kept from last time. Continue with it? [Y/n] ").strip().lower() in ("", "y", "yes"):
            with np.load(part) as z:
                rows, ups_t, ups = [list(r) for r in z["mag"]], list(z["acc_t"]), [list(r) for r in z["acc"]]
        xyz = lambda r: np.array([x[1:4] for x in r], float).reshape(-1, 3)
        self.ui.status(self.status_lines() + ["Now calibrating: HMC5883L compass (the IMU is only read as a motion and gravity reference, it is not calibrated now)."])
        try:
            self.ui.instruct("HMC5883L compass", 1, n_steps, "Rest face-down on the table", "Put the device on the table **face-down** (camera side down) and take your hands off it. "
                             "Keep everything that normally sits next to it (Pi, battery, cables) in place and keep phones, laptops and magnets away.")
            self.ask("Press Enter when it is face-down and still: ")
            A, M = self.rest("rest_facedown", 4.0, "rest")
            self.u_down = calimu_up(A)
            ax = int(np.argmax(np.abs(self.u_down)))
            ok_face = ax == 0
            self.ui.status([f"face-down check: gravity is on IMU axis {'XYZ'[ax]} ({self.u_down[ax]:+.2f} g); expected the front/back axis X: " + ("OK" if ok_face else
                            "NOT as expected: the sensor mounting differs from the reference mapping, or it is not face-down")])
            log.info("cal.mag.rest", gravity_axis="XYZ"[ax], u_down=np.round(self.u_down, 3).tolist(), as_expected=ok_face)
            for i, (key, title, axis, text) in enumerate(MAG_STEPS, start=2):
                self.ui.instruct("HMC5883L compass", i, n_steps, title, text)
                self.ask("Press Enter, then do the movement: ")
                self.cloud_base = [list(r) for r in rows]
                st = {"cov": 0.0, "last": -1e9, "deg": 0.0, "ok": False}

                def tick(imu, mag, el, key=key, axis=axis, title=title, st=st):
                    t = imu[-1][0] if imu else (mag[-1][0] if mag else 0.0)
                    if t - st["last"] < 0.25:                                  # the requirement and the panels are re-evaluated four times a second of sample time
                        return
                    st["last"] = t
                    pts = xyz(rows + mag)
                    cov, missing = calmag.direction_coverage(pts) if len(pts) > 50 else (0.0, [])
                    st["cov"] = cov
                    ok, need, st["deg"] = self.step_requirements(key, axis, imu, mag, el, cov, missing, len(pts), spread=float(pts.std(axis=0).max()) if len(pts) > 1 else 0.0)
                    st["ok"] = ok
                    self.ui.progress(t, coverage=cov, **({"movement_done": min(1.0, st["deg"] / GYRO_TURN_DONE_DEG)} if axis is not None else {}))
                    self.ui.status([f"RECORDING step: {title}", f"compass readings {len(pts)} ({len(mag)} in this step), coverage {cov * 100:.0f}% of 26 directions" +
                                    (f", turned {st['deg']:.0f} of {GYRO_TURN_DONE_DEG:.0f} deg about {AXIS_NAMES[axis]}" if axis is not None else ""),
                                    ("all requirements met: finishing this step" if ok else "STILL NEEDED: " + "; ".join(need))])
                    log.debug("cal.mag.progress", step=key, ok=ok, deg=round(st["deg"], 1), coverage=round(cov, 3), step_mag=len(mag), seconds=round(el, 1), still_needed=need)

                imu, mag = self.collect("mag_" + key, until=lambda imu, mag, st=st: st["ok"], tick=tick)
                rows += [list(r) for r in mag]
                ups_t += [r[0] for r in imu]
                ups += [list(r[4:7]) for r in imu]
                log.info("cal.mag.step_done", step=key, turned_deg=round(st["deg"], 1), coverage=round(st["cov"], 3), step_mag=len(mag), total_mag=len(rows))
                if axis is not None and imu:
                    dom, share = calimu.dominant_axis([r[0] for r in imu], [r[1:4] for r in imu])
                    if dom != axis:
                        self.ui.status([f"WARNING: you turned mainly about IMU axis {'XYZ'[dom]} ({share * 100:.0f}% of the rotation); the {title.lower()} movement is expected about axis {'XYZ'[axis]}. "
                                        "Either the movement was different, or the sensor axes are not as in the reference mapping."])
                    log.info("cal.mag.step", step=key, dominant_axis="XYZ"[dom], share=round(share, 2), samples=len(mag))
                self._save_partial(part, rows, ups_t, ups)
        except StepIncomplete as e:
            self._save_partial(part, rows, ups_t, ups)
            self.ui.status([f"Step not completed ({e}). Nothing advanced and nothing was saved; the finished steps are kept and offered next time."])
            log.warning("cal.mag.step_incomplete", error=str(e))
            return None
        except KeyboardInterrupt:
            self._save_partial(part, rows, ups_t, ups)
            self.ui.status(["Stopped. Completed results are kept; the capture so far is kept and offered next time."])
            raise
        return self._finish_mag(np.array(rows, float).reshape(-1, 4), np.array(ups_t, float), np.array(ups, float).reshape(-1, 3), part)

    def step_requirements(self, key, axis, imu, mag, elapsed, coverage, missing, n_points, spread=1e9):
        """(met, what is still needed, degrees swept). A step finishes only when ALL of these hold; there is no timer that ends it.
        movement steps: the running angle about the step's axis spans >= GYRO_TURN_DONE_DEG deg (extent, so wobble and noise do not count), >= MIN_STEP_MAG compass readings in
        the step, >= MIN_STEP_S of samples. figure-eights: 90% of the 26 directions covered, >= 400 compass readings, >= MIN_STEP_S."""
        need, deg = [], 0.0
        if axis is not None:
            if len(imu) > 2:
                deg = calimu.sweep([r[0] for r in imu], [r[1:4] for r in imu], axis)[1]
                dom, share = calimu.dominant_axis([r[0] for r in imu], [r[1:4] for r in imu])
                if dom != axis and sum(calimu.angle_about([r[0] for r in imu], [r[1:4] for r in imu], a)[1] for a in range(3)) > 90:
                    need.append(f"you are turning mainly about {AXIS_NAMES[dom]}; turn about {AXIS_NAMES[axis]}")
            if deg < GYRO_TURN_DONE_DEG:
                need.append(f"turn {GYRO_TURN_DONE_DEG - deg:.0f} more degrees about {AXIS_NAMES[axis]} (slow and smooth)")
        else:
            if coverage < COVERAGE_GOAL:
                need.append(f"cover more directions ({coverage * 100:.0f}% of {COVERAGE_GOAL * 100:.0f}%): missing " + (" ".join(missing[:8]) if missing else "?"))
            if n_points < 400:
                need.append(f"{400 - n_points} more compass readings")
            if spread < MIN_FIELD_SPREAD:
                need.append(f"the compass hardly changes ({spread:.0f} counts): the device is not being turned, lift it and draw the figure-eights")
        if axis is not None and len(mag) < MIN_STEP_MAG:
            need.append(f"{MIN_STEP_MAG - len(mag)} more compass readings in this step (keep moving slowly)")
        if elapsed < MIN_STEP_S:
            need.append(f"at least {MIN_STEP_S - elapsed:.0f} s more (too fast to give enough readings)")
        return (not need), need, deg

    def _save_partial(self, path, rows, ups_t, ups):
        os.makedirs(self.cal_dir, exist_ok=True)
        np.savez_compressed(path, mag=np.array(rows, float).reshape(-1, 4), acc_t=np.array(ups_t, float), acc=np.array(ups, float).reshape(-1, 3))

    def _finish_mag(self, rows, acc_t, acc, part):
        if len(rows) < 200 or len(acc_t) < 20:
            self.ui.status([f"Only {len(rows)} compass readings: too few to calibrate."])
            return None
        t_mag, pts = rows[:, 0], rows[:, 1:4]
        cal = calmag.fit(pts)
        corrected = apply_cal(cal, pts)
        up = calmag.pair_up(t_mag, acc_t, acc)
        P, score, nxt, ok_map, mean_ang = calmag.estimate_axis_map(corrected, up)
        cl = calmag.checks(cal, pts, up=up, P=P)
        cl.append(("compass-to-IMU axis map", "pass" if ok_map else "fail",
                   f"measured {P.astype(int).tolist()} (score {score:.1f}, next best {nxt:.1f})" + ("" if ok_map else ": not clearly better than the alternatives, turn through more orientations")))
        matches = calmag.same_permutation(P, calmag.REF_PERM_IMU_FROM_MAG)
        cl.append(("axis map matches the reference", "pass" if matches else "warn",
                   "compass X<->IMU Y, compass Y<->IMU X, compass Z<->IMU Z, as the reference mounting implies (IMU X front/back, Y left/right, Z up/down) (signs measured: " + str(P.astype(int).tolist()) + ")" if matches else
                   f"the permutation differs from the reference (compass X/Y/Z to IMU Y/X/Z): measured {P.astype(int).tolist()}. The mounting is not what you described: the measured map is what will be saved"))
        frames = load_frames(self.data_dir + "/imu_sync.json", self.data_dir + "/mag_sync.json", self.data_dir + "/device_frame.json")
        R_dev_mag = None
        if not frames.sources["imu_to_device"].startswith("none"):
            R_dev_mag = frames.R_dev_imu @ P
            dev_ok = calmag.same_permutation(R_dev_mag, calmag.REF_PERM_DEV_FROM_MAG)
            cl.append(("compass axes in the device frame", "pass" if dev_ok else "warn", f"device <- compass:\n{np.round(R_dev_mag, 2).tolist()}  " +
                       ("(compass X = device x left/right, Y = device z front/back, Z = device y up/down, signs as shown)" if dev_ok else "(does not follow the reference: X, Y, Z are not left/right, front/back, up/down)")))
        else:
            cl.append(("compass axes in the device frame", "warn", "unknown until the IMU is calibrated (phase 2 measures the device axes from gravity)"))
        self.ui.cloud(0.0, pts, corrected=corrected, centre=cal["offset"])
        self.ui.report("HMC5883L calibration result", cl, [f"offset (counts): {np.round(cal['offset'], 1).tolist()}", f"method: {cal['method']}"])
        log.info("cal.mag.result", checks={n: s for n, s, _ in cl}, offset=np.round(cal["offset"], 1).tolist(), quality=round(cal["quality"], 4), method=cal["method"])
        if not calmag.can_save(cl):
            self.ui.status(["The result has FAIL checks: it is not offered for saving. Redo the movements (run the script again, answer n to resuming) and cover more directions."])
            return None
        if self.ask("Save this compass calibration? [y/N] ").strip().lower() not in ("y", "yes"):
            self.ui.status(["Not saved. The capture is kept so you can run again and continue it."])
            return None
        doc = calstore.mag_document(cal["offset"], cal["W"], cal["radius_counts"], cal["quality"], cal["coverage26"], cal["method"], P,
                                    {"labels": calmag.REFERENCE_TEXT, "permutation_device_from_mag": calmag.REF_PERM_DEV_FROM_MAG.tolist(), "permutation_imu_from_mag": calmag.REF_PERM_IMU_FROM_MAG.tolist()},
                                    matches, [t for _, s, t in cl if s != "pass"], {n: {"status": s, "text": t} for n, s, t in cl}, device_from_mag=R_dev_mag)
        calstore.save(self.mag_path, doc, calstore.validate_mag)
        calstore.write_legacy_mag(doc, self.data_dir)
        if os.path.exists(part):
            os.remove(part)
        self.ui.saved(f"Compass calibration SAVED at {doc['date']}: {self.mag_path}, and for the running system {self.data_dir}/mag_cal.json and mag_sync.json.")
        return doc

    # -- phase 2: IMU --------------------------------------------------------------------------------------------------------------------------------------------
    def run_imu(self):
        self.show_mag = False
        mag_doc, _ = self.stored()["mag"]
        if mag_doc is None:
            self.ui.status(["The compass calibration must be completed and saved before the IMU phase."])
            return None
        total = 4
        part = os.path.join(self.cal_dir, "imu_partial.npz")
        holds = []
        if os.path.exists(part) and self.ask("Unfinished still holds were kept from last time. Continue with them? [Y/n] ").strip().lower() in ("", "y", "yes"):
            with np.load(part) as z:
                holds = [list(r) for r in z["holds"]]
        self.ui.status(self.status_lines() + ["Now calibrating: BMI160 IMU. The compass is not used."])
        try:
            self.ui.instruct("BMI160 IMU", 1, total, "Rest face-down on the table", "Put the device **face-down** on the table, hands off, and keep it **perfectly still for 10 s**. "
                             "This measures the gyro's noise and what is left of its bias after the Pi's own correction, and which way gravity points.")
            self.ask("Press Enter when it is face-down and still: ")
            A, _ = self.rest("imu_rest_facedown", 10.0, "rest")
            gs, ac = calimu.gyro_stats(A[:, 0], A[:, 1:4]), calimu.accel_stats(A[:, 4:7])
            self.u_down = calimu_up(A)
            warns = calimu.warnings_for(gs, ac)
            self.ui.status([f"gyro noise {np.round(gs['noise_dps'], 3).tolist()} deg/s, residual {np.round(gs['residual_bias_dps'], 3).tolist()} deg/s", f"accelerometer |a| = {ac['norm_mean_g']:.3f} g at rest"] + warns)
            log.info("cal.imu.rest", gyro=gs, accel=ac, warnings=warns)

            self.ui.instruct("BMI160 IMU", 2, total, "Hold it still on all six sides",
                             "**LIFT the device and hold it dead still, 3 s each, with a DIFFERENT side facing down each time**: camera side, back, top, bottom, left side, right side. "
                             "The script captures each hold by itself when it stays still; the panel shows which of the six directions are still missing. Start with the camera side UP (turn it over).")
            self.ask("Press Enter, then start holding it in turn: ")
            cap = StillCapture(min_s=3.0, gyro_max_dps=STILL_GYRO_DPS, std_max=STILL_SHAKE_MS2)
            seed = [np.array(h) * G for h in holds] + [A[:, 4:7].mean(axis=0) * G]            # the face-down rest is the first of the six holds
            means = list(cluster_orientations(np.vstack(seed)))

            fed = {"n": 0}
            last = {"msg": ""}

            def tick(imu, mag, el):
                nonlocal means
                new, fed["n"] = imu[fed["n"]:], len(imu)                       # each sample goes to the hold detector exactly once
                for r in new:
                    m = cap.feed(r[0], np.array(r[4:7]) * G, np.radians(r[1:4]))
                    if m is not None:
                        before = len(means)
                        means = list(cluster_orientations(np.vstack(means + [m]))) if means else [m]
                        self._save_partial_holds(part, means)
                        last["msg"] = (f"LAST HOLD: new direction saved ({len(means)} so far), reading {np.round(m / G, 3).tolist()} g" if len(means) > before else
                                       f"LAST HOLD: captured, but it points the same way as an earlier hold (within 15 deg), so nothing new. Turn it to a side NOT marked ✓. Reading {np.round(m / G, 3).tolist()} g")
                        log.info("cal.imu.hold", new=len(means) > before, holds=len(means), reading_g=np.round(m / G, 3).tolist())
                        self.ui.status([last["msg"]])
                cov = faces_covered(np.array(means)) if means else {k: False for k in FACES}
                need = [k for k, v in cov.items() if not v]
                self.ui.progress(imu[-1][0] if imu else 0, faces_covered=(6 - len(need)) / 6.0)
                if len(imu) % 50 == 1 and len(imu) > 50:
                    w = np.array([r[1:7] for r in imu[-50:]], float)               # the last ~0.3 s: what the hold detector is judging right now
                    gy, sh, up = np.linalg.norm(w[:, :3], axis=1).max(), w[:, 3:].std(0).max() * G, w[:, 3:].mean(0)
                    still = gy < STILL_GYRO_DPS and sh < STILL_SHAKE_MS2
                    self.ui.status([last["msg"], f"faces covered: " + " ".join(f"{k}{'✓' if v else '·'}" for k, v in cov.items()), "still needed: " + (" ".join(need) or "none") +
                                    "  (a face is 'covered' when that IMU axis pointed at the ground)",
                                    f"now: gravity {np.round(up, 2).tolist()} g, turning {gy:.1f} deg/s (limit {STILL_GYRO_DPS:g}), shaking {sh:.2f} m/s2 (limit {STILL_SHAKE_MS2:g}): " +
                                    ("still enough, keep holding for 3 s" if still else "TOO MOVING to count, hold it steadier (rest it against something)")])
                    log.debug("cal.imu.hold_state", gyro_dps=round(float(gy), 2), shake_ms2=round(float(sh), 3), still=bool(still), holds=len(means), faces=sum(cov.values()))
            self.collect("imu_holds", until=lambda imu, mag: len(means) >= calimu.MIN_ORIENTATIONS and all(faces_covered(np.array(means)).values()), tick=tick)
            holds = [m / G for m in means]

            self.ui.instruct("BMI160 IMU", 3, total, "Hold it upright, as worn on the chest",
                             "Hold the device **upright the way it hangs on your chest** (camera facing forward, top up), left-right level, and keep it **still for 4 s**. "
                             "This and the face-down rest fix the device axes from gravity.")
            self.ask("Press Enter when it is upright and still: ")
            Au, _ = self.rest("imu_upright", 4.0, "upright")
            u_up = calimu_up(Au)

            self.ui.instruct("BMI160 IMU", 4, total, "Gyro direction check",
                             "Hold the device in front of you with the **camera side facing YOU** (like a mirror). Turn it **clockwise, one full slow turn (about 10 s), about the line between the camera and your face**.")
            self.ask("Press Enter, then turn it clockwise: ")
            cam = int(np.argmax(np.abs(self.u_down)))                          # the camera axis: where gravity was when it lay face-down

            def swept(imu):
                return calimu.sweep([r[0] for r in imu], [r[1:4] for r in imu], cam)[1] if len(imu) > 2 else 0.0

            def gyro_tick(imu, mag, el):
                deg = swept(imu)
                self.ui.progress(imu[-1][0] if imu else 0, movement_done=min(1.0, deg / GYRO_TURN_DONE_DEG))
                self.ui.status(["RECORDING step: gyro direction check", f"turned {deg:.0f} of {GYRO_TURN_DONE_DEG:.0f} deg about {AXIS_NAMES[cam]}" +
                                ("" if deg >= GYRO_TURN_DONE_DEG and el >= MIN_STEP_S else f"; STILL NEEDED: {max(0.0, GYRO_TURN_DONE_DEG - deg):.0f} more degrees, slowly")])

            imu_t, _ = self.collect("imu_gyro_clockwise", until=lambda imu, mag: swept(imu) >= GYRO_TURN_DONE_DEG and (imu[-1][0] - imu[0][0]) >= MIN_STEP_S, tick=gyro_tick)
            G_t, G_g = [r[0] for r in imu_t], [r[1:4] for r in imu_t]
            dom, share = calimu.dominant_axis(G_t, G_g)
            net = calimu.angle_about(G_t, G_g, dom)[0]
            return self._finish_imu(gs, ac, warns, holds, self.u_down, u_up, dom, share, net, part)
        except StepIncomplete as e:
            self.ui.status([f"Step not completed ({e}). Nothing advanced and nothing was saved; the holds so far are kept and offered next time."])
            log.warning("cal.imu.step_incomplete", error=str(e))
            return None
        except KeyboardInterrupt:
            self.ui.status(["Stopped. Completed results are kept; the holds so far are kept and offered next time."])
            raise

    def _save_partial_holds(self, path, means):
        os.makedirs(self.cal_dir, exist_ok=True)
        np.savez_compressed(path, holds=np.array(means, float).reshape(-1, 3) / G)

    def _finish_imu(self, gs, ac, warns, holds, u_down, u_up, gyro_axis, gyro_share, gyro_net, part):
        cl = []
        for w in warns:
            cl.append(("sensor stability", "warn", w))
        if not warns:
            cl.append(("sensor stability", "pass", f"gyro noise {max(gs['noise_dps']):.3f} deg/s, residual {np.linalg.norm(gs['residual_bias_dps']):.3f} deg/s, accelerometer {ac['norm_mean_g']:.3f} g, all normal"))
        accel = calimu.fit_accel(holds) if len(holds) else {"calibrated": False, "reason": "no holds captured", "n_orientations": 0, "faces_missing": []}
        cl.append(("accelerometer bias and scale", "pass" if accel["calibrated"] else "warn",
                   (f"bias {np.round(accel['bias_ms2'], 3).tolist()} m/s^2, scale {np.round(accel['scale'], 4).tolist()}; gravity error {accel['rms_before_ms2']:.3f} to {accel['rms_after_ms2']:.3f} m/s^2 rms"
                    if accel["calibrated"] else "NOT calibrated: " + accel["reason"])))
        ax_face = int(np.argmax(np.abs(u_down)))
        cl.append(("face-down gravity axis", "pass" if ax_face == 0 else "warn", f"gravity on IMU axis {'XYZ'[ax_face]}; expected the front/back axis X"))
        R = None
        try:
            R = calimu.device_from_imu(u_down, u_up)
            matches = calmag.same_permutation(R, calmag.REF_PERM_DEV_FROM_IMU)
            name, det = classify_mount(R @ (u_up / np.linalg.norm(u_up)))
            cl.append(("device axes from gravity", "pass" if matches else "warn", f"device <- IMU:\n{np.round(R, 2).tolist()}  " +
                       ("(IMU X = device z front/back, IMU Y = device x left/right, IMU Z = device y up/down, as in the reference)" if matches else "(the permutation differs from the reference mapping)") +
                       f"; the upright pose reads as {name}, camera {det['pitch_down_deg']:+.1f} deg below the horizon"))
        except ValueError as e:
            matches = False
            cl.append(("device axes from gravity", "fail", str(e)))
        if R is not None:
            exp_sign = float(np.sign((R @ np.eye(3)[gyro_axis])[2]))        # the device-z part of the turned IMU axis; clockwise seen from the camera side is negative about device z
            ok_dir = abs(R[2, gyro_axis]) > 0.7
            sense_ok = ok_dir and (gyro_net * exp_sign) < 0
            cl.append(("gyro direction", "pass" if sense_ok else "warn", f"turned mainly about IMU axis {'XYZ'[gyro_axis]} ({gyro_share * 100:.0f}%), net {gyro_net:+.0f} deg; " +
                       ("a clockwise turn seen from the camera side reads negative about the device's forward axis, as the mapping predicts" if sense_ok else "does not match the mapping's prediction: gyro axes or the direction of the turn differ from what the mapping implies")))
        self.ui.report("BMI160 calibration result", cl)
        log.info("cal.imu.result", checks={n: s for n, s, _ in cl}, accel=accel)
        if any(s == "fail" for _, s, _ in cl):
            self.ui.status(["The result has FAIL checks: it is not offered for saving."])
            return None
        if self.ask("Save this IMU calibration? [y/N] ").strip().lower() not in ("y", "yes"):
            self.ui.status(["Not saved."])
            return None
        doc = calstore.imu_document(dict(gs, residual_after_pi_bias_note="the Pi subtracts its own start-up bias; this is what remained"), accel, R,
                                    {"labels": calmag.REFERENCE_TEXT, "permutation_device_from_imu": calmag.REF_PERM_DEV_FROM_IMU.tolist()}, matches,
                                    [t for _, s, t in cl if s != "pass"], {n: {"status": s, "text": t} for n, s, t in cl})
        calstore.save(self.imu_path, doc, calstore.validate_imu)
        calstore.write_legacy_imu(doc, self.data_dir)
        self.refresh_model()
        if os.path.exists(part):
            os.remove(part)
        self.ui.saved(f"IMU calibration SAVED at {doc['date']}: {self.imu_path}" + (f", and {self.data_dir}/accel_calib.json" if accel["calibrated"] else " (no accelerometer calibration: the holds did not support one)") +
                      (f", {self.data_dir}/device_frame.json" if R is not None else ""))
        return doc


def calimu_up(A):
    """Mean accelerometer direction (unit 'up') of a still capture (rows t,gx,gy,gz,ax,ay,az)."""
    a = np.asarray(A, float)[:, 4:7].mean(axis=0)
    return a / np.linalg.norm(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--phase", choices=["mag", "imu"], help="run only this phase (the IMU phase needs a saved compass calibration)")
    ap.add_argument("--redo-mag", action="store_true")
    ap.add_argument("--redo-imu", action="store_true")
    ap.add_argument("--no-viewer", action="store_true", help="do not open the Rerun window (with --save-rrd to record it)")
    ap.add_argument("--save-rrd", help="also record the Rerun session to this file")
    ap.add_argument("--live-view", action="store_true", help="also draw the sensors in Rerun while waiting for Enter (Rerun stores whatever is drawn; off by default)")
    ap.add_argument("--log-level", default=None)
    a = ap.parse_args()
    run = dlog.configure(run_name="calibrate", level=a.log_level)
    print(f"diagnostic log: {run}")
    ui = UI(enabled=True, spawn=not a.no_viewer, save=a.save_rrd)
    try:
        src = LiveSource(a.port, a.token)
    except Exception as e:
        raise SystemExit(f"cannot open the link: {e}")
    s = Session(src, ui, data_dir=a.data_dir, live=a.live_view)
    print("\n".join(s.status_lines()))
    ui.status(s.status_lines() + ["Waiting for the Pi ... place the device face-down on the table."])
    try:
        stored = s.stored()
        if a.phase != "imu" and (a.redo_mag or stored["mag"][0] is None):
            s.run_mag()
        elif a.phase != "imu":
            print("compass already calibrated and valid: skipped (use --redo-mag to repeat it)")
        stored = s.stored()
        if a.phase == "mag":
            return
        if stored["mag"][0] is None:
            print("\nThe compass calibration is not saved, so the IMU phase is not started. Run again to finish it.")
            return
        if a.redo_imu or stored["imu"][0] is None:
            if a.phase == "imu" or s.ask("\nCompass done. Continue with the BMI160 IMU calibration now? [y/N] ").strip().lower() in ("y", "yes"):
                s.run_imu()
            else:
                print("Stopped before the IMU phase; the compass result is saved. Run again (or --phase imu) when you are ready.")
        else:
            print("IMU already calibrated and valid: skipped (use --redo-imu to repeat it)")
    except NoData as e:
        print("\nERROR:", e)
        log.error("cal.no_data", error=str(e))
    except KeyboardInterrupt:
        print("\nStopped. Completed results are saved; unfinished captures are kept.")
    finally:
        src.close()
        dlog.shutdown()


if __name__ == "__main__":
    main()
