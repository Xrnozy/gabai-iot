"""The calibration lab's controller: one recording, one working calibration, one RTAB-Map run at a time, a preview of the current picture, the auto-calibration, saved files.

Three things are kept apart on purpose:
  recording    data/recordings/<folder>, never written to (its files are fingerprinted at load and checked after every run);
  calibration  `manual` = the working copy the sliders edit, `auto` = what the automatic check proposed, `base` = the file the session started from; saved as JSON in data/calib_lab/configs;
  results      one folder per finished run in data/calib_lab/experiments: the calibration used, the clip, the metrics, the TSDF mesh, the measurements.

A change to the calibration updates the PREVIEW at once (rectified pair, disparity, row alignment of the picture on screen). RTAB-Map cannot take new camera numbers into a map it has already
built (f, cx, cy and the baseline enter the odometry and every depth in the TSDF), so the run is stopped and a new one starts from the clip start with a new bridge and an empty TSDF: at once
while playing (after 0.8 s without further edits), or at the next Play / Restart when paused. Until then the page says the map is STALE and shows which numbers it was built with.

Browser protocol: every message is   u32 length | JSON header | payload   (see send()); images are JPEG, meshes are float32 vertices | uint8 colours | uint32 triangles.
"""
import csv
import json
import os
import struct
import threading
import time

import cv2
import numpy as np

from rtabmap_lab.calib_lab import params as P
from rtabmap_lab.calib_lab.autocalib import AutoCalibration, epipolar_stats, explain_stats
from rtabmap_lab.calib_lab.recording import Recording
from rtabmap_lab.calib_lab.runner import Run, default_settings

SMEAR_OK_PX = 0.25                      # a pair turning faster than this (f * omega * dt) is not used to judge the calibration
DEBOUNCE_S = 0.8


def jpeg(img, q=75):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def disp_color(d, vmax=96.0):
    g = np.nan_to_num(np.clip(d / vmax, 0, 1) * 255, nan=0).astype(np.uint8)
    c = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
    c[~np.isfinite(d) | (d <= 0)] = (45, 45, 45)
    return c


class Lab:
    def __init__(self, folder, calib, root="data/calib_lab", clip=(0, 300), settings=None):
        self.root = root
        for sub in ("configs", "experiments", "measurements"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        self.base = P.CalibParams.from_npz(calib)
        self.manual, self.auto = self.base.copy(), None
        self.rec = Recording(folder).decide(self.base.rect())
        self.settings = dict(settings or default_settings())
        self.clip = (max(0, int(clip[0])), min(int(clip[1]), self.rec.n))
        self.cursor, self.run, self.ref = self.clip[0], None, None
        self.auto_restart, self.loop, self.batch_state = True, False, None
        self.sink = lambda data: None
        self.cache, self._cache_lock = {}, threading.Lock()
        self._ctl, self._timer, self._port = threading.RLock(), None, 0
        self._req, self._req_cv, self._alive = None, threading.Condition(), True
        self.quality, self.last_disp, self.log = {}, None, []
        self._last_heavy, self._ref_cache = 0.0, {}
        from rtabmap_lab.sgbm_depth import SgbmDepth
        self.sgbm = SgbmDepth(block=9, wls=False, max_std=0)
        self.last_experiment = None
        threading.Thread(target=self._preview_loop, daemon=True).start()
        self.request_preview()

    # -- output to the page ------------------------------------------------------------------------------------------------------------------------------------------------------------
    def send(self, header, payload=b"", keep=None):
        h = json.dumps(header, default=float).encode()
        data = struct.pack("<I", len(h)) + h + payload
        if keep:
            with self._cache_lock:
                self.cache[keep] = data
        self.sink(data)

    def note(self, text, level="info"):
        self.log.append((time.strftime("%H:%M:%S"), level, text))
        self.send({"t": "log", "time": self.log[-1][0], "level": level, "text": text})

    def snapshot(self):
        with self._cache_lock:
            return list(self.cache.values())

    def push_state(self):
        run, m = self.run, self.manual
        d = m.derived()
        state = {
            "t": "state",
            "recording": self.rec.info(), "clip": list(self.clip), "cursor": self.cursor, "auto_restart": self.auto_restart, "loop": self.loop,
            "params": {n: {"value": m.values[n], "original": m.original[n], "base": self.base.values[n], "changed": abs(m.values[n] - m.original[n]) > 1e-9} for n in P.NAMES},
            "derived": d, "changed": m.changed(), "fingerprint": m.fingerprint(),
            "run": None if run is None else {"state": run.state, "pos": run.pos, "clip": list(run.clip), "fingerprint": run.fingerprint, "error": run.error,
                                             "changed_since": [n for n in P.NAMES if abs(run.params.values[n] - m.values[n]) > 1e-9], "metrics": run.metrics},
            "stale": run is not None and run.fingerprint != m.fingerprint(),
            "reference": None if self.ref is None else {"name": self.ref["name"], "fingerprint": self.ref["params"].fingerprint(), "metrics": self.ref.get("metrics"), "changed": self.ref["params"].changed(self.base)},
            "auto": self._auto_summary(), "configs": self._lists()[0], "experiments": self._lists()[1], "batch": self.batch_state,
            "recording_unchanged": self.rec.unchanged(), "quality": self.quality,
        }
        self.send(state, keep=("state",))

    def _lists(self):
        """(configs, experiments), read from disk only after something was saved: the page state is pushed many times a second."""
        if getattr(self, "_lists_cache", None) is None:
            self._lists_cache = (self.list_configs(), self.list_experiments())
        return self._lists_cache

    def _auto_summary(self):
        a = self.auto
        if a is None:
            return None
        return {k: a.get(k) for k in ("verdict", "frames_asked", "frames_used", "proposed", "estimates", "std", "checks", "unobservable", "diagnostic", "skipped_count", "skipped_reasons", "row_err_current_px",
                                      "row_err_proposed_px", "note", "progress", "running")}

    @staticmethod
    def doc():
        """All parameter explanations for the page (sent once)."""
        out = {n: P.explain(n) for n in P.NAMES + P.DERIVED}
        return {"t": "doc", "docs": out, "groups": P.GROUPS, "spec": {n: P.SPEC[n] for n in P.NAMES}}

    # -- the preview: instant, independent of RTAB-Map ---------------------------------------------------------------------------------------------------------------------------
    def request_preview(self, idx=None, run=None):
        with self._req_cv:
            self._req = (self.cursor if idx is None else idx, self.manual.copy(), run)
            self._req_cv.notify()

    def _stats(self, l, r, idx, heavy):
        s = epipolar_stats(l, r)
        out = {"idx": idx, "rows": s, "reading": explain_stats(s), "smear_px": self.rec.smear_px(idx, self.manual.rect().f)}
        return out

    def _preview_loop(self):
        from rtabmap_lab.tools.stereo_stages import plane_noise
        while self._alive:
            with self._req_cv:
                while self._req is None and self._alive:
                    self._req_cv.wait(0.5)
                req, self._req = self._req, None
            if req is None:
                continue
            idx, params, run = req
            try:
                t0 = time.perf_counter()
                l, r = self.rec.prepare(idx, params.rect())
                disp = self.sgbm.stages(l, r)["disp"]
                self.last_disp, self.last_fb = disp, params.rect().f * params.rect().B
                heavy = time.monotonic() - self._last_heavy > 0.7 or run is None
                self._last_heavy = time.monotonic() if heavy else self._last_heavy
                q = self._stats(l, r, idx, heavy) if heavy else dict(self.quality)
                q["valid_share"] = float(np.isfinite(disp).mean())
                if heavy:
                    pn = plane_noise(disp)
                    q["sigma_d_px"], q["plane_share"] = (pn[1], pn[0]) if pn else (None, None)
                q["fingerprint"], q["ms"] = params.fingerprint(), (time.perf_counter() - t0) * 1000
                self.quality = q
                self.send({"t": "img", "name": "left", "idx": idx}, jpeg(l), keep=("img", "left"))
                self.send({"t": "img", "name": "right", "idx": idx}, jpeg(r), keep=("img", "right"))
                self.send({"t": "img", "name": "disp", "idx": idx, "fb": self.last_fb}, jpeg(disp_color(disp)), keep=("img", "disp"))
                if run is not None and run.fingerprint == params.fingerprint() and q.get("rows") and (q["smear_px"] is None or q["smear_px"] <= SMEAR_OK_PX):
                    run.samples["row_err_px"].append(q["rows"]["row_err_px"])
                    run.samples["valid_share"].append(q["valid_share"])
                    if q.get("sigma_d_px") is not None:
                        run.samples["sigma_d_px"].append(q["sigma_d_px"])
                if self.ref is not None:
                    self._ref_preview(idx)
                self.push_state()
            except Exception as e:
                self.note(f"preview failed: {type(e).__name__}: {e}", "error")

    def _ref_preview(self, idx):
        """The pinned 'before' calibration on the SAME pair: its rectified left, disparity and row alignment, for the wipe comparison."""
        rp = self.ref["params"]
        l, r = self.rec.prepare(idx, rp.rect())
        disp = self.sgbm.stages(l, r)["disp"]
        s = epipolar_stats(l, r)
        self.send({"t": "img", "name": "ref_left", "idx": idx}, jpeg(l), keep=("img", "ref_left"))
        self.send({"t": "img", "name": "ref_disp", "idx": idx}, jpeg(disp_color(disp)), keep=("img", "ref_disp"))
        self.ref["quality"] = {"idx": idx, "rows": s, "valid_share": float(np.isfinite(disp).mean())}
        self.send({"t": "ref_quality", **self.ref["quality"]}, keep=("ref_quality",))

    def depth_at(self, x, y):
        """Distance along the optical axis at one pixel of the preview disparity: f*B/d, median of a 7x7 window (a number from the pictures, not from the map)."""
        d = self.last_disp
        if d is None:
            return
        x, y = int(x), int(y)
        w = d[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        w = w[np.isfinite(w) & (w > 0.5)]
        if len(w) < 5:
            self.send({"t": "depth_readout", "x": x, "y": y, "z": None, "text": "no disparity here"})
            return
        dm = float(np.median(w))
        self.send({"t": "depth_readout", "x": x, "y": y, "disparity": dm, "z": self.last_fb / dm, "n": int(len(w))})

    # -- parameters ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def set_param(self, name, value):
        self.manual.set(name, value)
        self._changed(name)

    def reset_param(self, name):
        self.manual.reset(name)
        self._changed(name)

    def reset_all(self):
        self.manual = self.base.copy()
        self._changed("all")

    def load_params(self, p, why):
        self.manual = p.copy()
        self.manual.original = dict(self.base.original)
        self.note(f"calibration replaced: {why}")
        self._changed("load")

    def _changed(self, what):
        self.request_preview()
        self.push_state()
        if self._timer:
            self._timer.cancel()
        r = self.run
        if self.auto_restart and r is not None and r.alive() and not r.pause.is_set():
            self._timer = threading.Timer(DEBOUNCE_S, self.restart)
            self._timer.daemon = True
            self._timer.start()
        elif r is not None and r.fingerprint != self.manual.fingerprint():
            self.note("the map was built with other calibration numbers: it is STALE and stays as it was until Restart / Play", "warn")

    # -- the run -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def restart(self, start=None):
        """Stop the run, start a new one from the clip start (or `start`) with the working calibration and an empty map."""
        with self._ctl:
            old = self.run
            if old is not None:
                old.stop()
            if self._timer:
                self._timer.cancel()
            a = self.clip[0] if start is None else int(start)
            self._port = (self._port + 1) % 8
            run = Run(self.rec, self.manual, (a, self.clip[1]), self.settings, emit=lambda k, *args: self._on_run(run, k, *args), port=5710 + self._port)
            self.run = run
            self.cursor = a
            self.send({"t": "reset", "fingerprint": run.fingerprint, "start": a})
            with self._cache_lock:
                self.cache.pop(("mesh", "main"), None)
                self.cache.pop(("camera",), None)
            self.note(f"RTAB-Map restarted at pair {a} with calibration {run.fingerprint}" + (f" (changed: {', '.join(self.manual.changed(self.base)[:6])})" if self.manual.changed(self.base) else " (the saved calibration)"))
            run.start()
            self.push_state()

    def _on_run(self, run, kind, *a):
        if run is not self.run:
            return                                                              # an old run that is still winding down: nothing of it reaches the page
        if kind == "frame":
            self.cursor = a[0]
            self.request_preview(a[0], run)
        elif kind == "camera":
            T, st = a
            self.send({"t": "camera", "pose": T.tolist(), "run": run.fingerprint, "pos": run.pos, **st}, keep=("camera",))
        elif kind == "disparity":
            self.send({"t": "img", "name": "rtab_disp"}, jpeg(disp_color(a[0].astype(np.float32) / 4.0)), keep=("img", "rtab_disp"))
        elif kind == "mesh":
            self._send_mesh("main", a[0], run.fingerprint)
        elif kind == "log":
            self.note("bridge: " + a[0], "info" if "loop closure" in a[0] else "warn")
        elif kind == "progress":
            if a[0] % 5 == 0:
                self.push_state()
        elif kind == "done":
            self._run_done(run, a[0])

    def _send_mesh(self, which, m, fp):
        v, c, t = m["vertices"], m["colors"], m["triangles"]
        pad = b"\0" * ((-3 * len(v)) % 4)                                          # the uint32 triangles must start on a 4-byte boundary
        self.send({"t": "mesh", "which": which, "nv": int(len(v)), "nt": int(len(t)), "fingerprint": fp}, v.tobytes() + c.tobytes() + pad + t.tobytes(), keep=("mesh", which))

    def _run_done(self, run, metrics):
        if metrics.get("stopped"):
            return
        if "failed" in metrics:
            self.note("run failed: " + metrics["failed"], "error")
        elif run.state == "done":
            self.last_experiment = self.save_experiment(run)
            self.note(f"run finished; saved {self.last_experiment}")
            if self.loop and run.fingerprint == self.manual.fingerprint():
                threading.Timer(0.5, self.restart).start()
        if not self.rec.unchanged():
            self.note("WARNING: the recording's files changed while it was in use", "error")
        self.push_state()

    def play(self):
        r = self.run
        if r is None or not r.alive():
            c = self.cursor
            return self.restart(c if (r is None or c != r.pos) and self.clip[0] <= c < self.clip[1] - 5 else None)
        if r.pause.is_set():
            if self.cursor != r.pos:                                              # scrubbed somewhere else while paused: RTAB-Map cannot jump, so it starts again there (a new map)
                return self.restart(self.cursor)
            r.set_paused(False)
        self.push_state()

    def pause(self):
        if self.run is not None and self.run.alive():
            self.run.set_paused(True)
        self.push_state()

    def seek(self, idx):
        self.cursor = int(np.clip(idx, 0, self.rec.n - 1))
        self.request_preview()
        self.push_state()

    def set_clip(self, a, b):
        a, b = int(np.clip(a, 0, self.rec.n - 2)), int(np.clip(b, 1, self.rec.n))
        if b <= a:
            raise ValueError("the clip end must be after its start")
        self.clip = (a, b)
        self.push_state()

    # -- reference ("before") --------------------------------------------------------------------------------------------------------------------------------------------------------
    def pin_reference(self, name=""):
        run = self.run
        self.ref = {"name": name or f"calibration {self.manual.fingerprint()}", "params": self.manual.copy(), "metrics": run.metrics if run and run.fingerprint == self.manual.fingerprint() else None,
                    "mesh": run.mesh if run and run.fingerprint == self.manual.fingerprint() else None}
        if self.ref["mesh"] is not None:
            self._send_mesh("ref", self.ref["mesh"], self.ref["params"].fingerprint())
        self.note(f"pinned '{self.ref['name']}' as BEFORE" + ("" if self.ref["mesh"] is not None else " (no finished map for it yet: its pictures are compared, run it once to compare maps)"))
        self.request_preview()
        self.push_state()

    def clear_reference(self):
        self.ref = None
        with self._cache_lock:
            for k in [k for k in self.cache if k[-1] in ("ref", "ref_left", "ref_disp") or k[0] == "ref_quality"]:
                self.cache.pop(k)
        self.send({"t": "ref_clear"})
        self.push_state()

    # -- automatic calibration --------------------------------------------------------------------------------------------------------------------------------------------------------
    def auto_calibrate(self, n=40, whole=True):
        """Check / refine the working calibration from the footage of the clip (needs no board). Runs in a thread; the result goes to `auto`, never to the working copy."""
        if self.auto and self.auto.get("running"):
            return
        self.auto = {"running": True, "progress": 0.0, "verdict": "running ..."}
        self.push_state()

        def work():
            try:
                p, rec = self.manual.copy(), self.rec
                f = p.rect().f
                a0, a1 = (0, rec.n) if whole else self.clip
                cand = [int(i) for i in np.linspace(a0, a1 - 1, max(n * 6, 60))]
                gate = lambda i: None if (rec.smear_px(i, f) is not None and rec.smear_px(i, f) <= SMEAR_OK_PX) else "rig turning too fast (or no gyro) for a fair row measurement"
                ok = [i for i in cand if gate(i) is None][:n]
                if len(ok) < 10:
                    ok = cand[:n]                                           # not enough slow pairs: use what there is; the pooled-fit checks will say so if it is not enough
                    gate = None
                ac = AutoCalibration(p, rec.calibrated_pair, gate)
                res = ac.run(ok, progress=lambda x, t: self._auto_progress(x, t))
                res["skipped_count"] = len(res.get("skipped", []))
                import re, collections
                res["skipped_reasons"] = dict(collections.Counter(re.sub(r"\d+(\.\d+)?", "N", why) for _, why in res.get("skipped", [])))
                res["running"], res["source_fingerprint"], res["when"] = False, p.fingerprint(), time.strftime("%Y-%m-%d %H:%M:%S")
                self.auto = res
                if "params" in res:
                    res["params"].original = dict(self.base.original)
                self.note(f"automatic calibration: {res['verdict']}")
            except Exception as e:
                self.auto = {"running": False, "verdict": f"failed: {type(e).__name__}: {e}"}
                self.note(self.auto["verdict"], "error")
            self.push_state()
        threading.Thread(target=work, daemon=True).start()

    def _auto_progress(self, x, text):
        if self.auto is not None:
            self.auto["progress"] = x
            self.auto["verdict"] = text
        if int(x * 20) != getattr(self, "_ap", -1):
            self._ap = int(x * 20)
            self.push_state()

    def apply_auto(self):
        if not self.auto or "params" not in self.auto:
            raise ValueError("there is no automatic result with a calibration to apply")
        self.load_params(self.auto["params"], "the automatic result was copied into the working copy (the automatic result itself stays separate)")

    # -- files -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _path(self, sub, name):
        safe = "".join(c for c in name if c.isalnum() or c in "-_. ").strip().replace(" ", "_")
        if not safe:
            raise ValueError("give the file a name")
        return os.path.join(self.root, sub, safe)

    def save_config(self, name, kind="manual", notes=""):
        """Save the working copy ('manual') or the automatic result ('auto'): two kinds, two name prefixes, so they are never mixed."""
        if kind == "auto":
            if not self.auto or "params" not in self.auto:
                raise ValueError("no automatic result to save")
            p, name = self.auto["params"], name if name.startswith("auto_") else "auto_" + name
        else:
            p = self.manual
            name = name if not name.startswith("auto_") else name[5:]
        path = self._path("configs", name) + ".json"
        d = p.to_json(kind=kind, name=name, notes=notes)
        d["recording"] = {"folder": self.rec.folder, "fingerprint": self.rec.stamp}
        d["clip"] = list(self.clip)
        if kind == "auto" and self.auto:
            d["automatic_check"] = {k: self.auto.get(k) for k in ("verdict", "estimates", "std", "checks", "unobservable", "diagnostic", "row_err_current_px", "row_err_proposed_px", "when")}
        json.dump(d, open(path, "w"), indent=1, default=float)
        self._lists_cache = None
        self.note(f"saved {kind} calibration {os.path.relpath(path)}")
        self.push_state()
        return path

    def export_npz(self, name):
        """The working calibration as a .npz for the real pipeline (data/stereo_calib_lab_<name>.npz); the pipeline then takes it with source.calib / --pipeline-config."""
        safe = "".join(c for c in name if c.isalnum() or c in "-_").strip() or self.manual.fingerprint()
        path = os.path.join("data", f"stereo_calib_lab_{safe}.npz")
        self.manual.to_npz(path)
        self.note(f"exported {path}: set  calib = \"{path.replace(os.sep, '/')}\"  in a pipeline config to use it (the saved file {self.base.meta.get('npz')} is untouched)")
        return path

    def load_config(self, name):
        path = self._path("configs", name.replace(".json", "")) + ".json"
        d = json.load(open(path))
        p = P.CalibParams.from_json(d)
        p.meta["npz"] = p.meta.get("npz") or self.base.meta.get("npz")
        self.load_params(p, f"loaded {d.get('kind', 'manual')} calibration '{d.get('name')}'")

    def list_configs(self):
        out = []
        for f in sorted(os.listdir(os.path.join(self.root, "configs"))):
            if f.endswith(".json"):
                try:
                    d = json.load(open(os.path.join(self.root, "configs", f)))
                    out.append({"file": f[:-5], "kind": d.get("kind"), "created": d.get("created"), "fingerprint": d.get("fingerprint"), "notes": d.get("notes", "")})
                except Exception:
                    pass
        return out

    def list_experiments(self):
        out = []
        base = os.path.join(self.root, "experiments")
        for f in sorted(os.listdir(base), reverse=True)[:30]:
            p = os.path.join(base, f, "experiment.json")
            if os.path.exists(p):
                try:
                    d = json.load(open(p))
                    out.append({"id": f, "fingerprint": d["calibration"]["fingerprint"], "clip": d["clip"], "metrics": d["metrics"], "recording": os.path.basename(d["recording"]["folder"])})
                except Exception:
                    pass
        return out

    def save_experiment(self, run):
        """Everything needed to repeat the run and to compare it: the recording (path + file fingerprint), the clip, the calibration, the settings, the metrics, the TSDF mesh."""
        eid = time.strftime("%Y%m%d_%H%M%S_") + run.fingerprint
        d = os.path.join(self.root, "experiments", eid)
        os.makedirs(d, exist_ok=True)
        run.save_mesh(os.path.join(d, "tsdf.ply"))
        json.dump({"id": eid, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "recording": {"folder": self.rec.folder, "pairs": self.rec.n, "fingerprint": self.rec.stamp, "unchanged_after_run": self.rec.unchanged(),
                                                                                           "layout": self.rec.layout_note, "orientation": self.rec.flip_reason},
                   "clip": list(run.clip), "calibration": run.params.to_json(kind="manual", name=eid), "changed_from_saved": run.params.changed(self.base), "settings": {k: v for k, v in run.settings.items()},
                   "metrics": run.metrics, "camera_model": {k: v for k, v in run.cal.items() if k != "size"} | {"size": list(run.cal["size"])}},
                  open(os.path.join(d, "experiment.json"), "w"), indent=1, default=float)
        run.experiment_dir = d
        self._lists_cache = None
        return os.path.relpath(d)

    def export_measurements(self, points, note=""):
        """points: what the page holds (id, world xyz, camera position when picked, distances). Saved as JSON and CSV next to the experiment, with the calibration it was measured under."""
        run = self.run
        eid = time.strftime("%Y%m%d_%H%M%S") + "_" + (run.fingerprint if run else self.manual.fingerprint())
        base = os.path.join(self.root, "measurements", eid)
        pair_d = [{"a": a["id"], "b": b["id"], "distance_m": float(np.linalg.norm(np.array(a["world"]) - np.array(b["world"])))} for i, a in enumerate(points) for b in points[i + 1:]]
        json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note, "calibration": self.manual.to_json(kind="manual", name=eid), "run_fingerprint": run.fingerprint if run else None,
                   "run_matches_working_calibration": bool(run and run.fingerprint == self.manual.fingerprint()), "clip": list(run.clip) if run else None, "recording": self.rec.folder,
                   "frame": "RTAB-Map map frame: x forward, y left, z up, origin at the camera of the first node, metres", "points": points, "point_to_point": pair_d}, open(base + ".json", "w"), indent=1, default=float)
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "x_m", "y_m", "z_m", "distance_from_camera_at_pick_m", "distance_from_camera_when_exported_m", "distance_from_map_origin_m"])
            for q in points:
                w.writerow([q["id"], *[f"{v:.4f}" for v in q["world"]], q.get("dist_pick"), q.get("dist_now"), q.get("dist_origin")])
        self.note(f"measurements saved: {os.path.relpath(base)}.json/.csv")
        return base

    # -- comparison ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def compare(self, a, b):
        """Parameter differences and the stored metrics of two calibrations (saved config names, or 'saved' = the file the session started from, 'working' = the current copy, 'auto')."""
        def get(x):
            if x == "saved":
                return self.base
            if x == "working":
                return self.manual
            if x == "auto" and self.auto and "params" in self.auto:
                return self.auto["params"]
            return P.CalibParams.from_json(json.load(open(self._path("configs", x.replace(".json", "")) + ".json")))
        pa, pb = get(a), get(b)
        rows = [{"name": n, "a": pa.values[n], "b": pb.values[n], "delta": pb.values[n] - pa.values[n]} for n in P.NAMES if abs(pa.values[n] - pb.values[n]) > 1e-9]
        ex = {e["fingerprint"]: e for e in self.list_experiments()[::-1]}
        res = {"t": "compare", "a": a, "b": b, "rows": rows, "metrics_a": (ex.get(pa.fingerprint()) or {}).get("metrics"), "metrics_b": (ex.get(pb.fingerprint()) or {}).get("metrics"),
               "fingerprint_a": pa.fingerprint(), "fingerprint_b": pb.fingerprint()}
        self.send(res)
        return res

    def batch(self, names, clip=None):
        """Replay the clip once per saved calibration with the SAME recording and settings, one after the other; the metric tables of all of them end up in the experiments list."""
        if self.batch_state and self.batch_state.get("running"):
            return

        def work():
            self.batch_state = {"running": True, "done": 0, "total": len(names), "rows": []}
            with self._ctl:
                if self.run is not None:
                    self.run.stop()
            try:
                for k, nm in enumerate(names):
                    p = self.manual if nm == "working" else self.base if nm == "saved" else P.CalibParams.from_json(json.load(open(self._path("configs", nm.replace(".json", "")) + ".json")))
                    run = Run(self.rec, p, clip or self.clip, self.settings, port=5710 + k % 8)
                    run.start().thread.join()
                    if run.state == "failed":
                        raise RuntimeError(run.error)
                    self.save_experiment(run)
                    self.batch_state["rows"].append({"name": nm, "fingerprint": p.fingerprint(), "metrics": run.metrics})
                    self.batch_state["done"] = k + 1
                    self.push_state()
            except Exception as e:
                self.note(f"batch failed: {type(e).__name__}: {e}", "error")
            self.batch_state["running"] = False
            self.push_state()
        threading.Thread(target=work, daemon=True).start()

    def close(self):
        self._alive = False
        with self._req_cv:
            self._req_cv.notify_all()
        if self.run is not None:
            self.run.stop()
        self.rec.close()
