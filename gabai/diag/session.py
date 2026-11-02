"""Per-frame diagnostics for the depth and alert pipeline, shared by the live viewer and the offline replay.

Call order for one frame:
    tr = diag.begin(fid, t_capture, t_recv, link, left)         inputs and freshness checks
    diag.after_depth(tr, disp, trusted, model_ms, model, ...)   what the depth model produced and whether it looks usable
    diag.after_decision(tr, res, controller, said, ...)         zone evidence, the alert decision and why, then the trace is sunk

It only observes: it never changes a disparity, a distance or an alert. With enabled=False (or GABAI_DIAG=0) every method returns
at once, so the safety path runs exactly as before.
"""
import os

import numpy as np

from gabai.diag import checks as C
from gabai.diag import log as dlog
from gabai.diag.artifacts import ArtifactSaver
from gabai.diag.sink import TraceSink
from gabai.diag.trace import ESTIMATED, MEASURED, MODEL, FrameTrace, q, validated


class PerceptionDiag:
    def __init__(self, source, f, B, cx, cy, sink=None, saver=None, enabled=None, sample_every=15, system_every=30, image_every=5):
        self.enabled = (os.environ.get("GABAI_DIAG", "1") != "0") if enabled is None else enabled
        self.source, self.f, self.B, self.cx, self.cy = source, f, B, cx, cy
        self.sink = sink or TraceSink(every=sample_every)
        self.saver = saver
        self.system_every, self.image_every = system_every, image_every
        self.n = 0
        self._disp = None
        self._left = None
        self.log = dlog.get("pipeline")

    def set_geometry(self, f, B, cx, cy):
        """Camera geometry can change between sources (calibrated live rig vs assumed values in a video): set it per frame."""
        self.f, self.B, self.cx, self.cy = f, B, cx, cy

    # -- inputs ---------------------------------------------------------------------------------------------------------------
    def begin(self, fid, t_capture, t_recv, link, left=None):
        tr = FrameTrace(fid, t_capture, t_recv, self.source)
        if not self.enabled:
            return tr
        self.n += 1
        self._left = left
        with tr.stage("input") as s:
            s["age_s"] = q(link.get("age_s"), MEASURED, unit="s")
            s["pair_dt_ms"] = q(link.get("pair_dt_ms"), MEASURED, unit="ms", note="left/right capture offset reported by the Pi")
            s["turn_dps"] = q(link.get("turn_dps"), MEASURED, unit="deg/s", note="gyro peak over the last 0.3 s")
            s["latency_s"] = q(link.get("latency_s"), MEASURED, unit="s", note="capture to PC receipt, uses the NTP-style clock offset")
        tr.add_check(C.check_frame_age(link.get("age_s")))
        tr.add_check(C.check_pair_sync(link.get("pair_dt_ms"), link.get("turn_dps"), self.f))
        if left is not None and self.image_every and (self.n - 1) % self.image_every == 0:
            import cv2
            small = cv2.resize(left, (left.shape[1] // 2, left.shape[0] // 2), interpolation=cv2.INTER_AREA)
            c = tr.add_check(C.check_image(small))
            tr.put("input", image=c.value)
        return tr

    # -- depth ----------------------------------------------------------------------------------------------------------------
    def after_depth(self, tr, disp, trusted, model_ms, model, lr_check=False, options=None):
        if not self.enabled:
            return
        d = np.asarray(disp)
        self._disp = d
        sub = d[::4, ::4]
        frac_trusted = None if trusted is None else float(np.asarray(trusted)[::2, ::2].mean())
        ok = np.isfinite(sub) & (sub > 0.5)
        stats = {"valid_fraction": round(float(ok.mean()), 4)}
        if ok.any():
            p5, p50, p95 = np.percentile(sub[ok], [5, 50, 95])
            stats.update(p5_px=round(float(p5), 2), p50_px=round(float(p50), 2), p95_px=round(float(p95), 2),
                         median_depth_m=round(float(self.f * self.B / p50), 3))
        with tr.stage("depth") as s:
            s["model"] = model
            s["model_ms"] = round(float(model_ms), 2)
            s["options"] = options or {}
            s["disparity"] = q(stats, MODEL, unit="px", note="model output: a prediction, not a measurement")
            if frac_trusted is not None:
                s["trusted_fraction"] = validated(q(round(frac_trusted, 4), MODEL), by="left-right consistency")
                s["trusted_fraction"]["note"] = "two passes of the same model agree; this does not prove the metres are right"
        tr.add_check(C.check_disparity_cap(d[::2, ::2]))
        if frac_trusted is not None:
            tr.add_check(C.check_trusted_fraction(frac_trusted))

    # -- decision -------------------------------------------------------------------------------------------------------------
    def after_decision(self, tr, res, controller, said, loop_ms, speech_ms, cam_h, imu_calibrated, have_up_vector, up_cam_source=None,
                       grid=None, governor=None):
        if not self.enabled:
            return
        ev = res.evidence or {}
        tr.put("zones", **ev)
        tr.put("zones", up_vector_source=up_cam_source)
        tr.add_check(C.check_depth_coverage(res.valid_fraction))
        tr.add_check(C.check_floor_model(cam_h, imu_calibrated, have_up_vector))
        tr.add_check(C.check_latency(speech_ms))
        zb = (ev.get("bands") or {}).get(res.alert[0]) if res.alert and res.alert[0] in (ev.get("bands") or {}) else None
        if zb and zb.get("p5_depth_m") is not None:
            tr.add_check(C.check_range_vs_depth(zb["p5_depth_m"], zb.get("p5_horizontal_range_m")))
        if grid is not None:
            tr.put("grid", **{k: v for k, v in grid.items() if k not in ("image",)})
            tr.add_check(C.check_grid_agreement(res.alert, grid.get("corridor"), grid.get("frames")))
        if governor is not None:
            tr.put("governor", **{k: governor[k] for k in ("tier", "reason", "depth_every", "detect_every", "temps_unknown") if k in governor})
        dec = getattr(controller, "decision", {}) or {}
        zone_alert = res.alert
        reasons = [x["reason"] for x in dec.get("not_spoken", [])]
        if said:
            outcome, accepted = "spoken", True
        elif zone_alert or dec.get("link_alert"):
            outcome, accepted = "suppressed", False
        else:
            outcome, accepted = "none", None
        tr.decision("alert", outcome, reasons=reasons + list(ev.get("decision_path", [])), accepted=accepted, zone_alert=zone_alert,
                    spoken=list(said), not_spoken=dec.get("not_spoken", []), confirmed=dec.get("confirmed"))
        tr.put("timing", loop_ms=round(float(loop_ms), 2), speech_ms=speech_ms)

        if self.system_every and self.n % self.system_every == 0:
            from gabai.diag.sysmon import snapshot
            s = snapshot()
            tr.put("system", **s)
            tr.add_check(C.check_resources(s["cpu_pct"], s["ram_pct"], s["gpu_temp_c"]))

        fid = tr.fid
        for chk in tr.checks:
            if chk.status == "fail":
                self.log.warning_every(f"check:{chk.name}", 10.0, "check.fail", fid=fid, check=chk.name, value=chk.value, limit=chk.limit,
                                       detail=chk.detail, hint=chk.hint, component=chk.component)
            elif chk.status == "suspect":
                self.log.debug("check.suspect", fid=fid, check=chk.name, value=chk.value, detail=chk.detail)
        band = (ev.get("bands") or {}).get(zone_alert[0]) if zone_alert and zone_alert[0] in (ev.get("bands") or {}) else None
        for phrase in said:
            self.log.info("alert.spoken", fid=fid, phrase=phrase, zone_alert=zone_alert, valid_fraction=round(res.valid_fraction, 3),
                          nearest_px=band and band["nearest_px"], p5_depth_m=band and band["p5_depth_m"],
                          depth_uncertainty_m=band and band["depth_uncertainty_m"], depth_kind=ESTIMATED if band else None,
                          floor_model=ev.get("floor_model"), speech_ms=speech_ms, path=ev.get("decision_path"))
        for x in dec.get("not_spoken", []):
            self.log.info_every(f"supp:{x.get('key')}:{str(x.get('reason'))[:24]}", 2.0, "alert.suppressed", fid=fid, key=x.get("key"),
                                reason=x.get("reason"), zone_alert=zone_alert)
        self.log.debug("pipeline.frame", fid=fid, loop_ms=round(float(loop_ms), 2), alert=zone_alert, valid_fraction=round(res.valid_fraction, 3),
                       failed=tr.failed())
        self.sink.offer(tr)
        if self.saver is not None:
            self.saver.maybe_save(fid, frame_left=self._left, disp=self._disp, interesting=tr.interesting(), grid_img=None if grid is None else grid.get("image"))
