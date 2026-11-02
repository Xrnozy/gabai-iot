"""Diagnostics for one live (or replayed) SLAM run: what was pushed into Basalt, what came out, and what looked wrong.

The live tool calls note_imu / note_image / note_frame_pushed / note_pose as things happen and tick() about every loop; finish() at
the end writes slam_summary.json (trajectory summary and findings, see gabai.slam.diag) and logs each finding. Everything is
logged under component "slam". Frame ids (fid) are the Pi capture time in ms, the same number the Pi and the depth viewer log:
Basalt's own frame stamps have the camera delay removed, so it is added back here.
"""
import json
import os
from collections import deque

import numpy as np

from gabai.diag import checks as C
from gabai.diag import log as dlog
from gabai.slam.diag import findings_for, summarize_trajectory

MAX_IMU_KEPT = 200_000          # about 16 minutes at 200 Hz; older samples are dropped from the end-of-run summary only
VERDICT_CLEAN = "nothing implausible was found in the trajectory (plausible, not whether it is correct: there is no ground truth)"


class SlamRunDiag:
    def __init__(self, out_dir, f_px, cam_delay_s=0.0, period_s=5.0, no_pose_s=1.0):
        self.out, self.f, self.delay, self.period, self.no_pose_s = out_dir, f_px, cam_delay_s, period_s, no_pose_s
        self.log = dlog.get("slam")
        self.traj = []
        self.imu = deque(maxlen=MAX_IMU_KEPT)
        self.last_img = None
        self.last_pose_wall, self.frames_since_pose = None, 0
        self._win = self._new_window(0.0)
        self._t_tick = None
        self._corner_every, self._n_img = 10, 0

    @staticmethod
    def _new_window(now):
        return {"t0": now, "poses": 0, "delay": [], "timing": {}, "speeds": []}

    def fid_of(self, t_ns):
        return int(round(t_ns / 1e6 + self.delay * 1000))

    # -- inputs ---------------------------------------------------------------------------------------------------------------
    def note_imu(self, t_s, gyro_rad, accel_ms2):
        self.imu.append((t_s, *gyro_rad, *accel_ms2))

    def note_image(self, gray, fid=None):
        """Check the picture Basalt is about to see. The corner count is OUR measure of how trackable the scene is (Shi-Tomasi corners),
        not Basalt's own landmark count, which this Basalt build does not report."""
        self._n_img += 1
        if self._n_img % self._corner_every and self.last_img is not None:
            return
        import cv2
        small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
        chk = C.check_image(small)
        pts = cv2.goodFeaturesToTrack(small, 300, 0.01, 8)
        self.last_img = {"status": chk.status, "detail": chk.detail, "stats": chk.value, "trackable_corners": 0 if pts is None else len(pts), "fid": fid}
        if chk.status in ("fail", "suspect") or self.last_img["trackable_corners"] < 40:
            self.log.warning_every("img", 10.0, "slam.image_poor", fid=fid, status=chk.status, detail=chk.detail, trackable_corners=self.last_img["trackable_corners"],
                                   hint="Basalt needs light and texture; this picture will track badly")

    def note_frame_pushed(self, fid):
        self.frames_since_pose += 1

    # -- outputs --------------------------------------------------------------------------------------------------------------
    def note_pose(self, p, push_wall, now):
        """p: the pose dict from BasaltTracker.pop_pose(); push_wall: when its frame was pushed (monotonic) or None; now: monotonic."""
        t = p["t_ns"] * 1e-9
        prev = self.traj[-1] if self.traj else None
        self.traj.append((t, *p["p"], *p["q"]))
        self.last_pose_wall, self.frames_since_pose = now, 0
        w = self._win
        w["poses"] += 1
        if push_wall is not None:
            w["delay"].append((now - push_wall) * 1000)
        for k, v in (p.get("timing") or {}).items():
            if v is not None:
                w["timing"].setdefault(k, []).append(v)
        fid = self.fid_of(p["t_ns"])
        if prev is not None:
            dt = t - prev[0]
            step = float(np.linalg.norm(np.array(p["p"]) - np.array(prev[1:4])))
            w["speeds"].append(step / max(dt, 1e-6))
            chk = C.check_pose_step(step, dt)
            if chk.status == "fail":
                self.log.warning_every("jump", 5.0, "slam.jump", fid=fid, step_m=round(step, 3), dt_ms=round(dt * 1000, 1), speed_mps=chk.value["speed_mps"],
                                       image=self.last_img, imu_recent=self._imu_context(t), hint=chk.hint)
        self.log.debug("slam.pose", fid=fid, t=round(t, 3), p=[round(x, 3) for x in p["p"]], v=[round(x, 3) for x in p["v"]], timing=p.get("timing"))

    def _imu_context(self, t):
        rows = [r for r in list(self.imu)[-600:] if t - 0.5 <= r[0] <= t + 0.1]
        if not rows:
            return None
        a = np.array(rows)
        return {"gyro_peak_dps": round(float(np.degrees(np.linalg.norm(a[:, 1:4], axis=1).max())), 1), "n": len(rows)}

    def tick(self, now, gate=None, n_frames=0, n_imu=0, started_wall=None):
        if self._t_tick is None:
            self._t_tick = now
            self._win["t0"] = now
        if gate is not None and not gate.ready and started_wall is not None and now - started_wall > 20.0:
            self.log.warning_every("gate", 20.0, "slam.gate_waiting", seconds=round(now - started_wall, 1), progress=round(gate.progress, 2),
                                   last_gyro_dps=None if gate.last_gyro_dps is None else round(gate.last_gyro_dps, 1),
                                   last_accel_err_ms2=None if gate.last_accel_err is None else round(gate.last_accel_err, 2),
                                   hint="Basalt waits for the rig to be still: hold it dead still on a table. If the gyro reads high while still, the Pi measured "
                                        "a wrong gyro bias at start: restart the sender with the rig still. If the accelerometer error is large, calibrate it.")
        if self.last_pose_wall is not None and self.frames_since_pose and now - self.last_pose_wall > self.no_pose_s:
            self.log.warning_every("nopose", 5.0, "slam.no_pose", seconds_without_pose=round(now - self.last_pose_wall, 1),
                                   frames_pushed_since_last_pose=self.frames_since_pose, image=self.last_img,
                                   hint="Basalt received pictures but produced no pose: tracking lost (blur, no texture) or Basalt is overloaded")
        if now - self._t_tick >= self.period:
            w, span = self._win, max(now - self._win["t0"], 1e-6)
            d = np.array(w["delay"]) if w["delay"] else None
            self.log.info("slam.stats", window_s=round(span, 1), poses_in_window=w["poses"], pose_rate_hz=round(w["poses"] / span, 2),
                          pose_delay_ms_p50=None if d is None else round(float(np.percentile(d, 50)), 1),
                          pose_delay_ms_p95=None if d is None else round(float(np.percentile(d, 95)), 1),
                          basalt_ms={k: round(float(np.median(v)), 2) for k, v in w["timing"].items()},
                          speed_mps_p95=round(float(np.percentile(w["speeds"], 95)), 2) if w["speeds"] else None,
                          position=[round(x, 3) for x in self.traj[-1][1:4]] if self.traj else None, frames_pushed=n_frames, imu_pushed=n_imu,
                          image=self.last_img, gate_ready=None if gate is None else gate.ready)
            self._win, self._t_tick = self._new_window(now), now

    # -- end ------------------------------------------------------------------------------------------------------------------
    def finish(self):
        poses = np.array(self.traj) if self.traj else np.zeros((0, 8))
        imu = np.array(self.imu) if len(self.imu) > 10 else None
        s = summarize_trajectory(poses, imu=imu)
        findings = findings_for(s)
        summary = {"trajectory": s, "findings": findings, "verdict": VERDICT_CLEAN if not findings else
                   f"{sum(f['level'] == 'confirmed' for f in findings)} confirmed and {sum(f['level'] == 'possible' for f in findings)} possible problem(s)"}
        with open(os.path.join(self.out, "slam_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        self.log.info("slam.summary", **{k: v for k, v in s.items() if k not in ("jumps", "gaps")})
        for f in findings:
            (self.log.warning if f["level"] == "confirmed" else self.log.info)("slam.finding", code=f["code"], level=f["level"], message=f["message"],
                                                                                   evidence=f["evidence"], possible_causes=f["possible_causes"])
        return summary
