"""Rerun drawing of the Pi's KLT tracks and of the synchronisation of the three streams (used by gabai/pipeline/viz.py).

Entities
  klt/left                    the pipeline's left picture (rectified, as the rest of the pipeline sees it)
  klt/left/tracked            followed points, coloured by the Pi's LK patch error (green good, yellow fair, red poor); the feature id is the label
  klt/left/new, /lost         found in this frame (yellow); lost in this frame, drawn where they were last seen (red)
  klt/left/trails             where each followed point has been over the last `TRAIL` frames
  klt/left/motion             the last step of each point, drawn 3x longer
  klt/features/left           every point as its own entity, without the picture; hover a point for "id (x, y) quality"
  klt/count/*                 tracked, new, lost, mean patch error, the Pi's tracking time
  sync/*                      what the server received and released for each frame (see sync())
"""
from collections import deque

import numpy as np
import rerun as rr

TRAIL, GOOD_Q, FAIR_Q = 15, 24, 64                    # patch error x 8: under 3 grey levels is good, under 8 is fair
GREEN, YELLOW, RED, CYAN = (60, 220, 60), (255, 220, 0), (255, 60, 60), (0, 220, 255)


class KltPanel:
    def __init__(self, log, clear, scalar, scale):
        self._log, self._clear, self.scalar, self.scale = log, clear, scalar, scale
        self.trail = {}                               # feature id -> deque of its last positions (full picture pixels)
        self.shown = False

    def draw(self, left_bgr, obs, right_bgr=None):
        import cv2
        sc = self.scale
        img = cv2.resize(left_bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1.0 else left_bgr
        self._log("klt/left", rr.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=65))
        if obs is None or not len(obs.ids):
            for name in ("tracked", "new", "lost", "trails", "motion"):
                self._clear(f"klt/left/{name}", False)
            self._clear("klt/features/left", False)
            self.trail.clear()
            return
        ids, uv = obs.ids, obs.uv.astype(float)
        new = np.zeros(len(ids), bool) if obs.new is None else np.asarray(obs.new, bool)
        q = np.zeros(len(ids)) if obs.quality is None else np.asarray(obs.quality, float)
        prev = np.array([self.trail[i][-1] if i in self.trail and not n else (np.nan, np.nan) for i, n in zip(ids, new)])
        for i, p in zip(ids, uv):
            self.trail.setdefault(int(i), deque(maxlen=TRAIL)).append(p)
        for dead in set(self.trail) - set(int(i) for i in ids):
            del self.trail[dead]
        half = 0.5                                    # Rerun puts pixel centres at half integers, OpenCV at integers
        col = np.array([GREEN if v < GOOD_Q else YELLOW if v < FAIR_Q else RED for v in q], np.uint8)
        m = ~new
        self._points("klt/left/tracked", (uv[m] + half) * sc, col[m], ids[m], 2.5)
        self._points("klt/left/new", (uv[new] + half) * sc, [YELLOW] * int(new.sum()), ids[new], 3.5)
        lost = obs.lost_uv if obs.lost_uv is not None else np.empty((0, 2))
        self._points("klt/left/lost", (np.asarray(lost, float) + half) * sc, [RED] * len(lost), obs.lost_ids if obs.lost_ids is not None else [], 4.0)
        strips = [np.asarray(t) * sc + half * sc for t in (self.trail[int(i)] for i in ids[m]) if len(t) > 1]
        self._log("klt/left/trails", rr.LineStrips2D(strips, colors=CYAN, radii=0.6)) if strips else self._clear("klt/left/trails", False)
        ok = m & np.isfinite(prev[:, 0])
        if ok.any():
            self._log("klt/left/motion", rr.Arrows2D(origins=(prev[ok] + half) * sc, vectors=(uv[ok] - prev[ok]) * sc * 3.0, colors=YELLOW, radii=0.5))
        else:
            self._clear("klt/left/motion", False)
        labels = [f"{int(i)} ({x:.0f},{y:.0f}) q{int(v)}" for i, (x, y), v in zip(ids, uv, q)]
        self._log("klt/features/left", rr.Points2D(uv + half, colors=[YELLOW if n else GREEN for n in new], radii=2.0, labels=labels, show_labels=False))
        self.scalar("klt/count/tracked", float(len(ids)))
        self.scalar("klt/count/new", float(new.sum()))
        self.scalar("klt/count/lost", float(len(lost)))
        self.scalar("klt/count/mean_patch_error_x8", float(q[m].mean()) if m.any() else 0.0)
        self.scalar("klt/count/pi_ms", float(obs.klt_ms))
        self._right(right_bgr, obs, ids)

    def _right(self, right_bgr, obs, ids):
        """The same points in the right picture (the Pi follows them left to right with --klt-stereo); a point it could not find there is simply not drawn."""
        import cv2
        ruv = getattr(obs, "right_uv", None)
        if right_bgr is None or ruv is None:
            self._clear("klt/right/tracked", False)
            return
        sc = self.scale
        img = cv2.resize(right_bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1.0 else right_bgr
        self._log("klt/right", rr.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=65))
        ok = np.isfinite(ruv).all(axis=1)
        self.scalar("klt/count/in_right", float(ok.sum()))
        self._points("klt/right/tracked", (ruv[ok].astype(float) + 0.5) * sc, [GREEN] * int(ok.sum()), ids[ok], 2.5)

    def _points(self, path, pos, colors, ids, radius):
        if len(pos):
            self._log(path, rr.Points2D(pos, colors=colors, radii=radius, labels=[str(int(i)) for i in ids], show_labels=len(pos) <= 60))
        else:
            self._clear(path, False)


SYNC_SCALARS = (("frame_seq", "seq"), ("klt_frame_seq", "klt_seq"), ("imu_samples_in_interval", "imu_n"), ("imu_last_minus_frame_ms", "imu_margin_ms"), ("imu_span_ms", "imu_span_ms"),
                ("imu_interval_s", "interval_s"), ("imu_gap_s", "imu_gap_s"), ("backlog_pairs", "backlog"), ("imu_queue", "imu_queue"), ("wait_for_parts_ms", "wait_ms"),
                ("latency_ms", "latency_ms"), ("skipped_before", "skipped_before"))
TOTALS = ("pi_skipped", "dup_frames", "imu_missing", "imu_dup", "imu_dropped", "overflow", "recovery_skipped", "recoveries", "incomplete", "klt_missing", "klt_mismatch", "klt_orphans")


def log_sync(scalar, event, info, pose_lag_frames=None):
    """Scalars under sync/ for one released group, and a text event when something is wrong. info = SyncBuffer's Group.info (+ latency_ms)."""
    if not info:
        return
    for name, key in SYNC_SCALARS:
        if info.get(key) is not None:
            scalar(f"sync/{name}", float(info[key]))
    if info.get("klt_seq") is not None:
        scalar("sync/klt_seq_minus_frame_seq", float(info["klt_seq"] - info["seq"]))      # 0 = the points belong to this very pair
    for k, v in (info.get("totals") or {}).items():
        if k in TOTALS:
            scalar(f"sync/total_{k}", float(v))
    if pose_lag_frames is not None:
        scalar("sync/pose_lag_frames", float(pose_lag_frames))                          # how many frames behind the newest one the pose just drawn is
    problems = []
    if not info.get("imu_complete", True):
        problems.append("IMU did not reach the frame in time")
    if info.get("imu_gap_s"):
        problems.append(f"IMU hole of {info['imu_gap_s'] * 1000:.0f} ms in this interval")
    if info.get("klt_missing"):
        problems.append("no KLT points for this frame")
    if info.get("skipped_before"):
        problems.append(f"{info['skipped_before']} older pictures skipped to catch up (their IMU kept)")
    scalar("sync/ok", 0.0 if problems else 1.0)
    if problems:
        event(f"sync frame {info['seq']}: " + "; ".join(problems), "WARN")
