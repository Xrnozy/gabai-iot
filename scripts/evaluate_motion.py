"""How well does stereo_only track fast motion, with and without the gyro prior and the time-aware prediction? On a EuRoC recording that has an IMU (data/slam/rec_*, live_*).

py scripts/evaluate_motion.py data/slam/rec_20261026_155458 [--model s] [--variants "tracking.gyro_prior=false" "tracking.gyro_prior=true"]

Each variant replays the recording through the real pipeline (stereo_only; the gyro is only read when tracking.gyro_prior is true) and reports
  held        frames whose pose was held (tracking lost)          inconsistent   frames kept out of the map because the pose and the depth disagreed (> map.max_misalign_cm)
  align       median distance between the map and each frame's own depth (cm)     inliers   median number of features that agreed on each pose
  vs Basalt   position error of the path against data/slam/<rec>/trajectory.txt (Basalt's visual-inertial path, a reference, NOT ground truth) after the best rigid fit:
              rms and largest error in cm, and the error at the end as a share of the path length
"""
import argparse
import json
import os
import tempfile

import numpy as np

from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.run import run
from gabai.pipeline.viz import Viz


class PoseViz(Viz):
    last = None

    def __init__(self, cal, size, save=None, viewer=False, **kw):
        PoseViz.last = self
        self.poses, self.cur_t, self.scalars, self.n_logged = {}, 0.0, {}, 0

    def at(self, idx, t):
        self.cur_t = t

    def pose(self, R_wc, t_wc, traj, left_bgr=None):
        self.poses[round(self.cur_t, 4)] = np.array(t_wc, float)

    def scalar(self, name, v, *a):
        self.scalars.setdefault(name, []).append(float(v) if np.isscalar(v) else float("nan"))

    def __getattr__(self, name):
        return lambda *a, **k: None


def rigid_fit(P, Q):
    """Rotation and translation that best map P onto Q (no scale)."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def versus_reference(poses, ref_file):
    ref = np.loadtxt(ref_file, comments="#")
    rt, rp = ref[:, 0], ref[:, 1:4]
    ts = np.array(sorted(poses))
    P, Q = [], []
    for t in ts:
        i = int(np.argmin(np.abs(rt - t)))
        if abs(rt[i] - t) < 0.03:
            P.append(poses[t]), Q.append(rp[i])
    if len(P) < 20:
        return None
    P, Q = np.array(P), np.array(Q)
    R, tr = rigid_fit(P, Q)
    e = np.linalg.norm(P @ R.T + tr - Q, axis=1)
    path = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
    return dict(rms_cm=float(np.sqrt((e ** 2).mean()) * 100), max_cm=float(e.max() * 100), end_pct=float(e[-1] / max(path, 1e-6) * 100), n=len(P), path_m=path)


def parse(kv):
    out = {}
    for item in kv.split():
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="s")
    ap.add_argument("--variants", nargs="+", default=["tracking.gyro_prior=false", "tracking.gyro_prior=true"])
    ap.add_argument("--frames", type=int, default=0)
    a = ap.parse_args()
    ref = os.path.join(a.folder, "trajectory.txt")
    for v in a.variants:
        ov = {"pipeline.mode": "stereo_only", "source.kind": "recording", "source.folder": a.folder, "source.skip": 1, "source.max_frames": a.frames, "stereo.model": a.model,
              "stereo.mono": "", "viz.viewer": False, "hands.enabled": False, "detector.enabled": False, "perf.async_viz": False, "perf.staged": False, "perf.low_priority": False,
              "tracking.loop_closure": False}
        ov.update(parse(v))
        cfg = PipelineConfig.load("config/pipeline.toml", ov)
        s = run(cfg, viz_factory=PoseViz, log_dir=tempfile.mkdtemp(), quiet=True)
        inl = [x for x in PoseViz.last.scalars.get("tracking/inliers", []) if np.isfinite(x)]
        r = versus_reference(PoseViz.last.poses, ref) if os.path.exists(ref) else None
        print(f"[{v}] frames {s['frames']} held {s['poses_held']} inconsistent {s['frames_skipped_inconsistent']} resets {s['map_resets']} align {s['map_alignment_median_cm']['median'] if s.get('map_alignment_median_cm') else None} cm "
              f"inliers {np.median(inl) if inl else None} frame_ms {s['frame_ms_median']}")
        if r:
            print(f"    vs Basalt over {r['n']} frames, path {r['path_m']:.1f} m: rms {r['rms_cm']:.1f} cm  max {r['max_cm']:.1f} cm  end {r['end_pct']:.1f}% of the path")


if __name__ == "__main__":
    main()
