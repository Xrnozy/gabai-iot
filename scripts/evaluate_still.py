"""Does the pose stay put when the camera is not moved? Replays a recording in which the rig sits still (data/recordings/still_*) through the visual-only tracker and measures
how far the reported camera wanders: the true motion is zero, so every metre or degree is error.

py scripts/evaluate_still.py data/recordings/still_20261030_010616 [--model h] [--set tracking.refine=false ...]

Prints the largest distance and angle from the first pose, the largest single-frame step, and each frame where a step above 1 cm / 0.5 deg happened with what the tracker reported
at that frame (keyframe change, relocalisation, correction from the map, loop closure), so a jump can be traced to the mechanism that made it.
"""
import argparse
import json
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.pipeline.config import PipelineConfig
from gabai.pipeline.run import run
from gabai.pipeline.viz import Viz


class Log(Viz):
    last = None

    def __init__(self, cal, size, save=None, viewer=False, **kw):
        Log.last = self
        self.cur_t, self.poses, self.events, self.sc, self.n_logged = 0.0, [], [], {}, 0

    def at(self, idx, t):
        self.cur_t = t

    def pose(self, R_wc, t_wc, traj, left_bgr=None):
        self.poses.append((self.cur_t, np.array(R_wc, float), np.array(t_wc, float)))

    def scalar(self, name, v, *a):
        self.sc.setdefault(name, []).append((self.cur_t, float(v) if np.isscalar(v) else float("nan")))

    def event(self, text, level="INFO"):
        self.events.append((self.cur_t, text))

    def __getattr__(self, name):
        return lambda *a, **k: None


def parse(kv):
    out = {}
    for item in kv:
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="h")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--mono", action="store_true", help="monocular fill on (default off: only tracking is tested)")
    a = ap.parse_args()
    ov = {"pipeline.mode": "stereo_only", "source.kind": "replay", "source.folder": a.folder, "source.max_frames": a.frames, "stereo.model": a.model, "viz.viewer": False,
          "hands.enabled": False, "detector.enabled": False, "perf.async_viz": False, "perf.staged": False, "perf.low_priority": False, "tracking.gyro_prior": False}
    if not a.mono:
        ov["stereo.mono"] = ""
    ov.update(parse(a.set))
    s = run(PipelineConfig.load("config/pipeline.toml", ov), viz_factory=Log, log_dir=tempfile.mkdtemp(), quiet=True)
    v = Log.last
    P = np.array([p[2] for p in v.poses])
    Rs = [R.from_matrix(p[1]) for p in v.poses]
    ts = np.array([p[0] for p in v.poses])
    dist = np.linalg.norm(P - P[0], axis=1) * 100
    ang = np.array([(Rs[0].inv() * r).magnitude() for r in Rs]) * 180 / np.pi
    step = np.linalg.norm(np.diff(P, axis=0), axis=1) * 100
    astep = np.array([(Rs[i].inv() * Rs[i + 1]).magnitude() for i in range(len(Rs) - 1)]) * 180 / np.pi
    print(f"{a.folder}: {len(P)} poses over {ts[-1] - ts[0]:.1f} s; frames {s['frames']} held {s['poses_held']} inconsistent {s['frames_skipped_inconsistent']} resets {s['map_resets']} keyframes {s.get('vo')} vo {s.get('vo_stats')}")
    print(f"  wander from the first pose: max {dist.max():.1f} cm / {ang.max():.2f} deg, at the end {dist[-1]:.1f} cm / {ang[-1]:.2f} deg")
    print(f"  single-frame step: median {np.median(step):.2f} cm / {np.median(astep):.3f} deg, p99 {np.percentile(step, 99):.2f} cm / {np.percentile(astep, 99):.2f} deg, max {step.max():.2f} cm / {astep.max():.2f} deg")
    corr = np.array([x[1] for x in v.sc.get("tracking/map_correction_cm", [])])
    if len(corr):
        print(f"  correction of the pose from the map: {len(corr)} frames, median {np.median(corr):.2f} cm, p99 {np.percentile(corr, 99):.2f} cm, max {corr.max():.2f} cm")
    big = [i for i in range(len(step)) if step[i] > 1.0 or astep[i] > 0.5]
    print(f"  {len(big)} steps above 1 cm or 0.5 deg")
    ev = v.events
    for i in big[:25]:
        t = ts[i + 1]
        near = [e[1][:110] for e in ev if abs(e[0] - t) < 1e-6]
        print(f"    t={t - ts[0]:6.2f}s step {step[i]:5.2f} cm {astep[i]:5.2f} deg  {near}")


if __name__ == "__main__":
    main()
