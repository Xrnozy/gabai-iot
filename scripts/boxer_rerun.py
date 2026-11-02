"""Watch Boxer's 3D object boxes in Rerun, on top of the normal pipeline view (camera, trajectory, map).

py scripts/boxer_rerun.py data/recordings/<folder> [--model s] [--stutter none|log] [--save out.rrd] [--no-viewer]

Plays the recording through the live pipeline (stereo_imu), and per frame runs OWLv2 + BoxerNet (third_party/boxer, see scripts/boxer_landmarks.py). In the 3D view:
  world/boxer/current     yellow boxes: what Boxer sees in this frame (class and confidence)
  world/boxer/landmarks   one coloured box per object seen at least --min-looks times within --max-range, at the median of its looks, labelled  #id class  looks  scatter cm
  world/boxer/looks       every look of those objects as a dot in the colour of its object: a tight cloud = a stable object, a smear = drift or an object that moved
Objects are matched across frames by class (tv = monitor) and distance (--gate m) to the object's median; this is the test of scripts/object_landmarks.py, drawn. No correction is
applied to the trajectory. A frame whose pose the pipeline did not trust is not drawn.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import rerun as rr                                                            # noqa: E402
from scipy.spatial.transform import Rotation                                  # noqa: E402

import boxer_landmarks as bl                                                  # noqa: E402  (puts third_party/boxer on the path)
from evaluate_stutter import make_source                                      # noqa: E402

from gabai.objects3d import Objects3D                                         # noqa: E402
from gabai.pipeline import sources                                            # noqa: E402
from gabai.pipeline.config import PipelineConfig                              # noqa: E402
from gabai.pipeline.run import run                                            # noqa: E402
from gabai.pipeline.viz import Viz                                            # noqa: E402

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128),
           (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128)]


class BoxerViz(Viz):
    objects, a = None, None

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.shot, self.pending, self.sec = None, None, 0.0

    def at(self, idx, t):
        super().at(idx, t)
        if self.pending is not None:                                          # the frame before was trusted (no "pose held" followed its pose): hand it over now
            BoxerViz.objects.submit(*self.pending)
            self.pending = None
        self.sec = t

    def stereo(self, left, right, disp, trusted, cal, *rest):
        super().stereo(left, right, disp, trusted, cal, *rest)
        self.shot = (left, disp, trusted, cal)

    def event(self, text, level="INFO"):
        super().event(text, level)
        if "pose held" in text:                                               # the pose of this frame is not trusted: do not lift it
            self.pending = None

    def pose(self, R_wc, t_wc, trajectory, left_bgr=None):
        super().pose(R_wc, t_wc, trajectory, left_bgr)
        if self.shot is not None:
            left, disp, trusted, cal = self.shot
            self.pending, self.shot = (self.sec, left, disp, trusted, cal, R_wc, t_wc), None
        snap = BoxerViz.objects.poll()                                        # whatever the worker has finished; never waits
        if snap is None:
            return
        dets, marks = snap["current"], snap["landmarks"]
        if dets:
            q = Rotation.from_matrix(np.array([d["R_obj"] for d in dets])).as_quat()
            self._log("world/boxer/current", rr.Boxes3D(centers=[d["p_w"] for d in dets], half_sizes=[d["dims"] / 2 for d in dets], quaternions=[rr.Quaternion(xyzw=x) for x in q],
                                                        labels=[f"{d['label']} {d['conf']:.2f}" for d in dets], colors=[255, 220, 0], radii=0.004))
        else:
            self._clear("world/boxer/current")
        if marks:
            q = Rotation.from_matrix(np.array([m["R"] for m in marks])).as_quat()
            col = [PALETTE[m["id"] % len(PALETTE)] for m in marks]
            self._log("world/boxer/landmarks", rr.Boxes3D(centers=[m["center"] for m in marks], half_sizes=[m["dims"] / 2 for m in marks], quaternions=[rr.Quaternion(xyzw=x) for x in q],
                                                          labels=[f"#{m['id']} {m['label']}  {m['looks']} looks  {m['sigma_cm']:.0f} cm" for m in marks], colors=col, radii=0.006))
            self._log("world/boxer/looks", rr.Points3D(np.concatenate([m["recent"] for m in marks]), colors=np.concatenate([[col[i]] * len(m["recent"]) for i, m in enumerate(marks)]),
                                                       radii=rr.Radius.ui_points(2.5)))
        rr.log("boxer/detections", rr.Scalars(float(len(dets))))
        rr.log("boxer/landmarks", rr.Scalars(float(len(marks))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="s")
    ap.add_argument("--stutter", choices=["log", "none"], default="none")
    ap.add_argument("--labels", default=",".join(bl.LABELS))
    ap.add_argument("--thresh2d", type=float, default=0.25), ap.add_argument("--thresh3d", type=float, default=0.5), ap.add_argument("--detector-hw", type=int, default=960)
    ap.add_argument("--gate", type=float, default=0.25), ap.add_argument("--max-range", type=float, default=2.0), ap.add_argument("--min-looks", type=int, default=5)
    ap.add_argument("--every", type=float, default=0.3, help="seconds between frames handed to Boxer (it runs on its own thread, one frame at a time)")
    ap.add_argument("--save", default="", help="also write the recording to this .rrd file"), ap.add_argument("--no-viewer", action="store_true"), ap.add_argument("--frames", type=int, default=0, help="stop after this many frames (0 = all)")
    a = ap.parse_args()
    BoxerViz.a = a
    bl.preload()                                                               # before the worker thread starts: see gabai/boxer_lift.preload
    BoxerViz.objects = Objects3D(lambda: bl.make_lifter([s.strip() for s in a.labels.split(",")], a.thresh2d, a.thresh3d, a.detector_hw), a.every, a.max_range, a.gate, a.min_looks)
    sent = []
    src, _, _ = make_source(a.folder, a.stutter, sent)
    sources.live_source = src
    ov = {"pipeline.mode": "stereo_imu", "source.kind": "live", "source.max_frames": a.frames, "stereo.model": a.model, "viz.viewer": not a.no_viewer, "viz.save": a.save, "hands.enabled": False,
          "detector.enabled": False, "perf.async_viz": False, "perf.staged": False, "perf.low_priority": False, "perf.viz_every": 1, "stereo.mono": "", "map.tsdf": False}
    import tempfile
    run(PipelineConfig.load("config/pipeline.toml", ov), viz_factory=BoxerViz, log_dir=tempfile.mkdtemp(), quiet=True)


if __name__ == "__main__":
    main()
