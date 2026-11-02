"""The object-landmark consistency test of scripts/object_landmarks.py, with Boxer (Meta / Project Aria, third_party/boxer) as the 3D object detector instead of YOLO + a depth median.

py scripts/boxer_landmarks.py data/recordings/<folder> [--model s] [--labels "cabinet,table,sofa"] [--save boxer_objs.npz] [--load boxer_objs.npz]

Per frame: OWLv2 finds 2D boxes for the text labels; BoxerNet lifts each to a metric, gravity-aligned 3D box from the left picture, the stereo depth (as sparse depth points) and the
pose of that frame. The box CENTRE (of the whole object, also its hidden part) is what is tracked, which is what the YOLO test could not do: a partial view shifted the median depth.
Same report as scripts/object_landmarks.py (scatter in one visit, visit shift, pair distances). It measures map consistency, not ground truth; read that script's docstring.
Needs the weights in third_party/boxer/ckpts (scripts/download_ckpts.sh) and a world with z up, which stereo_imu gives.
"""
import argparse
import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())



from gabai.boxer_lift import LABELS, make_lifter, preload                             # noqa: E402
import object_landmarks as ol                                                 # noqa: E402

def make_measure(*args):
    lift = make_lifter(*args)

    def measure(left, disp, trusted, cal, R_wc, t_wc):
        return [(d["label"], d["conf"], 0, R_wc.T @ (d["p_w"] - t_wc), d["p_w"]) for d in lift(left, disp, trusted, cal, R_wc, t_wc)]
    return measure


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="s")
    ap.add_argument("--stutter", choices=["log", "none"], default="none")
    ap.add_argument("--labels", default=",".join(LABELS), help="comma separated text prompts for OWLv2")
    ap.add_argument("--thresh2d", type=float, default=0.25), ap.add_argument("--thresh3d", type=float, default=0.5), ap.add_argument("--detector-hw", type=int, default=960)
    ap.add_argument("--gate", type=float, default=0.25), ap.add_argument("--gap", type=float, default=3.0), ap.add_argument("--min-obs", type=int, default=4)
    ap.add_argument("--max-range", type=float, default=2.0, help="ignore looks farther than this many metres (0 = all): stereo depth noise at 41.7 mm baseline is about 10 cm at 3 m"), ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--plot", help="write a picture of the object positions and their drift to this .png")
    ap.add_argument("--save"), ap.add_argument("--load")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    ov = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            ov[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            ov[k] = v
    if a.load:
        z = np.load(a.load, allow_pickle=False)
        o = {k: z[k] for k in z.files}
    else:
        o = ol.capture(a, ov, measure=make_measure([s.strip() for s in a.labels.split(",")], a.thresh2d, a.thresh3d, a.detector_hw))
        if a.save:
            np.savez(a.save, **o)
    ol.analyse(o, a)


if __name__ == "__main__":
    main()
