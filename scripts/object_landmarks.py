"""Do stationary objects stay put in the world the VIO builds? Offline consistency test, no correction is applied.

py scripts/object_landmarks.py data/recordings/vio_20261101_014854 [--model s] [--stutter none|log] [--save objs.npz] [--load objs.npz] [--set key=value ...]

Plays the recording through the live code path (as scripts/evaluate_stutter.py does), runs the YOLO detector on every pair, takes the median 3D point of each static-class
detection from the trusted stereo depth, and puts it in the world with the pose of that frame. Then, per object track (same class, within --gate of the track's running median):
  scatter        spread of the world position inside one visit (a visit = observations less than --gap s apart): the depth/partial-view noise floor of ONE look
  visit shift    distance between the medians of the first and last visit: the drift that the pose chain put into this object (or the object really moved)
  pair distance  distance between two tracks in each visit they share. Pose drift moves both objects together, so this distance should NOT change; if it does, an object moved or
                 its centroid is biased by what is visible. This is how a moved table is told from drift.
This measures map consistency, NOT absolute trajectory accuracy: the objects are placed with the same poses that may be wrong, and a revisit needs the camera to look at the
same object twice with a gap in between. A track seen in one visit says nothing about drift.
"""
import argparse
import ast
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from evaluate_stutter import make_source                                     # noqa: E402

from gabai.detect import UltralyticsDetector                                  # noqa: E402
from gabai.objects3d import ALIAS                                  # noqa: E402
from gabai.pipeline import sources, vio                                       # noqa: E402
from gabai.pipeline.config import PipelineConfig                              # noqa: E402
from gabai.pipeline.run import run                                            # noqa: E402
from gabai.pipeline.viz import Viz                                            # noqa: E402

# COCO classes that normally stay put. The shipped detector (yolo11n-seg, COCO) has no cabinet class.
STATIC = ("chair", "couch", "bed", "dining table", "tv", "refrigerator", "oven", "microwave", "sink", "toilet", "potted plant", "bench",
          "laptop", "keyboard", "book", "mouse", "monitor", "clock", "vase")      # desk objects, for a desk recording
MIN_PIX, MAX_Z, ERODE = 150, 3.5, 4


class Cap(Viz):
    """Stands in for the viewer: keeps the left picture, depth and pose of every frame and turns static-class detections into world points."""
    rows = []                                                                  # (frame, t, label, conf, n_pix, p_cam(3), p_w(3), ok)

    def __init__(self, cal, size, save=None, viewer=False, **kw):
        self.t, self.fr, self.shot, self.n_logged = 0.0, 0, None, 0

    def at(self, idx, t):
        self.fr, self.t = idx, t

    def stereo(self, left, right, disp, trusted, cal, *a):
        self.shot = (left, disp, trusted, cal)

    def pose(self, R_wc, t_wc, traj, left_bgr=None):
        if self.shot is None:
            return
        left, disp, trusted, cal = self.shot
        self.shot = None
        for label, conf, n, pc, pw in Cap.measure(left, disp, trusted, cal, np.asarray(R_wc), np.asarray(t_wc)):
            Cap.rows.append([self.fr, self.t, label, conf, n, *pc, *pw, True])

    measure = None                                                              # set by capture(): (left, disp, trusted, cal, R_wc, t_wc) -> [(label, conf, n, p_cam, p_w)]

    def event(self, text, level="INFO"):
        if "pose held" in text and Cap.rows and Cap.rows[-1][0] == self.fr:
            for r in reversed(Cap.rows):
                if r[0] != self.fr:
                    break
                r[-1] = False

    def __getattr__(self, name):
        return lambda *a, **k: None


def yolo_measure(det):
    def measure(left, disp, trusted, cal, R_wc, t_wc):
        import cv2
        f, cx, cy = float(cal["f"]), float(cal["cx"]), float(cal["cy"])
        ok_d = trusted & np.isfinite(disp) & (disp > 0.5)
        out = []
        for d in det.detect(left):
            if d.label not in STATIC or d.mask is None:
                continue
            m = cv2.erode(d.mask.astype(np.uint8), np.ones((2 * ERODE + 1,) * 2, np.uint8)).astype(bool) & ok_d       # the rim of a mask sees the background
            if m.sum() < MIN_PIX:
                continue
            v, u = np.nonzero(m)
            z = f * float(cal["B"]) / disp[v, u]
            k = z < MAX_Z
            if k.sum() < MIN_PIX:
                continue
            u, v, z = u[k], v[k], z[k]
            pc = np.median(np.c_[(u - cx) * z / f, (v - cy) * z / f, z], axis=0)
            out.append((d.label, d.conf, int(k.sum()), pc, R_wc @ pc + t_wc))
        return out
    return measure


def capture(a, ov, measure=None):
    Cap.rows = []
    Cap.measure = measure or yolo_measure(UltralyticsDetector(conf=a.conf, imgsz=a.imgsz))
    sent = []
    src, _, _ = make_source(a.folder, a.stutter, sent)
    orig = sources.live_source
    sources.live_source = src
    base = {"pipeline.mode": "stereo_imu", "source.kind": "live", "source.max_frames": 0, "stereo.model": a.model, "viz.viewer": False, "hands.enabled": False, "detector.enabled": False,
            "perf.async_viz": False, "perf.staged": False, "perf.low_priority": False, "perf.viz_every": 1, "stereo.mono": "", "map.tsdf": False}
    base.update(ov)
    try:
        run(PipelineConfig.load("config/pipeline.toml", base), viz_factory=Cap, log_dir=tempfile.mkdtemp(), quiet=True)
    finally:
        sources.live_source = orig
    r = Cap.rows
    return dict(frame=np.array([x[0] for x in r]), t=np.array([x[1] for x in r]), label=np.array([x[2] for x in r]), conf=np.array([x[3] for x in r]), n=np.array([x[4] for x in r]),
                p_cam=np.array([x[5:8] for x in r]).reshape(-1, 3), p_w=np.array([x[8:11] for x in r]).reshape(-1, 3), ok=np.array([x[11] for x in r], bool))


def tracks(o, gate):
    """Greedy association: an observation joins the nearest same-class track whose running median is within `gate` metres, else starts one."""
    tr = []
    for i in np.nonzero(o["ok"])[0]:
        best, bd = None, gate
        for k, T in enumerate(tr):
            if ALIAS.get(T["label"], T["label"]) != ALIAS.get(str(o["label"][i]), str(o["label"][i])):
                continue
            d = np.linalg.norm(np.median(o["p_w"][T["obs"]], axis=0) - o["p_w"][i])
            if d < bd:
                best, bd = k, d
        if best is None:
            tr.append(dict(label=str(o["label"][i]), obs=[i]))
        else:
            tr[best]["obs"].append(i)
    return tr


def visits(o, obs, gap):
    t = o["t"][obs]
    cut = np.nonzero(np.diff(t) > gap)[0] + 1
    return [obs[s:e] for s, e in zip(np.r_[0, cut], np.r_[cut, len(obs)])]


def plot(o, tr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    t0 = o["t"].min()
    for n, T in enumerate(tr):
        P, t = o["p_w"][T["obs"]], o["t"][T["obs"]] - t0
        c = plt.cm.tab20(n % 20)
        ax[0].scatter(P[:, 0], P[:, 1], s=8, color=c, alpha=0.5)
        m = np.median(P, axis=0)
        ax[0].annotate(f"{n} {T['label']}", m[:2], fontsize=8, color=c)
        ax[1].plot(t, np.linalg.norm(P - m, axis=1) * 100, ".", ms=3, color=c, label=f"{n} {T['label']}")
    ax[0].set_title("object centres in the world, seen from above (x, y m); each dot is one look"), ax[0].set_aspect("equal"), ax[0].set_xlabel("x m"), ax[0].set_ylabel("y m")
    ax[1].set_title("distance of each look from its object's median position: a slope = drift (or the object moved)"), ax[1].set_xlabel("seconds"), ax[1].set_ylabel("cm")
    ax[1].legend(fontsize=7, ncol=2, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"picture saved: {path}")


def analyse(o, a):
    o = dict(o, ok=o["ok"] & (np.linalg.norm(o["p_cam"], axis=1) <= (getattr(a, "max_range", 0) or 1e9)) & (o["conf"] >= getattr(a, "min_conf", 0)))       # keep only near, confident looks
    ok = o["ok"]
    print(f"detections of static classes: {len(ok)}  with a trusted pose: {int(ok.sum())}  frames with at least one: {len(set(o['frame'][ok]))}")
    tr = [T for T in tracks(o, a.gate) if len(T["obs"]) >= a.min_obs]
    if not tr:
        print("no object was seen often enough: nothing to measure (raise --conf lower, or this recording has no static-class objects in range)")
        return
    if getattr(a, "plot", None):
        plot(o, tr, a.plot)
    print(f"object tracks with >= {a.min_obs} looks: {len(tr)}   (gate {a.gate} m, visit gap {a.gap} s)\n")
    print(f"{'#':>2} {'class':<13}{'looks':>6}{'visits':>7}{'first-last t':>14}{'range m':>9}{'scatter cm (1 visit)':>22}{'visit shift cm':>16}")
    med = []
    for n, T in enumerate(tr):
        vs = visits(o, np.array(T["obs"]), a.gap)
        vm = [np.median(o["p_w"][v], axis=0) for v in vs if len(v) >= 3]
        sc = np.median([np.linalg.norm(o["p_w"][v].std(axis=0)) for v in vs if len(v) >= 3] or [np.nan]) * 100
        shift = np.linalg.norm(vm[-1] - vm[0]) * 100 if len(vm) >= 2 else np.nan
        rng = np.median(np.linalg.norm(o["p_cam"][T["obs"]], axis=1))
        t = o["t"][T["obs"]]
        print(f"{n:>2} {T['label']:<13}{len(T['obs']):>6}{len(vs):>7}{t[0]:>7.1f}-{t[-1]:<6.1f}{rng:>9.2f}{sc:>22.1f}{shift:>16.1f}")
        T["vs"], T["vm"] = vs, vm
        med.append((n, vs))
    print("\npair distances between objects seen together in two or more visits (cm change = an object moved, or a biased centroid; drift alone does not change them):")
    any_pair = False
    for i in range(len(tr)):
        for j in range(i + 1, len(tr)):
            ti, tj = tr[i], tr[j]
            ds = []
            for vi, vj in ((vi, vj) for vi in ti["vs"] for vj in tj["vs"]):
                if len(vi) < 3 or len(vj) < 3:
                    continue
                if abs(np.median(o["t"][vi]) - np.median(o["t"][vj])) < a.gap * 2:        # seen in the same stretch
                    ds.append((np.median(o["t"][vi]), np.linalg.norm(np.median(o["p_w"][vi], axis=0) - np.median(o["p_w"][vj], axis=0))))
            if len(ds) >= 2:
                any_pair = True
                d = [x[1] * 100 for x in ds]
                print(f"  {i}-{j} ({ti['label']}/{tj['label']}): distance {', '.join(f'{x:.0f}' for x in d)} cm over {len(ds)} visits -> change {max(d) - min(d):.1f} cm")
    if not any_pair:
        print("  none: no two objects were seen together in two separate visits")
    n_rev = sum(1 for T in tr if len(T["vm"]) >= 2)
    print(f"\nobjects with a revisit (>= 2 visits): {n_rev} of {len(tr)}. Fewer than 2 such objects means a landmark correction could not be tested on this recording.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="s")
    ap.add_argument("--stutter", choices=["log", "none"], default="none")
    ap.add_argument("--conf", type=float, default=0.3), ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--gate", type=float, default=0.6), ap.add_argument("--gap", type=float, default=3.0), ap.add_argument("--min-obs", type=int, default=5)
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
        o = capture(a, ov)
        if a.save:
            np.savez(a.save, **o)
    analyse(o, a)


if __name__ == "__main__":
    main()
