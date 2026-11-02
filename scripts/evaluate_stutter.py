"""How much does a stuttering link hurt the stereo_imu pose? Plays a record_pairs folder (with imu.csv) through the LIVE code path in virtual time (no sockets, no waiting): the
pairs and IMU samples reach the pipeline exactly as gabai.tools.fake_pi would send them, with the 'log' stutter or without.

py scripts/evaluate_stutter.py data/recordings/vio_20261101_014854 [--model s] [--stutter log|none] [--ref ref.npz] [--save out.npz] [--set tracking.gyro_prior=false ...]

Run once with --stutter none --save ref.npz (every recorded pair, the best this algorithm can do), then with --stutter log --ref ref.npz. The reference is NOT ground truth: it is the same
estimator without stutter, so the numbers say how much the stutter costs, not how accurate the pose is. Reported, per run:
  frames / ok / held                 pairs the pipeline got, poses it trusted, poses it held
  ATE rms / max (cm)                 position error against the reference after the best rigid fit (rotation + shift) over the frames both runs trusted
  rotation step error median / p90   per pair of consecutive trusted frames: angle between the estimated rotation change and the reference's (degrees)
  gyro step error median / p90       same against the raw gyro integrated over that interval (independent of the reference; bias taken from the quietest 1 s)
"""
import argparse
import ast
import csv
import json
import os
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from evaluate_still import Log                                              # noqa: E402  a Viz that records instead of drawing

from gabai.pipeline import sources                                           # noqa: E402
from gabai.pipeline.config import PipelineConfig                             # noqa: E402
from gabai.pipeline.run import run                                           # noqa: E402
from gabai.tools.fake_pi import LOG_PACE_S, LOG_STUTTER, stutter_times   # noqa: E402


def load_recording(folder):
    import cv2
    meta = json.load(open(os.path.join(folder, "meta.json")))
    rows = list(csv.DictReader(open(os.path.join(folder, "pairs.csv"))))
    tc = np.array([float(r["t_capture"]) for r in rows])
    dts = [None if r["dt_ms"] == "" else float(r["dt_ms"]) for r in rows]
    imu = np.loadtxt(os.path.join(folder, "imu.csv"), delimiter=",", skiprows=1)
    return meta, tc, dts, imu, cv2.VideoCapture(os.path.join(folder, "left.avi")), cv2.VideoCapture(os.path.join(folder, "right.avi"))


def send_schedule(tc, stutter):
    """Recorded pair index sent at each step."""
    rel = tc - tc[0]
    if stutter == "none":
        return list(range(len(tc)))
    times = stutter_times(LOG_STUTTER, 1.0 / LOG_PACE_S, float(rel[-1]))
    idx = [int(np.searchsorted(rel, s)) for s in times]
    return [i for k, i in enumerate(idx) if i < len(rel) and (k == 0 or i != idx[k - 1])]


def make_source(folder, stutter, sent_log):
    meta, tc, dts, imu, cap_l, cap_r = load_recording(folder)
    order = send_schedule(tc, stutter)

    def live_source(cfg, rect, delay_s, k=1.0):
        pos, last, last_imu_t, buf = 0, None, tc[0] - 1.0, []
        prep = sources.PairPrep(cfg, rect, k, delay_s)
        for n, i in enumerate(order):
            while pos <= i:
                okl, l = cap_l.read()
                okr, r = cap_r.read()
                if not (okl and okr):
                    return
                last, pos = (l, r), pos + 1
            sel = (imu[:, 0] > last_imu_t) & (imu[:, 0] <= tc[i])               # the IMU samples of the stall arrive with the next pair, as the Pi sends them
            last_imu_t = tc[i]
            buf += [(float(t), np.radians(g), np.array(a) * sources.G) for t, g, a in zip(imu[sel, 0], imu[sel, 1:4], imu[sel, 4:7])]
            t = float(tc[i]) - delay_s
            rows, buf = [r_ for r_ in buf if r_[0] <= t], [r_ for r_ in buf if r_[0] > t]
            sent_log.append(i)
            yield prep.frame(n, last[0], last[1], float(tc[i]), dts[i], t, rows, meta.get("pi_health_at_end") or {})

    return live_source, tc, imu


def umeyama(P, Q):
    """Rotation and shift that best map P onto Q (no scale)."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((Q - mq).T @ (P - mp))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    Rm = U @ S @ Vt
    return Rm, mq - Rm @ mp


def gyro_rotation(imu, t0, t1, bias):
    sel = (imu[:, 0] > t0) & (imu[:, 0] <= t1)
    r = R.identity()
    tt = np.r_[t0, imu[sel, 0]]
    for k in range(sel.sum()):
        r = r * R.from_rotvec(np.radians(imu[sel][k, 1:4] - bias) * (tt[k + 1] - tt[k]))
    return r


def run_once(folder, model, stutter, overrides):
    sent_log, got, calls = [], [], [0]
    src, tc, imu = make_source(folder, stutter, sent_log)
    from gabai.pipeline import vio
    orig_push, orig_live = vio.VisualInertialProvider.push, sources.live_source
    sources.live_source = src

    def wrap(self, *a, **k):
        out = orig_push(self, *a, **k)
        calls[0] += 1
        for p in out:
            got.append((calls[0] - 1, p.ok, np.array(p.R_wc), np.array(p.t_wc), dict(p.info)))
        return out
    vio.VisualInertialProvider.push = wrap
    ov = {"pipeline.mode": "stereo_imu", "source.kind": "live", "source.max_frames": 0, "stereo.model": model, "viz.viewer": False, "hands.enabled": False, "detector.enabled": False,
          "perf.async_viz": False, "perf.staged": False, "perf.low_priority": False, "stereo.mono": "", "map.tsdf": False}
    ov.update(overrides)
    try:
        s = run(PipelineConfig.load("config/pipeline.toml", ov), viz_factory=Log, log_dir=tempfile.mkdtemp(), quiet=True)
    finally:
        vio.VisualInertialProvider.push, sources.live_source = orig_push, orig_live
    idx = np.array([sent_log[g[0]] for g in got])                           # the recorded pair each pose belongs to (the first calls return no pose while the IMU warms up)
    return dict(idx=idx, ok=np.array([g[1] for g in got]), R=np.array([g[2] for g in got]), t=np.array([g[3] for g in got]), tc=tc, imu=imu, summary=s,
                reasons=[g[4].get('reason', '') for g in got], prior=np.array([g[4].get("prior_disagreement_deg", np.nan) if g[4].get("prior_disagreement_deg") is not None else np.nan for g in got]))


def report(a, ref):
    n, ok = len(a["idx"]), a["ok"]
    print(f"frames {n}  ok {int(ok.sum())}  held {int((~ok).sum())}  (pairs in the recording {len(a['tc'])})")
    from collections import Counter
    why = Counter(r.split(' (')[0][:70] for r, k in zip(a["reasons"], ok) if not k)
    print("held because:", dict(why.most_common(4)))
    g = a["imu"]
    w = int(1.0 * (len(g) - 1) / (g[-1, 0] - g[0, 0]))                      # gyro bias = the mean over the quietest 1 s of the recording (the rig may never rest at the start)
    quiet = min(range(0, len(g) - w, max(w // 4, 1)), key=lambda k: np.linalg.norm(g[k:k + w, 1:4].mean(axis=0)) + np.linalg.norm(g[k:k + w, 1:4].std(axis=0)))
    bias = g[quiet:quiet + w, 1:4].mean(axis=0)
    print(f"gyro bias taken from the quietest second: {np.round(bias, 2)} deg/s")
    gy = []
    sel = np.where(ok)[0]
    for x, y in zip(sel[:-1], sel[1:]):
        est = R.from_matrix(a["R"][x]).inv() * R.from_matrix(a["R"][y])
        g = gyro_rotation(a["imu"], a["tc"][a["idx"][x]], a["tc"][a["idx"][y]], bias)
        gy.append(np.degrees((est.inv() * R_ic_rel(g)).magnitude()))
    if gy:
        print(f"gyro step error (camera-axis rotation vs raw gyro): median {np.median(gy):.1f}  p90 {np.percentile(gy, 90):.1f} deg   [{len(gy)} steps]")
    if ref is None:
        return
    common = sorted(set(a["idx"][ok]) & set(ref["idx"][ref["ok"]]))
    if len(common) < 5:
        print("too few common trusted frames")
        return
    ia = {int(i): k for k, i in enumerate(a["idx"])}
    ir = {int(i): k for k, i in enumerate(ref["idx"])}
    P = np.array([a["t"][ia[i]] for i in common])
    Q = np.array([ref["t"][ir[i]] for i in common])
    Rm, tm = umeyama(P, Q)
    err = np.linalg.norm((P @ Rm.T + tm) - Q, axis=1) * 100
    rot = []
    for x, y in zip(common[:-1], common[1:]):
        ea = R.from_matrix(a["R"][ia[x]]).inv() * R.from_matrix(a["R"][ia[y]])
        er = R.from_matrix(ref["R"][ir[x]]).inv() * R.from_matrix(ref["R"][ir[y]])
        rot.append(np.degrees((ea.inv() * er).magnitude()))
    print(f"vs reference over {len(common)} common frames: ATE rms {np.sqrt((err ** 2).mean()):.1f} cm, max {err.max():.1f} cm | rotation step error median {np.median(rot):.2f} p90 {np.percentile(rot, 90):.2f} deg")


R_IC = None


def R_ic_rel(g):
    """The gyro's rotation (IMU axes) as a rotation of the camera axes."""
    M = R.from_matrix(R_IC.T @ g.as_matrix() @ R_IC)
    return M


def main():
    global R_IC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="s")
    ap.add_argument("--stutter", choices=["log", "none"], default="log")
    ap.add_argument("--ref"), ap.add_argument("--save")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--baseline", action="store_true", help="switch off the gyro-across-stalls handling (the behaviour before it existed)")
    a = ap.parse_args()
    ov = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            ov[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            ov[k] = v
    cfg = PipelineConfig.load("config/pipeline.toml", {})
    from gabai.slam.euroc import load_imu_extrinsics
    R_IC = load_imu_extrinsics(cfg.source.imu_sync)["R_ic"]
    from gabai.pipeline import vio
    vio.USE_GYRO_ACROSS_STALLS = not a.baseline
    res = run_once(a.folder, a.model, a.stutter, ov)
    ref = None
    if a.ref:
        z = np.load(a.ref)
        ref = {k: z[k] for k in z.files}
    report(res, ref)
    if a.save:
        np.savez(a.save, idx=res["idx"], ok=res["ok"], R=res["R"], t=res["t"])


if __name__ == "__main__":
    main()
