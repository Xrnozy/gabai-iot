"""How well does the tracker hold on when the video is blurry, late or stuttering? Plays a record_pairs folder (with imu.csv) through the LIVE code path in virtual time and reports tracking health.

py scripts/trackeval.py data/recordings/vio_20261101_014854 [--stutter log|none] [--blur 0|1|2] [--model s] [--save poses.npz] [--set key=value ...]
py scripts/rot_drift_check.py data/eval/gray_full_rate.npz poses.npz          (then: rotation error of those poses against the pictures themselves, frames 4-60 s apart)

--stutter log   the pairs the starved Pi really delivered (gaps up to 3.8 s, the IMU of a stall arriving with the next pair): late and missing frames
--blur M        extra motion blur on both pictures, M times what the camera moved in a 33 ms exposure: image speed = f * gyro rate, a line kernel along it. 0 = the recording as it is (it is
                already blurred in places: auto exposure in a dim room). The reference for the drift check stays the sharp pictures in data/eval/gray_full_rate.npz.
Reported: poses trusted / held and why; tracked features and RANSAC inliers (median, 10th percentile, frames under 40); time of one tracker call; tracker counters.
Not ground truth: inliers say how well the tracker is fed, the drift check says how far the rotation went wrong against the images.
"""
import argparse
import ast
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from evaluate_still import Log                                              # noqa: E402
from evaluate_stutter import make_source                                     # noqa: E402

from gabai.pipeline import sources, vio                                      # noqa: E402
from gabai.pipeline.config import PipelineConfig                             # noqa: E402
from gabai.pipeline.run import run                                           # noqa: E402
from gabai.slam.euroc import load_imu_extrinsics                             # noqa: E402

EXPOSURE_S = 0.033


def line_kernel(vx, vy, length):
    n = int(max(length, 1.0)) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    d = np.array([vx, vy]) / (np.hypot(vx, vy) + 1e-9)
    cv2.line(k, (int(round(c - d[0] * c)), int(round(c - d[1] * c))), (int(round(c + d[0] * c)), int(round(c + d[1] * c))), 1.0, 1)
    return k / max(k.sum(), 1e-9)


def blur_frame(fr, mult, R_ic, f):
    if mult <= 0 or not fr.imu:
        return
    w = np.mean([np.asarray(r[1]) for r in fr.imu[-4:]], axis=0)                 # rad/s, IMU axes
    wc = R_ic.T @ w                                                              # camera axes
    vx, vy = f * wc[1], f * wc[0]                                                # yaw moves the picture sideways, pitch up and down (px/s)
    length = mult * EXPOSURE_S * float(np.hypot(vx, vy))
    if length >= 2.0:
        k = line_kernel(vx, vy, length)
        fr.left, fr.right = cv2.filter2D(fr.left, -1, k), cv2.filter2D(fr.right, -1, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="s")
    ap.add_argument("--stutter", choices=["log", "none"], default="log")
    ap.add_argument("--blur", type=float, default=0.0)
    ap.add_argument("--save")
    ap.add_argument("--const", action="append", default=[], help="override a module constant for this run, e.g. kfvo.FIX_KEEP_FRAC=0 or vio.GRAV_MAX_DPS=0")
    ap.add_argument("--old", action="store_true", help="switch off the blur/stall handling of gabai/kfvo.py (wide retry, keyframe deferral, relocalisation gyro gate): the tracker as it was")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    ov = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            ov[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            ov[k] = v
    if a.old:
        from gabai import kfvo
        kfvo.WIDE_RETRY, kfvo.KF_DEFER_MAX, kfvo.RELOC_GYRO_GATE_DEG = False, 0, 1e9
    import importlib
    for c in a.const:
        name, val = c.split("=", 1)
        mod, attr = name.split(".")
        setattr(importlib.import_module({"kfvo": "gabai.kfvo", "vio": "gabai.pipeline.vio"}[mod]), attr, float(val))
    cfg0 = PipelineConfig.load("config/pipeline.toml", {})
    R_ic = load_imu_extrinsics(cfg0.source.imu_sync)["R_ic"]
    sent = []
    base_src, tc, _ = make_source(a.folder, a.stutter, sent)

    def live(cfg, rect, delay_s, k=1.0):
        for fr in base_src(cfg, rect, delay_s, k):
            blur_frame(fr, a.blur, R_ic, fr.cal["f"])
            yield fr
    sources.live_source = live
    rec, prov, calls, ms = [], [], [0], []
    orig_push, orig_init = vio.VisualInertialProvider.push, vio.VisualInertialProvider.__init__

    def pinit(self, *x, **k):
        orig_init(self, *x, **k)
        prov.append(self)

    def push(self, *x, **k):
        t0 = time.perf_counter()
        out = orig_push(self, *x, **k)
        ms.append((time.perf_counter() - t0) * 1000)
        for p in out:
            rec.append((calls[0], p.ok, np.array(p.R_wc), np.array(p.t_wc), dict(p.info)))
        calls[0] += 1
        return out
    vio.VisualInertialProvider.__init__, vio.VisualInertialProvider.push = pinit, push
    base = {"pipeline.mode": "stereo_imu", "source.kind": "live", "source.max_frames": 0, "stereo.model": a.model, "viz.viewer": False, "hands.enabled": False, "detector.enabled": False,
            "perf.async_viz": False, "perf.staged": False, "perf.low_priority": False, "stereo.mono": "", "map.tsdf": False, "objects3d.enabled": False}
    base.update(ov)
    t_run0 = time.perf_counter()
    try:
        run(PipelineConfig.load("config/pipeline.toml", base), viz_factory=Log, log_dir=__import__("tempfile").mkdtemp(), quiet=True)
    finally:
        vio.VisualInertialProvider.__init__, vio.VisualInertialProvider.push = orig_init, orig_push
    ok = np.array([r[1] for r in rec])
    inl = np.array([r[4]["inliers"] for r in rec])
    trk = np.array([r[4]["tracked"] for r in rec])
    idx = np.array([sent[r[0]] for r in rec])
    why = Counter(r[4]["reason"].split(" (")[0][:60] for r in rec if not r[1])
    print(f"stutter={a.stutter} blur x{a.blur:g}: frames {len(rec)}  trusted {int(ok.sum())}  held {int((~ok).sum())}   held because: {dict(why.most_common(3))}")
    if ok.any():
        print(f"features tracked median {np.median(trk[ok]):.0f}   RANSAC inliers median {np.median(inl[ok]):.0f}  10th percentile {np.percentile(inl[ok], 10):.0f}  frames under 40 inliers: {int((inl[ok] < 40).sum())}")
    b = np.array([r[4].get("gyro_bias_dps", np.nan) for r in rec])
    if np.isfinite(b).any():
        print(f"gyro bias in use: median {np.nanmedian(b):.2f} deg/s  max {np.nanmax(b):.2f}  at-rest frames {int(sum(bool(r[4].get('at_rest')) for r in rec))} of {len(rec)}")
    P = np.array([r[3] for r in rec])
    stp = np.linalg.norm(np.diff(P, axis=0), axis=1)
    lv = [r[4].get("loop_version") for r in rec if r[4].get("loop_version") is not None]
    print(f"path: length {stp.sum():.2f} m  extent {np.round(P.max(0) - P.min(0), 2)} m  largest step {stp.max() * 100:.1f} cm  steps over 5 cm: {int((stp > 0.05).sum())}   map re-lays (graph versions): {int(max(lv) - min(lv)) if lv else 0}")
    print("graph:", {k: v for k, v in prov[0].vo.stats.items() if k.startswith("landmark") or k in ("graph_runs",)}, f"  wall {time.perf_counter() - t_run0:.0f} s")
    print(f"tracker call: median {np.median(ms):.0f} ms  p95 {np.percentile(ms, 95):.0f} ms   tracker counters: " + str({k: v for k, v in prov[0].vo.stats.items() if k in ("relocalized", "predicted_ok", "wide_ok", "kf_deferred", "fixrot_used", "free_rot_used")}))
    if a.save:
        info = [r[4] for r in rec]
        np.savez(a.save, idx=idx, ok=ok, R=np.array([r[2] for r in rec]), t=np.array([r[3] for r in rec]), inliers=inl, tracked=trk,
                 prior_dis=np.array([np.nan if i.get("prior_disagreement_deg") is None else i["prior_disagreement_deg"] for i in info]),
                 reloc=np.array([bool(i.get("relocalized")) for i in info]), kf=np.array([bool(i.get("keyframe")) for i in info]), init=np.array([str(i.get("init")) for i in info]),
                 reason=np.array([i["reason"][:70] for i in info]), at_rest=np.array([bool(i.get("at_rest")) for i in info]),
                 bias_dps=np.array([i.get("gyro_bias_dps", np.nan) for i in info]))


if __name__ == "__main__":
    main()
