"""M2 acceptance: RTAB-Map odometry over a recording, with and without IMU help. No depth model, no TSDF: only the odometry and the map graph.

  py -m mapper.odom_eval data/recordings/walk_20261102_230115 --calib data/stereo_calib_charuco_45.npz [--clip 0 860]

Runs (each a fresh bridge process, frames sent one at a time as rtabmap_lab/calib_lab/runner.py does):
  baseline   rtabmap_lab bridge, pictures only
  gyro       rtabmap_lab bridge, gyro turns the pose at odometry restarts
  guess      mapper_bridge --guess 1: gyro rotation + last translation as the relative motion guess given to Odometry::process
  imu        mapper_bridge --imu 1: the gyro's attitude goes into RTAB-Map's odometry as IMU data (motion guess) + the same restart
Reports lost pairs, restarts, mean inliers, path length, start-end gap, and the heading error: the angle between the rotation the odometry gives from one tracked pair to the next and
the recorded gyro's (the gyro agrees with good visual odometry to 0.9 deg median, docs/room-reconstruction.md), median / 90th percentile / sum over the walk.
"""
import argparse
import os

import cv2
import numpy as np

from rtabmap_lab import live as livemod
from rtabmap_lab.bridge_client import Bridge
from rtabmap_lab.calib_lab import params as P
from rtabmap_lab.calib_lab.recording import Recording
from rtabmap_lab.tsdf import OPTICAL

NEW_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "mapper_bridge.exe")


def run(rec, rect, name, exe, gyro, extra, clip, port=5710, lost_reset=None):
    ns = livemod.add_args(__import__("argparse").ArgumentParser()).parse_args([])
    cal = rect.camera(flip=rec.flip, k=rec.k)
    w, h = cal["size"]
    kw = {} if exe is None else {"exe": exe}
    b = Bridge(db="", port=port, config=ns.config, extra=extra, **kw)
    b.init(cal["f"], cal["f"], cal["cx"], cal["cy"], cal["B"], w, h, ns.cloud_stride, ns.min_range, ns.max_range, ns.disparity_every, ns.lost_reset if lost_reset is None else lost_reset, ns.max_depth_std)
    st = {"lost": 0, "frames": 0, "restarts": 0, "inliers": [], "poses": {}, "lost_ids": []}
    last_sent = None
    def handle(m):
        if m["type"] != "status":
            return
        st["frames"] += 1; st["lost"] += int(m["lost"]); st["lost_ids"] += [m["id"]] if m["lost"] else []; st["restarts"] = m["restarts"]
        if not m["lost"]:
            st["inliers"].append(m["inliers"])
        if m["odom"] is not None and not m["lost"]:
            st["poses"][m["id"]] = m["odom"]
    for idx in range(*clip):
        left, right = rec.prepare(idx, rect, shutter=None)
        b.wait_idle()
        for m in b.poll():
            handle(m)
        g = rec.gyro_rotvec(last_sent, idx) if (gyro and last_sent is not None) else None
        if b.send_frame(idx, float(rec.t[idx] - rec.t[0]), left, cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), g):
            last_sent = idx
    b.wait_idle()
    for m in b.poll():
        handle(m)
    skipped = b.skipped
    b.close(save=False)
    ids = sorted(st["poses"])
    err = []
    for a, c in zip(ids[:-1], ids[1:]):
        gv = rec.gyro_rotvec(a, c)
        if gv is None:
            continue
        Ta, Tc = st["poses"][a] @ OPTICAL, st["poses"][c] @ OPTICAL
        Ro = (np.linalg.inv(Ta) @ Tc)[:3, :3]
        err.append(np.degrees(np.linalg.norm(cv2.Rodrigues(cv2.Rodrigues(gv)[0].T @ Ro)[0])))
    abs_err = []                                                                 # orientation of the odometry against the gyro since the first tracked frame (judge only, never an input)
    for c in ids[1:]:
        gv = rec.gyro_rotvec(ids[0], c)
        if gv is None:
            continue
        Ro = (np.linalg.inv(st["poses"][ids[0]] @ OPTICAL) @ (st["poses"][c] @ OPTICAL))[:3, :3]
        abs_err.append(np.degrees(np.linalg.norm(cv2.Rodrigues(cv2.Rodrigues(gv)[0].T @ Ro)[0])))
    xyz = np.array([st["poses"][i][:3, 3] for i in ids]) if ids else np.zeros((1, 3))
    path = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0
    return {"lost_ids": st["lost_ids"], "name": name, "frames": st["frames"], "skipped": skipped, "lost": st["lost"], "restarts": st["restarts"], "inliers": float(np.mean(st["inliers"])) if st["inliers"] else 0.0,
            "path": path, "gap": float(np.linalg.norm(xyz[-1] - xyz[0])), "err_med": float(np.median(err)) if err else float("nan"),
            "err_p90": float(np.percentile(err, 90)) if err else float("nan"), "err_sum": float(np.sum(err)),
            "abs_med": float(np.median(abs_err)) if abs_err else float("nan"), "abs_p90": float(np.percentile(abs_err, 90)) if abs_err else float("nan"), "abs_end": float(abs_err[-1]) if abs_err else float("nan")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--calib", default="data/stereo_calib_charuco_45.npz")
    ap.add_argument("--clip", type=int, nargs=2, default=(0, 10 ** 9))
    ap.add_argument("--save-lost", help="write the pair indices the FIRST run lost to this JSON file (for mapper.motion_check --lost)")
    ap.add_argument("--runs", nargs="+", default=["baseline", "gyro", "imu"])
    ap.add_argument("--sweep", nargs="?", const="params", default=None, help="RTAB-Map parameter sweep, pictures only (no gyro in the odometry; the recorded gyro is only the judge of the heading error)")
    a = ap.parse_args(argv)
    base = P.CalibParams.from_npz(a.calib)
    rect = base.rect()
    rec = Recording(a.folder).decide(rect)
    clip = (a.clip[0], min(a.clip[1], rec.n))
    specs = {"baseline": (None, False, ()), "gyro": (None, True, ()), "imu": (NEW_EXE, True, ("--imu", "1")), "guess": (NEW_EXE, True, ("--guess", "1")), "both": (NEW_EXE, True, ("--imu", "1", "--guess", "1"))}
    if a.sweep:
        sweep = {"never reset": (("--Odom/ResetCountdown", "0"), 10 ** 6), "reset 15": (("--Odom/ResetCountdown", "15"), 10 ** 6), "reset 40": (("--Odom/ResetCountdown", "40"), 10 ** 6),
                 "reset 3": (("--Odom/ResetCountdown", "3"), 10 ** 6)} if a.sweep == "reset" else {
            "never, no guess window": (("--Odom/ResetCountdown", "0", "--Vis/CorGuessWinSize", "0"), 10 ** 6),
            "never, no guess at all": (("--Odom/ResetCountdown", "0", "--Vis/CorGuessWinSize", "0", "--Odom/GuessMotion", "false"), 10 ** 6),
            "never, no guess, 4000 map": (("--Odom/ResetCountdown", "0", "--Vis/CorGuessWinSize", "0", "--Odom/GuessMotion", "false", "--OdomF2M/MaxSize", "4000"), 10 ** 6),
            "never, no guess, inl 8": (("--Odom/ResetCountdown", "0", "--Vis/CorGuessWinSize", "0", "--Odom/GuessMotion", "false", "--Vis/MinInliers", "8"), 10 ** 6)} if a.sweep == "recover" else {"default": ((), None), "reset1": (("--Odom/ResetCountdown", "1"), 10 ** 6), "reset3": (("--Odom/ResetCountdown", "3"), 10 ** 6),
                 "reset3+noguesswin": (("--Odom/ResetCountdown", "3", "--Vis/CorGuessWinSize", "0"), 10 ** 6),
                 "sift+reset3": (("--Odom/ResetCountdown", "3", "--Vis/FeatureType", "1"), 10 ** 6),
                 "gftt-orb+reset3": (("--Odom/ResetCountdown", "3", "--Vis/FeatureType", "8"), 10 ** 6),
                 "reset3+inl8": (("--Odom/ResetCountdown", "3", "--Vis/MinInliers", "8"), 10 ** 6),
                 "reset3+f3000": (("--Odom/ResetCountdown", "3", "--Vis/MaxFeatures", "3000"), 10 ** 6)}
        rows = [run(rec, rect, n, None, False, ex, clip, lost_reset=lr) for n, (ex, lr) in sweep.items()]
    else:
        rows = [run(rec, rect, n, *specs[n], clip) for n in a.runs]
    if a.save_lost:
        import json
        json.dump(rows[0]["lost_ids"], open(a.save_lost, "w"))
    print(f"\n{rec.folder}  pairs {clip[0]}..{clip[1]}  flip={rec.flip} swapped={rec.swapped}")
    print(f"{'run':10s} {'sent':>5s} {'skip':>5s} {'lost':>5s} {'restarts':>8s} {'inliers':>7s} {'path m':>7s} {'gap m':>6s} {'head err med':>12s} {'p90':>6s} {'sum deg':>8s}")
    print("absolute orientation error vs gyro since the first tracked frame (deg): median / p90 / at the end")
    for r in rows:
        print(f"  {r['name']:10s} {r['abs_med']:6.1f} {r['abs_p90']:6.1f} {r['abs_end']:6.1f}")
    for r in rows:
        print(f"{r['name']:10s} {r['frames']:5d} {r['skipped']:5d} {r['lost']:5d} {r['restarts']:8d} {r['inliers']:7.0f} {r['path']:7.2f} {r['gap']:6.2f} {r['err_med']:11.2f}° {r['err_p90']:5.2f}° {r['err_sum']:8.0f}")


if __name__ == "__main__":
    main()
