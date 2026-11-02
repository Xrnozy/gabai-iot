"""Does the visual odometry keep its position when the camera only turns? Rendered room (gabai.tools.vo_bench), known path, stereo noise like the real network's (part of the
depth error sticks to the surface), picture noise, optional motion blur.

py scripts/evaluate_rotation.py [--frames 300] [--seeds 3] [--noise 0.4] [--blur] [--set window_ba=False ...]
  case 'turn'   the camera stays at one point and turns: yaw +-35 deg, pitch +-20 deg, roll +-10 deg (the path the user tests by hand)
  case 'walk'   the camera moves 1.2 m forward and 0.25 m sideways while turning a little (translation must still be tracked)
Per case: how far the estimated position wanders from the true one (rms and largest, cm), the position error at the end, the rotation error rms (deg), frames lost, the largest
single-frame jump of the estimated position (cm), and the number of frames where it jumped more than 5 cm.
KeyframeVO is built with the options of config/pipeline.toml unless --set overrides them (name=value, python literals).
"""
import argparse
import ast

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.kfvo import KeyframeVO
from gabai.sim import look_pose
from gabai.tools import vo_bench as V

PIPELINE_OPTS = dict(predict=True, refine=True, relocalize=True, window_ba=True, ba_every=3, ba_window=6, ba_max_landmarks=250, loop_closure=True, loop_apply=True, loop_radius_m=1.0,
                     loop_min_path_m=2.0, loop_appearance=True, loop_every=2, loop_max_drift_m=2.0, fuse_depth=True)


def path_turn(n, seed=0):
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, 3)
    Rs, ts = [], []
    for i in range(n):
        s = i / max(n - 1, 1)
        yaw, pitch, roll = 35 * np.sin(2 * np.pi * 1.5 * s + ph[0]), 20 * np.sin(2 * np.pi * 2.0 * s + ph[1]), 10 * np.sin(2 * np.pi * 1.0 * s + ph[2])
        Rs.append(R.from_euler("YXZ", [yaw, pitch, roll], degrees=True).as_matrix())
        ts.append(np.zeros(3))
    return [Rs[0].T @ r for r in Rs], ts                        # the odometry's world is its first camera: the path starts at the identity


def path_walk(n, seed=0):
    Rs, ts = path_turn(n, seed)
    Rs = [R.from_rotvec(R.from_matrix(r).as_rotvec() * 0.3).as_matrix() for r in Rs]
    ts = [np.array([0.25 * np.sin(2 * np.pi * i / max(n - 1, 1)), 0.0, 1.2 * i / max(n - 1, 1)]) for i in range(n)]
    return Rs, ts


def holes(rng, valid):
    """Stereo rejects most of a real picture (blur, texture-less areas, left/right disagreement): a changing mask that keeps about `valid` of the pixels, in patches."""
    n = cv2.GaussianBlur(rng.standard_normal((V.H, V.W)).astype(np.float32), (0, 0), 9.0)
    return n > np.quantile(n, 1.0 - valid)


def run(path, opts, n, seed, noise, blur, valid=1.0, step=1):
    rng = np.random.default_rng(seed)
    Rs, ts = path(n * step, seed)
    Rs, ts = Rs[::step], ts[::step]                                  # step > 1: the camera turns `step` times farther between the frames the odometry gets (a link that drops frames)
    tex = V.texture(seed)
    vo = KeyframeVO(V.F, V.F, V.W / 2, V.H / 2, **opts)
    vo.set_baseline(V.B)
    trusted = np.ones((V.H, V.W), bool)
    R0w, t0w = look_pose(np.zeros(3))
    est, rot_err, lost = [], [], 0
    for i, (Rt, tt) in enumerate(zip(Rs, ts)):
        img, disp = V.render_room(tex, Rt, tt, noise_disp=noise, rng=rng)
        if blur and i > 0:
            ang = np.arccos(np.clip((np.trace(Rs[i - 1].T @ Rt) - 1) / 2, -1, 1))
            img = V.motion_blur(img, V.F * ang)
        disp = np.where(np.isfinite(disp), disp, 0.0).astype(np.float32)
        ok_mask = trusted & (disp > 0.5) & (holes(rng, valid) if valid < 1.0 else True)
        Rw, tw, info = vo.step(img, disp, ok_mask)
        lost += not info["ok"]
        est.append(R0w.T @ (tw - t0w))
        rot_err.append(np.degrees(np.arccos(np.clip((np.trace((R0w.T @ Rw).T @ Rt) - 1) / 2, -1, 1))))
    est, ts = np.array(est), np.array(ts)
    err = np.linalg.norm(est - ts, axis=1) * 100
    jump = np.linalg.norm(np.diff(est, axis=0), axis=1) * 100
    return dict(rms=float(np.sqrt((err ** 2).mean())), max=float(err.max()), end=float(err[-1]), rot=float(np.sqrt(np.mean(np.square(rot_err)))), lost=lost, jump=float(jump.max()), jumps=int((jump > 5).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--blur", action="store_true")
    ap.add_argument("--valid", type=float, default=1.0, help="share of pixels with usable depth")
    ap.add_argument("--step", type=int, default=1, help="use every n-th frame of the path")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--cases", nargs="+", default=["turn", "walk"])
    a = ap.parse_args()
    opts = dict(PIPELINE_OPTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        opts[k] = ast.literal_eval(v)
    for name in a.cases:
        rows = [run({"turn": path_turn, "walk": path_walk}[name], opts, a.frames, s, a.noise, a.blur, a.valid, a.step) for s in range(a.seeds)]
        m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        print(f"{name:5s} position error rms {m['rms']:5.1f} cm, max {m['max']:5.1f}, end {m['end']:5.1f} | rotation error rms {m['rot']:.2f} deg | lost {m['lost']:.1f} | largest jump {m['jump']:.1f} cm, jumps >5 cm: {m['jumps']:.1f}", flush=True)


if __name__ == "__main__":
    main()
