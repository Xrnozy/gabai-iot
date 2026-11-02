"""Drift benchmark: several laps around the same spot in a rendered room, true path known.

    py -m gabai.tools.slam_bench [--laps 3] [--seeds 2] [--noise 0.4]

Reports for each tracker setting the error of the ONLINE poses (what the system output at the time) and of the path REBUILT after loop corrections (what the
map can use), the cost per frame, and how many loops were found and used. The scene is a rendered flat-walled room with a persistent depth error, so the
numbers measure the method, not a real camera. Drift here comes from that depth error only; real drift also comes from blur, bad calibration and moving objects.
"""
import argparse
import time

import numpy as np

from gabai.kfvo import KeyframeVO
from gabai.sim import look_pose
from gabai.tools import vo_bench as vb


def lap_path(n, laps):
    Rs, ts = [], []
    for i in range(n):
        s = 2 * np.pi * laps * i / max(n - 1, 1)
        yaw = np.radians(25 * np.sin(s))
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rs.append(np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]))
        ts.append(np.array([0.4 * np.sin(s), 0.0, 0.35 * (1 - np.cos(s))]))
    return Rs, ts


def ate(est, truth):
    n = min(len(est), len(truth))
    e = np.linalg.norm(np.asarray(est[:n]) - np.asarray(truth[:n]), axis=1)
    return 100 * float(np.sqrt((e ** 2).mean())), 100 * float(e[-1])


def run_laps(make_vo, n=300, laps=3, noise=0.4, seed=0, render_cache=None):
    tex = vb.texture()
    rng = np.random.default_rng(seed)
    Rs, ts = lap_path(n, laps)
    R0w, t0w = look_pose(np.zeros(3))
    vo = make_vo()
    est, ms = [], []
    for i, (R, t) in enumerate(zip(Rs, ts)):
        if render_cache is not None and (seed, i) in render_cache:
            img, disp = render_cache[(seed, i)]
        else:
            img, disp = vb.render_room(tex, R, t, noise_disp=noise, rng=rng)
            if render_cache is not None:
                render_cache[(seed, i)] = (img, disp)
        ok = np.isfinite(disp)
        d = np.where(ok, disp, np.nan).astype(np.float32)
        a = time.perf_counter()
        Rw, tw, info = vo.step(img, d, ok)
        ms.append((time.perf_counter() - a) * 1000)
        est.append(R0w.T @ (tw - t0w))
    out = {"online_rms": ate(est, ts)[0], "online_final": ate(est, ts)[1], "ms_median": float(np.median(ms)), "ms_p95": float(np.percentile(ms, 95)),
           "ms_max": float(np.max(ms)), "path_m": float(np.linalg.norm(np.diff(np.array(ts), axis=0), axis=1).sum())}
    if hasattr(vo, "flush"):
        vo.flush()
        R_t, t_t = vo.trajectory()
        if len(t_t) >= n - 2:
            rebuilt = [R0w.T @ (p - t0w) for p in t_t[:n]]
            out["rebuilt_rms"], out["rebuilt_final"] = ate(rebuilt, ts)
        out["loops"] = vo.stats["loops_found"]
        out["graph_runs"] = vo.stats["graph_runs"]
        out["graph"] = vo.last_graph
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--noise", type=float, default=0.4)
    a = ap.parse_args()

    def mk(**kw):
        def f():
            v = KeyframeVO(vb.F, vb.F, vb.W / 2, vb.H / 2, **kw)
            v.set_baseline(vb.B)
            return v
        return f
    variants = [("no loop closure", mk()), ("loop detect only", mk(loop_closure=True, loop_min_path_m=1.0)),
                ("loop + pose graph", mk(loop_closure=True, loop_apply=True, loop_min_path_m=1.0))]
    cache = {}
    for name, f in variants:
        rs = [run_laps(f, a.frames, a.laps, a.noise, s, cache) for s in range(a.seeds)]
        m = lambda k: np.mean([r[k] for r in rs if k in r]) if any(k in r for r in rs) else float("nan")
        print(f"{name:20s} online rms {m('online_rms'):5.2f} final {m('online_final'):5.2f} cm | rebuilt path rms {m('rebuilt_rms'):5.2f} final {m('rebuilt_final'):5.2f} cm | "
              f"{m('ms_median'):4.1f} ms median, p95 {m('ms_p95'):5.1f}, max {m('ms_max'):6.1f} | loops {m('loops'):4.1f} graph runs {m('graph_runs'):3.1f} (path {rs[0]['path_m']:.1f} m)", flush=True)


if __name__ == "__main__":
    main()
