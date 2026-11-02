"""Matchers on room recordings (no ground truth): valid share, planarity of one depth picture, agreement with Fast-FoundationStereo.
  PYTHONPATH=. py scripts/scene_bench.py data/recordings/vio_20261101_014854 [frames] [step]"""
import sys, time
ARGS = sys.argv[1:]; sys.argv = sys.argv[:1]      # classical_bench reads argv at import
import cv2, numpy as np
from rtabmap_lab.tools.depth_bench import frames, to_depth, planarity
from rtabmap_lab.census_sgm import CensusSGM
from scripts.classical_bench import sgbm_cv, run_sgbm, run_lr, run_wls

folder = ARGS[0]; n = int(ARGS[1]) if len(ARGS) > 1 else 12; step = int(ARGS[2]) if len(ARGS) > 2 else 40
fr = frames(folder, n, step); cal = dict(fr[0].cal)
G = [(cv2.cvtColor(f.left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f.right, cv2.COLOR_BGR2GRAY)) for f in fr]
print(f"{len(G)} pairs, f={cal['f']:.1f} B={cal['B']*1000:.1f}mm")
from gabai.spike.ffs_runner import FfsDepth
ffs = FfsDepth(); ref = [to_depth(ffs.disparity(f.left[:, :, ::-1].copy(), f.right[:, :, ::-1].copy())[0], cal) for f in fr]; del ffs
M = {
 "SGBM rtabmap-like (bs9 3way)": run_sgbm(sgbm_cv(9)),
 "SGBM bs7 HH": run_sgbm(sgbm_cv(7, mode=cv2.STEREO_SGBM_MODE_HH)),
 "SGBM bs7 + LR": run_lr(sgbm_cv(7, d12=-1, uniq=5)),
 "CensusSGM default": CensusSGM(96).disparity,
 "CensusSGM uniq .03": CensusSGM(96, uniq=0.03).disparity,
 "CensusSGM p2 250": CensusSGM(96, p1=25, p2=250, p2_edge=80).disparity,
 "CensusSGM tol10": CensusSGM(96, tol=10).disparity,
}
print(f"{'method':30s} {'ms':>5s} {'valid':>6s} {'planar':>7s} {'thick mm':>9s} {'agreeFFS5%':>10s} {'MAEvsFFS%':>9s}")
for name, fn in M.items():
    Z, t = [], []
    for l, r in G:
        t0 = time.perf_counter(); d = fn(l, r); t.append(time.perf_counter() - t0); Z.append(to_depth(np.where(d > 0.5, d, np.nan), cal))
    pl = [x for x in (planarity(z, cal) for z in Z) if x]
    ag, er = [], []
    for z, q in zip(Z, ref):
        b = np.isfinite(z) & np.isfinite(q)
        if b.sum() > 500: ag.append(np.mean(np.abs(z[b] - q[b]) / q[b] < 0.05)); er.append(np.median(np.abs(z[b] - q[b]) / q[b]))
    print(f"{name:30s} {np.median(t)*1000:5.0f} {np.mean([np.isfinite(z).mean() for z in Z])*100:5.0f}% {np.mean([x[0] for x in pl])*100:6.0f}% {np.mean([x[1] for x in pl]):9.1f} {np.mean(ag)*100:9.0f}% {np.mean(er)*100:8.1f}%", flush=True)
