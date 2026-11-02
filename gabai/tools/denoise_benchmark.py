"""Does fusing a monocular metric model with LAS2 stereo reduce noise without hurting accuracy?

On the checkerboard pairs saved by calibrate_stereo (true board distance from solvePnP) it reports, for raw stereo and
for the fused map: median and 90th-percentile error (accuracy), the spread of the error over the flat board
(noise), and a roughness figure over the whole picture (high-frequency content of the depth, a noise proxy that needs
no ground truth). Lower is better for all. Roughness alone can be fooled by over-smoothing, so read it next to the error.

    py -m gabai.tools.denoise_benchmark --mono dav2
"""
import argparse
import glob
import os

import cv2
import numpy as np

from gabai.fuse_depth import fuse
from gabai.spike.las2_runner import Las2Depth
from gabai.spike.mono_depth import MonoDepth
from gabai.tools.benchmark_depth import board_depth
from gabai.tools.calibrate_stereo import Rectifier


def roughness(d, mask=None):
    """Median absolute high-frequency residual of a disparity map, in pixels."""
    hf = np.abs(d - cv2.medianBlur(np.ascontiguousarray(d, dtype=np.float32), 5))
    return float(np.median(hf if mask is None else hf[mask]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-dir", default="data/calib_pairs_swapped")
    ap.add_argument("--calib", default="data/stereo_calib_new.npz")
    ap.add_argument("--stereo", default="m")
    ap.add_argument("--mono", default="dav2")
    ap.add_argument("--scenes", type=int, default=6, help="real-scene frames from data/left.mp4 for roughness only")
    a = ap.parse_args()

    rect = Rectifier(a.calib)
    z = np.load(a.calib)
    P1, sq = z["P1"], float(z["square_mm"]) / 1000
    las, mono = Las2Depth(a.stereo), MonoDepth(a.mono)
    f, B = rect.f, rect.B
    res = {"stereo": [], "fused": [], "mono (aligned)": []}
    rough = {"stereo": [], "fused": [], "mono (aligned)": []}
    from gabai.fuse_depth import align_mono

    def process(l, r):
        rl, rr = cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        ds, ok, _ = las.disparity_lr(rl, rr)
        zm = np.clip(mono.depth(rl, f_px=f), 0.2, 30.0)
        dm = f * B / zm
        fused, info = fuse(ds, ok, dm)
        aligned, _ = align_mono(dm, ds, ok & (ds > 0.5))
        return ds, fused, aligned

    for lf in sorted(glob.glob(os.path.join(a.pairs_dir, "*_L.png"))):
        L, R = cv2.imread(lf), cv2.imread(lf[:-6] + "_R.png")
        l, r = rect.apply(L, R)
        truth = board_depth(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), P1, 9, 6, sq)
        if truth is None:
            continue
        m = np.isfinite(truth)
        ds, fused, aligned = process(l, r)
        for name, d in (("stereo", ds), ("fused", fused), ("mono (aligned)", aligned)):
            e = (f * B / np.maximum(d[m], 0.5) - truth[m]) / truth[m]
            res[name].append(e)
            rough[name].append(roughness(d))
    print(f"{len(res['stereo'])} board pairs; error = (estimated - true) / true over the board")
    print(f"{'':16s} {'median|e|':>9s} {'p90|e|':>8s} {'noise(std)':>11s} {'roughness px':>13s}")
    for name in res:
        e = np.concatenate(res[name])
        print(f"{name:16s} {np.median(np.abs(e)) * 100:8.2f}% {np.percentile(np.abs(e), 90) * 100:7.2f}% {np.std(e) * 100:10.2f}% {np.mean(rough[name]):13.3f}")

    cap = cv2.VideoCapture("data/left.mp4"), cv2.VideoCapture("data/right.mp4")
    if all(c.isOpened() for c in cap) and a.scenes:
        print(f"\nreal kitchen scenes (no ground truth): roughness of the depth map, lower = less noisy")
        from gabai.spike.demo_viewer import load_rectification
        M, _, _ = load_rectification("data/left.mp4", "data/right.mp4", 640, "data/autorect_640.npz")
        total = int(cap[0].get(cv2.CAP_PROP_FRAME_COUNT))
        rr_ = {"stereo": [], "fused": [], "mono (aligned)": []}
        for k in np.linspace(0.1, 0.9, a.scenes) * total:
            for c in cap:
                c.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            (okl, l0), (okr, r0) = cap[0].read(), cap[1].read()
            if not (okl and okr):
                continue
            l1 = cv2.resize(l0, (640, 360), interpolation=cv2.INTER_AREA)
            r1 = cv2.warpAffine(cv2.resize(r0, (640, 360), interpolation=cv2.INTER_AREA), M, (640, 360))
            rl, rr = cv2.cvtColor(l1, cv2.COLOR_BGR2RGB), cv2.cvtColor(r1, cv2.COLOR_BGR2RGB)
            ds, ok, _ = las.disparity_lr(rl, rr)
            dm = (0.6 * 640 * 0.065) / np.clip(mono.depth(rl), 0.2, 30.0)    # assumed f and B, as in the video viewer
            fused, _ = fuse(ds, ok, dm)
            from gabai.fuse_depth import align_mono
            aligned, _ = align_mono(dm, ds, ok & (ds > 0.5))
            for name, d in (("stereo", ds), ("fused", fused), ("mono (aligned)", aligned)):
                rr_[name].append(roughness(d))
        for name, v in rr_.items():
            print(f"  {name:16s} roughness {np.mean(v):.3f} px")
    print(f"\nmodel times: stereo (single pass) and mono {mono.name}: last mono {mono.last_ms:.0f} ms")


if __name__ == "__main__":
    main()
