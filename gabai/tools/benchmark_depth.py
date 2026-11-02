"""Measure depth accuracy on YOUR rig using the checkerboard pairs saved by calibrate_stereo.

The board is a flat object of known size, so solvePnP gives its true distance in every picture. We compare
each model's depth over the board area against that. Models: LiteAnyStereo S/M/H (Z = f*B/d) and, if the weights
can be loaded, Depth Anything V2 Metric (monocular, no baseline needed).

    py -m gabai.tools.benchmark_depth
Limits: board area only (a high-contrast, flat target: easy for stereo), distances 0.3-1.5 m, scale tied to the
square size you entered, so absolute scale errors in the SQUARE SIZE cancel for stereo but not for the monocular model.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from gabai.tools.calibrate_stereo import Rectifier, board_points, find_corners


def board_depth(rect_gray, P1, cols, rows, square_m):
    """True depth map (m) of the board area in the rectified left image, or None if the board is not found."""
    c = find_corners(rect_gray, (cols, rows))
    if c is None:
        return None
    K = P1[:3, :3].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(board_points(cols, rows, square_m), c, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2]                      # board normal in camera coordinates
    d0 = float(n @ tvec.ravel())     # plane: n . X = d0
    mask = np.zeros(rect_gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(c.reshape(-1, 2).astype(np.int32)), 1)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    v, u = np.nonzero(mask)
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, dtype=np.float64)], 1)
    z = d0 / (ray @ n)
    out = np.full(rect_gray.shape, np.nan, np.float32)
    out[v, u] = z
    return out


class DepthAnything:
    def __init__(self, repo="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo).half().cuda().eval()

    def depth(self, bgr):
        t = self.torch
        inp = self.proc(images=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), return_tensors="pt")
        x = inp["pixel_values"].half().cuda()
        t.cuda.synchronize()
        t0 = time.perf_counter()
        with t.no_grad():
            out = self.model(pixel_values=x).predicted_depth
        t.cuda.synchronize()
        dt = time.perf_counter() - t0
        d = t.nn.functional.interpolate(out[None].float(), size=bgr.shape[:2], mode="bilinear")[0, 0]
        return d.cpu().numpy(), dt


def stats(errs):
    e = np.array(errs)
    return f"median {np.median(np.abs(e))*100:5.1f}%  p90 {np.percentile(np.abs(e), 90)*100:5.1f}%  bias {np.median(e)*100:+5.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-dir", default="data/calib_pairs")
    ap.add_argument("--calib", default="data/stereo_calib_new.npz")
    ap.add_argument("--gray", action="store_true", help="feed grayscale to the models (default is colour)")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--models", default="s,m,h,da", help="comma list: LAS2 sizes s,m,h; 'ffs' for Fast-FoundationStereo; 'da' for Depth Anything")
    a = ap.parse_args()

    rect = Rectifier(a.calib)
    z = np.load(a.calib)
    P1, square_m = z["P1"], float(z["square_mm"]) / 1000
    models = a.models.split(",")
    las = {}
    from gabai.spike.las2_runner import Las2Depth
    for m in models:
        if m in "smh" and len(m) == 1:
            las[m] = Las2Depth(m)
        elif m == "ffs":
            from gabai.spike.ffs_runner import FfsDepth
            las[m] = FfsDepth()
        elif m == "defom":
            from gabai.spike.other_stereo import DefomDepth
            las[m] = DefomDepth()
        elif m == "prompt":
            from gabai.spike.other_stereo import PromptStereoDepth
            las[m] = PromptStereoDepth()
    da = DepthAnything() if "da" in models else None

    res = {m: dict(err=[], t=[], near=[], mid=[], far=[], scaled=[], trusted_err=[], trusted_frac=[]) for m in models}
    dist_log = []
    for lf in sorted(glob.glob(os.path.join(a.pairs_dir, "*_L.png"))):
        rf = lf[:-6] + "_R.png"
        L, R = cv2.imread(lf), cv2.imread(rf)
        if L is None or R is None:
            continue
        g = (lambda x: cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)) if a.gray else (lambda x: x)
        l, r = rect.apply(g(L), g(R))
        truth = board_depth(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), P1, a.cols, a.rows, square_m)
        if truth is None:
            continue
        m_ = np.isfinite(truth)
        zt = truth[m_]
        dist = float(np.median(zt))
        dist_log.append(dist)
        for m in models:
            if m in las:
                disp, ok_lr, dt = las[m].disparity_lr(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
                dt /= 2  # report the single-pass time; the check costs a second pass
                zp = rect.f * rect.B / np.maximum(disp, 0.5)
                keep = ok_lr[m_]
                res[m]["trusted_frac"].append(float(keep.mean()))
                res[m]["trusted_err"].extend((((zp[m_] - zt) / zt)[keep]).tolist())
            else:
                zp, dt = da.depth(l)
            e = (zp[m_] - zt) / zt
            res[m]["err"].extend(e.tolist())
            res[m]["t"].append(dt * 1000)
            res[m]["scaled"].extend(((zp[m_] * (np.median(zt) / np.median(zp[m_])) - zt) / zt).tolist())
            bucket = "near" if dist < 0.6 else "mid" if dist < 0.9 else "far"
            res[m][bucket].append(float(np.median(e)))
    if not dist_log:
        raise SystemExit("board not found in any saved pair")
    print(f"{len(dist_log)} pairs with the board found; true board distance {min(dist_log):.2f}-{max(dist_log):.2f} m "
          f"(median {np.median(dist_log):.2f} m); square size {square_m*1000:.1f} mm")
    print("error = (estimated - true) / true over the board area; 'scale-fixed' removes each picture's overall scale\n")
    for m in models:
        name = {"ffs": "Fast-FoundationStereo", "defom": "DEFOM-Stereo", "prompt": "PromptStereo"}.get(m, f"LAS2-{m.upper()}") if m in las else "DepthAnythingV2-Metric-Small"
        r = res[m]
        print(f"{name:28s} {stats(r['err'])}   | scale-fixed median {np.median(np.abs(r['scaled']))*100:4.1f}%   "
              f"| model {np.median(r['t']):.0f} ms")
        if r["trusted_err"]:
            print(f"{'':28s} left-right check keeps {np.mean(r['trusted_frac'])*100:.0f}% of board pixels; "
                  f"error on those: {stats(r['trusted_err'])}")
        buckets = [f"{k} {np.median(v)*100:+.1f}% (n={len(v)})" for k, v in (("<0.6m", r["near"]), ("0.6-0.9m", r["mid"]), (">0.9m", r["far"])) if v]
        print(f"{'':28s} bias by distance: {'   '.join(buckets)}")


if __name__ == "__main__":
    main()
