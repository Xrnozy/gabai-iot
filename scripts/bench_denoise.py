"""Does denoising the left/right pictures before LAS2 help the depth, and what does it cost?

    py scripts/bench_denoise.py                   uses data/noise_rec_*.npz (real still frames from your cameras) and data/left.mp4 + data/right.mp4

Three tests, every candidate applied to BOTH pictures the same way:
  A  real noise, still rig (the 30-frame noise recordings): no ground truth, so each frame is compared with the pixel-wise median of the 30 raw
     disparity maps (the converged answer of a still scene). Lower error = less frame-to-frame jitter. Edge retention = how much of the sharpest
     gradients survive in the picture (below ~0.9 the denoiser is eating detail).
  B  noise added to the sharp daylight video pairs (sigma 8 / 16 / 24 grey levels), compared with LAS2's own result on the clean pair.
  C  the same video pairs with a 33 ms left/right capture offset (a moving rig), with and without noise, with and without the offset correction:
     what matters more, noise or the offset?
Latency is the median CPU time of the candidate on a 640x360 pair. Writes data/bench/denoise.json.
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from gabai.skewcomp import SkewCompensator             # noqa: E402
from gabai.spike.las2_runner import Las2Depth           # noqa: E402

CANDIDATES = {
    "none": lambda im: im,
    "gauss 3x3": lambda im: cv2.GaussianBlur(im, (3, 3), 0.7),
    "bilateral": lambda im: cv2.bilateralFilter(im, 5, 20, 2),
    "median 3x3": lambda im: cv2.medianBlur(im, 3),
    "non-local means": lambda im: cv2.fastNlMeansDenoisingColored(im, None, 4, 4, 5, 15),
}


def edge_retention(raw, den):
    def top(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        return float(np.percentile(m, 99))
    return top(den) / max(top(raw), 1e-6)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    return cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def paint(d, ok, top):
    v = cv2.applyColorMap(np.clip(np.nan_to_num(d) / top * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    v[~ok] = 0
    return v


def top_view(d, ok, f=384.0, B=0.065, span=None, size=(480, 360)):
    ys, xs = np.nonzero(ok & (d > 1))
    Z = f * B / d[ys, xs]
    X = (xs - d.shape[1] / 2) * Z / f
    img = np.zeros((size[1], size[0], 3), np.uint8)
    ux = ((X - span[0]) / (span[1] - span[0]) * (size[0] - 1)).astype(int)
    uy = ((span[3] - Z) / (span[3] - span[2]) * (size[1] - 1)).astype(int)
    m = (ux >= 0) & (ux < size[0]) & (uy >= 0) & (uy < size[1])
    img[uy[m], ux[m]] = (80, 220, 80)
    return img


def figure(las, L, R, T, noisy):
    """Pictures for one frame: still rig with noise (raw / smoothed) and the moving rig (offset as before / corrected / corrected + smoothed / clean)."""
    def disp(l, r):
        d, ok, _ = las.disparity_lr_batched(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        return d, ok
    g = CANDIDATES["gauss 3x3"]
    i = 7
    d0, ok0 = disp(L[i], R[i])
    top = float(np.percentile(d0[ok0], 98))
    nl, nr = noisy(L[i], 16), noisy(R[i], 16)
    a, aok = disp(nl, nr)
    b, bok = disp(g(nl), g(nr))
    row1 = np.hstack([label(nl.copy(), "still rig, noise sigma 16: left picture"), label(g(nl).copy(), "after 3x3 smoothing (edge retention 0.83)"),
                      label(paint(a, aok, top), f"depth, raw: {aok.mean() * 100:.0f}% valid"), label(paint(b, bok, top), f"depth, smoothed: {bok.mean() * 100:.0f}% valid")])
    sc = SkewCompensator()
    sc.prev, sc.t_prev = sc._gray(L[i - 1]), T[i - 1]
    ln, rl = noisy(L[i], 12), noisy(R[i + 1], 12)
    fixed, _ = sc.apply(ln, rl, T[i], 1000 / 30.0)
    res = {"moving, 33 ms offset as before": disp(ln, rl), "offset corrected": disp(ln, fixed), "corrected + smoothed": disp(g(ln), g(fixed)), "clean pair (reference)": (d0, ok0)}
    row2 = np.hstack([label(paint(d, ok, top), f"{k}: {ok.mean() * 100:.0f}% valid") for k, (d, ok) in res.items()])
    ys, xs = np.nonzero(ok0 & (d0 > 1))
    Zr = 384.0 * 0.065 / d0[ys, xs]
    Xr = (xs - d0.shape[1] / 2) * Zr / 384.0
    span = (float(np.percentile(Xr, 1)) - 0.1, float(np.percentile(Xr, 99)) + 0.1, float(np.percentile(Zr, 1)) * 0.8, float(np.percentile(Zr, 99)) * 1.1)
    row3 = np.hstack([label(top_view(d, ok, span=span), f"{k}: top view (x {span[0]:.1f}..{span[1]:.1f} m, depth {span[2]:.1f}..{span[3]:.1f} m)") for k, (d, ok) in res.items()])
    cv2.imwrite(os.path.join("data", "bench", "denoise_compare.png"), np.vstack([row1, cv2.resize(row2, (row1.shape[1], row2.shape[0] * row1.shape[1] // row2.shape[1]))]))
    cv2.imwrite(os.path.join("data", "bench", "denoise_clouds.png"), cv2.resize(row3, (row1.shape[1], int(row3.shape[0] * row1.shape[1] / row3.shape[1]))))


def main():
    las = Las2Depth("h")

    def disp(l, r):
        d, ok, _ = las.disparity_lr_batched(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        return d, ok
    out = {}
    # latency
    z = np.load(sorted(glob.glob(os.path.join("data", "noise_rec_*.npz")))[0])
    probe = cv2.resize(z["left"][0], (640, 360))
    out["latency_ms_per_pair"] = {}
    for n, f in CANDIDATES.items():
        t = []
        for _ in range(15):
            t0 = time.perf_counter()
            f(probe), f(probe)
            t.append((time.perf_counter() - t0) * 1000)
        out["latency_ms_per_pair"][n] = round(float(np.median(t)), 2)
    # A: how noisy are your cameras? temporal noise of a still scene (the 30-frame recordings): std of the difference of two frames / sqrt 2, mid-tones only
    out["A_your_camera_noise_sigma_grey_levels"] = {}
    for path in sorted(glob.glob(os.path.join("data", "noise_rec_*.npz")))[:2]:
        z = np.load(path)
        g = np.stack([np.asarray(f, np.float32).mean(2) for f in z["left"]])
        mid = (g[0] > 20) & (g[0] < 235)
        out["A_your_camera_noise_sigma_grey_levels"][os.path.basename(path)] = {"sigma": round(float((g[1:] - g[:-1])[:, mid].std() / np.sqrt(2)), 2), "picture_median": round(float(np.median(g)), 0)}
    out["A_edge_retention_of_the_3x3_gauss_on_sharp_video"] = round(float(np.mean([edge_retention(l, CANDIDATES["gauss 3x3"](l)) for l in np.load(os.path.join("data", "bench", "seq_clean.npz"))["L"][:6]])), 2)
    # B and C on the sharp video
    seq = os.path.join("data", "bench", "seq_clean.npz")
    z = np.load(seq)
    L, R = z["L"][:10], z["R"][:10]
    ref = [disp(l, r) for l, r in zip(L, R)]
    rng = np.random.default_rng(0)
    noisy = lambda im, s: np.clip(im.astype(np.float32) + rng.normal(0, s, im.shape), 0, 255).astype(np.uint8)
    out["B_added_noise_vs_clean"] = {}
    for sigma in (8, 16, 24):
        nz = [(noisy(l, sigma), noisy(r, sigma)) for l, r in zip(L, R)]
        out["B_added_noise_vs_clean"][f"sigma {sigma}"] = {}
        for name, f in CANDIDATES.items():
            e, c = [], []
            for (l, r), (d0, ok0) in zip(nz, ref):
                d, ok = disp(f(l), f(r))
                e.append(np.abs(d - d0)[ok & ok0]), c.append(ok.mean())
            e = np.concatenate(e)
            out["B_added_noise_vs_clean"][f"sigma {sigma}"][name] = {"valid_pct": round(float(np.mean(c)) * 100, 1), "median_err_px": round(float(np.median(e)), 3), "over_2px_pct": round(float((e > 2).mean() * 100), 2)}
    # C: offset pair (right picture of the NEXT frame) -- needs consecutive frames: use the 30 fps pairs of data/bench/seq_skew-like set
    cfg_fr = os.path.join("data", "bench", "seq_consecutive.npz")
    if not os.path.exists(cfg_fr):
        from gabai.pipeline.config import PipelineConfig
        from gabai.pipeline.sources import open_source
        fr = list(open_source(PipelineConfig.load("config/pipeline.toml", {"source.start": 30.0, "source.skip": 1, "source.max_frames": 14, "stereo.model": "h"})))
        np.savez_compressed(cfg_fr, L=np.array([f.left for f in fr]), R=np.array([f.right for f in fr]), t=np.array([f.t for f in fr]))
    z = np.load(cfg_fr)
    L, R, T = z["L"], z["R"], z["t"]
    out["C_moving_rig_33ms_offset"] = {}
    for sigma in (0, 12):
        res = {"offset, as before": [], "offset corrected": [], "offset corrected + gauss": [], "offset + gauss (no correction)": []}
        sc = SkewCompensator()
        refs = []
        for i in range(1, len(L) - 1):
            l0, r0, rl = L[i], R[i], R[i + 1]
            ln = noisy(l0, sigma) if sigma else l0
            rn_true = noisy(r0, sigma) if sigma else r0
            rn_late = noisy(rl, sigma) if sigma else rl
            sc.prev, sc.t_prev = sc._gray(L[i - 1]), T[i - 1]
            fixed, _ = sc.apply(ln, rn_late, T[i], 1000 / 30.0)
            d0, ok0 = disp(l0, r0)
            for key, (a, b) in {"offset, as before": (ln, rn_late), "offset corrected": (ln, fixed),
                                "offset corrected + gauss": (CANDIDATES["gauss 3x3"](ln), CANDIDATES["gauss 3x3"](fixed)),
                                "offset + gauss (no correction)": (CANDIDATES["gauss 3x3"](ln), CANDIDATES["gauss 3x3"](rn_late))}.items():
                d, ok = disp(a, b)
                res[key].append((np.abs(d - d0)[ok & ok0], ok.mean()))
        out["C_moving_rig_33ms_offset"][f"noise sigma {sigma}"] = {k: {"valid_pct": round(float(np.mean([c for _, c in v])) * 100, 1), "median_err_px": round(float(np.median(np.concatenate([e for e, _ in v]))), 3),
                                                                      "over_2px_pct": round(float((np.concatenate([e for e, _ in v]) > 2).mean() * 100), 1)} for k, v in res.items()}
    figure(las, L, R, T, noisy)
    json.dump(out, open(os.path.join("data", "bench", "denoise.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
