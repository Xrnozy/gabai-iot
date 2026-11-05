"""Where does the stereo noise enter? Two jobs on a recording (or any folder of rectified pairs).

  py -m rtabmap_lab.tools.stereo_stages data/recordings/walk_20261102_230115 --config config/pipeline_calib45.toml            measure each depth variant
  py -m rtabmap_lab.tools.stereo_stages <folder> --config ... --dump out/wall1m --frames 3 --step 1                          save every stage of the first pairs

Measure: the disparity noise of each variant IN PIXELS, without a tape. A flat wall is a plane in disparity space (d = a*x + b*y + c), so the scatter of valid pixels around the best planes is the matcher's own
disparity error (sigma_d). Planes are RANSAC fits (inlier band +-1 px, so a noisy matcher reads slightly better than it is: compare the inlier share too). Reported: ms per pair, valid pixels, share of valid pixels
on a plane, sigma_d, and the depth noise that sigma_d means at 2 m and 4 m for this rig. Put the number in SgbmDepth(sigma_px=...) and --max-depth-std.

Dump: per pair left / right rectified, raw SGBM, texture mask, WLS-refined, final disparity, depth (colour) and a point cloud (.ply), so a bad pixel can be followed back to the stage that made it.
"""
import argparse
import os

import cv2
import numpy as np

from rtabmap_lab.sgbm_depth import SgbmDepth
from rtabmap_lab.tools.depth_bench import frames, run_source


def plane_noise(disp, n_planes=3, band=1.0, iters=300, min_inliers=1500, seed=0):
    """(share of valid pixels on a plane, RMS distance to those planes in px, inlier count) for one disparity map."""
    v, u = np.nonzero(np.isfinite(disp) & (disp > 3))
    rng = np.random.default_rng(seed)
    if len(v) > 30000:
        k = rng.choice(len(v), 30000, replace=False)
        v, u = v[k], u[k]
    X = np.stack([u, v, np.ones(len(u))], 1).astype(np.float64)
    d = disp[v, u].astype(np.float64)
    total, res = len(d), []
    for _ in range(n_planes):
        if len(d) < min_inliers:
            break
        best = None
        for _ in range(iters):
            i = rng.choice(len(d), 3, replace=False)
            try:
                w = np.linalg.solve(X[i], d[i])
            except np.linalg.LinAlgError:
                continue
            inl = np.abs(X @ w - d) < band
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < min_inliers:
            break
        w = np.linalg.lstsq(X[best], d[best], rcond=None)[0]
        res.append((X[best] @ w - d[best]))
        X, d = X[~best], d[~best]
    if not res:
        return None
    r = np.concatenate(res)
    return len(r) / total, float(np.sqrt(np.mean(r ** 2))), len(r)


def dump(model, fr, cal, out, idx):
    os.makedirs(out, exist_ok=True)
    s = model.stages(fr.left, fr.right, debug=True)
    fb = cal["f"] * cal["B"]
    col = lambda d: cv2.applyColorMap((np.clip(np.nan_to_num(d, nan=0) / 96.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO) * np.isfinite(d)[..., None].astype(np.uint8)
    z = np.where(s["disp"] > 0.5, fb / s["disp"], np.nan)
    pics = {"left": s["gray_l"], "right": s["gray_r"], "raw_nolr": col(s["raw_nolr"]), "raw": col(s["raw"]), "texture_mask": s["tex"].astype(np.uint8) * 255,
            "disp": col(s["disp"]), "depth": cv2.applyColorMap((np.clip(np.nan_to_num(z, nan=0) / 4.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO) * np.isfinite(z)[..., None].astype(np.uint8)}
    if s["wls"] is not None:
        pics["wls"] = col(s["wls"])
    for k, p in pics.items():
        cv2.imwrite(os.path.join(out, f"{idx:03d}_{k}.png"), p)
    h, w = z.shape
    v, u = np.nonzero(np.isfinite(z) & (z > 0.3) & (z < 6))
    zz = z[v, u]
    pts = np.stack([(u - cal["cx"]) * zz / cal["f"], (v - cal["cy"]) * zz / cal["f"], zz], 1)
    rgb = fr.left[v, u][:, ::-1] if fr.left.ndim == 3 else np.repeat(s["gray_l"][v, u][:, None], 3, 1)
    with open(os.path.join(out, f"{idx:03d}_cloud.ply"), "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(f, np.hstack([pts, rgb]), fmt="%.4f %.4f %.4f %d %d %d")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--config", default="config/pipeline.toml")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--models", action="store_true", help="also measure LAS2 and Fast-FoundationStereo (loads them one at a time)")
    ap.add_argument("--dump", help="write every stage of the pairs to this folder (uses the full variant: texture + WLS) instead of measuring")
    a = ap.parse_args(argv)
    fr = frames(a.folder, a.frames, a.step, a.config)
    cal = dict(fr[0].cal)
    fb = cal["f"] * cal["B"]
    if a.dump:
        m = SgbmDepth()
        for i, f in enumerate(fr):
            dump(m, f, cal, a.dump, i)
        print(f"{len(fr)} pairs written to {a.dump}")
        return
    pairs = [(f.left.copy(), f.right.copy()) for f in fr]
    import cv2 as _cv
    variants = {
        "SGBM, the bench's settings": SgbmDepth(block=7, mode=_cv.STEREO_SGBM_MODE_HH, texture=0, wls=False, speckle_range=4),
        "SGBM block 5, 3-way, no extra checks": SgbmDepth(texture=0, wls=False),
        "  + texture mask": SgbmDepth(wls=False),
        "  + texture mask + WLS refinement": SgbmDepth(),
    }
    rows = []
    for name, m in variants.items():
        out, ms, _, _ = run_source(name, lambda: None, lambda _, p, m=m: m.stages(*p)["disp"], pairs, cal)
        rows.append((name, ms, out))
    if a.models:
        from gabai.spike.las2_runner import Las2Depth
        out, ms, _, _ = run_source("LAS2", lambda: Las2Depth("s"), lambda mm, p: mm.disparity_lr_batched(p[0][:, :, ::-1].copy(), p[1][:, :, ::-1].copy()), pairs, cal)
        rows.append(("LAS2 (+ left-right check)", ms, [np.where(t, d, np.nan).astype(np.float32) for d, t, _ in out]))
        from gabai.spike.ffs_runner import FfsDepth
        out, ms, _, _ = run_source("FFS", lambda: FfsDepth(), lambda mm, p: mm.disparity(p[0][:, :, ::-1].copy(), p[1][:, :, ::-1].copy())[0], pairs, cal)
        rows.append(("Fast-FoundationStereo", ms, [np.where(d > 0.5, d, np.nan).astype(np.float32) for d in out]))
    print(f"\n{len(pairs)} pairs, f*B = {fb:.1f} px*m\n{'variant':40s} {'ms':>5s} {'valid':>6s} {'on plane':>9s} {'sigma_d px':>10s} {'noise@2m':>9s} {'noise@4m':>9s}")
    for name, ms, out in rows:
        valid = np.mean([np.isfinite(d).mean() for d in out])
        pn = [x for x in (plane_noise(d) for d in out) if x]
        if not pn:
            print(f"{name:40s} {ms:5.0f} {valid * 100:5.0f}%   (no plane found)")
            continue
        share, sig = np.mean([p[0] for p in pn]), float(np.sqrt(np.mean([p[1] ** 2 for p in pn])))
        print(f"{name:40s} {ms:5.0f} {valid * 100:5.0f}% {share * 100:8.0f}% {sig:10.3f} {4 * sig / fb * 100:7.1f}cm {16 * sig / fb * 100:7.1f}cm")


if __name__ == "__main__":
    main()
