"""Find out what kind of noise your depth has, and whether fusing a monocular model reduces it.

HOLD THE RIG STILL on a static scene (or put it on a table) for ~10 s, pointing at a room with a flat wall or floor in
the middle of the picture. The Pi must be streaming. Then:

    py -m gabai.tools.noise_test --mono dav2            (later: --mono moge-s)

It records frames, then for raw stereo, the aligned monocular map, and the fused map reports:
  * temporal noise: how much each pixel's disparity jumps from frame to frame on a static scene, in pixels and in
    centimetres of depth at 1, 2 and 3 m (the baseline makes the same pixel noise cost much more at range);
  * spatial noise: how rough a flat patch is (residual after fitting a plane);
  * how many pixels were usable.
This separates "flickery" from "bumpy" noise, which need different cures.
"""
import argparse
import time

import cv2
import numpy as np


def temporal_stats(stack, ok, min_fraction=0.8):
    """stack, ok: (T, H, W) disparity and trusted mask. Pixels trusted in at least `min_fraction` of the frames only."""
    good = ok.mean(axis=0) >= min_fraction
    if not good.any():
        return {"temporal_std_px": float("nan"), "frame_to_frame_px": float("nan"), "usable_fraction": 0.0}
    sd = np.std(np.where(ok, stack, np.nan), axis=0)
    f2f = np.abs(np.diff(stack, axis=0))
    return {"temporal_std_px": float(np.nanmedian(sd[good])), "frame_to_frame_px": float(np.median(f2f[:, good])),
            "usable_fraction": float(good.mean())}


def plane_residual(d, roi):
    """Standard deviation (px) of disparity over roi=(x0, y0, x1, y1) after fitting a plane d = a x + b y + c."""
    x0, y0, x1, y1 = roi
    p = d[y0:y1, x0:x1]
    y, x = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    ok = np.isfinite(p)
    A = np.column_stack([x[ok], y[ok], np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(A, p[ok].astype(np.float64), rcond=None)
    return float(np.std(p[ok] - A @ coef))


def depth_jitter_cm(sigma_px, fb, z):
    """Depth error in cm that a disparity error of sigma_px causes at distance z (m), with fb = focal * baseline."""
    return float(100 * z * z * sigma_px / fb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--stereo", default="m", help="LAS2 size")
    ap.add_argument("--mono", default="dav2")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--calib", default="data/stereo_calib_new.npz")
    ap.add_argument("--roi", default="", help="x0,y0,x1,y1 of a flat patch; default is the middle third of the picture")
    a = ap.parse_args()

    from gabai.fuse_depth import align_mono, fuse
    from gabai.pc.receiver import LiveReceiver
    from gabai.spike.las2_runner import Las2Depth
    from gabai.spike.mono_depth import MonoDepth
    from gabai.tools.calibrate_stereo import Rectifier

    rect = Rectifier(a.calib)
    rx = LiveReceiver(a.port, a.token, rotate_left=rect.rotation[0], rotate_right=rect.rotation[1])
    rx.start()
    print(f"waiting for the Pi on port {rx.port} ...", flush=True)
    frames, seq, t0 = [], 0, None
    while True:
        p = rx.latest()
        if p is not None and p.seq != seq:
            seq = p.seq
            if t0 is None:
                t0 = time.monotonic()
                print(f"recording {a.seconds:.0f} s: KEEP THE RIG STILL", flush=True)
            l, r = rect.apply(p.left, p.right)
            frames.append((l, r))
        if t0 is not None and time.monotonic() - t0 > a.seconds:
            break
        time.sleep(0.005)
    rx.stop()
    if len(frames) < 20:
        raise SystemExit("fewer than 20 frames recorded")
    gray = np.stack([cv2.cvtColor(l, cv2.COLOR_BGR2GRAY) for l, _ in frames]).astype(np.float32)
    moved = float(np.median(np.std(gray, axis=0)))
    print(f"{len(frames)} frames; picture changed by {moved:.1f} grey levels between frames "
          f"({'steady' if moved < 4 else 'THE SCENE OR RIG MOVED: results will overstate the noise'})", flush=True)

    las, mono = Las2Depth(a.stereo), MonoDepth(a.mono)
    f, B = rect.f, rect.B
    S, OK, FU, AL = [], [], [], []
    for l, r in frames:
        rl, rr = cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        ds, ok, _ = las.disparity_lr(rl, rr)
        dm = f * B / np.clip(mono.depth(rl, f_px=f), 0.2, 30.0)
        fused, _info = fuse(ds, ok, dm)
        aligned = _info["aligned"]
        S.append(ds); OK.append(ok); FU.append(fused); AL.append(aligned)
    S, OK, FU, AL = (np.stack(v) for v in (S, OK, FU, AL))
    h, w = S.shape[1:]
    roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else (w // 3, h // 3, 2 * w // 3, 2 * h // 3)
    allok = np.ones_like(OK)
    fb = f * B
    print(f"\nscene: median disparity {np.median(S[OK]):.1f} px (about {fb / np.median(S[OK]):.1f} m); flat patch for the plane test: {roi}")
    print(f"{'':18s} {'temporal px':>11s} {'f2f px':>7s} {'plane px':>9s} {'usable':>7s}   depth jitter at 1 / 2 / 3 m (cm)")
    for name, st, ok in (("raw stereo", S, OK), ("mono (aligned)", AL, allok), ("fused", FU, allok)):
        t = temporal_stats(st, ok)
        pr = float(np.mean([plane_residual(st[i], roi) for i in range(0, len(st), 5)]))
        j = [depth_jitter_cm(t["temporal_std_px"], fb, z) for z in (1, 2, 3)]
        print(f"{name:18s} {t['temporal_std_px']:11.3f} {t['frame_to_frame_px']:7.3f} {pr:9.3f} {t['usable_fraction'] * 100:6.0f}%   "
              f"{j[0]:5.1f} / {j[1]:5.1f} / {j[2]:5.1f}")
    print("\ntemporal = per-pixel std over time (flicker); f2f = typical change between consecutive frames; plane = bumpiness of a flat patch.")


if __name__ == "__main__":
    main()
