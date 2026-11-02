"""Live noise viewer: watch, in real time, how noisy the depth is while you keep the rig still (or move it).

    py -m gabai.tools.noise_view --mono dav2
Four pictures: the left camera, the stereo depth, the fused (stereo + monocular) depth, and a NOISE MAP showing, for each
pixel, how much its disparity jumped over the last ~3 seconds (blue = steady, red = jumpy, black = often untrusted).
Under them a rolling table: per-pixel flicker in pixels and what it costs in centimetres at 1, 2 and 3 m, for raw stereo,
the aligned monocular map and the fused map. Hold the rig still on a table to measure sensor noise; move it to see how
motion hurts. The Pi must be streaming (same sender as the viewer).

Keys:
  n  noise map of the next method (stereo -> mono -> fused)      r  reset the window
  g  brighten dark pictures (gamma 0.6)      c  contrast boost (CLAHE)      t  average 3 frames (only for a still rig)
     (g, c, t are applied to both cameras before the depth model: compare the flicker numbers with and without)
  w  save the last window of pictures to data/noise_rec_<time>.npz so the problem can be studied offline
  s  screenshot      q  quit
"""
import argparse
import os
import time
from collections import deque

import cv2
import numpy as np

from gabai.tools.noise_test import depth_jitter_cm, temporal_stats

NAMES = ("raw stereo", "mono (aligned)", "fused", "stereo + time avg")
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def colour_depth(d, ok=None):
    v = d.copy()
    good = np.isfinite(v) & (v > 0.5) if ok is None else ok & np.isfinite(v)
    if not good.any():
        return np.zeros((*d.shape, 3), np.uint8)
    lo, hi = np.percentile(v[good], [2, 98])
    img = cv2.applyColorMap(np.clip((v - lo) * 255 / max(hi - lo, 1e-3), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~good] = 0
    return img


def noise_map(stack, ok, full_scale=0.6):
    """Per-pixel temporal std (px) as a colour picture; black where the pixel was mostly untrusted."""
    good = ok.mean(axis=0) >= 0.8
    sd = np.std(np.where(ok, stack, np.nan), axis=0)
    sd = np.where(good, np.nan_to_num(sd), 0)
    img = cv2.applyColorMap(np.clip(sd / full_scale * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[~good] = 0
    return img


def preprocess(img, gamma, clahe):
    """Same treatment for both cameras. img is RGB uint8."""
    if gamma:
        lut = (255 * (np.arange(256) / 255.0) ** 0.6).astype(np.uint8)
        img = cv2.LUT(img, lut)
    if clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[..., 0] = CLAHE.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stereo", default="m")
    ap.add_argument("--mono", default="dav2")
    ap.add_argument("--window", type=int, default=30, help="frames in the rolling noise window (about 3 s at 10 fps)")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--calib", default="data/stereo_calib_new.npz")
    ap.add_argument("--no-lr", action="store_true", help="skip the left-right check (about twice as fast, trusts every pixel)")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="", help="save the view as an mp4")
    a = ap.parse_args()

    from gabai.fuse_depth import DepthFuser, TemporalFilter
    from gabai.pc.receiver import LiveReceiver
    from gabai.spike.las2_runner import Las2Depth
    from gabai.spike.mono_depth import MonoDepth
    from gabai.tools.calibrate_stereo import Rectifier

    rect = Rectifier(a.calib)
    las, mono, fuser, smoother = Las2Depth(a.stereo), MonoDepth(a.mono), DepthFuser(), TemporalFilter()
    f, B = rect.f, rect.B
    fb = f * B
    rx = LiveReceiver(a.port, a.token, rotate_left=rect.rotation[0], rotate_right=rect.rotation[1])
    rx.start()
    print(f"waiting for the Pi on port {rx.port} ... (q quits, n changes the noise map, r resets)", flush=True)
    win = "Gabai noise view (n map, g gamma, c clahe, t avg3, w save, r reset, q quit)"
    S, OK, AL, FU, G, IM, SM = (deque(maxlen=a.window) for _ in range(7))
    recent = deque(maxlen=3)
    SMOK = deque(maxlen=a.window)
    which, seq, n, writer, last_t = 0, 0, 0, None, time.time()
    gamma = clahe = avg3 = False
    note = ""
    while True:
        p = rx.latest()
        if p is None or p.seq == seq:
            k = 255 if a.no_window else cv2.waitKey(5) & 0xFF
            if k in (ord("q"), 27):
                break
            time.sleep(0.005)
            continue
        seq = p.seq
        l, r = rect.apply(p.left, p.right)
        rl, rr = cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        IM.append((l, r))
        recent.append((rl, rr))
        if avg3 and len(recent) == 3:
            rl = np.mean([x[0] for x in recent], axis=0).astype(np.uint8)
            rr = np.mean([x[1] for x in recent], axis=0).astype(np.uint8)
        rl, rr = preprocess(rl, gamma, clahe), preprocess(rr, gamma, clahe)
        if a.no_lr:
            ds, _ = las.disparity(rl, rr)
            ok = ds > 0.5
        else:
            ds, ok, _ = getattr(las, "disparity_lr_batched", las.disparity_lr)(rl, rr)
        dm = f * B / np.clip(mono.depth(rl, f_px=f), 0.2, 30.0)
        fused, info = fuser.fuse(ds, ok, dm)
        aligned = info["aligned"]
        small = lambda x: cv2.resize(np.ascontiguousarray(x, dtype=np.float32), (192, 144), interpolation=cv2.INTER_AREA)
        S.append(small(ds)); OK.append(cv2.resize(ok.astype(np.uint8), (192, 144), interpolation=cv2.INTER_NEAREST) > 0)
        AL.append(small(aligned)); FU.append(small(fused))
        sm = smoother.update(ds, ok)
        SM.append(small(np.where(np.isfinite(sm), sm, 0)))
        SMOK.append(cv2.resize(np.isfinite(sm).astype(np.uint8), (192, 144), interpolation=cv2.INTER_NEAREST) > 0)
        G.append(cv2.resize(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), (192, 144), interpolation=cv2.INTER_AREA).astype(np.float32))
        stacks = (np.stack(S), np.stack(AL), np.stack(FU), np.stack(SM))
        oks = (np.stack(OK), np.ones_like(np.stack(OK)), np.ones_like(np.stack(OK)), np.stack(SMOK))
        enough = len(S) >= 10
        rows = []
        if enough:
            for st, o in zip(stacks, oks):
                t = temporal_stats(st, o)
                rows.append((t, [depth_jitter_cm(t["temporal_std_px"], fb, z) for z in (1, 2, 3)]))
            moved = float(np.median(np.std(np.stack(G), axis=0)))
            nm = noise_map(stacks[which], oks[which])
        bright = float(np.mean(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)))
        trusted_pct = float(ok.mean() * 100)
        pw, ph = 480, 360
        pan = lambda im: cv2.resize(im, (pw, ph), interpolation=cv2.INTER_AREA)
        left = pan(l.copy())
        cv2.putText(left, "left camera", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        d_st = pan(colour_depth(ds, ok)); d_fu = pan(colour_depth(fused))
        cv2.putText(d_st, f"raw stereo depth ({trusted_pct:.0f}% trusted)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(d_fu, "fused depth", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        nmp = pan(cv2.resize(nm, (640, 480), interpolation=cv2.INTER_NEAREST)) if enough else np.zeros((ph, pw, 3), np.uint8)
        cv2.putText(nmp, f"NOISE MAP: {NAMES[which]} (blue steady, red jumpy)" if enough else f"collecting {len(S)}/10 frames",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        top = np.hstack([left, d_st])
        bottom = np.hstack([d_fu, nmp])
        pad = np.zeros((230, pw * 2, 3), np.uint8)
        lines = []
        modes = ("gamma " if gamma else "") + ("clahe " if clahe else "") + ("avg3 " if avg3 else "")
        lines.append((f"picture brightness {bright:.0f}/255" + ("  <- DARK: add light, or raise gain/exposure on the Pi" if bright < 70 else "")
                      + f"    input fixes on: {modes or 'none'}" + (f"   [{note}]" if note else ""), (0, 140, 255) if bright < 70 else (200, 200, 200)))
        if enough:
            steady = moved < 4
            lines.append((f"picture changed by {moved:.1f} grey levels over the window: " + ("STEADY, numbers are sensor noise" if steady
                          else "MOVING, numbers include motion"), (0, 220, 0) if steady else (0, 140, 255)))
            lines.append((f"{'':16s}{'flicker px':>11s}{'f2f px':>9s}   {'cm of depth jitter at 1 / 2 / 3 m':>36s}", (200, 200, 200)))
            for (t, j), name in zip(rows, NAMES):
                lines.append((f"{name:16s}{t['temporal_std_px']:11.3f}{t['frame_to_frame_px']:9.3f}   {j[0]:8.1f} / {j[1]:5.1f} / {j[2]:5.1f}",
                              (255, 255, 255)))
            lines.append((f"mono scale a={info['a']:.2f} b={info['b']:+.1f}px (smoothed)   mono-only pixels {info['from_mono'].mean() * 100:.0f}%   "
                          f"{len(S)} frames, {1 / max(time.time() - last_t, 1e-3):.1f} fps", (200, 200, 200)))
        last_t = time.time()
        for i, (txt, col) in enumerate(lines):
            cv2.putText(pad, txt, (10, 24 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        frame = np.vstack([top, bottom, pad])
        if a.out:
            if writer is None:
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 8, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        n += 1
        if not a.no_window:
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            reset = False
            if k == ord("n"):
                which = (which + 1) % 4
            if k == ord("g"):
                gamma, reset = not gamma, True
            if k == ord("c"):
                clahe, reset = not clahe, True
            if k == ord("t"):
                avg3, reset = not avg3, True
            if k == ord("r"):
                reset = True
            if reset:
                for q in (S, OK, AL, FU, G, SM, SMOK):
                    q.clear()
                smoother.reset()
                note = "window reset"
            if k == ord("s"):
                os.makedirs("data", exist_ok=True)
                cv2.imwrite(f"data/noise_{int(time.time())}.png", frame)
            if k == ord("w") and IM:
                os.makedirs("data", exist_ok=True)
                path = f"data/noise_rec_{int(time.time())}.npz"
                np.savez_compressed(path, left=np.stack([x[0] for x in IM]), right=np.stack([x[1] for x in IM]), f=f, B=B)
                note = f"saved {len(IM)} pairs to {path}"
                print(note, flush=True)
        if a.max_frames and n >= a.max_frames:
            break
    rx.stop()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
