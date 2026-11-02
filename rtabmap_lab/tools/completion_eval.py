"""How accurate is each layer of the depth completion? A hold-out test on real node pictures: blocks of depth that IS measured are hidden, the layer under test fills them, and the fill is compared with what was
hidden. The hidden value is RTAB-Map's own depth, itself noisy (about 0.5 px), so the numbers are an agreement, not an absolute error; the comparison with plain inpainting (cv2.inpaint, no second source)
shows what the extra source adds.

  LAS2 layer: RTAB-Map depth hidden in blocks, LAS2 allowed to fill.
  mono layer: blocks of the RTAB-Map + LAS2 depth (the truth proxy here) hidden in both, only the two monocular maps may fill (scale fitted to everything outside the blocks). Blocks are --block rows x cols pixels:
              a small hole is easy for inpainting; the monocular maps are for the big ones.

  py -m rtabmap_lab.tools.completion_eval map.db --calib data/stereo_calib_charuco.npz
"""
import argparse

import cv2
import numpy as np


def blocks(valid, rng, n=14, h=36, w=48):
    """n random hidden blocks inside the area where `valid` (a boolean map) is mostly true."""
    H, W = valid.shape
    m = np.zeros((H, W), bool)
    tries = 0
    k = 0
    while k < n and tries < 400:
        tries += 1
        y, x = rng.integers(0, H - h), rng.integers(0, W - w)
        if valid[y:y + h, x:x + w].mean() > 0.9:
            m[y:y + h, x:x + w] = True
            k += 1
    return m


def score(fill_z, truth, hidden):
    ok = hidden & np.isfinite(truth)
    got = ok & np.isfinite(fill_z)
    if not got.any():
        return {"filled": 0.0, "med": float("nan"), "p90": float("nan")}
    e = np.abs(fill_z[got] - truth[got]) / truth[got]
    return {"filled": float(got.sum() / ok.sum()), "med": float(np.median(e) * 100), "p90": float(np.percentile(e, 90) * 100)}


def inpaint_baseline(z, hidden):
    zz = np.nan_to_num(z, nan=0.0).astype(np.float32)
    mask = (hidden | ~np.isfinite(z)).astype(np.uint8)
    out = cv2.inpaint((zz * 1000).clip(0, 65000).astype(np.uint16).astype(np.float32).astype(np.uint16) if False else (zz * 50).clip(0, 255).astype(np.uint8), mask, 5, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 50.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db")
    ap.add_argument("--calib")
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--block", type=int, nargs=2, default=[72, 96], metavar=("ROWS", "COLS"), help="size of the hidden blocks of the mono test")
    ap.add_argument("--orientation", default="as-is", choices=["as-is", "flip"])
    a = ap.parse_args(argv)
    from gabai.pipeline.config import PipelineConfig
    from gabai.spike.las2_runner import Las2Depth
    from gabai.spike.mono_depth import MonoDepth
    from gabai.tools.calibrate_stereo import Rectifier
    from rtabmap_lab.completion import complete
    from rtabmap_lab.tools.surface_lab import load_nodes
    rect = Rectifier(a.calib or PipelineConfig.load("config/pipeline.toml").source.calib)
    w, h = rect.size
    cal = {"f": rect.f, "B": rect.B, "cx": rect.cx, "cy": rect.cy}
    fb = cal["f"] * cal["B"]
    nodes = load_nodes(a.db, cal, (w, h))
    ids = sorted(nodes)[::max(1, len(nodes) // a.nodes)][:a.nodes]
    las, mono = Las2Depth("s"), MonoDepth("dav2")
    rng = np.random.default_rng(0)
    rows = {"las2": [], "las2 inpaint": [], "mono": [], "mono inpaint": []}
    for i in ids:
        depth, bgr, T, right = nodes[i]
        left_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB) if right.ndim == 2 else cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        r = np.where(depth > 0, depth.astype(np.float32) / 1000.0, np.nan)
        d, ok, _ = las.disparity_lr_batched(left_rgb, right_rgb)
        g = np.where(ok & (d > 0.5), fb / np.maximum(d, 0.5), np.nan).astype(np.float32)
        ml, mr = mono.depth(left_rgb), mono.depth(right_rgb)
        hidden = blocks(np.isfinite(r) & np.isfinite(g), rng)
        if hidden.any():
            r_h = np.where(hidden, np.nan, r)
            z, cls, info, _ = complete(r_h, g, fb)
            rows["las2"].append(score(z, r, hidden))
            rows["las2 inpaint"].append(score(inpaint_baseline(r_h, hidden), r, hidden))
        base, _, _, _ = complete(r, g, fb)                                   # RTAB-Map + LAS2: the truth proxy of the mono test
        hid2 = blocks(np.isfinite(base), rng, n=6, h=a.block[0], w=a.block[1])
        if hid2.any():
            r_h, g_h = np.where(hid2, np.nan, r), np.where(hid2, np.nan, g)
            z, cls, info, _ = complete(r_h, g_h, fb, mono_left=ml, mono_right=mr)
            rows["mono"].append(score(z, base, hid2))
            rows["mono inpaint"].append(score(inpaint_baseline(np.where(np.isfinite(r_h), r_h, g_h), hid2), base, hid2))
    print(f"{len(rows['las2'])} nodes for the LAS2 test, {len(rows['mono'])} for the mono test; error = |filled - hidden| / hidden")
    print(f"{'layer':16s} {'filled':>8s} {'median err':>11s} {'p90 err':>9s}")
    for k, v in rows.items():
        f = np.mean([x["filled"] for x in v]) * 100
        med = np.nanmedian([x["med"] for x in v])
        p90 = np.nanmedian([x["p90"] for x in v])
        print(f"{k:16s} {f:7.0f}% {med:10.1f}% {p90:8.1f}%")


if __name__ == "__main__":
    main()
