"""Does running LAS2 on a larger copy of the 640x360 pictures give better depth (thin structures, fan grills, bars)? Upscaling adds no information, but the model
matches on a coarse feature grid and a bar that is 2 px wide at 640 is 4 px wide at 1280.

    py scripts/bench_input_scale.py [--frames 10]

data/left.mp4 and right.mp4 are 1280x720, so a better reference exists: LAS2-H run on the native 1280x720 pair. The pictures the Pi sends are simulated by shrinking them to 640x360.
Variants run on the 640x360 pair: as it is, upscaled to 960x540, upscaled to 1280x720 (all results brought back to 640 wide, disparities scaled). Each is compared with the
native-1280 result on the pixels both trust. Writes data/bench/input_scale.json.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from gabai.pipeline.config import PipelineConfig      # noqa: E402
from gabai.pipeline.sources import open_source        # noqa: E402
from gabai.spike.las2_runner import Las2Depth         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=10), ap.add_argument("--model", default="h"), ap.add_argument("--ref-model", default="h", help="the model that makes the native-1280 reference")
    a = ap.parse_args()
    cfg = PipelineConfig.load("config/pipeline.toml", {"source.start": 60.0, "source.skip": 12, "source.max_frames": a.frames, "stereo.model": a.ref_model, "stereo.width": 1280})
    frames = list(open_source(cfg))
    las = Las2Depth(a.model)
    las_ref = las if a.ref_model == a.model else Las2Depth(a.ref_model)

    def run(l, r, m=None):
        t = time.perf_counter()
        d, ok, _ = (m or las).disparity_lr_batched(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        return d, ok, (time.perf_counter() - t) * 1000
    small = lambda im: cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA)
    res = {"as it is (640x360)": [], "upscaled to 960x540": [], "upscaled to 1280x720": []}
    ref_ok_total = []
    for fr in frames:
        dref, okref, _ = run(fr.left, fr.right, las_ref)
        dref640 = cv2.resize(dref, (640, 360), interpolation=cv2.INTER_AREA) * 0.5
        okref640 = cv2.resize(okref.astype(np.uint8), (640, 360), interpolation=cv2.INTER_AREA) > 0.99
        l, r = small(fr.left), small(fr.right)
        for name, w in (("as it is (640x360)", 640), ("upscaled to 960x540", 960), ("upscaled to 1280x720", 1280)):
            if w == 640:
                ll, rr = l, r
            else:
                ll, rr = (cv2.resize(x, (w, w * 9 // 16), interpolation=cv2.INTER_CUBIC) for x in (l, r))
            d, ok, ms = run(ll, rr)
            if w != 640:
                d = cv2.resize(d, (640, 360), interpolation=cv2.INTER_AREA) * (640.0 / w)
                ok = cv2.resize(ok.astype(np.uint8), (640, 360), interpolation=cv2.INTER_AREA) > 0.99
            both = ok & okref640
            err = np.abs(d - dref640)[both]
            res[name].append((ok.mean(), float(np.median(err)), float((err > 1).mean()), float((ok & okref640).sum() / max(okref640.sum(), 1)), ms))
    out = {k: {"valid_pct": round(float(np.mean([x[0] for x in v])) * 100, 1), "median_err_vs_native_1280_px_at_640": round(float(np.mean([x[1] for x in v])), 3),
               "over_1px_pct": round(float(np.mean([x[2] for x in v])) * 100, 1), "share_of_native_1280_trusted_pixels_also_trusted_pct": round(float(np.mean([x[3] for x in v])) * 100, 1),
               "ms_per_pair": round(float(np.median([x[4] for x in v])), 1)} for k, v in res.items()}
    json.dump(out, open(os.path.join("data", "bench", f"input_scale_{a.model}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
