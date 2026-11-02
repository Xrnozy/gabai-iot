"""Distance correction of the stereo model, learned from the checkerboard pairs of the stereo calibration (data/calib_pairs_15).

The board is flat and its square size is known, so solvePnP gives its true distance in every picture (gabai.tools.benchmark_depth.board_depth). The model's
disparity over the board is compared with the true disparity f*B/Z, and a ratio curve true/predicted is stored per model size. LAS2 compresses disparity a
little (it reads too far at the near end of the board range and too near at the far end), a bias that depends on distance, so one scale factor does not fix it.

The curve is only known where the board was seen (0.4 to 0.7 m in data/calib_pairs_15). Outside that range it tapers to 1 over 25% of the range
end, so nothing is extrapolated: a far wall is not 'corrected' by data that never saw one. Extend the range with a bigger board (gabai.tools.range_test) and refit.
"""
import json
import os

import numpy as np

DEFAULT_PATH = "data/calibration/depth_correction.json"


class DisparityCorrection:
    def __init__(self, knots_px, ratio, taper=0.25):
        self.knots, self.ratio, self.taper = np.asarray(knots_px, float), np.asarray(ratio, float), float(taper)

    def __call__(self, disp):
        """Corrected disparity; NaN and non-positive values pass through unchanged."""
        d = np.asarray(disp, np.float32)
        lo, hi = float(self.knots[0]), float(self.knots[-1])
        r = np.interp(d, self.knots, self.ratio)
        t = np.where(d < lo, (d - lo * (1 - self.taper)) / (lo * self.taper), np.where(d > hi, (hi * (1 + self.taper) - d) / (hi * self.taper), 1.0))
        out = d * (1.0 + (r - 1.0) * np.clip(t, 0.0, 1.0))
        return np.where(np.isfinite(d) & (d > 0), out, d).astype(np.float32)

    def to_dict(self):
        return {"knots_px": [round(float(k), 3) for k in self.knots], "ratio": [round(float(r), 5) for r in self.ratio], "taper": self.taper}


def fit_knots(pred, true, n_knots=3, min_samples=50):
    """pred, true: matching predicted and true disparities (px). Knots sit at quantiles of the predicted disparity; the ratio at each is the median of true/pred
    over the pixels near it."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    qs = np.quantile(pred, np.linspace(0.05, 0.95, n_knots))
    ratio = []
    for q in qs:
        s = np.abs(pred - q) < max(2.0, 0.08 * q)
        ratio.append(float(np.median(true[s] / pred[s])) if s.sum() >= min_samples else 1.0)
    return DisparityCorrection(qs, ratio)


def load(path=DEFAULT_PATH, model=None):
    """The correction stored for `model` ('s', 'm' or 'h'), or None when the file or that model's entry is missing."""
    if not path or not os.path.exists(path):
        return None
    entry = json.load(open(path, encoding="utf-8")).get("models", {}).get(model)
    return None if entry is None else DisparityCorrection(entry["knots_px"], entry["ratio"], entry.get("taper", 0.25))
