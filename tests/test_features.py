import cv2
import numpy as np

from rtabmap_lab.features import FeatureTracker


def textured(shift=0.0):
    rng = np.random.default_rng(3)
    big = cv2.GaussianBlur((rng.random((420, 700)) * 255).astype(np.uint8), (0, 0), 1.5)
    M = np.float32([[1, 0, -20 - shift], [0, 1, -20]])
    return cv2.warpAffine(big, M, (640, 360))


def test_features_are_scored_tracked_and_aged():
    ft = FeatureTracker(max_features=400, show=100)
    r0 = ft.update(textured(0))
    assert r0["stats"]["features"] > 100 and r0["stats"]["tracked"] == 0 and r0["score"].max() == 1.0 and (np.diff(r0["score"]) <= 1e-6).all()
    for k in range(1, 6):
        r = ft.update(textured(2.0 * k))                       # the picture slides 2 px a step
    st = r["stats"]
    assert st["tracked_share"] > 0.8 and st["age_max"] >= 4 and st["empty_cells"] <= 2
    assert st["top20_survive"] is not None and st["top20_survive"] > 0.8
    assert len(r["trails"][0]) >= 2
