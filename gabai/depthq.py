"""Why a pixel has no depth.

A stereo hole has several unrelated causes and each needs a different fix, so the pipeline names the cause for every pixel it drops:
  outside the right view  the left pixel x would land left of the right picture's edge (x < disparity): no camera sees it twice. The band is as
                          wide as the disparity, so it grows as the scene gets closer. Nothing can recover it from one pair.
  no texture              left/right disagree on a flat patch (blank wall, black plastic, glare): the picture holds nothing to match.
  left/right disagree     the same test failed where there IS texture: occlusion next to an edge, a repeated pattern, a pair that is out of
                          step in time (motion), or a bad rectification.
  small island removed    a speckle filter dropped a few trusted pixels that jump away from every neighbour.
  monocular veto          the optional monocular depth model contradicted the stereo value.
  filled by monocular     no stereo measurement, and the monocular map (scaled to the stereo pixels) supplies the depth: marked so it is never mistaken for a measurement.
"""
import cv2
import numpy as np

LABELS = ("trusted", "outside the right view", "no texture", "left/right disagree", "small island removed", "monocular veto", "filled by monocular")
COLORS = np.array([(60, 200, 60), (70, 110, 240), (130, 130, 130), (230, 60, 60), (255, 170, 40), (190, 100, 255), (60, 220, 220)], np.uint8)   # RGB
TRUSTED, OUTSIDE, FLAT, DISAGREE, SPECKLE, MONO, FILLED = range(7)
FLAT_GRAD = 4.0      # mean |gradient| (grey levels per pixel) in a 9x9 patch below which a patch counts as textureless; camera noise alone gives about 2


def texture(gray):
    g = gray.astype(np.float32)
    return cv2.blur(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)), (9, 9)) / 8.0


def reject_reasons(disp, lr_ok, after_mono, final, gray, filled=None):
    """uint8 label image (see LABELS). disp: the model's disparity; lr_ok: passed the left-right check; after_mono: still trusted after the
    monocular check; final: trusted after the speckle filter (what the map gets)."""
    h, w = disp.shape
    out = np.full((h, w), DISAGREE, np.uint8)
    out[texture(gray) < FLAT_GRAD] = FLAT
    out[(np.arange(w, dtype=np.float32)[None, :] - np.nan_to_num(disp)) < 0] = OUTSIDE
    out[lr_ok & ~after_mono] = MONO
    out[after_mono & ~final] = SPECKLE
    if filled is not None:
        out[filled] = FILLED
    out[final] = TRUSTED
    return out


def reason_fractions(labels):
    n = np.bincount(labels.ravel(), minlength=len(LABELS)) / labels.size
    return {name: float(v) for name, v in zip(LABELS, n)}


def paint(labels):
    """RGB picture of the label image."""
    return COLORS[labels]
