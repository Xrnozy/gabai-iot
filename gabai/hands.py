"""Hand keypoints (spec phase 3): MediaPipe Hands 21 points, lifted to 3D with the stereo depth, plus an open/closed reading.

  * HandTracker wraps the MediaPipe HandLandmarker. It needs checkpoints/hand_landmarker.task and never downloads anything itself.
  * lift_hand() gives each keypoint a camera-frame xyz in metres from the median TRUSTED disparity in a small window. A keypoint with
    too few trusted pixels, or whose depth is far from the rest of the hand (background bleeding through between fingers), is NaN
    instead of a guess. Lifted values are `estimated`; the pixel positions are `model_prediction`.
  * finger_state() reads open/closed from how far the four fingertips are from the wrist compared with the knuckles, in image pixels.
    It is a rule of thumb with named thresholds, not a trained classifier, and a hand pointing at the camera is foreshortened, so the
    state can be wrong then. In between the two thresholds the state is neither open nor closed.
  * MediaPipe's left/right label assumes a mirrored selfie image. This camera is not mirrored, so the label is kept as given
    (`side`) and not used by any rule here.
"""
import os
from dataclasses import dataclass

import numpy as np

from gabai.diag.trace import ESTIMATED

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
TIPS, KNUCKLES = (8, 12, 16, 20), (5, 9, 13, 17)
OPEN_RATIO, CLOSED_RATIO = 1.6, 1.2
WINDOW = 2                  # half-size of the depth window around a keypoint (5 x 5 pixels)
MIN_WINDOW_PX = 5
MAX_OFF_HAND_M = 0.25       # a keypoint this far in depth from the hand's median is treated as background
DEFAULT_MODEL = os.path.join("checkpoints", "hand_landmarker.task")


class HandsUnavailable(RuntimeError):
    pass


@dataclass
class Hand:
    side: str               # MediaPipe's label, see the module note
    kp2d: object            # (21, 2) pixels
    score: float            # MediaPipe's own handedness score; not a probability that the hand is real
    z_rel: object = None    # (21,) MediaPipe's relative depth of each keypoint, wrist-relative, in about the same units as x (a fraction of the image width); a model estimate


class HandTracker:
    def __init__(self, model_path=None, num_hands=2, min_conf=0.5):
        model_path = model_path or os.environ.get("GABAI_HAND_MODEL") or DEFAULT_MODEL
        if not os.path.exists(model_path):
            raise HandsUnavailable(
                f"hand model not found at '{model_path}'. Gabai does not download models by itself: download hand_landmarker.task (about 8 MB) from "
                "storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task, save it as "
                "checkpoints/hand_landmarker.task, or set GABAI_HAND_MODEL to its path.")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise HandsUnavailable(f"mediapipe is not installed ({e}). pip install mediapipe") from e
        self._mp = mp
        opts = vision.HandLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.VIDEO,
                                            num_hands=num_hands, min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
                                            min_tracking_confidence=min_conf)
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ms = -1

    def detect(self, bgr, t_ms):
        """Hands in one frame. t_ms must increase from call to call (MediaPipe's video mode requirement)."""
        import cv2
        t_ms = int(max(t_ms, self._last_ms + 1))
        self._last_ms = t_ms
        h, w = bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(img, t_ms)
        out = []
        for lms, cats in zip(res.hand_landmarks, res.handedness):
            out.append(Hand(cats[0].category_name, np.array([[p.x * w, p.y * h] for p in lms], float), float(cats[0].score), np.array([p.z for p in lms], float)))
        return out


def open_ratio(hand):
    """Mean over the four fingers of (fingertip to wrist) / (knuckle to wrist) in the image. About 1.7-2 for a flat open hand, below 1.2 for a fist."""
    k = np.asarray(hand.kp2d, float)
    wr = k[0]
    r = [np.linalg.norm(k[t] - wr) / max(np.linalg.norm(k[m] - wr), 1e-6) for t, m in zip(TIPS, KNUCKLES)]
    return float(np.mean(r))


def finger_state(hand):
    r = open_ratio(hand)
    return {"ratio": round(r, 3), "open": r >= OPEN_RATIO, "closed": r <= CLOSED_RATIO, "open_threshold": OPEN_RATIO, "closed_threshold": CLOSED_RATIO}


def lift_hand(hand, disp, f, B, cx, cy):
    """Camera-frame xyz (x right, y down, z forward, metres) per keypoint, NaN where depth is not trusted."""
    d = np.asarray(disp)
    h, w = d.shape
    xyz = np.full((21, 3), np.nan)
    z = np.full(21, np.nan)
    for i, (u, v) in enumerate(np.asarray(hand.kp2d, float)):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            continue
        win = d[max(vi - WINDOW, 0):vi + WINDOW + 1, max(ui - WINDOW, 0):ui + WINDOW + 1]
        ok = win[np.isfinite(win) & (win > 0.5)]
        if ok.size >= MIN_WINDOW_PX:
            z[i] = f * B / float(np.median(ok))
    med = np.nanmedian(z) if np.isfinite(z).any() else np.nan
    for i, (u, v) in enumerate(np.asarray(hand.kp2d, float)):
        if np.isfinite(z[i]) and abs(z[i] - med) <= MAX_OFF_HAND_M:
            xyz[i] = ((u - cx) / f * z[i], (v - cy) / f * z[i], z[i])
    valid = float(np.isfinite(xyz[:, 2]).mean())
    return {"xyz": xyz, "valid_fraction": round(valid, 3), "kind": ESTIMATED,
            "note": "median trusted disparity in a 5x5 window per keypoint; keypoints more than %.2f m from the hand's median depth are dropped" % MAX_OFF_HAND_M}


def hand_mask(hands, shape, grow_px=10):
    """Pixels covered by the hands (convex hull of the keypoints, grown). Cut these out before fusing depth into a map or before tracking features."""
    import cv2
    m = np.zeros(shape, np.uint8)
    for hnd in hands:
        pts = np.round(np.asarray(hnd.kp2d)).astype(np.int32)
        cv2.fillConvexPoly(m, cv2.convexHull(pts), 1)
    if grow_px and m.any():
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1)))
    return m.astype(bool)
