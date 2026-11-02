"""Object detection adapter (spec 2a: YOLO masks for people) and the link between a detection and the depth map.

  * UltralyticsDetector wraps a YOLO segmentation model. It needs the weights file (models/yolo11n-seg.pt) and will NOT download
    anything by itself: if the file is missing it raises DetectorUnavailable with what to do. It has not been run on real footage
    here (no weights on this machine); the code path is exercised in tests only through stand-in detectors.
  * person_mask() turns detections into a pixel mask to keep people out of the obstacle grid.
  * det_depth() answers "how far is this detection": the median depth of the TRUSTED pixels inside it, with how many there were, how
    spread out they are, and a flag when the evidence is weak. Depth from the model is `estimated` (and `unavailable` when there are no
    trusted pixels), never a confidence score invented from the detector's own confidence.
"""
import os
from dataclasses import dataclass

import numpy as np

from gabai.diag.trace import ESTIMATED, q

PEOPLE = ("person",)
WEAK_VALID_FRACTION = 0.30
WIDE_SPREAD_M = 1.0


class DetectorUnavailable(RuntimeError):
    pass


@dataclass
class Det:
    label: str
    conf: float
    box: tuple              # x0, y0, x1, y1 in pixels
    mask: object = None     # bool array the size of the image, or None


class UltralyticsDetector:
    def __init__(self, weights=None, conf=0.35, imgsz=416, device="cuda"):
        weights = weights or os.environ.get("GABAI_YOLO") or os.path.join("models", "yolo11n-seg.pt")
        if not os.path.exists(weights):
            raise DetectorUnavailable(
                f"detector weights not found at '{weights}'. Gabai does not download models by itself. Download yolo11n-seg.pt (about 6 MB) from the "
                "Ultralytics releases page (github.com/ultralytics/assets/releases), save it as models/yolo11n-seg.pt or set GABAI_YOLO to its path. "
                "Without it people are not masked out of the grid and no detection-to-depth evidence is recorded.")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise DetectorUnavailable(f"ultralytics is not installed ({e})") from e
        self.model, self.conf, self.imgsz, self.device = YOLO(weights), conf, imgsz, device
        self.names = self.model.names

    def detect(self, bgr):
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        h, w = bgr.shape[:2]
        for i, b in enumerate(r.boxes):
            m = None
            if r.masks is not None:
                import cv2
                m = cv2.resize(r.masks.data[i].cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            out.append(Det(self.names[int(b.cls)], float(b.conf), tuple(float(x) for x in b.xyxy[0].tolist()), m))
        return out


def person_mask(dets, shape, classes=PEOPLE, grow_px=8):
    """Pixels covered by detected people (their mask, else their box), grown by `grow_px` so edges do not leak into the map."""
    m = np.zeros(shape, bool)
    for d in dets:
        if d.label not in classes:
            continue
        if d.mask is not None:
            m |= np.asarray(d.mask, bool)
        else:
            x0, y0, x1, y1 = (int(round(v)) for v in d.box)
            m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    if grow_px and m.any():
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1))
        m = cv2.dilate(m.astype(np.uint8), k).astype(bool)
    return m


def det_depth(det, disp, f, B, min_pixels=30):
    """Depth of a detection from the depth map. disp NaN = no trusted depth."""
    d = np.asarray(disp)
    h, w = d.shape
    if det.mask is not None:
        region = np.asarray(det.mask, bool)
    else:
        region = np.zeros((h, w), bool)
        x0, y0, x1, y1 = (int(round(v)) for v in det.box)
        region[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
    n_region = int(region.sum())
    ok = region & np.isfinite(d) & (d > 0.5)
    n = int(ok.sum())
    frac = n / n_region if n_region else 0.0
    out = {"label": det.label, "detector_conf": det.conf, "n_pixels": n, "region_pixels": n_region, "valid_fraction": round(frac, 3)}
    if n < min_pixels:
        out.update(depth_m=q(None, ESTIMATED), spread_m=None, weak=True, note=f"only {n} trusted depth pixels inside the detection: no depth given")
        return out
    z = f * B / d[ok]
    med, p10, p90 = (float(x) for x in (np.median(z), np.percentile(z, 10), np.percentile(z, 90)))
    spread = p90 - p10
    note = "ok"
    if spread > WIDE_SPREAD_M:
        note = f"depths inside the detection span {spread:.1f} m: part of it is probably background, so the median may not be the object"
    out.update(depth_m=q(round(med, 3), ESTIMATED, formula="median of f*B/d over trusted pixels inside the detection"), p10_m=round(p10, 3), p90_m=round(p90, 3),
               spread_m=round(spread, 3), weak=frac < WEAK_VALID_FRACTION, note=note)
    if out["weak"]:
        out["note"] = f"only {frac * 100:.0f}% of the detection has trusted depth; " + ("" if note == "ok" else note)
    return out
