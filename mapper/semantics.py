"""Per-pixel class evidence from an instance segmenter, and what to read out of a labelled surfel map.

The CNN of SemanticFusion gives a class distribution per pixel. Here a YOLO11n-seg instance mask with score c gives its pixels the distribution  p_class = min(c, 0.9),  the rest spread evenly;
a pixel in no mask gets a modest background vote; a pixel in a mask of a class we do not track gives no evidence (a uniform distribution multiplies nothing). Classes are the indoor things the egocentric
scene graph needs. The map stores only (class index, confidence) per grid cell and expands to probabilities at integration (`expand`), so 260 nodes do not hold 260 full probability volumes.
"""
import numpy as np

CLASSES = ["background", "screen", "keyboard", "mouse", "cell phone", "book", "clock", "chair", "couch", "potted plant", "bed", "dining table", "bottle", "cup", "person"]
K = len(CLASSES)
COCO_TO_IDX = {n: i for i, n in enumerate(CLASSES) if i} | {"tv": 1, "laptop": 1}      # a monitor and a laptop screen are one class: the detector flips between them on the same surface
NONE = 255                                   # grid cell with no evidence (a mask of an untracked class)
BACKGROUND_VOTE = 0.5
PALETTE = np.array([[60, 60, 60], [230, 60, 60], [240, 220, 40], [160, 230, 60], [60, 220, 120], [40, 220, 220], [60, 160, 240], [60, 80, 240], [140, 80, 240],
                    [220, 80, 240], [240, 80, 160], [160, 110, 60], [200, 200, 120], [120, 200, 200], [255, 255, 255]], np.uint8)       # RGB per class


def expand(cls, conf, k=K):
    """Compact segmentation -> (h, w, k) float32 probabilities. cls uint8 (NONE = no evidence), conf float."""
    h, w = cls.shape
    out = np.full((h, w, k), 1.0 / k, np.float32)
    bg = cls == 0
    out[bg] = (1.0 - BACKGROUND_VOTE) / (k - 1)
    out[bg, 0] = BACKGROUND_VOTE
    for c in np.unique(cls):
        if c in (0, NONE) or c >= k:
            continue
        m = cls == c
        pc = np.minimum(conf[m].astype(np.float32), 0.9)
        out[m] = ((1.0 - pc) / (k - 1))[:, None]
        out[m, c] = pc
    return out


class Segmenter:
    """YOLO11n-seg (models/yolo11n-seg.pt, already used by gabai/detect.py) -> compact class grid at the surfel grid resolution."""

    def __init__(self, stride=2, conf=0.25, imgsz=640):
        from gabai.detect import UltralyticsDetector
        self.det, self.stride = UltralyticsDetector(conf=conf, imgsz=imgsz), stride

    def analyse(self, bgr):
        """One detector run for both uses: (class grid, detections). The detections feed the person tracker (mapper/dynamic.py)."""
        dets = self.det.detect(bgr)
        return self.grid(bgr, dets), dets

    def grid(self, bgr, dets=None):
        """Returns (cls uint8 (h/stride, w/stride), conf float16). The most confident instance wins an overlap."""
        import cv2
        h, w = bgr.shape[:2]
        cls = np.zeros((h, w), np.uint8)
        score = np.zeros((h, w), np.float32)
        for d in sorted(self.det.detect(bgr) if dets is None else dets, key=lambda d: d.conf):             # ascending: the last (most confident) is painted on top
            if d.mask is None:
                continue
            idx = COCO_TO_IDX.get(d.label, NONE)
            cls[d.mask] = idx
            score[d.mask] = d.conf
        cls[(cls == 0) & (score == 0)] = 0
        s = self.stride
        return cv2.resize(cls, (w // s, h // s), interpolation=cv2.INTER_NEAREST), cv2.resize(score, (w // s, h // s), interpolation=cv2.INTER_NEAREST).astype(np.float16)


def plane_fit(P):
    """Least-squares plane of points (n, 3): (normal, rms distance m, extent along the two in-plane axes m, 2nd-98th percentile)."""
    c = P.mean(axis=0)
    U, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = (P - c) @ Vt[2]
    ext = [float(np.percentile((P - c) @ Vt[i], 98) - np.percentile((P - c) @ Vt[i], 2)) for i in (0, 1)]
    return Vt[2], float(np.sqrt(np.mean(d ** 2))), ext


def split_valleys(P, bin_m=0.02, min_gap_bins=2, depth_frac=0.25, min_pts=200):
    """Index groups of one cluster cut at density valleys along its long horizontal axis: two monitors side by side that DBSCAN joined across the bezel become two. A valley is a run of
    at least min_gap_bins bins whose count is below depth_frac of the median of the non-empty bins, with min_pts on each side. Returns a list of index arrays (one when nothing splits)."""
    xy = P[:, :2] - P[:, :2].mean(0)
    ev, evec = np.linalg.eigh(np.cov(xy.T))
    t = xy @ evec[:, -1]
    if np.ptp(t) < 0.5:
        return [np.arange(len(P))]
    edges = np.arange(t.min(), t.max() + bin_m, bin_m)
    h, _ = np.histogram(t, edges)
    thr = depth_frac * np.median(h[h > 0])
    low = h < thr
    cuts, i = [], 0
    while i < len(h):
        if low[i]:
            j = i
            while j < len(h) and low[j]:
                j += 1
            if j - i >= min_gap_bins and i > 0 and j < len(h):
                cuts.append((edges[i] + edges[j]) / 2)
            i = j
        else:
            i += 1
    if not cuts:
        return [np.arange(len(P))]
    g = np.digitize(t, cuts)
    parts = [np.nonzero(g == k)[0] for k in range(len(cuts) + 1)]
    return [q for q in parts if len(q) >= min_pts] or [np.arange(len(P))]


def _emit(out, points, probs, m, k):
    Pm = points[m]
    lo, hi = np.percentile(Pm, 2, axis=0), np.percentile(Pm, 98, axis=0)
    out.append({"label": CLASSES[k], "conf": float(probs[m].mean()), "p_w": (lo + hi) / 2, "dims": hi - lo, "R_obj": np.eye(3), "members": m})


def instances(points, labels, probs, eps=0.10, min_pts=200):
    """Connected groups (DBSCAN, eps m) of surfels that share a class: one dict per group, ready for gabai.objects3d.ObjectMap.update (label, conf, p_w, dims, R_obj)
    plus the member indices. Background and unknown surfels do not form instances."""
    import open3d as o3d
    out = []
    for k in range(1, K):
        sel = np.nonzero(labels == k)[0]
        if len(sel) < min_pts:
            continue
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[sel].astype(np.float64)))
        lab = np.asarray(pc.cluster_dbscan(eps=eps, min_points=10))
        for c in range(lab.max() + 1 if len(lab) else 0):
            m = sel[lab == c]
            if len(m) < min_pts:
                continue
            for q in split_valleys(points[m], min_pts=min_pts):
                _emit(out, points, probs, m[q], k)
    return out
