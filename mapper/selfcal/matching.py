"""From a recording to observations: which frames to use, and feature tracks across the left and right pictures and across frames.

Frame selection (each rule is a reason a frame would mislead the solve):
  synchronised  |left-right capture offset| <= max_dt_ms (pairs.csv dt_ms). The two cameras read their rows in opposite directions, so ANY motion turns into a row error; on top of that only
  still         the median optical flow from the previous frame is under max_flow_px: the rig was not moving, so the opposite rolling shutters and the offset do not bend the geometry
  sharp, lit    Laplacian variance >= min_sharp and mean brightness >= min_brightness (dark blank walls and blur give no reliable corners)
  spread        at least min_gap_s apart, and greedily the frames that add the most new image coverage
Correspondences: SIFT on a contrast-equalised picture, ratio test 0.75, mutual best match, then a fundamental-matrix RANSAC per picture pair, so a moving object or a wrong match that does not fit
the rigid two-view geometry is rejected before the solve; the bundle adjustment rejects what is left (> 3 px). Tracks: union of the matches; a track that would put two features of one picture on one
point is dropped whole (an ambiguous match), and tracks seen fewer than 3 times are not used (they constrain nothing).
"""
import cv2
import numpy as np

from mapper.selfcal.core import Obs


def _gray(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)


def frame_stats(rec, i, prev_gray=None):
    l, r = rec.raw(i)
    g = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
    flow = np.inf
    if prev_gray is not None:
        p0 = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 10)
        if p0 is not None and len(p0) >= 20:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, g, p0, None, winSize=(15, 15), maxLevel=2)
            ok = st[:, 0] == 1
            if ok.sum() >= 20:
                flow = float(np.median(np.linalg.norm((p1 - p0)[ok, 0], axis=1)))
    return {"i": i, "sharp": float(cv2.Laplacian(g, cv2.CV_64F).var()), "bright": float(g.mean()), "flow": flow, "dt": rec.dt_ms[i]}, g


def select_frames(rec, n=40, clip=None, max_flow_px=0.5, max_dt_ms=3.0, min_sharp=120.0, min_brightness=40.0, min_gap_s=0.4, exclude=(), log=print):
    a, b = clip or (0, rec.n)
    stats, prev = [], None
    for i in range(a, b):
        s, prev = frame_stats(rec, i, prev)
        stats.append(s)
    reasons = {"not still": 0, "not synchronised": 0, "blurred or dark": 0}
    ok = []
    for s in stats:
        if s["i"] in exclude:
            continue
        if not s["flow"] < max_flow_px:
            reasons["not still"] += 1
        elif s["dt"] is None or abs(s["dt"]) > max_dt_ms:
            reasons["not synchronised"] += 1
        elif s["sharp"] < min_sharp or s["bright"] < min_brightness:
            reasons["blurred or dark"] += 1
        else:
            ok.append(s)
    log(f"frames {a}..{b}: {len(ok)} usable; rejected {reasons}")
    ok.sort(key=lambda s: -s["sharp"])
    chosen = []
    for s in ok:                                                                 # sharpest first, at least min_gap_s from every chosen one
        if all(abs(rec.t[s["i"]] - rec.t[c["i"]]) >= min_gap_s for c in chosen):
            chosen.append(s)
        if len(chosen) >= n:
            break
    return sorted(c["i"] for c in chosen), stats


def _match(da, db, ka, kb, ratio=0.75, ransac_px=1.0):
    """Mutual ratio-test SIFT matches that fit one fundamental matrix. Returns index pairs (ia, ib)."""
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    ab = bf.knnMatch(da, db, k=2)
    ba = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(db, da, k=2) if len(m) == 2 and m[0].distance < ratio * m[1].distance}
    pairs = np.array([(m[0].queryIdx, m[0].trainIdx) for m in ab if len(m) == 2 and m[0].distance < ratio * m[1].distance and ba.get(m[0].trainIdx) == m[0].queryIdx], int)
    if len(pairs) < 15:
        return np.zeros((0, 2), int)
    pa = np.float32([ka[i].pt for i in pairs[:, 0]])
    pb = np.float32([kb[j].pt for j in pairs[:, 1]])
    F, m = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, ransac_px, 0.999)
    if F is None or m is None:
        return np.zeros((0, 2), int)
    return pairs[m.ravel() > 0]


def build_obs(rec, frames, n_features=2000, temporal_window=6, min_temporal=40, log=print):
    """SIFT tracks over the chosen frames. Returns Obs (frame index = position in `frames`)."""
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02)
    kps, des = {}, {}
    for k, i in enumerate(frames):
        l, r = rec.raw(i)
        for side, img in ((0, l), (1, r)):
            kp, d = sift.detectAndCompute(_gray(img), None)
            kps[(k, side)], des[(k, side)] = kp, d
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    n_stereo, n_temporal, links = 0, 0, 0
    for k in range(len(frames)):
        m = _match(des[(k, 0)], des[(k, 1)], kps[(k, 0)], kps[(k, 1)])
        n_stereo += len(m)
        for a, b in m:
            union((k, 0, a), (k, 1, b))
        for j in range(k + 1, min(len(frames), k + 1 + temporal_window)):
            for side in (0, 1):
                m = _match(des[(k, side)], des[(j, side)], kps[(k, side)], kps[(j, side)])
                if len(m) < min_temporal:
                    continue
                links += 1
                n_temporal += len(m)
                for a, b in m:
                    union((k, side, a), (j, side, b))
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    frame, side, xy, point = [], [], [], []
    pid, conflicts = 0, 0
    for nodes in groups.values():
        imgs = [(f, s) for f, s, _ in nodes]
        if len(set(imgs)) != len(imgs):
            conflicts += 1
            continue
        if len(nodes) < 3:
            continue
        for f, s, kpi in nodes:
            frame.append(f); side.append(s); xy.append(kps[(f, s)][kpi].pt); point.append(pid)
        pid += 1
    log(f"matches: {n_stereo} stereo, {n_temporal} across frames ({links} frame pairs); {pid} tracks of 3+ views, {conflicts} ambiguous tracks dropped")
    if pid == 0:
        return None
    return Obs(frame, side, np.array(xy), point, len(frames), rec.size)
