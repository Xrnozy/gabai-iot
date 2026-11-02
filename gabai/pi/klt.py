"""Optional Shi-Tomasi + pyramidal Lucas-Kanade tracking on the Pi (`sender --klt`). Off unless asked for.

The sender forwards the camera's JPEG bytes without decoding. Tracking has to decode the left picture (and the right one with --klt-stereo), so it
asks the JPEG decoder for a reduced-size grayscale picture (IMREAD_REDUCED_GRAYSCALE_2 at scale 0.5): the decoder skips most of the work, no BGR picture
and no resize are made. Features are detected only when too few are left, and only on the part of the picture not already covered by a track.

Coordinates leave this module in the full sent picture, OpenCV convention (pixel centre at the integer). A point found at p in a picture reduced by s
sits at (p + 0.5) * s - 0.5 in the full one; the Rerun viewer adds the 0.5 back because Rerun puts pixel centres at half integers.

  python3 -m gabai.pi.klt            time every stage on synthetic frames, no cameras needed (use it on the Pi)

MotionEstimator (sender --klt-motion) turns the stereo tracks into the camera's own motion between two sent pairs, on the Pi, so the server gets a finished measurement
and not only points: the previous pair's tracked points get a depth from their left/right disparity, the current pair's tracked points are their pixels in the new picture,
and one PnP gives the motion. It costs one more point rectification and one PnP of about 100 points per pair.
"""
import time

import cv2
import numpy as np

from gabai.net.protocol import KLT_LOST, KLT_NEW, KLT_TRACKED

REDUCED = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4, 8: cv2.IMREAD_REDUCED_GRAYSCALE_8}


class KltResult:
    __slots__ = ("ids", "xy", "status", "xy_right", "ms", "n_tracked", "n_new", "n_lost", "stage_ms", "quality")

    def __init__(self, ids, xy, status, xy_right, ms, stage_ms, quality):
        self.ids, self.xy, self.status, self.xy_right, self.ms, self.stage_ms, self.quality = ids, xy, status, xy_right, ms, stage_ms, quality
        self.n_tracked = int((status == KLT_TRACKED).sum())
        self.n_new = int((status == KLT_NEW).sum())
        self.n_lost = int((status == KLT_LOST).sum())


class KltTracker:
    def __init__(self, width=640, height=360, scale=0.5, max_features=200, levels=2, win=15, min_dist=10, quality=0.01,
                 fb_px=0.0, replenish_frac=0.6, detect_gap=5, max_err=20.0, stereo=False):
        """max_err: drop a track whose patch differs from the old one by more than this many grey levels per pixel (LK's own error; right tracks measure 1-12,
        wrong ones 25-38 on the test scene). scale: tracking size / sent size (1, 0.5, 0.25 or 0.125: the sizes a JPEG decoder can produce directly).
        fb_px: forward-backward check threshold in tracking pixels, 0 = off (it doubles the LK cost).
        replenish_frac: detect again when fewer than this fraction of max_features is left; detect_gap: and at most every this many frames."""
        self.k = int(round(1.0 / scale))
        if self.k not in REDUCED:
            raise ValueError("scale must be 1, 0.5, 0.25 or 0.125")
        self.w, self.h, self.stereo = width, height, stereo
        self.tw, self.th = width // self.k, height // self.k
        self.max, self.min_dist, self.quality = int(max_features), int(min_dist), float(quality)
        self.lk = dict(winSize=(win, win), maxLevel=max(0, levels - 1), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        self.fb_px, self.max_err, self.detect_gap = float(fb_px), float(max_err), int(detect_gap)
        self.want = int(replenish_frac * self.max)
        self.mask = np.empty((self.th, self.tw), np.uint8)               # reused every detection
        self.pts = np.empty((self.max, 1, 2), np.float32)                # live tracks, first self.n rows
        self.ids = np.empty(self.max, np.int64)
        self.err = np.zeros(self.max, np.uint8)                          # LK patch error x 8 of each live track, 255 = worst; 0 for a point found this frame
        self.reset()

    def reset(self):
        """Forget all tracks (a new connection: the old picture is too old to track from)."""
        self.n, self.prev, self.next_id, self.since_detect = 0, None, 0, 10 ** 9

    def decode(self, jpeg):
        g = cv2.imdecode(np.frombuffer(jpeg, np.uint8), REDUCED[self.k])
        if g is None or g.shape != (self.th, self.tw):
            raise ValueError(f"cannot decode the picture at {self.tw}x{self.th}")
        return g

    def to_full(self, p):
        return (p + 0.5) * self.k - 0.5

    def _detect(self, gray, room):
        self.mask.fill(255)
        for x, y in self.pts[:self.n, 0]:
            cv2.circle(self.mask, (int(x), int(y)), self.min_dist, 0, -1)
        c = cv2.goodFeaturesToTrack(gray, room, self.quality, self.min_dist, mask=self.mask)
        return np.empty((0, 1, 2), np.float32) if c is None else c

    def update(self, left_jpeg, right_jpeg=None):
        t0 = time.perf_counter()
        gray = self.decode(left_jpeg)
        t_dec = time.perf_counter()
        lost_xy = np.empty((0, 2), np.float32)
        lost_id = np.empty(0, np.int64)
        t_trk = t_dec
        if self.prev is not None and self.n:
            p0 = self.pts[:self.n]
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None, **self.lk)
            ok = (st[:, 0] == 1) & (err[:, 0] < self.max_err)
            if self.fb_px > 0:
                pb, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **self.lk)
                ok &= (stb[:, 0] == 1) & (np.abs(pb - p0).max(axis=(1, 2)) < self.fb_px)
            x, y = p1[:, 0, 0], p1[:, 0, 1]
            ok &= (x >= 0) & (y >= 0) & (x <= self.tw - 1) & (y <= self.th - 1)
            lost_xy, lost_id = p0[~ok, 0], self.ids[:self.n][~ok]
            keep = int(ok.sum())
            self.pts[:keep] = p1[ok]
            self.ids[:keep] = self.ids[:self.n][ok]
            self.err[:keep] = np.clip(err[ok, 0] * 8.0, 0, 255)
            self.n = keep
            t_trk = time.perf_counter()
        n_old = self.n
        self.since_detect += 1
        if self.n < self.want and self.since_detect >= self.detect_gap:
            new = self._detect(gray, self.max - self.n)
            self.since_detect = 0
            m = len(new)
            if m:
                self.pts[self.n:self.n + m] = new
                self.ids[self.n:self.n + m] = np.arange(self.next_id, self.next_id + m)
                self.err[self.n:self.n + m] = 0
                self.next_id += m
                self.n += m
        t_det = time.perf_counter()
        self.prev = gray
        n = self.n
        ids = np.concatenate([self.ids[:n], lost_id])
        status = np.concatenate([np.where(np.arange(n) < n_old, KLT_TRACKED, KLT_NEW), np.full(len(lost_id), KLT_LOST)]).astype(np.uint8)
        xy = self.to_full(np.concatenate([self.pts[:n, 0], lost_xy])).astype(np.float32)
        quality = np.concatenate([self.err[:n], np.full(len(lost_id), 255, np.uint8)])
        right = None
        if self.stereo and right_jpeg is not None:
            right = np.full((len(ids), 2), np.nan, np.float32)
            if n:
                gr = self.decode(right_jpeg)
                pr, st, err = cv2.calcOpticalFlowPyrLK(gray, gr, self.pts[:n], None, **self.lk)
                ok = (st[:, 0] == 1) & (err[:, 0] < self.max_err) & (pr[:, 0, 0] >= 0) & (pr[:, 0, 1] >= 0) & (pr[:, 0, 0] <= self.tw - 1) & (pr[:, 0, 1] <= self.th - 1)
                right[:n][ok] = self.to_full(pr[ok, 0])
        t_end = time.perf_counter()
        stage = (round((t_dec - t0) * 1e3, 2), round((t_trk - t_dec) * 1e3, 2), round((t_det - t_trk) * 1e3, 2), round((t_end - t_det) * 1e3, 2))
        return KltResult(ids, xy, status, right, (t_end - t0) * 1e3, stage, quality)


class MotionEstimator:
    def __init__(self, calib, sent_size, min_inliers=15, max_rms=1.5, z_min=0.3, z_max=6.0, row_tol=2.0):
        """calib: the stereo calibration file (the rectification of the sent pictures). min_inliers / max_rms: the motion counts only with this many points agreeing within this
        many pixels (RMS). z_min / z_max: depths (m) a stereo point may have. row_tol: a left/right pair of points must lie on the same row within this many pixels once rectified
        (a camera order that does not match the calibration fails this and gives no motion, never a wrong one)."""
        from gabai.ptmap import RectPoints
        self.rp = RectPoints(calib, sent_size)
        self.min_in, self.max_rms, self.z_min, self.z_max, self.row_tol = int(min_inliers), float(max_rms), float(z_min), float(z_max), float(row_tol)
        self.prev = None                                                  # (ids ascending, X in the previous left camera's frame, its time, its frame number)

    def reset(self):
        self.prev = None

    def update(self, r, t, seq):
        """r: KltResult of this pair, t: its capture time (s), seq: its frame number. Returns the motion dict of protocol.pack_klt, or None."""
        live = r.status != KLT_LOST
        ids, ul = r.ids[live], self.rp(r.xy[live], "L")
        tracked = np.nonzero(r.status[live] == KLT_TRACKED)[0]
        motion = None
        if self.prev is not None and len(tracked):
            pid, X, tp, sp = self.prev
            _, ia, ib = np.intersect1d(pid, ids[tracked], return_indices=True)
            if len(ia) >= self.min_in:
                motion = self._pnp(X[ia], ul[tracked[ib]], t - tp, sp)
        self.prev = self._depth(ids, ul, r.xy_right[live] if r.xy_right is not None else None, t, seq)
        return motion

    def _pnp(self, X, uv, dt, prev_seq):
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(X, uv, self.rp.K, None, iterationsCount=60, reprojectionError=2.0, confidence=0.99, flags=cv2.SOLVEPNP_SQPNP)
            if not ok or inl is None or len(inl) < self.min_in:
                return None
            k = inl.ravel()
            rvec, tvec = cv2.solvePnPRefineLM(X[k], uv[k], self.rp.K, None, rvec, tvec)
        except cv2.error:
            return None
        proj = cv2.projectPoints(X[k], rvec, tvec, self.rp.K, None)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((proj - uv[k]) ** 2, axis=1))))
        if rms > self.max_rms:
            return None
        Rm = cv2.Rodrigues(rvec)[0]                                       # X_cur = Rm X_prev + tvec; the motion is the current camera seen from the previous one
        return {"prev_seq": int(prev_seq), "dt": float(dt), "rvec": -rvec.ravel(), "t": -Rm.T @ tvec.ravel(), "n_inl": len(k), "rms": rms}

    def _depth(self, ids, ul, xy_right, t, seq):
        if xy_right is None:
            return None
        ur = self.rp(xy_right, "R")
        d = ul[:, 0] - ur[:, 0]
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(d) & (d > 0.5) & (np.abs(ul[:, 1] - ur[:, 1]) < self.row_tol)
            z = np.where(ok, self.rp.f * self.rp.B / np.where(ok, d, 1.0), np.nan)
            ok &= (z >= self.z_min) & (z <= self.z_max)
        if ok.sum() < self.min_in:
            return None
        o = np.argsort(ids[ok])
        zz, u = z[ok][o], ul[ok][o]
        X = np.stack([(u[:, 0] - self.rp.cx) * zz / self.rp.f, (u[:, 1] - self.rp.cy) * zz / self.rp.f, zz], 1)
        return ids[ok][o], X, t, seq


def _synthetic(w, h, shift, seed=1):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 255, (h + 40, w + 40), dtype=np.uint8), (0, 0), 2.0)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    out = base[20:20 + h, 20 + shift:20 + shift + w]
    return cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--fb-px", type=float, default=0.0)
    ap.add_argument("--stereo", action="store_true")
    a = ap.parse_args()
    cv2.setNumThreads(1)
    jp = [_synthetic(640, 360, i) for i in range(12)]
    order = list(range(12)) + list(range(10, 0, -1))                  # back and forth, so every step is a small move
    tr = KltTracker(scale=a.scale, max_features=a.max, fb_px=a.fb_px, stereo=a.stereo)
    rows = []
    for i in range(a.frames):
        r = tr.update(jp[order[i % len(order)]], jp[order[(i + 3) % len(order)]] if a.stereo else None)
        rows.append((r.ms, *r.stage_ms, r.n_tracked, r.n_new))
    m = np.median(np.array(rows[10:]), axis=0)
    print(f"median over {a.frames - 10} frames: total {m[0]:.2f} ms = decode {m[1]:.2f} + track {m[2]:.2f} + detect {m[3]:.2f} + right {m[4]:.2f}; "
          f"tracked {m[5]:.0f}, new per frame {m[6]:.0f}  (scale {a.scale}, max {a.max}, fb {a.fb_px}, stereo {a.stereo})")


if __name__ == "__main__":
    main()
