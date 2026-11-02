"""Undo the capture-time offset between the left and right camera.

The two USB webcams are not synchronised. gabai/pi/cams.py pairs the nearest frames, so the right picture is still up to ~17 ms
older or newer than the left one. While the rig moves, the scene shifts between those two instants and the stereo model reads that
shift as depth: f * omega * dt px for a turn (10 px at 57 deg/s and 16 ms), a depth-dependent amount for a sideways move. Rows that
no longer line up (a pitch turn) hurt the matching even more than the column error.

The Pi sends the signed offset dt = t_right - t_left with every pair. The motion between the two last left pictures gives the image
velocity of every pixel (dense DIS flow on half-size copies, 2-3 ms); the right picture is sampled where that velocity says the scene
was at the left camera's instant. Only the right picture changes and only when the offset and the motion are big enough to matter.
"""
import cv2
import numpy as np


class SkewCompensator:
    def __init__(self, scale=0.5, max_gap_s=0.15, min_dt_ms=1.5, min_shift_px=0.3, max_frac=1.5):
        self.scale, self.max_gap, self.min_dt, self.min_shift, self.max_frac = scale, max_gap_s, min_dt_ms, min_shift_px, max_frac
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.prev = self.t_prev = self.grid = None

    def _gray(self, img):
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def apply(self, left, right, t, dt_ms):
        """(right picture aligned to the left camera's instant, info). t = left capture time in s, dt_ms = right minus left in ms or None.
        info['why'] says why nothing was changed; info['shift_px'] is the 95th percentile of the correction otherwise."""
        g = self._gray(left)
        prev, t_prev = self.prev, self.t_prev
        self.prev, self.t_prev = g, t
        if dt_ms is None:
            return right, {"applied": False, "why": "the sender did not send the left/right offset"}
        if abs(dt_ms) < self.min_dt:
            return right, {"applied": False, "why": f"offset {dt_ms:.1f} ms is below {self.min_dt} ms"}
        if prev is None or prev.shape != g.shape or not 0.0 < t - t_prev <= self.max_gap:
            return right, {"applied": False, "why": "no left picture close enough in time to measure the motion"}
        frac = float(np.clip(dt_ms / 1000.0 / (t - t_prev), -self.max_frac, self.max_frac))
        f = self.flow.calc(prev, g, None) * (frac / self.scale)          # px the scene moved at full size during dt
        h, w = right.shape[:2]
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        mag = np.hypot(f[..., 0], f[..., 1])
        p95 = float(np.percentile(mag[::4, ::4], 95))
        if p95 < self.min_shift:
            return right, {"applied": False, "why": f"motion during the offset is {p95:.2f} px, too small to matter", "shift_px": p95}
        if self.grid is None or self.grid[0].shape != (h, w):
            self.grid = tuple(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)))
        out = cv2.remap(right, self.grid[0] + f[..., 0], self.grid[1] + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out, {"applied": True, "shift_px": p95, "dt_ms": float(dt_ms)}
