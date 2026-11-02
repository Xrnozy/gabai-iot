"""Optional intermediate outputs (depth pictures, left frames) saved for later inspection. OFF unless asked.

    GABAI_DIAG_SAVE="depth,frame"   what to save: depth = half-size coloured depth picture, frame = left image (JPEG), grid = top-down grid picture,
                                    disp = raw disparity as float16 .npz (half size)
    GABAI_DIAG_EVERY=30             one regular sample per this many frames
    GABAI_DIAG_MAX_MB=200           stop saving after this much disk is used (logged once)
`interesting=True` (a failed check, an alert) saves the frame even between samples, at most one per `min_interval_s`.
Raw camera footage is therefore only stored when you ask for "frame", and never beyond the budget.
"""
import os
import time

import cv2
import numpy as np

from gabai.diag import log as dlog


class ArtifactSaver:
    def __init__(self, out_dir, what=None, every=None, max_mb=None, min_interval_s=1.0):
        env = os.environ.get
        if what is None:
            what = [w.strip() for w in env("GABAI_DIAG_SAVE", "").split(",") if w.strip()]
        self.what = set(what)
        self.every = int(every or env("GABAI_DIAG_EVERY", 30))
        self.max_bytes = float(max_mb if max_mb is not None else env("GABAI_DIAG_MAX_MB", 200)) * 1e6
        self.min_interval = min_interval_s
        self.dir, self.used, self.n, self._last = out_dir, 0, 0, -1e9
        self.stopped_for_budget = False
        self._log = dlog.get("diag")
        if self.what:
            os.makedirs(out_dir, exist_ok=True)
            self._log.info("artifacts.on", what=sorted(self.what), every=self.every, max_mb=self.max_bytes / 1e6, dir=out_dir)

    def maybe_save(self, fid, frame_left=None, disp=None, interesting=False, grid_img=None):
        """Returns the list of files written (empty when off, not due, or over budget)."""
        if not self.what:
            return []
        due = self.n % self.every == 0
        self.n += 1
        now = time.monotonic()
        if not (due or (interesting and now - self._last >= self.min_interval)):
            return []
        if self.used >= self.max_bytes:
            if not self.stopped_for_budget:
                self.stopped_for_budget = True
                self._log.warning("artifacts.budget_reached", used_mb=self.used / 1e6, note="no more diagnostic images will be saved this run")
            return []
        self._last = now
        out = []
        tag = f"{fid}"
        if "depth" in self.what and disp is not None:
            d = np.asarray(disp, np.float32)
            ok = np.isfinite(d) & (d > 0.5)
            hi = float(np.percentile(d[ok], 98)) if ok.any() else 1.0
            img = cv2.applyColorMap(np.clip(np.where(ok, d, 0) * 255.0 / max(hi, 1.0), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            img[~ok] = 0
            out.append(self._write(f"{tag}_depth.png", cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), ".png"))
        if "frame" in self.what and frame_left is not None:
            out.append(self._write(f"{tag}_left.jpg", frame_left, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 70]))
        if "grid" in self.what and grid_img is not None:
            out.append(self._write(f"{tag}_grid.png", grid_img() if callable(grid_img) else grid_img, ".png"))   # a callable is drawn only when saving
        if "disp" in self.what and disp is not None:
            p = os.path.join(self.dir, f"{tag}_disp.npz")
            np.savez_compressed(p, disp=np.asarray(disp, np.float16)[::2, ::2])
            self.used += os.path.getsize(p)
            out.append(p)
        if out:
            self._log.debug("artifacts.saved", fid=fid, files=[os.path.basename(x) for x in out], used_mb=round(self.used / 1e6, 2))
        return out

    def _write(self, name, img, ext, params=()):
        ok, buf = cv2.imencode(ext, img, list(params))
        p = os.path.join(self.dir, name)
        with open(p, "wb") as f:
            f.write(buf.tobytes())
        self.used += len(buf)
        return p
