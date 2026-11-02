"""Runs the depth sources on one rectified pair and fuses them (rtabmap_lab/depthfusion.py): Fast-FoundationStereo, LAS2 (with its left-right check), RTAB-Map's own stereo depth, and Depth Anything V2 metric on BOTH pictures.

All models stay resident (about 1.2 GB of the 16 GB together: FFS 0.75, LAS2 0.3, Depth Anything 0.13, see tools/depth_bench.py) but are never run at the same time: they run one after the other on the caller's thread,
so no second copy of anything is loaded and the GPU never holds more than one model's working memory. Per pair about 40 ms (LAS2) + 76 ms (FFS) + 26 ms (two monocular maps), which is why this runs only for the pairs
RTAB-Map turns into nodes (about 2 a second at most), on a worker thread that never makes the frame loop wait.
"""
import time

import cv2
import numpy as np

from rtabmap_lab.depthfusion import FusionParams, StereoIn, fuse

SIGMA_PX = {"ffs": 0.25, "las2": 0.30, "rtabmap": 0.50}          # disparity noise per source (px): FFS and LAS2 from the benchmark's agreement with each other and with the board, RTAB-Map's SGBM is the noisiest


class FusedSource:
    def __init__(self, ffs=True, las2=True, mono=True, params=None):
        self.params, self.models = params or FusionParams(), {}
        if ffs:
            from gabai.spike.ffs_runner import FfsDepth
            self.models["ffs"] = FfsDepth()
        if las2:
            from gabai.spike.las2_runner import Las2Depth
            self.models["las2"] = Las2Depth("s")
        if mono:
            from gabai.spike.mono_depth import MonoDepth
            self.models["mono"] = MonoDepth("dav2")
        self.stats = {"ms": 0.0}

    @staticmethod
    def _z(disp, cal, lo=0.3, hi=4.0):
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(disp > 0.5, cal["f"] * cal["B"] / disp, np.nan)
        return np.where((z >= lo) & (z <= hi), z, np.nan).astype(np.float32)

    def process(self, left_bgr, right_bgr, cal, rtab_depth_mm=None):
        """Returns (FusedDepth, parts): parts = {source name: depth map in metres, NaN = none} for the display, plus 'ms' per source."""
        rgb_l, rgb_r = left_bgr[:, :, ::-1].copy(), right_bgr[:, :, ::-1].copy()
        if rgb_r.ndim == 2:
            rgb_r = cv2.cvtColor(right_bgr, cv2.COLOR_GRAY2RGB)
        parts, ms, stereo = {}, {}, []
        t0 = time.perf_counter()
        if "ffs" in self.models:
            d, dt = self.models["ffs"].disparity(rgb_l, rgb_r)
            parts["ffs"], ms["ffs"] = self._z(d, cal), dt * 1000
            stereo.append(StereoIn(parts["ffs"], SIGMA_PX["ffs"], "ffs"))
        if "las2" in self.models:
            d, ok, dt = self.models["las2"].disparity_lr_batched(rgb_l, rgb_r)
            parts["las2"], ms["las2"] = self._z(np.where(ok, d, np.nan), cal), dt * 1000
            stereo.append(StereoIn(parts["las2"], SIGMA_PX["las2"], "las2"))
        if rtab_depth_mm is not None:
            z = np.where(rtab_depth_mm > 0, rtab_depth_mm.astype(np.float32) / 1000.0, np.nan).astype(np.float32)
            parts["rtabmap"] = z
            stereo.append(StereoIn(z, SIGMA_PX["rtabmap"], "rtabmap"))
        ml = mr = None
        if "mono" in self.models:
            ml, mr = self.models["mono"].depth(rgb_l), self.models["mono"].depth(rgb_r)
            parts["mono_left"], parts["mono_right"] = np.where((ml > 0.3) & (ml < 4.0), ml, np.nan).astype(np.float32), np.where((mr > 0.3) & (mr < 4.0), mr, np.nan).astype(np.float32)
        fused = fuse(stereo, ml, mr, cal, self.params)
        parts["ms"] = {**ms, "fusion": (time.perf_counter() - t0) * 1000 - sum(ms.values())}
        self.stats["ms"] = (time.perf_counter() - t0) * 1000
        return fused, parts

    @staticmethod
    def to_mm(z):
        return (np.nan_to_num(z, nan=0.0) * 1000.0).astype(np.uint16)


class FusionDepth:
    """DepthUpgrader adapter. mode 'fused': FFS + LAS2 + RTAB-Map + two monocular maps, confidence-aware fusion, class and confidence maps for the weighted TSDF and surfels. mode 'ffs': Fast-FoundationStereo alone
    (the best single source on the recording: see tools/surface_lab.py --fusion), no fusion, every pixel the same weight."""

    def __init__(self, mode="fused", params=None):
        self.mode = mode
        self.src = FusedSource(ffs=True, las2=mode == "fused", mono=mode == "fused", params=params)

    def depth_mm(self, left_bgr, right_bgr, cal, min_range=0.3, max_range=4.0, rtab_mm=None):
        fused, parts = self.src.process(left_bgr, right_bgr, cal, rtab_mm if self.mode == "fused" else None)
        if self.mode == "ffs":
            z = parts["ffs"]
            z = np.where((z >= min_range) & (z <= max_range), z, np.nan)
            return FusedSource.to_mm(z), self.src.stats["ms"], {"parts": parts, "fused": fused, "cls": None, "conf": None, "z": z}
        z = np.where((fused.z >= min_range) & (fused.z <= max_range), fused.z, np.nan)
        return FusedSource.to_mm(z), self.src.stats["ms"], {"parts": parts, "fused": fused, "cls": fused.cls, "conf": fused.conf, "z": z}
