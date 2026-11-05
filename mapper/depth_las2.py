"""LAS2 (LiteAnyStereo) as the mapper's depth source, with the same interface and the same per-pixel confidence as the TensorRT FoundationStereo path (`mapper.depth_trt.StereoTrt`).

One batched model call runs (left, right) and the mirrored (flipped right, flipped left), which gives the right view's disparity for the left-right agreement in confidence.py. Weights:
third_party/LiteAnyStereo/checkpoints/LAS2_{S,M,H}.pth; S is the default (the fastest, 36 ms for the pair on the RTX 5060 Ti in docs/rtabmap-noise.md), M and H are slower and more accurate.
LAS2 cannot match a disparity above 192 px (closer than f*B/192, 0.15 m on this rig): such pixels are dropped by the confidence stage, not guessed.
"""
import time

import numpy as np
import torch

from mapper.confidence import confidence
from mapper.depth_trt import DepthResult


class Las2Stereo:
    def __init__(self, fx, baseline_m, size="s", max_range=2.5, max_disp=192):
        from gabai.spike.las2_runner import Las2Depth
        self.model, self.fb, self.max_range = Las2Depth(size, max_disp=max_disp), fx * baseline_m, max_range

    @torch.no_grad()
    def __call__(self, left_rgb, right_rgb):
        """left_rgb, right_rgb: rectified uint8 HxWx3 RGB. Returns DepthResult (depth m, confidence 0..1, sigma_z m, left disparity px, ms)."""
        t0 = time.perf_counter()
        fl, fr = np.ascontiguousarray(right_rgb[:, ::-1]), np.ascontiguousarray(left_rgb[:, ::-1])
        (dl, drf), _ = self.model.disparity_batch([(left_rgb, right_rgb), (fl, fr)])
        dev = self.model.device
        dl_t = torch.as_tensor(np.ascontiguousarray(dl), device=dev).float().clamp(min=0)
        dr_t = torch.as_tensor(np.ascontiguousarray(drf[:, ::-1]), device=dev).float().clamp(min=0)
        gray = torch.as_tensor(left_rgb, device=dev).float().mean(-1)
        z, c, sz = confidence(dl_t, dr_t, gray, self.fb, self.max_range)
        res = [x.float().cpu().numpy() for x in (z, c, sz, dl_t)]
        return DepthResult(*res, ms=(time.perf_counter() - t0) * 1e3)
