"""Fast-FoundationStereo (NVIDIA) wrapper with the same interface as Las2Depth.

Both repos use top-level packages named `core` and `Utils`, so loading them in one process would clash. The
model is unpickled with the FFS repo first on sys.path and any previously imported `core`/`Utils` modules set
aside, then those are restored; the loaded model keeps working because it holds its own classes.
Weights: third_party/Fast-FoundationStereo/weights/hf-c-fast (NVIDIA C-Fast-FoundationStereo, Hugging Face,
NVIDIA Open Model Agreement, research and evaluation use).
"""
import os
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "Fast-FoundationStereo"))
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "hf-c-fast", "model_best_bp2_serialize.pth")


class FfsDepth:
    def __init__(self, weights=DEFAULT_WEIGHTS, iters=8, max_disp=192, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.iters = iters
        stash = {k: v for k, v in sys.modules.items() if k == "core" or k.startswith("core.") or k == "Utils"}
        for k in stash:
            del sys.modules[k]
        sys.path.insert(0, ROOT)
        try:
            from core.utils.utils import InputPadder  # noqa: E402  (FFS's own copy)
            self._padder_cls = InputPadder
            model = torch.load(weights, map_location="cpu", weights_only=False)
        finally:
            sys.path.remove(ROOT)
            for k in [k for k in sys.modules if k == "core" or k.startswith("core.") or k == "Utils"]:
                del sys.modules[k]
            sys.modules.update(stash)
        from omegaconf import open_dict
        with open_dict(model.args):  # the HF checkpoint's saved args lack 'normalize'; the repo default is True
            model.args.valid_iters = iters
            model.args.max_disp = max_disp
            if "normalize" not in model.args:
                model.args.normalize = True
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def disparity(self, left, right):
        """left/right: rectified RGB uint8 HxWx3. Returns (disparity float32 HxW in pixels, seconds)."""
        h, w = left.shape[:2]
        l = torch.as_tensor(left, device=self.device).float()[None].permute(0, 3, 1, 2)
        r = torch.as_tensor(right, device=self.device).float()[None].permute(0, 3, 1, 2)
        padder = self._padder_cls(l.shape, divis_by=32, force_square=False)
        l, r = padder.pad(l, r)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda", dtype=torch.float16):
            d = self.model.forward(l, r, iters=self.iters, test_mode=True, optimize_build_volume="pytorch1")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        d = padder.unpad(d.float()).cpu().numpy().reshape(h, w).clip(0, None)
        return d, dt


def _disparity_lr(self, left, right):
    from gabai.spike.las2_runner import lr_consistency
    dl, t1 = self.disparity(left, right)
    dr_f, t2 = self.disparity(np.ascontiguousarray(right[:, ::-1]), np.ascontiguousarray(left[:, ::-1]))
    return dl, lr_consistency(dl, dr_f[:, ::-1]), t1 + t2


FfsDepth.disparity_lr = _disparity_lr
