"""PromptStereo and DEFOM-Stereo wrappers with the same interface as Las2Depth / FfsDepth.

Each repo has top-level packages (`core`, `utils`, `model`, `util`, `dataset`, `depth_anything_v2`, ...) whose
names clash with the other repos, so each model is built inside `_Isolated`, which puts its repo first on sys.path,
sets any clashing modules aside, and restores them afterwards. The built model keeps working because it holds
its own classes.

Weights (downloaded by hand, see docs/other-stereo-guide.md):
  third_party/PromptStereo/checkpoint/unlimited_192.safetensors + depth_anything_v2_vitl.pth
  third_party/DEFOM-Stereo/checkpoints/defomstereo_vitl_sceneflow.pth
"""
import os
import sys
import time

import numpy as np
import torch

THIRD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party"))


def _top_level_names(root):
    names = set()
    for n in os.listdir(root):
        p = os.path.join(root, n)
        if os.path.isdir(p) and (os.path.exists(os.path.join(p, "__init__.py")) or n in ("core", "utils", "util")):
            names.add(n)
        elif n.endswith(".py"):
            names.add(n[:-3])
    names.discard("gabai")
    return names


class _Isolated:
    def __init__(self, root, extra_paths=()):
        self.root, self.extra = root, list(extra_paths)

    def _clashing(self):
        names = _top_level_names(self.root)
        return [k for k in sys.modules if k.split(".")[0] in names]

    def __enter__(self):
        self.stash = {k: sys.modules.pop(k) for k in self._clashing()}
        for p in [self.root] + self.extra:
            sys.path.insert(0, p)
        return self

    def __exit__(self, *exc):
        for p in [self.root] + self.extra:
            if p in sys.path:
                sys.path.remove(p)
        for k in self._clashing():
            del sys.modules[k]
        sys.modules.update(self.stash)


class _Base:
    def _pad(self, left, right):
        l = torch.as_tensor(left, device=self.device).float()[None].permute(0, 3, 1, 2)
        r = torch.as_tensor(right, device=self.device).float()[None].permute(0, 3, 1, 2)
        padder = self._padder_cls(l.shape, divis_by=32)
        l, r = padder.pad(l, r)
        return l, r, padder

    def disparity_lr(self, left, right):
        from gabai.spike.las2_runner import lr_consistency
        dl, t1 = self.disparity(left, right)
        drf, t2 = self.disparity(np.ascontiguousarray(right[:, ::-1]), np.ascontiguousarray(left[:, ::-1]))
        return dl, lr_consistency(dl, drf[:, ::-1]), t1 + t2


class PromptStereoDepth(_Base):
    def __init__(self, iters=16, max_disp=192, ckpt=None, device="cuda"):
        root = os.path.join(THIRD, "PromptStereo")
        ckpt = ckpt or os.path.join(root, "checkpoint", "unlimited_192.safetensors")
        self.device, self.iters = torch.device(device), iters
        with _Isolated(root):
            from accelerate import load_checkpoint_and_dispatch
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from util.padder import InputPadder
            cfg = OmegaConf.load(os.path.join(root, "config", "model", "promptstereo.yaml"))
            cfg.instance.cfg.pretrained_model.checkpoint = os.path.join(root, "checkpoint", "depth_anything_v2_vitl.pth")
            cfg.instance.cfg.gwc_max_disp = max_disp
            model = instantiate(cfg.instance)
            model = load_checkpoint_and_dispatch(model, ckpt)
            self._padder_cls = InputPadder
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def disparity(self, left, right):
        h, w = left.shape[:2]
        l, r, padder = self._pad(left, right)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            d = self.model(l, r, self.iters, test_mode=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        d = padder.unpad(d.float()).cpu().numpy().reshape(h, w).clip(0, None)
        return d, dt


class DefomDepth(_Base):
    def __init__(self, iters=16, scale_iters=8, ckpt=None, device="cuda", mixed_precision=False):  # fp16 gives all-NaN output
        import argparse
        root = os.path.join(THIRD, "DEFOM-Stereo")
        ckpt = ckpt or os.path.join(root, "checkpoints", "defomstereo_vitl_sceneflow.pth")
        self.device, self.iters, self.scale_iters = torch.device(device), iters, scale_iters
        args = argparse.Namespace(
            mixed_precision=mixed_precision, dinov2_encoder="vitl", idepth_scale=0.5, hidden_dims=[128] * 3,
            corr_implementation="reg", shared_backbone=False, corr_levels=2, corr_radius=4,
            scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], scale_corr_radius=2, n_downsample=2,
            context_norm="batch", n_gru_layers=3)
        with _Isolated(root, extra_paths=[os.path.join(root, "core")]):
            from core.defom_stereo import DEFOMStereo
            from utils.utils import InputPadder
            model = DEFOMStereo(args)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"] if "model" in state else state)
            self._padder_cls = InputPadder
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def disparity(self, left, right):
        h, w = left.shape[:2]
        l, r, padder = self._pad(left, right)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d = self.model(l, r, iters=self.iters, scale_iters=self.scale_iters, test_mode=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        d = padder.unpad(d.float()).cpu().numpy().reshape(h, w).clip(0, None)
        return d, dt
