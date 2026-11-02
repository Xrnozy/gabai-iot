"""Thin wrapper around third_party/LiteAnyStereo for PC-side tests. Inputs are rectified RGB uint8 images."""
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "LiteAnyStereo")
ROOT = os.path.abspath(ROOT)


class Las2Depth:
    def __init__(self, size="s", ckpt=None, max_disp=192, device="cuda"):
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from core.models import build_model, load_model_weights
        from core.utils.utils import InputPadder

        self._padder_cls = InputPadder
        self.max_disp = max_disp
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt = ckpt or os.path.join(ROOT, "checkpoints", f"LAS2_{size.upper()}.pth")
        model = build_model("las2", fnet_pretrained=False, model_size=size, max_disp=max_disp)
        load_model_weights(model, torch.load(ckpt, map_location=self.device), strict=True)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def disparity(self, left, right):
        """Return (disparity float32 HxW in pixels, seconds for the model call incl. sync)."""
        h, w = left.shape[:2]
        l = torch.as_tensor(left, device=self.device).float()[None].permute(0, 3, 1, 2)
        r = torch.as_tensor(right, device=self.device).float()[None].permute(0, 3, 1, 2)
        padder = self._padder_cls(l.shape, divis_by=32)
        l, r = padder.pad(l, r)
        import time
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda", dtype=torch.float16):
            d = self.model(l, r, max_disp=self.max_disp, test_mode=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        d = padder.unpad(d.float()).cpu().numpy().reshape(h, w)
        return d, dt


def lr_consistency(disp_l, disp_r, tol_px=1.0, tol_frac=0.05):
    """Mask of left-image pixels whose disparity agrees with the right image's own estimate.

    A left pixel x with disparity d should land on right pixel x-d, which should report the same d.
    Occlusions, glare and textureless patches fail this check, so they are marked untrusted instead of wrong.
    """
    h, w = disp_l.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    xr = np.rint(xs - disp_l).astype(np.int32)
    inside = (xr >= 0) & (xr < w) & (disp_l > 0.5)
    d_back = np.take_along_axis(disp_r, np.clip(xr, 0, w - 1), axis=1)
    return inside & (np.abs(d_back - disp_l) <= np.maximum(tol_px, tol_frac * disp_l))


def _disparity_lr(self, left, right):
    """Disparity (left reference), trusted mask, seconds. Runs the model twice: (L,R) and mirrored (R,L)."""
    dl, t1 = self.disparity(left, right)
    dr_flipped, t2 = self.disparity(np.ascontiguousarray(right[:, ::-1]), np.ascontiguousarray(left[:, ::-1]))
    dr = dr_flipped[:, ::-1]
    return dl, lr_consistency(dl, dr), t1 + t2


Las2Depth.disparity_lr = _disparity_lr


@torch.no_grad()
def _disparity_batch(self, pairs):
    """Run several same-sized (left, right) pairs through the model in ONE call. Returns (list of disparity maps, seconds)."""
    h, w = pairs[0][0].shape[:2]
    l = torch.as_tensor(np.stack([p[0] for p in pairs]), device=self.device).float().permute(0, 3, 1, 2)
    r = torch.as_tensor(np.stack([p[1] for p in pairs]), device=self.device).float().permute(0, 3, 1, 2)
    padder = self._padder_cls(l.shape, divis_by=32)
    l, r = padder.pad(l, r)
    import time
    if self.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda", dtype=torch.float16):
        d = self.model(l, r, max_disp=self.max_disp, test_mode=True)
    if self.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    d = padder.unpad(d.float()).cpu().numpy().reshape(len(pairs), h, w)
    return [d[i] for i in range(len(pairs))], dt


def _disparity_lr_batched(self, left, right):
    """Same result as disparity_lr but the normal and the mirrored pass share one batched model call."""
    (dl, drf), dt = _disparity_batch(self, [(left, right), (np.ascontiguousarray(right[:, ::-1]), np.ascontiguousarray(left[:, ::-1]))])
    return dl, lr_consistency(dl, drf[:, ::-1]), dt


Las2Depth.disparity_batch = _disparity_batch
Las2Depth.disparity_lr_batched = _disparity_lr_batched


NEAR_FROM_PX = 180.0       # full-resolution disparity above this is at the model's 192 px limit: ask the half-resolution pass


def merge_near(disp, trusted, disp_half, trusted_half, from_px=NEAR_FROM_PX):
    """LAS2 cannot match a disparity above 192 px (closer than f*B/192 = 0.23 m on this rig). A half-resolution copy of
    the pair halves every disparity, so it reaches 384 px (0.12 m). Where that pass sees something at least `from_px`
    away at full scale and trusts it, its value (x2) replaces the full-resolution one."""
    import cv2
    h, w = disp.shape
    d2 = cv2.resize(disp_half, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
    t2 = cv2.resize(trusted_half.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    use = t2 & np.isfinite(d2) & (d2 >= from_px)
    return np.where(use, d2, disp), np.where(use, True, trusted), use


def _disparity_lr_near(self, left, right):
    """Like disparity_lr_batched, plus the close-range half-resolution pass. Returns disp, trusted, seconds, near_mask."""
    import cv2
    dl, tr, dt = self.disparity_lr_batched(left, right)
    h, w = left.shape[:2]
    half = lambda im: cv2.resize(im, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    dh, th, dt2 = self.disparity_lr_batched(half(left), half(right))
    d, t, near = merge_near(dl, tr, dh, th)
    return d, t, dt + dt2, near


Las2Depth.disparity_lr_near = _disparity_lr_near
