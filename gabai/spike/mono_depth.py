"""Monocular METRIC depth models behind one interface: depth(rgb) -> metres (float32, same size as the image).

  dav2   Depth Anything V2 Metric Indoor Small (transformers, 99 MB, indoor-trained, scale is only roughly right)
  moge-s / moge-b / moge-l   MoGe-2 ViT-S / B / L (35 / 104 / 326 M parameters, metric scale and sharp detail;
                             needs `pip install git+https://github.com/microsoft/MoGe.git` and Hugging Face weights)
The scale is corrected against stereo afterwards (gabai.fuse_depth), so only the structure has to be good.
"""
import contextlib
import time

import cv2
import numpy as np
import torch

MOGE_IDS = {"moge-s": "Ruicheng/moge-2-vits-normal", "moge-b": "Ruicheng/moge-2-vitb-normal", "moge-l": "Ruicheng/moge-2-vitl"}


class MonoDepth:
    def __init__(self, name="dav2", device="cuda"):
        self.name, self.device = name, torch.device(device if torch.cuda.is_available() else "cpu")
        self.stream = torch.cuda.Stream() if self.device.type == "cuda" else None          # its own stream, so it can run while the stereo model runs
        if name == "dav2":
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            repo = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
            try:                                                   # the cached copy first: asking the hub whether it changed costs about 2 s on every start
                self.proc = AutoImageProcessor.from_pretrained(repo, local_files_only=True)
                model = AutoModelForDepthEstimation.from_pretrained(repo, local_files_only=True)
            except OSError:
                self.proc = AutoImageProcessor.from_pretrained(repo)
                model = AutoModelForDepthEstimation.from_pretrained(repo)
            self.model = model.half().to(self.device).eval()
        elif name in MOGE_IDS:
            from moge.model.v2 import MoGeModel
            self.model = MoGeModel.from_pretrained(MOGE_IDS[name]).to(self.device).eval().half()
        else:
            raise ValueError(f"unknown monocular model {name!r}")

    def _gpu_prep(self, images, side=350):
        """Resize and normalise on the GPU (the Hugging Face processor does it on the CPU and costs about 15 ms).
        The model needs sizes that are multiples of 14: the shorter side becomes `side` (350 = 25 patches)."""
        x = torch.from_numpy(np.stack(images)).to(self.device).permute(0, 3, 1, 2).half() / 255.0
        h, w = x.shape[-2:]
        s = side / min(h, w)
        nh, nw = max(int(round(h * s / 14)), 1) * 14, max(int(round(w * s / 14)), 1) * 14
        x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def depth(self, rgb, f_px=None):
        """rgb: HxWx3 uint8. f_px: horizontal focal length in pixels if known (helps MoGe's metric scale)."""
        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        ctx = torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()
        with ctx:
            d = self._infer(rgb, f_px, h, w)
            d = d.cpu().numpy().astype(np.float32)                    # waits for this stream only
        self.last_ms = (time.perf_counter() - t0) * 1000
        return d

    def _infer(self, rgb, f_px, h, w):
        if self.name == "dav2":
            out = self.model(pixel_values=self._gpu_prep([rgb])).predicted_depth
            d = torch.nn.functional.interpolate(out[None].float(), size=(h, w), mode="bilinear")[0, 0]
        else:
            x = torch.from_numpy(rgb).to(self.device).float().permute(2, 0, 1) / 255.0
            fov_x = None if f_px is None else float(np.degrees(2 * np.arctan(w / (2 * f_px))))
            out = self.model.infer(x.half(), fov_x=fov_x, use_fp16=True)
            d = out["depth"].float()
            d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
        return d


@torch.no_grad()
def _depth_pair(self, rgb_left, rgb_right, f_px=None):
    """Both camera pictures in one batched call, on this model's own CUDA stream. Returns (depth_left, depth_right) in metres."""
    h, w = rgb_left.shape[:2]
    t0 = time.perf_counter()
    ctx = torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()
    with ctx:
        if self.name == "dav2":
            out = self.model(pixel_values=self._gpu_prep([rgb_left, rgb_right])).predicted_depth
            d = torch.nn.functional.interpolate(out[:, None].float(), size=(h, w), mode="bilinear")[:, 0]
        else:
            d = torch.stack([self._infer(im, f_px, h, w) for im in (rgb_left, rgb_right)])
        d = d.cpu().numpy().astype(np.float32)                        # waits for this stream only
    self.last_ms = (time.perf_counter() - t0) * 1000
    return d[0], d[1]


MonoDepth.depth_pair = _depth_pair
