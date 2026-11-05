"""retinify (https://github.com/retinify/retinify, Apache-2.0) as the mapper's depth source, same interface and same per-pixel confidence as `mapper.depth_trt.StereoTrt` and `depth_las2.Las2Stereo`.

retinify's library builds only on Linux (Ubuntu + CUDA + TensorRT 10, Debian packages, a hard-coded /usr/local/cuda path) and has no pip package, so it is not linked here. Its stereo model is a plain 10 MB ONNX
(retinify-models v0.1.7, third_party/retinify/stereo-matching.onnx) and its pipeline is a few lines, which are reproduced from its source (retinify/src/pipeline.cpp, matproc.cpp):
  input     left and right as float32 GRAY in 0..255 (no normalisation), NHWC [B, H, W, 1], resized to the matching size: FAST 640x320, BALANCED 640x480, ACCURATE 1280x720
  output    disparity [B, H, W, 1] at the matching size; resized to the picture size and multiplied by (picture width / matching width)
  then      an occlusion filter; here the left-right agreement of confidence.py does that job (batch 2: the mirrored pair gives the right view's disparity)
The engine is built with this machine's TensorRT (fp16) on first use and cached under models/trt/.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from mapper.confidence import confidence
from mapper.depth_trt import DepthResult, TRT_DIR

ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "retinify", "stereo-matching.onnx")
MODES = {"fast": (320, 640), "balanced": (480, 640), "accurate": (720, 1280)}
PROFILE = {"min": (1, 320, 640, 1), "opt": (2, 480, 640, 1), "max": (2, 720, 1280, 1)}      # retinify's own engine profile (batch 2 added for the mirrored pass)


def engine_path(fp16=True):
    return os.path.join(TRT_DIR, "retinify_stereo_matching_" + ("fp16" if fp16 else "fp32") + ".engine")


def build_engine(onnx=ONNX, engine=None, fp16=True):
    import tensorrt as trt
    engine = engine or engine_path(fp16)
    log = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(log)
    net = b.create_network()
    p = trt.OnnxParser(net, log)
    if not p.parse_from_file(os.path.abspath(onnx)):
        raise RuntimeError("; ".join(str(p.get_error(i)) for i in range(p.num_errors)))
    cfg = b.create_builder_config()
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    prof = b.create_optimization_profile()
    for name in ("left", "right"):
        prof.set_shape(name, PROFILE["min"], PROFILE["opt"], PROFILE["max"])
    cfg.add_optimization_profile(prof)
    blob = b.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    os.makedirs(os.path.dirname(engine), exist_ok=True)
    with open(engine, "wb") as f:
        f.write(blob)
    return engine


class RetinifyStereo:
    def __init__(self, fx, baseline_m, mode="balanced", max_range=2.5, fp16=True):
        import tensorrt as trt
        if not os.path.exists(ONNX):
            raise FileNotFoundError(f"{ONNX} missing: download stereo-matching.onnx from github.com/retinify/retinify-models (release v0.1.7)")
        path = engine_path(fp16)
        if not os.path.exists(path):
            build_engine(fp16=fp16)
        with open(path, "rb") as f:
            self.eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        if self.eng is None:
            raise RuntimeError("engine does not match this TensorRT version: delete it and it is rebuilt")
        self.ctx, self.fb, self.max_range, self.hm, self.wm = self.eng.create_execution_context(), fx * baseline_m, max_range, *MODES[mode]
        self.dev = torch.device("cuda")

    def _gray(self, rgb):
        t = torch.from_numpy(rgb).to(self.dev).float()
        return (0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2])

    @torch.no_grad()
    def __call__(self, left_rgb, right_rgb):
        """left_rgb, right_rgb: rectified uint8 HxWx3 RGB. Returns DepthResult (depth m, confidence 0..1, sigma_z m, left disparity px, ms)."""
        t0 = time.perf_counter()
        gl, gr = self._gray(left_rgb), self._gray(right_rgb)
        h, w = gl.shape
        rs = lambda g: F.interpolate(g[None, None], size=(self.hm, self.wm), mode="bilinear", align_corners=False)[0, 0]
        l, r = rs(gl), rs(gr)
        a = torch.stack([l, r.flip(-1)])[..., None].contiguous()                  # left input:  left, mirrored right
        b = torch.stack([r, l.flip(-1)])[..., None].contiguous()                  # right input: right, mirrored left
        self.ctx.set_input_shape("left", tuple(a.shape))
        self.ctx.set_input_shape("right", tuple(b.shape))
        out = torch.empty(tuple(self.ctx.get_tensor_shape("disparity")), device=self.dev, dtype=torch.float32)
        for n, t in (("left", a), ("right", b), ("disparity", out)):
            self.ctx.set_tensor_address(n, t.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        scale = w / self.wm
        up = lambda d: F.interpolate(d[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0] * scale
        dl, dr = up(out[0, ..., 0]).clamp(min=0), up(out[1, ..., 0].flip(-1)).clamp(min=0)
        z, c, sz = confidence(dl, dr, gl, self.fb, self.max_range)
        res = [x.float().cpu().numpy() for x in (z, c, sz, dl)]
        return DepthResult(*res, ms=(time.perf_counter() - t0) * 1e3)
