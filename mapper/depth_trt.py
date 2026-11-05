"""Fast-FoundationStereo as a TensorRT fp16 engine: metric depth + per-pixel confidence for one rectified pair.

One engine call, batch 2: (left, right) and the mirrored (flip right, flip left). The second item is the right view's disparity, which gives the left-right agreement in confidence.py.
Build once: `py -m mapper.depth_trt build` (exports the ONNX in a child process, then builds the engine; cached under models/trt/).
"""
import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

from .confidence import confidence

HERE = os.path.dirname(os.path.abspath(__file__))
TRT_DIR = os.path.join(HERE, "..", "models", "trt")


def paths(iters=8, max_disp=192):
    tag = ("" if iters == 8 else f"_it{iters}") + ("" if max_disp == 192 else f"_md{max_disp}")   # the first files keep their names
    return (os.path.join(TRT_DIR, f"ffs_b2_384x640{tag}.onnx"), os.path.join(TRT_DIR, f"ffs_b2_384x640{tag}_fp16.engine"))


ONNX, ENGINE = paths()
H_ENG, W_ENG = 384, 640
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)


@dataclass
class DepthResult:
    depth: np.ndarray      # float32 m, 0 = none
    conf: np.ndarray       # float32 0..1
    sigma_z: np.ndarray    # float32 m
    disp: np.ndarray       # float32 px, left view
    ms: float


def build_engine(onnx=ONNX, engine=ENGINE):
    import tensorrt as trt
    log = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(log)
    net = b.create_network()
    p = trt.OnnxParser(net, log)
    if not p.parse_from_file(os.path.abspath(onnx)):
        raise RuntimeError("; ".join(str(p.get_error(i)) for i in range(p.num_errors)))
    cfg = b.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    blob = b.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    with open(engine, "wb") as f:
        f.write(blob)


class StereoTrt:
    def __init__(self, fx, baseline_m, engine=ENGINE, max_range=2.5):   # engine=paths(iters)[1] for another iteration count
        import tensorrt as trt
        self.trt, self.fb, self.max_range = trt, fx * baseline_m, max_range
        with open(engine, "rb") as f:
            self.eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        if self.eng is None:
            raise RuntimeError("engine does not match this TensorRT version: rebuild with `py -m mapper.depth_trt build`")
        self.ctx = self.eng.create_execution_context()
        self.mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
        self.out = torch.empty(2, 1, H_ENG, W_ENG, device="cuda")
        self.inp = torch.empty(2, 2, 3, H_ENG, W_ENG, device="cuda")   # [left_image|right_image][batch]
        for i, n in enumerate(("left_image", "right_image")):
            self.ctx.set_tensor_address(n, self.inp[i].data_ptr())
        self.ctx.set_tensor_address("disparity", self.out.data_ptr())

    def _prep(self, rgb):
        t = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float()
        h, w = t.shape[2:]
        if (h, w) != (H_ENG, W_ENG):
            assert w == W_ENG and h <= H_ENG, f"engine is {W_ENG}x{H_ENG}, got {w}x{h}"
            # symmetric: padding only at the bottom shifted disparity by 2 px median (measured against the PyTorch model)
            top = (H_ENG - h) // 2
            t = torch.nn.functional.pad(t, (0, 0, top, H_ENG - h - top), mode="replicate")
        return (t - self.mean) / self.std

    @torch.no_grad()
    def __call__(self, left_rgb, right_rgb):
        """left_rgb, right_rgb: rectified uint8 HxWx3 RGB, H <= 384, W = 640."""
        h = left_rgb.shape[0]
        top = (H_ENG - h) // 2
        t0 = time.perf_counter()
        l, r = self._prep(left_rgb), self._prep(right_rgb)
        self.inp[0].copy_(torch.cat([l, r.flip(-1)]))
        self.inp[1].copy_(torch.cat([r, l.flip(-1)]))
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        dl, dr = self.out[0, 0, top:top + h], self.out[1, 0, top:top + h].flip(-1)
        gray = l[0, :, top:top + h].mul(torch.tensor(STD, device="cuda").view(3, 1, 1)).add(torch.tensor(MEAN, device="cuda").view(3, 1, 1)).mean(0)
        z, c, sz = confidence(dl, dr, gray, self.fb, self.max_range)
        res = [x.float().cpu().numpy() for x in (z, c, sz, dl)]
        return DepthResult(*res, ms=(time.perf_counter() - t0) * 1e3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--iters", type=int, default=8, help="GRU refinement iterations")
    ap.add_argument("--max-disp", type=int, default=192, help="disparity search range px; the cost volume shrinks with it (96 px = 0.3 m at f*B 28.9)")
    a = ap.parse_args()
    os.makedirs(TRT_DIR, exist_ok=True)
    onnx, engine = paths(a.iters, a.max_disp)
    if not os.path.exists(onnx):
        subprocess.check_call([sys.executable, "-m", "mapper.export_onnx", onnx, "--iters", str(a.iters), "--max-disp", str(a.max_disp)], cwd=os.path.join(HERE, ".."))
    build_engine(onnx, engine)
    print("engine", engine, os.path.getsize(engine))
