"""One-off: export Fast-FoundationStereo as a single ONNX with batch 2 (left/right and mirrored right/left in one call).

Reuses NVIDIA's cost-volume patches from third_party/Fast-FoundationStereo/scripts/make_single_onnx.py. Run in its own process: that repo's `core` package clashes with LAS2's.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "Fast-FoundationStereo"))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
os.environ["TORCH_COMPILE_DISABLE"] = os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch  # noqa: E402
import core.foundation_stereo as fs  # noqa: E402
import make_single_onnx as ms  # noqa: E402


def export(out, height=384, width=640, iters=8, max_disp=192, batch=2,
           weights=os.path.join(ROOT, "weights", "hf-c-fast", "model_best_bp2_serialize.pth")):
    model = torch.load(weights, map_location="cpu", weights_only=False)
    model.args.max_disp, model.args.valid_iters, model.args.mixed_precision = max_disp, iters, False
    if "normalize" not in model.args:
        model.args.normalize = True
    wrapper = ms.FastFoundationStereoSingleOnnx(model.cuda().eval()).cuda().eval()
    fs.normalize_image = lambda img: img                       # normalisation is done outside the engine
    fs.build_gwc_volume_optimized_pytorch1 = ms._build_gwc_volume_onnx
    fs.build_concat_volume_optimized_pytorch1 = ms._build_concat_volume_onnx
    x = torch.randn(batch, 3, height, width, device="cuda")
    with torch.no_grad():
        torch.onnx.export(wrapper, (x, x.clone()), out, opset_version=17, input_names=["left_image", "right_image"],
                          output_names=["disparity"], do_constant_folding=True, dynamo=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--max-disp", type=int, default=192)
    p.add_argument("--batch", type=int, default=2)
    a = p.parse_args()
    export(a.out, a.height, a.width, a.iters, max_disp=a.max_disp, batch=a.batch)
    print("ok", a.out, os.path.getsize(a.out))
