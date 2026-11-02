"""Boxer (Meta / Project Aria, third_party/boxer): OWLv2 finds 2D boxes for text labels, BoxerNet lifts each to a metric, gravity-aligned 3D box from the picture, sparse depth and pose.

make_lifter() loads the models (several seconds, 14 s for the first frame's kernels) and returns lift(left_bgr, disp, trusted, cal, R_wc, t_wc) -> [dict(label, conf, p_w, dims, R_obj)].
Needs the weights in third_party/boxer/ckpts (that repo's scripts/download_ckpts.sh) and a world with z up (stereo_imu). Licence of the weights: mostly CC-BY-NC.
"""
import os
import sys

import numpy as np

BOXER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "boxer")
LABELS = ["cabinet", "refrigerator", "table", "desk", "sofa", "chair", "tv", "monitor", "shelf", "bookshelf", "bed", "sink", "oven", "microwave", "laptop", "keyboard", "plant"]
MAX_Z, N_SDP = 4.0, 10000


def preload():
    """Import torch's heavy modules on the CALLING thread. The lifter loads on a worker thread while the main thread loads the stereo model: two threads importing torch._inductor at the same
    moment deadlock ('_DeadlockError: deadlock detected by _ModuleLock'). Call this on the main thread before starting that worker."""
    import timm      # noqa: F401
    import torch     # noqa: F401
    import torchvision  # noqa: F401


def make_lifter(labels=LABELS, thresh2d=0.25, thresh3d=0.5, hw2d=960):
    """lift(left, disp, trusted, cal, R_wc, t_wc) -> [dict(label, conf, p_w, dims, R_obj)] with the whole object's box in the world."""
    sys.path.insert(0, BOXER)                                                # boxer's own modules are named boxernet, owl, utils, loaders
    import torch
    from boxernet.boxernet import BoxerNet
    from loaders.base_loader import BaseLoader
    from owl.owl_wrapper import OwlWrapper
    from utils.demo_utils import CKPT_PATH, DEFAULT_BOXERNET_CKPT
    from utils.tw.camera import CameraTW
    from utils.tw.pose import PoseTW
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owl = OwlWrapper(dev, text_prompts=labels, min_confidence=thresh2d, precision=None)
    net = BoxerNet.load_from_checkpoint(os.path.join(CKPT_PATH, DEFAULT_BOXERNET_CKPT), device=dev)
    hw = net.hw
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    def lift(left, disp, trusted, cal, R_wc, t_wc):
        import cv2
        f, cx, cy, B = (float(cal[k]) for k in ("f", "cx", "cy", "B"))
        H, W = left.shape[:2]
        rgb = cv2.resize(cv2.cvtColor(left, cv2.COLOR_BGR2RGB), (hw, hw), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
        ok = trusted & np.isfinite(disp) & (disp > 0.5)
        z = np.zeros(disp.shape, np.float32)
        z[ok] = f * B / disp[ok]
        z[z > MAX_Z] = 0.0
        datum = {"img0": img,
                 "cam0": CameraTW.from_surreal(width=W, height=H, type_str="pinhole", params=torch.tensor([f, f, cx, cy], dtype=torch.float32)).scale((hw / W, hw / H)).float(),
                 "T_world_rig0": PoseTW.from_Rt(torch.from_numpy(R_wc.astype(np.float32)), torch.from_numpy(t_wc.astype(np.float32))).float(),
                 "sdp_w": BaseLoader.sdp_from_depth(z, f, f, cx, cy, R_wc.astype(np.float32), t_wc.astype(np.float32), N_SDP).float()}
        bb2d, s2d, lab, _ = owl.forward(img.clone() * 255.0, resize_to_HW=(hw2d, hw2d))
        if bb2d.shape[0] == 0:
            return []
        datum["bb2d"] = bb2d
        with torch.autocast(device_type=dev, dtype=dtype):
            out = net.forward(datum)
        obb = out["obbs_pr_w"].cpu()[0]
        p3 = obb.prob.squeeze(-1)
        ctr = obb.bb3_center_world.float().numpy()
        dims = obb.bb3_diagonal.float().numpy()
        R_obj = obb.T_world_object.R.float().numpy()
        res = []
        for i in range(len(obb)):
            if float(p3[i]) < thresh3d or not np.isfinite(ctr[i]).all():
                continue
            pw = ctr[i].astype(np.float64)
            res.append(dict(label=labels[int(lab[i])], conf=0.5 * (float(s2d[i]) + float(p3[i])), p_w=pw, dims=dims[i].astype(np.float64), R_obj=R_obj[i].astype(np.float64)))
        return res
    return lift
