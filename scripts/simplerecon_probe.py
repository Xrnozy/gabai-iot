"""Offline probe of Niantic's SimpleRecon (third_party/simplerecon, hero model) on a recording: speed, GPU memory, and how its multi-view depth compares with the stereo depth.

py scripts/simplerecon_probe.py capture data/recordings/<folder> frames.npz [--model s] [--stride 5]     (pipeline pass: rectified left, stereo depth and VIO pose of every Nth frame)
py scripts/simplerecon_probe.py run frames.npz [--sources 7] [--fast]                                      (SimpleRecon on each keyframe with earlier frames as source views)

Two processes on purpose: SimpleRecon's top-level packages (utils, modules, options ...) must not meet the pipeline's. Its three small pure-python dependencies live in
third_party/simplerecon/_pylibs (pip --no-deps --target), nothing was installed into the project's Python.
The stereo depth is NOT ground truth: the comparison says where the two agree and where one fills what the other cannot, not which is right. Needs the weights in
third_party/simplerecon/weights/hero_model.ckpt. Frames are squeezed from 640x360 to the 512x384 the model was trained at (intrinsics scaled the same way).
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SR = os.path.join(ROOT, "third_party", "simplerecon")


def capture(a):
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.getcwd())
    from evaluate_stutter import make_source
    from gabai.pipeline import sources
    from gabai.pipeline.config import PipelineConfig
    from gabai.pipeline.run import run
    from gabai.pipeline.viz import Viz
    out = {"idx": [], "t": [], "left": [], "disp": [], "trusted": [], "R": [], "t_wc": []}
    cal_ = {}

    class Cap(Viz):
        def __init__(self, cal, size, save=None, viewer=False, **kw):
            self.t, self.fr, self.shot, self.n_logged = 0.0, 0, None, 0
            cal_.update(f=float(cal["f"]), cx=float(cal["cx"]), cy=float(cal["cy"]), B=float(cal["B"]))

        def at(self, idx, t):
            self.fr, self.t = idx, t

        def stereo(self, left, right, disp, trusted, cal, *r):
            self.shot = (left.copy(), disp.copy(), trusted.copy())

        def pose(self, R_wc, t_wc, traj, left_bgr=None):
            if self.shot is None or self.fr % a.stride:
                self.shot = None
                return
            l, d, tr = self.shot
            self.shot = None
            for k, v in zip(out, (self.fr, self.t, l, d, tr, np.array(R_wc, float), np.array(t_wc, float))):
                out[k].append(v)
            self.last = self.fr

        def event(self, text, level="INFO"):
            if "pose held" in text and out["idx"] and out["idx"][-1] == self.fr:               # not a trusted pose: take the frame back
                for k in out:
                    out[k].pop()

        def __getattr__(self, name):
            return lambda *x, **k: None

    sent = []
    src, _, _ = make_source(a.folder, "none", sent)
    sources.live_source = src
    ov = {"pipeline.mode": "stereo_imu", "source.kind": "live", "source.max_frames": a.frames, "stereo.model": a.model, "viz.viewer": False, "hands.enabled": False,
          "detector.enabled": False, "perf.async_viz": False, "perf.staged": False, "perf.low_priority": False, "perf.viz_every": 1, "stereo.mono": "", "map.tsdf": False}
    run(PipelineConfig.load("config/pipeline.toml", ov), viz_factory=Cap, log_dir=tempfile.mkdtemp(), quiet=True)
    np.savez(a.out, **{k: np.array(v) for k, v in out.items()}, **cal_)
    print(f"saved {len(out['idx'])} frames to {a.out}")


def pick_sources(R, t, i, n, min_t=0.04, min_deg=3.0, back=40):
    """Earlier frames that look at the same place from a somewhat different spot: the closest in a pose distance, at least min_t metres or min_deg degrees away."""
    from scipy.spatial.transform import Rotation
    sc = []
    for j in range(max(0, i - back), i):
        dt = np.linalg.norm(t[j] - t[i])
        dr = np.degrees((Rotation.from_matrix(R[j]).inv() * Rotation.from_matrix(R[i])).magnitude())
        if (dt >= min_t or dr >= min_deg) and dr < 40:
            sc.append((dt + 0.01 * dr, j))
    return [j for _, j in sorted(sc)[:n]]


def run_model(a):
    sys.path.insert(0, os.path.join(SR, "_pylibs"))
    sys.path.insert(0, SR)
    import torch
    import options
    from experiment_modules.depth_model import DepthModel
    torch.serialization.add_safe_globals([options.Options])                    # the checkpoint stores its training options object; allow that class only, not arbitrary pickles
    from modules import cost_volume

    z = np.load(a.frames)
    f0, cx0, cy0, B = (float(z[k]) for k in ("f", "cx", "cy", "B"))
    left, disp, trusted, R, t = z["left"], z["disp"], z["trusted"], z["R"], z["t_wc"]
    N, H, W = len(left), left.shape[1], left.shape[2]
    ck = torch.load(os.path.join(SR, "weights", "hero_model.ckpt"), map_location="cpu")        # built by hand: Lightning 2.x refuses this stripped checkpoint (no version key)
    model = DepthModel(opts=ck["hyper_parameters"]["opts"])
    bad = model.load_state_dict(ck["state_dict"], strict=False)
    print("weights loaded; missing:", len(bad.missing_keys), "unexpected:", len(bad.unexpected_keys))
    if a.fast and isinstance(model.cost_volume, cost_volume.FeatureVolumeManager):
        model.cost_volume = model.cost_volume.to_fast()
    model = model.cuda().eval()
    o = model.run_opts
    iw, ih = o.image_width, o.image_height
    print(f"model input {iw}x{ih}, depth bins {o.matching_num_depth_bins} in [{o.min_matching_depth}, {o.max_matching_depth}] m, matching scale {o.matching_scale}")
    sx, sy = iw / 2 / W, ih / 2 / H                                               # the model's K_s0 is the intrinsics at its DEPTH resolution (half the image), see datasets/scannet_dataset.py
    K = np.array([[f0 * sx, 0, cx0 * sx, 0], [0, f0 * sy, cy0 * sy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def img(i):
        import cv2
        x = cv2.resize(cv2.cvtColor(left[i], cv2.COLOR_BGR2RGB), (iw, ih), interpolation=cv2.INTER_AREA)
        return ((torch.from_numpy(x).permute(2, 0, 1)[None].float().cuda() / 255.0) - mean) / std

    def pose(i):
        w_T_c = np.eye(4, dtype=np.float32)
        w_T_c[:3, :3], w_T_c[:3, 3] = R[i], t[i]
        return torch.from_numpy(w_T_c).cuda(), torch.from_numpy(np.linalg.inv(w_T_c)).cuda()

    def Ks(batch_shape):
        d = {}
        for s in range(5):
            k = K.copy()
            k[:2] /= 2 ** s
            kk = torch.from_numpy(k).cuda()
            d[f"K_s{s}_b44"], d[f"invK_s{s}_b44"] = kk.expand(*batch_shape, 4, 4).clone(), torch.linalg.inv(kk).expand(*batch_shape, 4, 4).clone()
        return d

    times, rows, mem = [], [], 0
    for i in range(1, N):
        src = pick_sources(R, t, i, a.sources)
        if len(src) < a.sources:                                                    # the feature-volume MLP is built for exactly 7 source views
            continue
        w_T_c, c_T_w = pose(i)
        cur = dict(image_b3hw=img(i), world_T_cam_b44=w_T_c[None], cam_T_world_b44=c_T_w[None], **Ks((1,)))
        sp = [pose(j) for j in src]
        srcd = dict(image_b3hw=torch.cat([img(j) for j in src])[None], world_T_cam_b44=torch.stack([p[0] for p in sp])[None], cam_T_world_b44=torch.stack([p[1] for p in sp])[None], **Ks((1, len(src))))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out = model("test", cur, srcd, return_mask=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        mem = max(mem, torch.cuda.max_memory_allocated() / 1e9)
        sr = out["depth_pred_s0_b1hw"][0, 0].float().cpu().numpy()                 # metres, model's depth resolution
        mask = out["overall_mask_bhw"][0].float().cpu().numpy() > 0.5
        h, w = sr.shape
        st = np.zeros((H, W), np.float32)
        ok = trusted[i] & np.isfinite(disp[i]) & (disp[i] > 0.5)
        st[ok] = f0 * B / disp[i][ok]
        import cv2
        st_s = cv2.resize(st, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_s = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) if mask.shape != (h, w) else mask
        both = (st_s > 0) & mask_s & (st_s < 5.0)
        if both.sum() < 500:
            continue
        rel = np.abs(sr[both] - st_s[both]) / st_s[both]
        near, far = both & (st_s < 1.0), both & (st_s >= 2.0)
        rows.append([both.sum() / (st_s > 0).sum(), np.median(rel), np.median(np.abs(sr[near] - st_s[near]) / st_s[near]) if near.sum() > 200 else np.nan,
                     np.median(np.abs(sr[far] - st_s[far]) / st_s[far]) if far.sum() > 200 else np.nan, mask_s.mean(), (st_s > 0).mean(), np.median(sr[both] / st_s[both])])
    rows = np.array(rows)
    t_ = np.array(times[3:] if len(times) > 6 else times)                           # the first calls compile kernels
    print(f"\n{len(times)} keyframes ({'fast cost volume' if a.fast else 'normal cost volume'}, {a.sources} sources): {np.median(t_):.0f} ms median, p90 {np.percentile(t_, 90):.0f} ms, "
          f"peak GPU memory {mem:.2f} GB")
    print(f"stereo depth valid on {np.nanmean(rows[:, 5]) * 100:.0f}% of pixels; SimpleRecon had source-view evidence on {np.nanmean(rows[:, 4]) * 100:.0f}%; "
          f"stereo pixels inside that evidence: {np.nanmean(rows[:, 0]) * 100:.0f}%")
    print(f"|SimpleRecon - stereo| / stereo, median over keyframes: all {np.nanmedian(rows[:, 1]) * 100:.1f}%  near (<1 m) {np.nanmedian(rows[:, 2]) * 100:.1f}%  far (>=2 m) {np.nanmedian(rows[:, 3]) * 100:.1f}%")
    print(f"median ratio SimpleRecon / stereo per keyframe: median {np.median(rows[:, 6]):.2f}, 10th-90th percentile {np.percentile(rows[:, 6], 10):.2f} to {np.percentile(rows[:, 6], 90):.2f}; "
          f"keyframes within 10% of stereo: {(rows[:, 1] < 0.10).mean() * 100:.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("folder"), c.add_argument("out"), c.add_argument("--model", default="s"), c.add_argument("--stride", type=int, default=5), c.add_argument("--frames", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("frames"), r.add_argument("--sources", type=int, default=7), r.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    capture(a) if a.cmd == "capture" else run_model(a)


if __name__ == "__main__":
    main()
