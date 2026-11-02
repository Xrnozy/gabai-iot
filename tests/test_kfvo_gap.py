"""A stalled link: the pair after the stall comes seconds late, the camera moved meanwhile, and the gyro (also stalled) saw none of it.
Rendered scene, true path known; shows the method, not the real world."""
import numpy as np

from gabai import kfvo
from gabai.kfvo import KeyframeVO
from gabai.sim import look_pose
from gabai.tools import vo_bench as vb

TEX = vb.texture()
SKIP, GAP_AT = 12, 40                                       # frames the camera moves while nothing arrives, and where


def run(seed):
    vo = KeyframeVO(vb.F, vb.F, vb.W / 2, vb.H / 2)
    vo.set_baseline(vb.B)
    rng = np.random.default_rng(seed)
    Rs, ts = vb.truth_path_fast(90 + SKIP, seed)
    keep = [i for i in range(len(Rs)) if not GAP_AT < i < GAP_AT + SKIP]
    ok = np.ones((vb.H, vb.W), bool)
    R0w, _ = look_pose(np.zeros(3))
    prev, err = None, []
    for i in keep:
        img, disp = vb.render(TEX, Rs[i], ts[i], rng=rng)
        gap = prev is not None and i - prev > 1
        kw = dict(frame_s=1.5 if gap else 1 / 15)
        if prev is not None:
            kw["dR_prior"] = np.eye(3) if gap else Rs[prev].T @ Rs[i]          # the gyro missed the stall
        R, _, _ = vo.step(img, disp, ok, **kw)
        prev = i
        if i > GAP_AT + SKIP + 5:
            err.append(np.degrees(np.arccos(np.clip((np.trace((R0w.T @ R).T @ Rs[i]) - 1) / 2, -1, 1))))
    return float(np.mean(err))


def test_the_pose_comes_back_after_a_stall_the_gyro_missed(monkeypatch):
    new = np.mean([run(s) for s in range(3)])
    monkeypatch.setattr(kfvo, "GAP_S", 1e9)                 # the old behaviour: a stall is a normal frame
    old = np.mean([run(s) for s in range(3)])
    print(f"rotation error after the stall: new {new:.2f} deg, old {old:.2f} deg")
    assert new < 5.0 and new <= old, (new, old)
