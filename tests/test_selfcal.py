import json
import os
import cv2
import numpy as np
import pytest

from mapper.selfcal.core import Insufficient, Obs, calibrate, project

SIZE = (640, 360)
CL = np.array([612.0, 316.0, 176.0, 0.12, -0.25])        # f, cx, cy, k1, k2
CR = np.array([598.0, 327.0, 167.0, 0.10, -0.20])
RS = cv2.Rodrigues(np.radians([0.8, -1.5, 0.6]))[0]
TS = np.array([-0.0468, 0.0008, 0.0010])                 # metres; only its direction is recoverable
B = np.linalg.norm(TS)


def _scene(n_frames=14, rot_deg=18.0, n_points=2500, noise=0.3, seed=0, outliers=0.0):
    """A room of points 0.8-4 m away; the rig turns and moves between frames. Every point visible in a picture is an observation (a perfect tracker + noise)."""
    rng = np.random.default_rng(seed)
    X = np.c_[rng.uniform(-3, 3, n_points), rng.uniform(-2, 2, n_points), rng.uniform(0.8, 4.0, n_points)]
    frame, side, xy, point = [], [], [], []
    for i in range(n_frames):
        R = cv2.Rodrigues(np.radians(rng.uniform(-rot_deg, rot_deg, 3) * np.array([0.6, 1.0, 0.3])))[0]
        t = rng.uniform(-0.3, 0.3, 3)
        Xc = X @ R.T + t
        for s, (c, Xs) in enumerate(((CL, Xc), (CR, Xc @ RS.T + TS))):
            ok = Xs[:, 2] > 0.3
            uv = np.full((len(X), 2), np.nan)
            uv[ok] = project(Xs[ok], c)
            ok &= (uv[:, 0] > 2) & (uv[:, 0] < SIZE[0] - 2) & (uv[:, 1] > 2) & (uv[:, 1] < SIZE[1] - 2)
            idx = np.flatnonzero(ok)
            p = uv[idx] + rng.normal(0, noise, (len(idx), 2))
            bad = rng.random(len(idx)) < outliers
            p[bad] = rng.uniform([0, 0], SIZE, (bad.sum(), 2))
            frame += [i] * len(idx); side += [s] * len(idx); xy.append(p); point += list(idx)
    o = Obs(frame, side, np.concatenate(xy), point, n_frames, SIZE)
    cnt = np.bincount(o.point, minlength=n_points)
    return o.subset(cnt[o.point] >= 3)[0]


def test_recovers_a_known_rig_without_any_target():
    ba = calibrate(_scene(outliers=0.03), log=lambda *_: None)
    cL, cR = ba["cL"], ba["cR"]
    assert abs(cL[0] - CL[0]) / CL[0] < 0.01 and abs(cR[0] - CR[0]) / CR[0] < 0.01      # focal within 1%
    assert np.abs(cL[1:3] - CL[1:3]).max() < 3 and np.abs(cR[1:3] - CR[1:3]).max() < 3  # principal point within 3 px
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(ba["Rs"].T @ RS)[0])) < 0.1         # stereo rotation within 0.1 deg
    ang = np.degrees(np.arccos(np.clip(ba["ts"] @ TS / B, -1, 1)))
    assert ang < 1.5 and ang < 3 * np.degrees(np.linalg.norm(ba["std"][13:15])) + 0.1   # baseline direction: within 1.5 deg, and inside its own reported uncertainty
    assert ba["inlier_rms_px"] < 0.5
    assert ba["std"][0] / cL[0] < 0.01                                                  # and it knows it is well determined


def test_refuses_when_views_do_not_constrain_the_focal_length():
    with pytest.raises(Insufficient):
        calibrate(_scene(n_frames=6, rot_deg=0.0), log=lambda *_: None)                # no rotation between frames: f is not observable


# ---- output, validation, gating -----------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def solved():
    return calibrate(_scene(outliers=0.03), log=lambda *_: None)


def test_npz_loads_in_the_pipeline_and_rectifies_rows(solved, tmp_path):
    from mapper.selfcal.output import to_npz
    from rtabmap_lab.calib_lab.params import CalibParams
    p = to_npz(str(tmp_path / "c.npz"), solved, SIZE, baseline_mm=46.8)
    cp = CalibParams.from_npz(p)
    r = cp.rect()
    assert abs(r.B - 0.0468) < 1e-6 and 550 < r.f < 800                      # alpha=0 rectification crops, so the rectified focal exceeds the raw one
    rng = np.random.default_rng(5)
    X = np.c_[rng.uniform(-2, 2, 500), rng.uniform(-1, 1, 500), rng.uniform(1, 4, 500)]
    uvl, uvr = project(X, CL), project(X @ RS.T + TS, CR)                       # true pixels of the same points in the two raw pictures
    vis = ((uvl > 5) & (uvl < [635, 355])).all(1) & ((uvr > 5) & (uvr < [635, 355])).all(1)         # only points both cameras actually see
    uvl, uvr = uvl[vis], uvr[vis]
    z = np.load(p)
    pl = cv2.undistortPoints(uvl.reshape(-1, 1, 2), z["K1"], z["D1"], R=z["R1"], P=z["P1"]).reshape(-1, 2)
    pr = cv2.undistortPoints(uvr.reshape(-1, 1, 2), z["K2"], z["D2"], R=z["R2"], P=z["P2"]).reshape(-1, 2)
    assert np.median(np.abs(pl[:, 1] - pr[:, 1])) < 0.3 and np.percentile(np.abs(pl[:, 1] - pr[:, 1]), 95) < 1.0   # the estimated calibration puts both views on the same rows
    assert (pl[:, 0] - pr[:, 0] > 0).mean() > 0.95                                # positive disparity (the validation threshold; the misses are points at the picture border)


def test_update_gate_rejects_jumps_and_less_certain_estimates(solved):
    from mapper.selfcal.output import gate_update
    ok, why = gate_update(solved, solved)
    assert ok, why
    jumped = dict(solved, cL=solved["cL"] + np.array([40.0, 0, 0, 0, 0]))
    ok, why = gate_update(solved, jumped)
    assert not ok and any("sigma" in w for w in why)
    vague = dict(solved, std=solved["std"] * 4)
    assert not gate_update(solved, vague)[0]


def test_cli_refuses_without_writing_a_calibration(tmp_path):
    from mapper.selfcal.__main__ import main
    root = str(tmp_path / "selfcal")
    rc = main(["calibrate", "data/recordings/still_20261030_010616", "--root", root, "--frames", "30", "--max-dt-ms", "20"])
    run = os.listdir(root)[0]
    files = os.listdir(os.path.join(root, run))
    assert rc == 2 and "calib.npz" not in files and "report.json" in files             # a static camera has one view: the focal length is not observable, so nothing is produced
    assert json.load(open(os.path.join(root, run, "report.json")))["refused"]
