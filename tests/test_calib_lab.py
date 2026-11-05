"""The calibration lab: the editable calibration is exactly the saved one when unchanged, every parameter is explained, the automatic check recovers a known error and refuses weak data,
and an edited calibration really reaches RTAB-Map (the last test needs the bridge and a recording; it is skipped without them)."""
import json
import os

import cv2
import numpy as np
import pytest

from rtabmap_lab.calib_lab import params as P
from rtabmap_lab.calib_lab.autocalib import AutoCalibration

CALIB = "data/stereo_calib_charuco_45.npz"
REC = "data/recordings/walk_20261102_230115"
have_calib = pytest.mark.skipif(not os.path.exists(CALIB), reason="saved calibration missing")
have_all = pytest.mark.skipif(not (os.path.exists(CALIB) and os.path.exists(REC) and os.path.exists("rtabmap_lab/bin/rtabmap_bridge.exe")), reason="needs the recording and the built bridge")


def synthetic_npz(tmp_path, rot_deg=(0.3, 0.1, 0.2)):
    K = np.array([[600.0, 0, 320], [0, 600, 180], [0, 0, 1]])
    R = cv2.Rodrigues(np.radians(np.array(rot_deg)))[0]
    T = np.array([[-0.047], [0.0004], [0.0]])
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, np.zeros(5), K, np.zeros(5), (640, 360), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    path = str(tmp_path / "syn.npz")
    np.savez(path, K1=K, D1=np.zeros((1, 5)), K2=K, D2=np.zeros((1, 5)), R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, size=np.array([640, 360]), baseline=0.047, focal=600.0, cx=320.0, cy=180.0)
    return path, K, R, T


def correspondences(K, R, T, n=3000, seed=0, region=None, depth=(0.8, 4.0), noise_px=0.2):
    rng = np.random.default_rng(seed)
    X = np.stack([rng.uniform(-1.6, 1.6, n), rng.uniform(-1.0, 1.0, n), rng.uniform(*depth, n)], 1)
    xl = X[:, :2] / X[:, 2:]
    Xr = X @ R.T + T.ravel()
    xr = Xr[:, :2] / Xr[:, 2:] + rng.normal(0, noise_px / 600.0, (n, 2))
    px = xl * 600.0 + [320, 180]
    ok = (px[:, 0] > 0) & (px[:, 0] < 640) & (px[:, 1] > 0) & (px[:, 1] < 360)
    if region is not None:
        ok &= (px[:, 0] < region[0]) & (px[:, 1] < region[1])
    return [{"i": k, "na": xl[ok][k::10], "nb": xr[ok][k::10], "px": px[ok][k::10], "disp": (px[ok][k::10, 0] - (xr[ok][k::10, 0] * 600 + 320)), "sharp": 1.0} for k in range(10)]


# ---- the parameters ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
@have_calib
def test_unchanged_calibration_is_exactly_the_saved_one():
    from gabai.tools.calibrate_stereo import Rectifier
    p, saved = P.CalibParams.from_npz(CALIB), Rectifier(CALIB)
    im = np.random.default_rng(1).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    for a, b in zip(p.rect().apply(im, im), saved.apply(im, im)):
        assert np.array_equal(a, b)
    assert p.rect().f == pytest.approx(saved.f) and p.rect().B == pytest.approx(saved.B) and p.changed() == []


@have_calib
def test_baseline_scales_translation_and_keeps_direction():
    p = P.CalibParams.from_npz(CALIB)
    d0 = np.array([p.values[k] for k in ("T.x_mm", "T.y_mm", "T.z_mm")])
    p.set("baseline_mm", 50.0)
    d1 = np.array([p.values[k] for k in ("T.x_mm", "T.y_mm", "T.z_mm")])
    assert np.linalg.norm(d1) == pytest.approx(50.0) and np.allclose(d1 / 50.0, d0 / np.linalg.norm(d0)) and p.rect().B == pytest.approx(0.05)
    p.reset("baseline_mm")                                                    # resetting the derived baseline alone does nothing; its parts are reset by name
    p.reset()
    assert p.changed() == [] and p.values["baseline_mm"] == pytest.approx(p.original["baseline_mm"])


@have_calib
def test_invalid_values_are_refused():
    p = P.CalibParams.from_npz(CALIB)
    for bad in (("K1.fx", float("nan")), ("K1.fx", 10.0), ("baseline_mm", 0.5)):
        with pytest.raises(ValueError):
            p.set(*bad)
    with pytest.raises(KeyError):
        p.set("K1.nonsense", 1.0)


def test_every_parameter_is_explained_in_all_parts():
    for n in P.NAMES + P.DERIVED:
        d = P.explain(n)
        for k in ("what", "changes", "why", "used", "behavior", "guidance", "label"):
            assert len(d[k]) > 15, (n, k)
    assert {P.SPEC[n]["group"] for n in P.NAMES} == {P.INTRINSIC, P.DISTORTION, P.EXTRINSIC, P.RECT}


@have_calib
def test_json_and_npz_round_trip(tmp_path):
    from gabai.tools.calibrate_stereo import Rectifier
    p = P.CalibParams.from_npz(CALIB).with_(K2__cx=330.5, R__rx_deg=-0.5, baseline_mm=49.0)
    q = P.CalibParams.from_json(json.loads(json.dumps(p.to_json(kind="auto"))))
    assert q.fingerprint() == p.fingerprint() and q.changed() == p.changed(P.CalibParams.from_npz(CALIB)) or q.values == p.values
    out = p.to_npz(str(tmp_path / "x.npz"))
    r = Rectifier(out)
    im = np.random.default_rng(2).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    for a, b in zip(r.apply(im, im), p.rect().apply(im, im)):
        assert np.array_equal(a, b)
    assert r.B == pytest.approx(0.049)
    with pytest.raises(ValueError):
        P.CalibParams.from_json({"format": "something-else"})


# ---- the automatic check --------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_auto_solver_recovers_a_known_rotation_error(tmp_path):
    path, K, R_true, T = synthetic_npz(tmp_path)
    p = P.CalibParams.from_npz(path).with_(R__rx_deg=0.3 + 0.35, R__rz_deg=0.2 - 0.25)         # the "current" calibration is off by +0.35 deg pitch and -0.25 deg roll
    th, std, rms, n, _ = AutoCalibration(p, None).solve(correspondences(K, R_true, T))
    assert np.degrees(th[0]) == pytest.approx(-0.35, abs=0.03) and np.degrees(th[2]) == pytest.approx(0.25, abs=0.03)
    assert np.degrees(std[0]) < 0.05 and rms < 0.4                                    # pitch is well determined; the yaw (index 1) is allowed to be uncertain
    assert std[1] > std[0]


def test_auto_refuses_weak_data(tmp_path):
    path, K, R_true, T = synthetic_npz(tmp_path)
    p = P.CalibParams.from_npz(path)
    ac = AutoCalibration(p, None)
    corner = correspondences(K, R_true, T, n=60000, region=(160, 90))                       # every match in one corner of the picture
    ac.collect = lambda frames, progress=None: (corner, [])
    r = ac.run(range(10))
    assert r["verdict"] == "INSUFFICIENT DATA" and r["proposed"] == {} and "params" not in r
    assert any(not c[1] and c[0] == "covers the picture" for c in r["checks"])
    few = correspondences(K, R_true, T)[:3]
    ac.collect = lambda frames, progress=None: (few, [])
    r = ac.run(range(3))
    assert r["verdict"] == "INSUFFICIENT DATA" and r["proposed"] == {}
    assert "baseline length" in r["note"]


# ---- the recording and RTAB-Map -------------------------------------------------------------------------------------------------------------------------------------------------------
@have_all
def test_edited_calibration_reaches_rtabmap_and_the_recording_is_untouched():
    from rtabmap_lab.calib_lab.recording import Recording
    from rtabmap_lab.calib_lab.runner import run_headless
    base = P.CalibParams.from_npz(CALIB)
    rec = Recording(REC).decide(base.rect())
    a = run_headless(rec, base, (0, 25), port=5731)
    wide = base.with_(baseline_mm=base.values["baseline_mm"] * 1.10)
    b = run_headless(rec, wide, (0, 25), port=5732)
    assert a.fingerprint != b.fingerprint and a.tsdf is not b.tsdf                       # two runs, two maps: nothing is shared
    assert b.cal["B"] == pytest.approx(a.cal["B"] * 1.10)
    ids = sorted(set(a.tsdf.frames) & set(b.tsdf.frames))
    assert ids
    da, db = a.tsdf.frames[ids[0]][0].astype(float), b.tsdf.frames[ids[0]][0].astype(float)
    ok = (da > 0) & (db > 0)
    assert ok.sum() > 1000 and np.median(db[ok] / da[ok]) == pytest.approx(1.10, abs=0.01)  # RTAB-Map's own depth scaled with the baseline it was given
    assert rec.unchanged()


@have_all
def test_gyro_keeps_the_heading_when_odometry_is_lost():
    """Pairs 3..68 of the walk: the odometry is lost while the rig turns about 40 deg. Restarted at the old heading (no gyro) the next node is about 35 deg off; with the gyro it is not."""
    import cv2
    from rtabmap_lab.calib_lab.recording import Recording
    from rtabmap_lab.calib_lab.runner import default_settings, run_headless
    from rtabmap_lab.tsdf import OPTICAL
    base = P.CalibParams.from_npz(CALIB)
    rec = Recording(REC).decide(base.rect())

    def worst(gyro, port):
        s = default_settings()
        s["gyro"] = gyro
        r = run_headless(rec, base, (0, 100), s, port=port)
        ids = sorted(i for i in r.tsdf.poses if i in r.node_pair)
        err = []
        for a, b in zip(ids, ids[1:]):
            Ra, Rb = (r.tsdf.poses[a] @ OPTICAL)[:3, :3], (r.tsdf.poses[b] @ OPTICAL)[:3, :3]
            Rg = cv2.Rodrigues(rec.gyro_rotvec(r.node_pair[a], r.node_pair[b]))[0]
            err.append(np.degrees(np.linalg.norm(cv2.Rodrigues((Ra.T @ Rb).T @ Rg)[0])))
        return max(err)
    assert worst(False, 5733) > 25.0
    assert worst(True, 5734) < 15.0
