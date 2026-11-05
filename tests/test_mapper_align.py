import numpy as np
from scipy.spatial.transform import Rotation as Rot

from mapper.align import OPTICAL, alignment, apply, camera_up_prior


def _room(R_tilt, rng, n=6000):
    """A box room (floor, ceiling, 4 walls) with an arbitrary tilt applied to everything: what RTAB-Map's map looks like when the first camera was not level."""
    P, N = [], []
    def face(axis, val, lo, hi, n_):
        u = rng.uniform(lo, hi, (n_, 3)); u[:, axis] = val
        nrm = np.zeros(3); nrm[axis] = 1.0
        return u + rng.normal(0, 0.003, u.shape), np.tile(nrm, (n_, 1))
    for axis, val in ((2, 0.0), (2, 2.6), (0, -2.0), (0, 2.0), (1, -1.5), (1, 1.5)):
        p, nn = face(axis, val, [-2, -1.5, 0], [2, 1.5, 2.6], n)
        P.append(p); N.append(nn)
    P, N = np.concatenate(P), np.concatenate(N)
    return P @ R_tilt.T, N @ R_tilt.T


def test_recovers_up_and_walls_of_a_tilted_map():
    rng = np.random.default_rng(0)
    tilt = Rot.from_euler("xyz", [-35, 12, 40], degrees=True).as_matrix()          # the map is pitched 35 deg, rolled 12, turned 40
    P, N = _room(tilt, rng)
    # cameras: upright on average in the TRUE room, so their up vectors in the tilted map are tilt @ z, scattered by hand-held wobble
    poses = {}
    for i in range(60):
        up = tilt @ (Rot.from_euler("xyz", rng.normal([20, 0, 0], 12, 3), degrees=True).as_matrix() @ np.array([0, 0, 1.0]))
        z = np.array([0.0, 0, 1])
        Rc = np.eye(3)
        # build a camera whose optical -y axis is `up`: optical x = right, y = down, z = forward
        fwd = np.cross(up, [0.3, 1, 0]); fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        Rc = np.stack([right, -up, fwd], axis=1)
        T = np.eye(4); T[:3, :3] = Rc @ np.linalg.inv(OPTICAL[:3, :3]); poses[i] = T
    A, info = alignment(poses, P, N, np.ones(len(P)), refine=True, log=lambda *_: None)
    assert A is not None, info
    Pa = apply(A, P)
    assert np.degrees(np.arccos(np.clip(np.abs(np.mean(apply(A, N)[:6000, 2])), 0, 1))) < 1.0       # floor normals now along z
    # every wall is on an axis: the box faces map to +-x, +-y, +-z within 1 degree
    Na = apply(A, N)
    assert np.abs(np.abs(Na).max(axis=1) - 1).max() < 0.02
    assert info["refinement"] == "accepted" and info["tilt_corrected_deg"] > 10                                                           # the cameras were held pitched 20 deg on average: the prior alone would be 20 deg off, the planes correct it


def test_refuses_without_planes():
    rng = np.random.default_rng(1)
    P = rng.normal(0, 1, (20000, 3)); N = rng.normal(0, 1, (20000, 3)); N /= np.linalg.norm(N, axis=1, keepdims=True)
    poses = {0: np.eye(4)}
    A, info = alignment(poses, P, N, np.ones(len(P)), refine=True, log=lambda *_: None)
    assert A is None and "reason" in info


def test_planes_that_contradict_the_camera_prior_are_not_believed():
    rng = np.random.default_rng(3)
    tilt = Rot.from_euler("xyz", [-35, 12, 40], degrees=True).as_matrix()
    P, N = _room(tilt, rng)
    wrong = Rot.from_euler("x", 45, degrees=True).as_matrix()                       # cameras whose mean up is 45 deg off the room vertical: no room axis within the 25 deg limit
    poses = {}
    for i in range(40):
        up = tilt @ wrong @ np.array([0, 0, 1.0])
        fwd = np.cross(up, [0.3, 1, 0]); fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        T = np.eye(4); T[:3, :3] = np.stack([right, -up, fwd], axis=1) @ np.linalg.inv(OPTICAL[:3, :3]); poses[i] = T
    A, info = alignment(poses, P, N, np.ones(len(P)), refine=True, log=lambda *_: None)
    assert A is not None and info["refinement"].startswith("rejected")
    assert np.allclose(A[2, :3], info["prior_up"], atol=1e-6)                         # kept the prior


def test_default_uses_the_camera_prior_only():
    rng = np.random.default_rng(4)
    tilt = Rot.from_euler("xyz", [-35, 12, 40], degrees=True).as_matrix()
    P, N = _room(tilt, rng)
    poses = {}
    for i in range(30):
        up = tilt @ np.array([0, 0, 1.0])
        fwd = np.cross(up, [0.3, 1, 0]); fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        T = np.eye(4); T[:3, :3] = np.stack([right, -up, fwd], axis=1) @ np.linalg.inv(OPTICAL[:3, :3]); poses[i] = T
    A, info = alignment(poses, P, N, np.ones(len(P)), log=lambda *_: None)
    assert info["refinement"].startswith("off") and np.allclose(A[2, :3], tilt @ np.array([0, 0, 1.0]), atol=1e-6)
