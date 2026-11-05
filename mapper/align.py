"""Vision-only world alignment: which way is up, and where are the walls, in RTAB-Map's map.

RTAB-Map's world frame is the first camera pose taken as level. If the rig starts pitched (this walk starts looking up 26 degrees and is at 56 within three seconds, measured with the accelerometer
used as a judge only) the whole map is tilted by that: a monitor facing the camera is drawn facing up. Odometry and loop closure are not what is wrong here: linked nodes agree with the gyro's relative
rotation to 1.6 degrees median (47 + 82 loop links). What is missing is the orientation of the WORLD, and it cannot come from the odometry. No IMU is used to get it here:
  1. prior for up: the mean of the camera's own up direction over the whole trajectory (a hand-held rig is upright on average; the bias is the average pitch, tens of degrees at worst)
  2. planes of the map (floor, ceiling, table tops, walls, monitor faces) found by weighted RANSAC on the surfels
  3. a Manhattan fit refines up: normals within 30 degrees of the prior are horizontal surfaces, normals within 30 degrees of perpendicular are vertical surfaces; up is the direction that makes the first
     as parallel and the second as perpendicular to it as possible (largest eigenvector of  sum_H w n n^T - sum_V w n n^T ), iterated
  4. heading: vertical plane normals are turned onto the x / y axes (circular mean of 4 x azimuth), so walls run along the axes
The result is a rigid transform A (world -> aligned world) that is applied to poses and points when they are written or drawn; the map itself is not changed. It refuses (returns None) when there are too few
planes, and it reports how well the planes support it. Validation without the IMU in the loop: the accelerometer's gravity direction in each node's camera frame against the aligned map's z axis.
"""
import numpy as np

from mapper.flatten import find_plane

OPTICAL = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])


def camera_up_prior(poses):
    """Mean of the cameras' up directions (optical -y) in the world, unit length; None for no poses."""
    ups = [(T @ OPTICAL)[:3, :3] @ np.array([0.0, -1.0, 0.0]) for T in poses.values() if T is not None]
    if not ups:
        return None
    u = np.mean(ups, axis=0)
    return u / np.linalg.norm(u)


def planes_of(P, N, w, n_planes=8, tol=0.04, min_points=800, seed=0):
    """Sequential weighted RANSAC. Returns [(unit normal, d, weight sum)] largest first. P (n,3), N (n,3), w (n,)."""
    P, N, w = np.asarray(P, np.float64), np.asarray(N, np.float64), np.asarray(w, np.float64)
    free = np.ones(len(P), bool)
    out = []
    for k in range(n_planes):
        idx = np.nonzero(free)[0]
        if len(idx) < min_points:
            break
        r = find_plane(P[idx], N[idx], w[idx], tol, seed=seed + k)
        if r is None or r[2].sum() < min_points:
            break
        n, d, m, _ = r
        out.append((n, d, float(w[idx[m]].sum()), int(m.sum())))
        free[idx[m]] = False
    return out


def manhattan(planes, up0, cone_deg=30.0, iters=10):
    """planes: [(normal, d, weight, count)]. Returns (up, yaw_axes, support) or None: up unit vector refined from up0, and the horizontal unit vector the walls should map to x."""
    if not planes:
        return None
    Nn = np.array([p[0] for p in planes])
    W = np.array([p[2] for p in planes], float)
    W = W / W.sum()
    cone = np.cos(np.radians(cone_deg))
    u = up0 / np.linalg.norm(up0)
    for _ in range(iters):
        c = Nn @ u
        H = np.abs(c) > cone                                           # horizontal surfaces: normal parallel to up
        V = np.abs(c) < np.sin(np.radians(cone_deg))                   # vertical surfaces: normal perpendicular to up
        if not H.any() and V.sum() < 2:
            return None
        M = (Nn[H].T * W[H]) @ Nn[H] - (Nn[V].T * W[V]) @ Nn[V]
        ev, evec = np.linalg.eigh(M)
        un = evec[:, -1] if H.any() else evec[:, 0]                    # without a floor or ceiling, up is the direction the vertical normals are most perpendicular to: smallest eigenvalue
        un = un if un @ u >= 0 else -un
        if np.degrees(np.arccos(np.clip(un @ u, -1, 1))) < 0.05:
            u = un
            break
        u = un
    c = Nn @ u
    V = np.abs(c) < np.sin(np.radians(cone_deg))
    H = np.abs(c) > cone
    support = float(W[V | H].sum())
    e1 = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    if V.any():
        az = np.arctan2(Nn[V] @ e2, Nn[V] @ e1)
        yaw = np.arctan2(np.sum(W[V] * np.sin(4 * az)), np.sum(W[V] * np.cos(4 * az))) / 4.0
    else:
        yaw = 0.0
    x = np.cos(yaw) * e1 + np.sin(yaw) * e2
    return u, x, support


def _basis(u, x_hint):
    """Rotation with rows (x, y, u): u up, x the horizontal direction closest to x_hint."""
    x = x_hint - (x_hint @ u) * u
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(u, x), u])


def alignment(poses, P, N, w, min_support=0.5, max_refine_deg=25.0, min_poses=20, refine=False, log=print):
    """4x4 transform A (map -> aligned map) with +z up. Returns (A, info), or (None, info) when there is nothing to base it on.
    Up = the camera-up prior. With refine=True the planes of the map may refine it, if they agree with it to within max_refine_deg and are well supported. Off by default: measured on the walk (accelerometer as
    judge) the prior alone is 6.5 deg from gravity, while the plane refinement ended 67 deg away in one run and 20 deg in the next (a monitor face and its stacked sheets dominate the planes of this scene).
    Heading: with accepted planes, walls on the x / y axes; otherwise x is the horizontal part of the first camera's forward direction (RTAB-Map's own convention)."""
    if len(poses) < min_poses:
        return None, {"reason": f"{len(poses)} poses (need {min_poses}+ for a camera-up prior)"}
    up0 = camera_up_prior(poses)
    first = poses[min(poses)]
    fwd0 = (first @ OPTICAL)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    info = {"prior_up": up0.tolist(), "planes": 0, "support": 0.0, "refinement": "off (camera-up prior only)", "tilt_corrected_deg": 0.0}
    u, x_hint = up0, fwd0
    if refine and len(P) >= 5000:
        sel = np.random.default_rng(0).choice(len(P), min(len(P), 60000), replace=False)
        planes = planes_of(P[sel], N[sel], w[sel])
        info["planes"] = len(planes)
        m = manhattan(planes, up0)
        if m is None:
            info["refinement"] = "rejected: no usable planes"
        else:
            ur, xr, support = m
            moved = float(np.degrees(np.arccos(np.clip(ur @ up0, -1, 1))))
            info["support"] = support
            if support < min_support:
                info["refinement"] = f"rejected: only {support * 100:.0f}% of the plane weight is horizontal or vertical"
            elif moved > max_refine_deg:
                info["refinement"] = f"rejected: the planes put up {moved:.0f} deg from the camera-up prior (limit {max_refine_deg:.0f})"
            else:
                u, x_hint = ur, xr
                info["refinement"], info["tilt_corrected_deg"] = "accepted", moved
    A = np.eye(4)
    A[:3, :3] = _basis(u, x_hint)                                      # rows: the new axes in the old frame
    info["up"] = u.tolist()
    return A, info


def apply(A, T):
    """Aligned pose / points: poses 4x4 (T), points (n,3) -> aligned."""
    if np.shape(T)[-2:] == (4, 4):
        return A @ T
    return np.asarray(T) @ A[:3, :3].T


def up_error_vs_accel(rec, poses, node_frame, A=None, window=3, g_tol=0.12):
    """Judge only: angle (deg) between the accelerometer's up in each node's camera frame and the map's z axis in that camera frame, for nodes where the rig is near 1 g (not accelerating).
    rec: calib_lab Recording (needs imu, R_ci). Returns (errors array, n nodes used)."""
    from rtabmap_lab.orientation import load_R_ic
    R_ic = load_R_ic()
    imu = rec.imu
    errs = []
    for nid, T in poses.items():
        f = node_frame.get(nid)
        if f is None or imu is None:
            continue
        t = rec.t[f]
        m = np.abs(imu[:, 0] - t) < 0.06
        if m.sum() < 3:
            continue
        a = imu[m, 4:7].mean(0)
        if abs(np.linalg.norm(a) - 1.0) > g_tol:
            continue
        up_cam = R_ic.T @ (a / np.linalg.norm(a))                      # up in the camera's optical frame, measured
        Tc = (A @ T if A is not None else T) @ OPTICAL                 # camera -> (aligned) world
        z_cam = Tc[:3, :3].T @ np.array([0.0, 0.0, 1.0])              # the map's up in the camera frame
        errs.append(np.degrees(np.arccos(np.clip(up_cam @ z_cam, -1, 1))))
    return np.array(errs)
