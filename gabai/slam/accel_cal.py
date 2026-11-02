"""Accelerometer bias and scale from stretches where the rig is held still in different orientations.

A still accelerometer must read exactly gravity (9.80665 m/s^2) whichever way it points. So bias b and per-axis scale s are the
values for which |raw*(1+s) - b| = g for every still orientation. That needs the rig held still in several directions: from one
direction a bias along gravity and a scale error look the same. The six faces (each axis up and down) are the standard set.

Basalt applies `calibrated = raw + S raw - bias`; with S diagonal this is the same model (see to_basalt_params).
"""
import numpy as np
from scipy.optimize import least_squares

G = 9.80665
FACES = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}


def still_windows(t, accel, gyro, win_s=0.25, gyro_max_dps=3.0, std_max=0.08):
    """Mean accelerometer reading of every window in which the gyro is quiet and the accelerometer is steady (m/s^2)."""
    t, accel, gyro = np.asarray(t, float), np.asarray(accel, float), np.asarray(gyro, float)
    out, i = [], 0
    while i < len(t):
        j = np.searchsorted(t, t[i] + win_s)
        if j - i < 5 or j > len(t):
            break
        w = accel[i:j]
        if np.linalg.norm(gyro[i:j], axis=1).max() < np.radians(gyro_max_dps) and w.std(0).max() < std_max:
            out.append(w.mean(0))
        i = j
    return np.array(out)


def cluster_orientations(means, min_sep_deg=15.0):
    """Merge readings that point within `min_sep_deg` of each other; returns one mean reading per orientation."""
    means = np.asarray(means, float)
    groups = []
    for m in means:
        u = m / np.linalg.norm(m)
        for g in groups:
            c = np.mean(g, axis=0)
            if np.degrees(np.arccos(np.clip(u @ (c / np.linalg.norm(c)), -1, 1))) < min_sep_deg:
                g.append(m)
                break
        else:
            groups.append([m])
    return np.array([np.mean(g, axis=0) for g in groups])


def faces_covered(means, min_cos=0.85):
    """Which of the six faces (+x -x +y -y +z -z pointing at gravity) are covered by the readings."""
    u = np.asarray(means, float)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    return {name: bool(np.any(sign * u[:, axis] > min_cos)) for name, (axis, sign) in FACES.items()}


def fit_bias_scale(means):
    """Fit bias (3) and per-axis scale (3) so that |raw*(1+s) - b| = g. Returns a dict with the fit and whether it is trustworthy."""
    m = np.asarray(means, float)

    def residual(p):
        return np.linalg.norm(m * (1 + p[3:]) - p[:3], axis=1) - G

    rms = lambda r: float(np.sqrt(np.mean(r ** 2)))
    sol = least_squares(residual, np.zeros(6), loss="soft_l1", f_scale=0.05)
    J = sol.jac
    sv = np.linalg.svd(J, compute_uv=False)
    cond_ok = len(m) >= 6 and sv[-1] / sv[0] > 0.02          # a direction the data does not constrain shows up as a tiny singular value
    cov = np.linalg.pinv(J.T @ J) * max(rms(sol.fun) ** 2, 1e-6)
    return {"bias": sol.x[:3], "scale": sol.x[3:], "rms_before": rms(residual(np.zeros(6))), "rms_after": rms(sol.fun),
            "std": np.sqrt(np.diag(cov)), "observable": bool(cond_ok), "n": len(m), "sv_ratio": float(sv[-1] / sv[0])}


def to_basalt_params(bias, scale):
    """Basalt's 9 accelerometer parameters: bias (3), then S = [[p3,0,0],[p4,p6,0],[p5,p7,p8]] with only the diagonal used."""
    return [float(bias[0]), float(bias[1]), float(bias[2]), float(scale[0]), 0.0, 0.0, float(scale[1]), 0.0, float(scale[2])]


class StillCapture:
    """Watches a live IMU stream; returns the mean accelerometer reading once per hold, after the rig has been still for `min_s`."""

    def __init__(self, min_s=1.5, settle_s=0.4, gyro_max_dps=3.0, std_max=0.08):
        self.min_s, self.settle_s, self.gyro_max, self.std_max = min_s, settle_s, np.radians(gyro_max_dps), std_max
        self.run, self.captured = [], False

    def feed(self, t, accel, gyro):
        """t in seconds, accel in m/s^2, gyro in rad/s. Returns the mean reading of the hold, or None."""
        if np.linalg.norm(gyro) > self.gyro_max:
            self.run, self.captured = [], False
            return None
        self.run.append((t, np.asarray(accel, float)))
        if self.captured or t - self.run[0][0] < self.min_s:
            return None
        a = np.array([x[1] for x in self.run if x[0] - self.run[0][0] >= self.settle_s])
        if a.std(0).max() > self.std_max:
            self.run = self.run[-1:]                    # steady gyro but a drifting reading: start counting again
            return None
        self.captured = True
        return a.mean(0)
