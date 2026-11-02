"""IMU (BMI160) calibration maths for gabai.tools.calibrate_sensors. Raw in, numbers out; nothing here is applied to a reading.

Gyro: the Pi sender measures the gyro bias at every start and subtracts it before sending, so what arrives here is already bias-corrected. A saved bias would be
subtracted a second time. What is measured and saved is the gyro's noise and the residual left after the Pi's correction (a check: a large residual means the
sender was started while the device moved), and the online bias follower in gabai.fusion handles the slow change after that.
Accelerometer: bias and scale are only produced when the still holds cover the six faces and make the fit well-conditioned (gabai.slam.accel_cal); otherwise the
result says why and no accelerometer calibration is saved.
"""
import numpy as np

from gabai.frames import nearest_rotation, triad
from gabai.slam.accel_cal import G, cluster_orientations, faces_covered, fit_bias_scale

MIN_ORIENTATIONS = 6
NOISE_WARN_DPS = 0.4
RESIDUAL_WARN_DPS = 1.0


def gyro_stats(t, gyro_dps):
    """Noise (std per axis), residual bias (mean) and drift between the two halves, all deg/s, from a still stretch."""
    g, t = np.asarray(gyro_dps, float), np.asarray(t, float)
    h = len(g) // 2
    return {"noise_dps": g.std(axis=0).tolist(), "residual_bias_dps": g.mean(axis=0).tolist(), "drift_dps": (g[h:].mean(axis=0) - g[:h].mean(axis=0)).tolist(),
            "seconds": float(t[-1] - t[0]), "samples": int(len(g))}


def accel_stats(accel_g):
    a = np.asarray(accel_g, float)
    n = np.linalg.norm(a, axis=1)
    return {"mean_g": a.mean(axis=0).tolist(), "std_g": a.std(axis=0).tolist(), "norm_mean_g": float(n.mean()), "norm_std_g": float(n.std())}


def warnings_for(gs, ac):
    w = []
    if max(gs["noise_dps"]) > NOISE_WARN_DPS:
        w.append(f"gyro noise {max(gs['noise_dps']):.2f} deg/s is high (normal below {NOISE_WARN_DPS}): vibration, or the device was being touched")
    if np.linalg.norm(gs["residual_bias_dps"]) > RESIDUAL_WARN_DPS:
        w.append(f"gyro residual {np.linalg.norm(gs['residual_bias_dps']):.2f} deg/s after the Pi's own correction: the sender was probably started while the device moved. Restart it, keeping the device still for its first 3 s")
    if abs(max(map(abs, gs["drift_dps"]))) > 0.3:
        w.append("the gyro readings drift during the still stretch: it was not still, or the sensor is still warming up")
    if not 0.85 <= ac["norm_mean_g"] <= 1.15:
        w.append(f"the accelerometer reads {ac['norm_mean_g']:.3f} g at rest (should be 1.000): its scale or bias needs calibrating, or the device was moving")
    if ac["norm_std_g"] > 0.02:
        w.append(f"the accelerometer is noisy at rest (std {ac['norm_std_g']:.3f} g)")
    return w


def fit_accel(holds_g):
    """holds_g: one mean reading (g) per distinct still orientation. Returns a dict with calibrated (bool), the numbers when supported, and the reason when not."""
    means = cluster_orientations(np.asarray(holds_g, float) * G) if len(holds_g) else np.empty((0, 3))
    cov = faces_covered(means) if len(means) else {k: False for k in ("+x", "-x", "+y", "-y", "+z", "-z")}
    missing = [k for k, v in cov.items() if not v]
    res = {"calibrated": False, "n_orientations": int(len(means)), "faces_missing": missing, "reason": None}
    if len(means) < MIN_ORIENTATIONS or missing:
        res["reason"] = f"only {len(means)} distinct orientations and faces missing {' '.join(missing) or 'none'}: bias and scale cannot be told apart from gravity without all six faces"
        return res
    fit = fit_bias_scale(means)
    res.update(rms_before_ms2=fit["rms_before"], rms_after_ms2=fit["rms_after"], observable=fit["observable"])
    if not fit["observable"]:
        res["reason"] = "the orientations do not pin down bias and scale separately (poorly conditioned fit)"
        return res
    if fit["rms_after"] > 0.5 * max(fit["rms_before"], 1e-9) and fit["rms_before"] > 0.05:
        res["reason"] = f"the fit did not clearly improve the gravity error ({fit['rms_before']:.3f} to {fit['rms_after']:.3f} m/s^2): the holds were not still enough"
        return res
    res.update(calibrated=True, bias_ms2=fit["bias"].tolist(), scale=fit["scale"].tolist(), bias_std=fit["std"][:3].tolist(), scale_std=fit["std"][3:].tolist(),
               improvement_note="gravity error rms before and after")
    return res


def device_from_imu(u_face_down_imu, u_upright_imu):
    """R_dev_imu (v_device = R @ v_imu; device x right, y down, z forward = the camera) from gravity measured in the IMU axes in two poses: face down on the table
    (front = down, so 'up' = -z) and upright as worn (up = -y). triad() takes the front-up pose, which is the face-down one with 'up' reversed."""
    return triad(-np.asarray(u_face_down_imu, float), np.asarray(u_upright_imu, float))


def angle_about(t, gyro_dps, axis):
    """Net and total rotation (degrees) about IMU axis `axis` (0,1,2): trapezoid integral of the gyro. Net keeps the sign (the sense of the turn)."""
    t, g = np.asarray(t, float), np.asarray(gyro_dps, float)[:, axis]
    dt = np.diff(t)
    inc = 0.5 * (g[1:] + g[:-1]) * dt
    return float(inc.sum()), float(np.abs(inc).sum())


def sweep(t, gyro_dps, axis):
    """(net, extent) in degrees about IMU axis `axis`. net = signed turn from the first sample; extent = largest minus smallest running angle, i.e. how much of a circle
    was really swept. Hand tremor, back-and-forth wobble and noise add to the total path but barely to the extent, so extent is what decides that a turn was done."""
    t, g = np.asarray(t, float), np.asarray(gyro_dps, float)[:, axis]
    if len(t) < 2:
        return 0.0, 0.0
    run = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(t))])
    return float(run[-1]), float(run.max() - run.min())


def dominant_axis(t, gyro_dps):
    """(axis index, share of the total rotation about it): which IMU axis the device was really turned about."""
    tot = [angle_about(t, gyro_dps, a)[1] for a in range(3)]
    s = sum(tot) or 1.0
    k = int(np.argmax(tot))
    return k, float(tot[k] / s)
