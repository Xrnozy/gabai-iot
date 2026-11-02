"""One-time frame calibration from four short captures: no tumbling, no camera, nothing but resting and turning on the spot.

  1. still, lying on its back with the front (camera side) UP        -> gravity in the IMU axes = device z
  2. turning slowly one full turn about the vertical in that pose     -> compass readings on a ring
  3. still, standing upright as worn on the chest                     -> gravity in the IMU axes, whose perpendicular part to z is device -y
  4. turning slowly one full turn about the vertical in that pose     -> a second ring, about a different axis

From these: R_dev_imu (triad, see gabai.frames), the compass's hard-iron offset (sphere through both rings), and the compass->IMU axis map (the one of the 24
axis-aligned rotations under which the field keeps a constant angle to gravity while turning about gravity, in BOTH poses: one pose alone cannot tell the
axes apart because turning about gravity leaves rotations about gravity unseen; two different vertical axes resolve it).

What it does not give: soft iron (a stretch of the compass readings along tilted axes) needs readings from every direction, which two rings do not provide;
gabai.tools.sync_sensors does that with a tumble. The sphere fit here removes hard iron, which is the large part (about 50 uT on your board).
"""
import numpy as np

from gabai.frames import triad
from gabai.mag import MG_PER_COUNT, fit_sphere
from gabai.mag_sync import rotations_24

MIN_SWEEP_DEG = 270.0


def mean_up(accel):
    a = np.asarray(accel, float)
    m = a.mean(axis=0)
    return m / np.linalg.norm(m)


def still_ok(gyro_dps, accel, gyro_max=3.0, accel_sd_max=0.03):
    g, a = np.asarray(gyro_dps, float), np.asarray(accel, float)
    return bool(np.linalg.norm(g.mean(axis=0)) < gyro_max and np.linalg.norm(a.std(axis=0)) < accel_sd_max)


def sweep_about(m, up):
    """Total angle (degrees) the field vector turns about the axis `up`, following the shortest step each time."""
    u = np.asarray(up, float)
    u = u / np.linalg.norm(u)
    e1 = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    m = np.asarray(m, float)
    ang = np.unwrap(np.arctan2(m @ e2, m @ e1))
    return float(abs(np.degrees(ang[-1] - ang[0])))


def solve(still_front_up, still_upright, spin_front_up, spin_upright):
    """Each still_*: dict(accel=(n,3) in g). Each spin_*: dict(accel=(n,3), mag=(k,3) raw counts). Returns a dict with R_dev_imu, R_imu_mag, mag calibration
    and a list of problems (empty when everything looked right)."""
    problems = []
    uB, uA = mean_up(still_front_up["accel"]), mean_up(still_upright["accel"])
    R_dev_imu = triad(uB, uA)
    sB, sA = mean_up(spin_front_up["accel"]), mean_up(spin_upright["accel"])
    for name, s, ref in (("lying-on-its-back turn", sB, uB), ("upright turn", sA, uA)):
        drift = float(np.degrees(np.arccos(np.clip(s @ ref, -1, 1))))
        if drift > 15.0:
            problems.append(f"during the {name} the device's tilt changed by {drift:.0f} degrees from the still reading: keep the same pose while turning")
    mB, mA = np.asarray(spin_front_up["mag"], float), np.asarray(spin_upright["mag"], float)
    both = np.vstack([mB, mA])
    centre, _ = fit_sphere(both)
    cB, cA = mB - centre, mA - centre
    radius = float(np.mean(np.linalg.norm(np.vstack([cB, cA]), axis=1)))
    quality = float(np.std(np.linalg.norm(np.vstack([cB, cA]), axis=1)) / radius)
    swept = {}
    best = None
    scores = []
    for P in rotations_24():
        e = []
        for c, u in ((cB, sB), (cA, sA)):
            v = c @ P.T
            ang = np.degrees(np.arccos(np.clip(v @ u / np.linalg.norm(v, axis=1), -1, 1)))
            e.append((float(np.var(ang)), float(np.mean(ang))))
        score = e[0][0] + e[1][0] + (e[0][1] - e[1][1]) ** 2
        scores.append((score, P, e[0][1], e[1][1]))
    scores.sort(key=lambda x: x[0])
    score, R_imu_mag, angB, angA = scores[0]
    second = scores[1][0]
    for name, c, P, u in (("lying-on-its-back turn", cB, R_imu_mag, sB), ("upright turn", cA, R_imu_mag, sA)):
        swept[name] = sweep_about(c @ P.T, u)
        if swept[name] < MIN_SWEEP_DEG:
            problems.append(f"the {name} covered only {swept[name]:.0f} degrees: turn through a full circle slowly")
    if second < 3.0 * score + 1.0:
        problems.append(f"the compass axis map is not clearly better than the next candidate (score {score:.2f} against {second:.2f}): turn more slowly and through a full circle in both poses")
    if quality > 0.1:
        problems.append(f"the compass readings do not lie on a sphere (quality {quality:.2f}): something magnetic moves relative to the sensor, or the turns were not about the vertical")
    ut = radius * MG_PER_COUNT / 10.0
    if not 20.0 <= ut <= 70.0:
        problems.append(f"the field after removing the offset is {ut:.0f} uT, Earth's is 25 to 65 uT: soft iron (needs sync_sensors) or a magnet that moves with the sensor")
    return {"R_dev_imu": R_dev_imu, "R_imu_mag": R_imu_mag, "offset": centre, "radius_counts": radius, "quality": quality,
            "field_uT": ut, "dip_deg": 0.5 * (angB + angA) - 90.0, "dip_gap_deg": abs(angB - angA), "map_score": score, "map_score_next": second,
            "swept_deg": swept, "problems": problems}
