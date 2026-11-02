import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.tools.check_extrinsics import fit_axes, gyro_rotvec


def test_fit_axes_recovers_a_rotation_that_is_not_a_perfect_permutation_despite_outliers():
    rng = np.random.default_rng(0)
    true = R.from_euler("zyx", [8.0, 3.0, -4.0], degrees=True).as_matrix() @ np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])
    vis = rng.normal(0, np.radians(2.0), (300, 3))
    gyro = vis @ true.T + rng.normal(0, np.radians(0.15), vis.shape)
    gyro[:60] = rng.normal(0, np.radians(3.0), (60, 3))                       # 20% wrong image rotations
    Rf, inl, _ = fit_axes(vis, gyro)
    assert np.degrees(R.from_matrix(Rf @ true.T).magnitude()) < 1.0
    snapped = np.round(true)
    assert np.degrees(R.from_matrix(snapped @ true.T).magnitude()) > 5.0       # a snapped permutation would be several degrees off
    assert inl[60:].mean() > 0.8


def test_gyro_rotvec_integrates_rate_over_the_interval():
    ti = np.arange(0, 1, 0.005)
    w = np.tile([0.0, 0.0, 1.0], (len(ti), 1))                                  # 1 rad/s about z
    g = gyro_rotvec(ti, w, np.gradient(ti), 0.2, 0.4)
    assert abs(g[2] - 0.2) < 0.01 and abs(g[0]) < 1e-9
    assert gyro_rotvec(ti, w, np.gradient(ti), 5.0, 5.2) is None                # no samples there: not zero
