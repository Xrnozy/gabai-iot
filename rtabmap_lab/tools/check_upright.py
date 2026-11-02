"""Is the picture RTAB-Map is given the right way up? A stereo-only SLAM has no gravity: it starts with the camera assumed level, z up, and builds a map with z along the picture's
'up'. A picture that is upside down therefore gives an UPSIDE-DOWN map. This check uses a recording's accelerometer ONLY as an outside judge (RTAB-Map never sees it): it measures where
'up' is in the camera's own axes at the start of the recording and says how far that is from the 'up' RTAB-Map assumes, for the picture as is and turned 180 deg.

  py -m rtabmap_lab.tools.check_upright data/recordings/vio_20261101_014854 [--seconds 3]
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--imu-sync", default="data/imu_sync_corrected.json", help="camera-IMU rotation for the pictures as received (the file the pipeline uses)")
    ap.add_argument("--seconds", type=float, default=3.0)
    a = ap.parse_args()
    imu = np.loadtxt(os.path.join(a.folder, "imu.csv"), delimiter=",", skiprows=1)
    imu = imu[imu[:, 0] <= imu[0, 0] + a.seconds]
    acc = imu[:, 4:7].mean(axis=0)
    up_imu = acc / np.linalg.norm(acc)                                   # an accelerometer reads +up when still
    R_ic = np.array(json.load(open(a.imu_sync))["R_ci"], float).T          # camera axes in IMU axes, for the pictures as received
    D = np.diag([-1.0, -1.0, 1.0])
    assumed = np.array([0.0, -1.0, 0.0])                                 # RTAB-Map's start: the camera level and upright, so 'up' is minus the optical y axis
    print(f"{len(imu)} IMU samples from the first {a.seconds:g} s, mean |a| = {np.linalg.norm(acc):.2f} g, gyro spread {imu[:, 1:4].std(axis=0).round(1)} deg/s")
    for name, up_c in (("pictures as received", R_ic.T @ up_imu), ("pictures turned 180 deg (--flip180)", D @ R_ic.T @ up_imu)):
        ang = float(np.degrees(np.arccos(np.clip(up_c @ assumed, -1, 1))))
        verdict = "UPRIGHT" if ang < 60 else ("UPSIDE DOWN" if ang > 120 else "sideways/uncertain")
        print(f"  {name:40s} up in camera axes {np.round(up_c, 2)}  angle from RTAB-Map's assumed up: {ang:5.1f} deg  -> {verdict}")


if __name__ == "__main__":
    main()
