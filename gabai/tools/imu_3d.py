"""Live 3D view of the Pi's IMU and compass in Rerun: a box that turns with the board, plus gyro, accelerometer and compass curves.

PC:  py -m gabai.tools.imu_3d
Pi:  python3 -m gabai.pi.sender --host <PC_IP> --left /dev/video2 --right /dev/video0      (IMU and compass on, i.e. no --no-imu / --no-mag)

Orientation is gyro + gravity (Madgwick), so tilt is held by gravity but the HEADING (yaw) has no reference and drifts slowly; the gyro's residual bias is
followed while the board is still (gabai.orient.GyroBias). The heading is the azimuth of one sensor axis that is not near vertical, shown wrapped and continuous
(an Euler yaw was used before and went wild on a board standing on its side). The compass is shown as its own raw arrow (uncalibrated:
no offset or scale, so its direction is only roughly right near iron or magnets). World z is up. The box is drawn in the sensor frame: red x, green y, blue z.
Units: gyro in degrees per second, accelerometer as the sender gives it (about 1.0 at rest means g), compass in raw counts.
"""
import argparse
import time

import numpy as np
import rerun as rr

from gabai.frames import load_frames
from gabai.fusion import SensorFusion
from gabai.mag import field_ut, load_accel_cal, load_mag_cfg
from gabai.orient import q_to_R
from gabai.pc.receiver import LiveReceiver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--forward", default="auto", choices=["auto", "x", "y", "z"], help="which sensor axis is 'forward' for the heading curve; auto = the device's forward axis (camera z) from the saved frames, else the most horizontal one at start")
    ap.add_argument("--no-auto-bias", action="store_true", help="do not follow the gyro bias while still (to see the raw drift)")
    ap.add_argument("--mag-cal", default="data/mag_cal.json")
    ap.add_argument("--mag-sync", default="data/mag_sync.json")
    ap.add_argument("--accel-cal", default="data/accel_calib.json")
    ap.add_argument("--no-compass", action="store_true", help="ignore the compass (gyro + gravity only)")
    ap.add_argument("--no-accel-cal", action="store_true")
    ap.add_argument("--beta", type=float, default=0.05, help="filter gain: tilt follows gravity with time constant 1/(20*beta) s (0.05: 1 s), heading follows the compass with 1/(6*beta) s (3.3 s). Higher follows the sensors faster, lower trusts the gyro more")
    a = ap.parse_args()
    rr.init("gabai_imu_3d", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/board", rr.Boxes3D(half_sizes=[[0.06, 0.04, 0.01]], colors=[[120, 170, 255]], fill_mode="solid"), static=True)
    for name, vec, col in (("x", [0.12, 0, 0], [255, 60, 60]), ("y", [0, 0.12, 0], [60, 220, 60]), ("z", [0, 0, 0.12], [80, 120, 255])):
        rr.log(f"world/board/axis_{name}", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[vec], colors=[col]), static=True)
    rx = LiveReceiver(a.port, a.token)
    rx.start()
    print(f"listening on port {rx.port}. Start the sender on the Pi with the IMU on. Ctrl+C stops.")
    mag_cfg = None if a.no_compass else load_mag_cfg(a.mag_cal, a.mag_sync)
    accel_cal = None if a.no_accel_cal else load_accel_cal(a.accel_cal)
    print("compass:", "calibrated and axis-mapped: the heading is tied to magnetic north and returns after the rig is lifted and put back" if mag_cfg is not None else
          f"NOT USED ({a.mag_cal} / {a.mag_sync} missing): the heading only integrates the gyro, so it drifts and cannot return after a lift. Run: py -m gabai.tools.sync_sensors")
    print("accelerometer calibration:", "applied" if accel_cal is not None else f"not applied ({a.accel_cal} missing: py -m gabai.tools.calibrate_accel)")
    frames = load_frames()
    known = not frames.sources["imu_to_device"].startswith("none")
    print("device frame:", f"IMU->camera axes from {frames.sources['imu_to_device']}; the heading is that of the camera's forward axis" if known else
          "UNKNOWN (no data/imu_sync.json or data/device_frame.json): heading axis is guessed. Run py -m gabai.tools.frame_calibrate")
    for n in frames.notes:
        print("  note:", n)
    fusion = SensorFusion(mag_cfg=mag_cfg, beta=a.beta, auto_bias=not a.no_auto_bias, accel_cal=accel_cal, frames=frames if known else None)
    mount_shown = False
    if a.forward != "auto":
        fusion.heading.forward = np.eye(3)["xyz".index(a.forward)]                 # an explicit sensor axis overrides the device's forward axis
    t0, t_drawn, last_raw = None, -1.0, None
    try:
        while True:
            imu, mag = rx.drain_imu(), rx.drain_mag()
            fusion.add_mag(mag)
            if imu:
                t0 = imu[0][0] if t0 is None else t0
                last_raw = imu[-1]
                fusion.add_imu(imu)
                fusion.step()
            t = fusion.track.t[-1] if fusion.track.t else None
            if t is not None and t - t_drawn >= 1 / 30:                       # about 30 draws a second whatever the sensor rate
                t_drawn = t
                q = fusion.filt.q
                h = fusion.heading
                rr.set_time("t", duration=t - t0)
                rr.log("world/board", rr.Transform3D(rotation=rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]])))
                if h.valid:
                    rr.log("heading/wrapped_deg", rr.Scalars(h.wrapped))
                    rr.log("heading/continuous_deg", rr.Scalars(h.unwrapped))
                rr.log("heading/valid", rr.Scalars(float(h.valid)))
                if fusion.mount is not None:
                    if not mount_shown:
                        mount_shown = True
                        print(f"pose recognised: {fusion.mount[0]}; camera looks {fusion.mount[1]['pitch_down_deg']:+.1f} deg below the horizon, sideways lean {fusion.mount[1]['roll_deg']:+.1f} deg")
                    rr.log("mount/pitch_down_deg", rr.Scalars(fusion.mount[1]["pitch_down_deg"]))
                if mag_cfg is not None:
                    rr.log("compass/used_in_filter", rr.Scalars(float(fusion.used_compass)))
                    rr.log("compass/gate_ok", rr.Scalars(float(fusion.mag_ok)))
                    if fusion.mag_field_ratio is not None:
                        rr.log("compass/field_vs_calibrated", rr.Scalars(fusion.mag_field_ratio))
                    if fusion.mag_heading is not None:
                        rr.log("compass/heading_deg", rr.Scalars(fusion.mag_heading))
                rr.log("filter/accel_trust", rr.Scalars(fusion.filt.tilt_trust))             # 1 = the accelerometer reads gravity alone; 0 = shaken, ignored for tilt
                rr.log("filter/tilt_error_deg", rr.Scalars(fusion.filt.tilt_error_deg))      # how far the estimate is from the accelerometer's idea of up
                rr.log("filter/compass_weight", rr.Scalars(fusion.filt.yaw_trust))
                up = q_to_R(q).T @ np.array([0.0, 0.0, 1.0])            # which way is up, in the sensor's own axes
                rr.log("tilt/up_in_sensor_x", rr.Scalars(up[0]))
                rr.log("tilt/up_in_sensor_y", rr.Scalars(up[1]))
                rr.log("tilt/up_in_sensor_z", rr.Scalars(up[2]))
                _, gx, gy, gz, ax, ay, az = last_raw
                for k, v in zip("xyz", (gx, gy, gz)):
                    rr.log(f"gyro_dps/{k}", rr.Scalars(v))
                for k, v in zip("xyz", fusion.bias.bias if fusion.bias is not None else (0, 0, 0)):
                    rr.log(f"gyro_bias_estimate_dps/{k}", rr.Scalars(v))
                for k, v in zip("xyz", (ax, ay, az)):
                    rr.log(f"accel/{k}", rr.Scalars(v))
                rr.log("accel/magnitude", rr.Scalars(float(np.linalg.norm((ax, ay, az)))))
            for (tm, mx, my, mz) in mag:
                if t0 is None:
                    continue
                rr.set_time("t", duration=tm - t0)
                for k, v in zip("xyz", (mx, my, mz)):
                    rr.log(f"compass_raw/{k}", rr.Scalars(v))
                rr.log("compass_raw/field_uT", rr.Scalars(field_ut((mx, my, mz))))
                rr.log("world/board/compass", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[np.array([mx, my, mz]) / 4000.0], colors=[[255, 200, 0]]))
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()


if __name__ == "__main__":
    main()
