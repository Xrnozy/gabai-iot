"""Check that the compass, the IMU, the camera and the SLAM agree on one frame, in the pose you actually wear the device in.

    py -m gabai.tools.frame_check          (the Pi sender running, IMU and compass on; wear the device, stand still)

It holds still for 4 s, then prints: which pose it recognises (chest_upright, flat_front_up, ...), where gravity and the Earth's field point in the DEVICE
frame (camera x right, y down, z forward), the camera's pitch below the horizon (the number for [tracking] initial_pitch_deg, which makes the SLAM world z
truly up), what the two IMU-to-camera maps say, and whether the compass field agrees with gravity as a calibrated compass must.
Nothing is written. Frames and their files are described in gabai/frames.py.
"""
import argparse
import time

import numpy as np

from gabai.frames import classify_mount, load_frames
from gabai.mag import apply_cal, load_mag_cfg


def report(frames, accel, mag_cfg, mag_raw):
    """Lines of text about one still capture (accel (n,3) in g, mag_raw (k,3) counts or empty)."""
    out = []
    up_imu = np.mean(accel, axis=0)
    up_imu = up_imu / np.linalg.norm(up_imu)
    up_dev = frames.up_in_device(up_imu)
    name, d = classify_mount(up_dev)
    out.append(f"pose: {name}   (gravity 'up' in the device frame x right, y down, z forward: {np.round(up_dev, 3)})")
    out.append(f"camera pitch below the horizon {d['pitch_down_deg']:+.1f} deg, sideways lean {d['roll_deg']:+.1f} deg  ->  [tracking] initial_pitch_deg = {d['pitch_down_deg']:.0f}")
    if name != "chest_upright":
        out.append("  note: not the worn pose. Stand upright wearing it to get the number for the chest")
    out.append(f"IMU->device map from: {frames.sources.get('imu_to_device')}; compass->IMU map from: {frames.sources.get('compass_to_imu')}")
    out += ["note: " + n for n in frames.notes]
    if mag_cfg is not None and len(mag_raw):
        m_imu = frames.R_imu_mag @ np.mean(apply_cal(mag_cfg, mag_raw), axis=0)
        strength = float(np.linalg.norm(m_imu))
        cos = float(m_imu @ up_imu / strength)
        dip = float(np.degrees(np.arccos(np.clip(cos, -1, 1)))) - 90.0
        ut = strength * 0.092
        m_dev = frames.R_dev_imu @ m_imu
        out.append(f"compass: field {ut:.0f} uT after calibration, dip {dip:+.0f} deg, in the device frame {np.round(m_dev / strength, 3)}")
        if not 20.0 <= ut <= 70.0:
            out.append("  PROBLEM: the field is not Earth-sized: redo frame_calibrate or sync_sensors, or move the compass away from magnets/current")
        if abs(dip) < 5 or abs(dip) > 85:
            out.append("  PROBLEM: dip near 0 or 90 deg is implausible for most places: the compass-to-IMU map or the offset is wrong")
        horizontal = m_dev - (m_dev @ up_dev) * up_dev
        fwd = np.array([0.0, 0.0, 1.0])
        fh = fwd - (fwd @ up_dev) * up_dev
        az = float(np.degrees(np.arctan2(np.cross(horizontal, fh) @ up_dev, horizontal @ fh)))
        out.append(f"the device's forward axis points {abs(az):.0f} deg {'west' if az > 0 else 'east'} of magnetic north (heading {az % 360:.0f} counted towards west, 0 = north)")
    else:
        out.append("compass: not calibrated or no readings: run py -m gabai.tools.frame_calibrate")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--seconds", type=float, default=4.0)
    a = ap.parse_args()
    from gabai.pc.receiver import LiveReceiver
    rx = LiveReceiver(a.port, a.token)
    rx.start()
    frames, mag_cfg = load_frames(), load_mag_cfg()
    print("hold still ...")
    imu, mag = [], []
    rx.drain_imu(), rx.drain_mag()
    end = time.time() + a.seconds
    while time.time() < end:
        imu += rx.drain_imu()
        mag += rx.drain_mag()
        time.sleep(0.02)
    rx.stop()
    if len(imu) < 20:
        raise SystemExit("no IMU readings: does the Pi sender print 'IMU found'?")
    A = np.array(imu, float)
    if np.linalg.norm(A[:, 4:7].std(axis=0)) > 0.03 or np.linalg.norm(A[:, 1:4].mean(axis=0)) > 3.0:
        print("WARNING: it moved during the capture; the numbers below are less reliable")
    for line in report(frames, A[:, 4:7], mag_cfg, np.array(mag, float)[:, 1:4] if mag else np.zeros((0, 3))):
        print(line)


if __name__ == "__main__":
    main()
