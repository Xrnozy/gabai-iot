"""Record the Pi's raw IMU and compass for a while, to judge the orientation filter on your own sensors (scripts/evaluate_orientation.py --sensors <file>).

PC:  py -m gabai.tools.record_sensors --name still --seconds 30        (hold the rig still: the noise and the resting drift)
     py -m gabai.tools.record_sensors --name small --seconds 40        (small hand movements, the case that looked unstable)
     py -m gabai.tools.record_sensors --name turn --seconds 40         (slow turns through every direction, then back to where you began)
Pi:  python3 -m gabai.pi.sender --host <PC_IP> --left /dev/video2 --right /dev/video0       (IMU and compass on)

Saves data/sensors/<name>_<date>.npz with imu (t, gx, gy, gz, ax, ay, az: deg/s, g, as sent) and mag (t, x, y, z: raw counts), both on the Pi's clock, untouched:
calibration and filtering are applied later, by the evaluation, exactly as the live system does."""
import argparse
import os
import time

import numpy as np

from gabai.pc.receiver import LiveReceiver


def gap_report(t, nominal_hz):
    """(rate in Hz, longest gap in ms, number of gaps over 2.5 nominal periods) of a time stamp list."""
    if len(t) < 3:
        return 0.0, 0.0, 0
    d = np.diff(t)
    return (len(t) - 1) / (t[-1] - t[0]), float(d.max() * 1000), int((d > 2.5 / nominal_hz).sum())


def record(rx, seconds, out, clock=time.monotonic, sleep=time.sleep, say=print):
    imu, mag = [], []
    t_end, t_say = clock() + seconds, clock()
    while clock() < t_end:
        imu += rx.drain_imu()
        mag += rx.drain_mag()
        if clock() - t_say >= 2.0:
            t_say = clock()
            say(f"  {len(imu)} IMU samples, {len(mag)} compass readings, link: {rx.stats().get('connected')}")
        sleep(0.02)
    imu += rx.drain_imu()
    mag += rx.drain_mag()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, imu=np.array(imu, float).reshape(-1, 7), mag=np.array(mag, float).reshape(-1, 4))
    return np.array(imu, float).reshape(-1, 7), np.array(mag, float).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="sensors")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--out-dir", default="data/sensors")
    a = ap.parse_args()
    rx = LiveReceiver(a.port, a.token)
    rx.start()
    print(f"listening on port {rx.port}; waiting for the Pi ...")
    t0 = time.monotonic()
    while not rx.stats().get("connected") and time.monotonic() - t0 < 120:
        time.sleep(0.2)
    if not rx.stats().get("connected"):
        rx.stop()
        raise SystemExit("no Pi connected within 2 minutes. Start the sender on the Pi (see the top of this file).")
    time.sleep(2.0)                                                                # let the first batches arrive
    out = os.path.join(a.out_dir, f"{a.name}_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    print(f"RECORDING {a.seconds:.0f} s -> {out}")
    try:
        imu, mag = record(rx, a.seconds, out)
    finally:
        rx.stop()
    ri, gi, ni = gap_report(imu[:, 0], 200.0)
    rm, gm, nm = gap_report(mag[:, 0], 50.0)
    print(f"IMU: {len(imu)} samples, {ri:.0f} Hz, longest gap {gi:.0f} ms, {ni} gaps over 12 ms.   Compass: {len(mag)} readings, {rm:.0f} Hz, longest gap {gm:.0f} ms, {nm} gaps over 50 ms")
    print(f"saved {out}\nevaluate: py scripts/evaluate_orientation.py --sensors {out}")


if __name__ == "__main__":
    main()
