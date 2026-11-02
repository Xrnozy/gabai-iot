"""One-time frame calibration: how the IMU and the compass axes sit in the device, and the compass hard-iron offset. Saved once, loaded on every later run.

    PC:  py -m gabai.tools.frame_calibrate           (the Pi sender must be running, with IMU and compass on)
         py -m gabai.tools.frame_calibrate --from-npz data/frame_calibration_raw.npz      (solve again from a saved capture, no hardware)

Four captures, each started by pressing Enter:
  1. lay the device on a table on its BACK, the front (camera side) facing UP, and keep it still for 4 s
  2. keep it flat on the table and turn it slowly, one full turn about the vertical, in 20 s
  3. stand it UPRIGHT the way it hangs on your chest (against a wall, or hold it against your chest), left-right level, and keep it still for 4 s
  4. in that pose turn on the spot slowly, one full turn, in 20 s (wearing it, or turning it by hand about the vertical)
Keep it away from phones, laptops and anything magnetic during the turns, and at the same distance from the Pi and battery as when worn (they are part of
what is being calibrated).

Writes data/device_frame.json (IMU -> device axes from gravity), data/mag_cal.json (hard-iron offset) and data/mag_sync.json (compass -> IMU axes; the compass
delay is kept if the file existed, else 0). The previous files are copied to *.bak first. gabai.tools.frame_check then verifies the result in the pose you wear it in.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

from gabai.frames import classify_mount
from gabai.framecal import solve, still_ok


def capture(rx, seconds, label):
    """Collect IMU and compass samples for `seconds`. Returns dict(gyro (n,3) deg/s, accel (n,3) g, mag (k,3) raw counts)."""
    imu, mag = [], []
    rx.drain_imu(), rx.drain_mag()
    end = time.time() + seconds
    last = 0
    while time.time() < end:
        imu += rx.drain_imu()
        mag += rx.drain_mag()
        left = int(end - time.time())
        if left != last:
            print(f"\r  {label}: {left:2d} s left, {len(imu)} IMU and {len(mag)} compass readings   ", end="", flush=True)
            last = left
        time.sleep(0.02)
    print()
    a = np.array(imu, float)
    m = np.array(mag, float)
    if len(a) < seconds * 20 or len(m) < seconds * 10:
        raise SystemExit(f"too little data ({len(a)} IMU, {len(m)} compass readings): does the Pi sender print 'IMU found' and 'compass found'?")
    return {"gyro": a[:, 1:4], "accel": a[:, 4:7], "mag": m[:, 1:4]}


def live(a):
    from gabai.pc.receiver import LiveReceiver
    rx = LiveReceiver(a.port, a.token)
    rx.start()
    print(f"listening on port {rx.port}. Start the sender on the Pi (IMU and compass on).")
    steps = [("still_front_up", 4, "1/4  Lay the device on its BACK, front (camera side) UP. Press Enter, then keep it STILL for 4 s."),
             ("spin_front_up", a.turn_seconds, f"2/4  Keep it flat. Press Enter, then turn it slowly ONE full turn in {a.turn_seconds:.0f} s."),
             ("still_upright", 4, "3/4  Stand it UPRIGHT as worn on the chest, left-right level. Press Enter, then keep it STILL for 4 s."),
             ("spin_upright", a.turn_seconds, f"4/4  Same pose. Press Enter, then turn on the spot ONE full turn in {a.turn_seconds:.0f} s.")]
    seg = {}
    for name, secs, text in steps:
        while True:
            input("\n" + text + "\n  [Enter] ")
            seg[name] = capture(rx, secs, name)
            if name.startswith("still") and not still_ok(seg[name]["gyro"], seg[name]["accel"]):
                print("  it moved during the still capture: again")
                continue
            break
    rx.stop()
    return seg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--turn-seconds", type=float, default=20)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--from-npz", help="solve from a capture saved earlier instead of using the hardware")
    ap.add_argument("--raw-out", default="data/frame_calibration_raw.npz", help="where the captured readings are kept (so the solve can be repeated)")
    ap.add_argument("--no-save", action="store_true", help="show the result, write nothing")
    a = ap.parse_args()
    if a.from_npz:
        z = np.load(a.from_npz)
        seg = {n: {k: z[f"{n}__{k}"] for k in ("gyro", "accel", "mag") if f"{n}__{k}" in z.files} for n in ("still_front_up", "spin_front_up", "still_upright", "spin_upright")}
    else:
        seg = live(a)
        os.makedirs(os.path.dirname(a.raw_out) or ".", exist_ok=True)
        np.savez_compressed(a.raw_out, **{f"{n}__{k}": v for n, d in seg.items() for k, v in d.items()})
        print(f"raw readings kept in {a.raw_out}")
    r = solve(seg["still_front_up"], seg["still_upright"], seg["spin_front_up"], seg["spin_upright"])
    print("\n=== result ===")
    print("device <- IMU axes (v_device = R @ v_imu; device x right, y down, z forward):\n", np.round(r["R_dev_imu"], 3))
    print("IMU <- compass axes (v_imu = R @ v_compass):\n", r["R_imu_mag"].astype(int))
    print(f"  axis-map score {r['map_score']:.3f} (next best {r['map_score_next']:.3f}); turns covered " + ", ".join(f"{k} {v:.0f} deg" for k, v in r["swept_deg"].items()))
    print(f"compass hard-iron offset {np.round(r['offset'], 1)} counts; field after removing it {r['field_uT']:.0f} uT (Earth: 25-65); sphere quality {r['quality']:.3f}; "
          f"dip {r['dip_deg']:.0f} deg (field points {'down' if r['dip_deg'] > 0 else 'up'})")
    up_worn = np.mean(seg["still_upright"]["accel"], axis=0)
    name, det = classify_mount(r["R_dev_imu"] @ (up_worn / np.linalg.norm(up_worn)))
    print(f"the upright pose reads as: {name}, camera looks {det['pitch_down_deg']:+.1f} deg below the horizon (put this in [tracking] initial_pitch_deg)")
    if r["problems"]:
        print("\nPROBLEMS (nothing saved):" if not a.no_save else "\nPROBLEMS:")
        for p in r["problems"]:
            print("  -", p)
        if not a.no_save:
            raise SystemExit(1)
        return
    if a.no_save:
        return
    d = a.data_dir
    for f in ("device_frame.json", "mag_cal.json", "mag_sync.json"):
        if os.path.exists(os.path.join(d, f)):
            shutil.copy2(os.path.join(d, f), os.path.join(d, f + ".bak"))
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump({"R_dev_imu": r["R_dev_imu"].tolist(), "date": stamp, "source": "frame_calibrate: gravity with the front up, and upright"}, open(os.path.join(d, "device_frame.json"), "w"), indent=2)
    json.dump({"offset": r["offset"].tolist(), "scale": [1.0, 1.0, 1.0], "W": np.eye(3).tolist(), "radius_counts": r["radius_counts"], "quality": r["quality"],
               "coverage": 0.5, "method": "two_rings_sphere", "date": stamp,
               "note": "hard iron only. Soft iron needs py -m gabai.tools.sync_sensors (a tumble through every direction)"}, open(os.path.join(d, "mag_cal.json"), "w"), indent=2)
    delay = 0.0
    old = os.path.join(d, "mag_sync.json.bak")
    if os.path.exists(old):
        delay = float(json.load(open(old)).get("delay_s", 0.0))
    json.dump({"delay_s": delay, "R_im": r["R_imu_mag"].tolist(), "score": r["map_score"], "date": stamp, "source": "frame_calibrate"}, open(os.path.join(d, "mag_sync.json"), "w"), indent=2)
    print(f"\nsaved {d}/device_frame.json, mag_cal.json, mag_sync.json. They are loaded automatically by imu_3d, demo_viewer and the fusion from now on.")
    print("next: wear it, and run  py -m gabai.tools.frame_check")


if __name__ == "__main__":
    main()
