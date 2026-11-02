# Running the Pi on a power bank

Your Phase M logs show `throttled = 0x50005` on every frame: the Pi reports **under-voltage right now** (bit 0x1) and **throttling right now** (bit 0x4), and it has happened since boot (bits 0x10000 and 0x40000). You chose to keep the power bank, so this guide lowers the Pi's current draw in software. **I could not measure the effect: the Pi 3B has no current sensor and I have no access to your Pi.** The test below tells you whether it worked.

## What draws power, and what I changed
| Source | Before | Now |
|---|---|---|
| Sender loop waiting for the next pair | woke up 1000 times a second, all the time | sleeps until the next pair is due (always on, no flag) |
| `vcgencmd` health reads | 2 new processes every second, spawned from a big Python process | every 5 s with `--low-power` (the 1 s health message itself is unchanged) |
| IMU polling (I2C) | 200 Hz | 100 Hz with `--low-power` |
| Compass polling | 50 Hz | 25 Hz with `--low-power` |
| Camera capture | 30 FPS on both cameras, half of the frames thrown away | 20 FPS with `--low-power` (fewer USB transfers and wake-ups) |
| HDMI, Bluetooth, activity LED, CPU clock | on, up to 1200 MHz | `scripts/pi_power_save.sh`: off, off, off, capped at 800 MHz |
The depth model runs on the PC, so it costs the Pi nothing.

## Cost: pairing gets coarser
At 20 FPS capture the left and right pictures can be up to about 25 ms apart (half a frame period), against about 17 ms at 30 FPS. Phase S measured a median of 3.8 ms at 30 FPS. The viewer's `L/R pairing offset` line shows the real value; the depth is a little worse in fast turns, and the system's "reduced accuracy" warning already covers fast turns. IMU and compass at lower rates: the sync tools (`sync_sensors`) were verified at 200 and 50 Hz, so rerun `sync_sensors` once if you rely on the camera, gyro or compass delays.

## How to use it
On the Pi, after a **reboot** (the "since boot" flag only clears on a reboot, so you can see the effect):
```bash
sh scripts/pi_power_save.sh
python3 -m gabai.pi.sender --host 192.168.1.64 --left /dev/video2 --right /dev/video0 --low-power
```
For a different cap use `sh scripts/pi_power_save.sh 600`. Put the same `--gain`, `--brightness` and `--exposure` options you use normally on the sender line.

## How to find out if it worked (about 10 minutes)
1. Reboot the Pi on the power bank. Wait 1 minute, then run `vcgencmd get_throttled`. If it already shows `0x1`, the bank cannot even idle the Pi: no software helps; use another cable or another port of the bank.
2. Run the sender **without** `--low-power` and without the script for 3 minutes, with the viewer logging (`--log data/runs/P1_normal.csv`). Reboot.
3. Run the script, then the sender **with** `--low-power` for 3 minutes (`--log data/runs/P2_lowpower.csv`).
4. Compare the two logs:
```powershell
py -m gabai.tools.make_report --run data/runs/P1_normal.csv --run data/runs/P2_lowpower.csv --title "Pi power: normal vs low-power"
```
Look at "Pi under-voltage now (share of frames)", "Pi under-voltage since boot", the number of gaps over 0.5 s, and the L/R pairing offset. An inline USB power meter between bank and Pi (a few pounds) shows the real current if you have one.

## Honest limits
- Under-voltage comes from the power bank and cable not holding 5 V when the Pi's current peaks (two cameras on USB, Wi-Fi bursts, CPU). Software lowers the average and the peaks; it **cannot** raise the voltage. If the bank sags at about 0.7 A it will still trip.
- Two USB cameras are the biggest fixed load and cannot be switched off.
- The free hardware fixes (a shorter or thicker cable, a different port of the bank, a bank with a 2.4 A or higher output) are not software, but they cost nothing to try if the test above still shows `0x1`.
- Until the flag stays clear, the M6 stability result is the result of a Pi that was under-voltage the whole time. The report states that.
