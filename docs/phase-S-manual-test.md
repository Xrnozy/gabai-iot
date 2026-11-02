# Phase S: Hardware spike, manual test guide

Purpose: decide go/no-go on the camera hardware before any more software. Nothing here has been run on your Pi yet; the sync analysis logic is unit-tested on synthetic data only (4/4 pass), the Pi scripts are only syntax-checked.

## Prerequisites
- Pi 3B on Raspberry Pi OS with SSH access, both webcams plugged in, powered from a bank/charger rated >= 2.5 A with a good cable.
- Pi and PC joined by the Ethernet cable (fixed IPs, e.g. PC 192.168.50.1, Pi 192.168.50.2) or on your usual LAN for now. Tell me the IP you use.
- On the Pi: `sudo apt install python3-opencv v4l-utils python3-gpiozero`
- Copy the folder to the Pi: from the PC, `scp -r gabai pi@<PI_IP>:~/` (PowerShell, in the project folder).
- Optional LED for test 1: GPIO17 -> 330 ohm -> LED -> GND, visible to both cameras. Any flashing light source also works (phone strobe app), then skip the pulser.

## Test 0: IMU and compass wiring check (10 min, optional for the go/no-go but cheap)
Wiring read from your diagram: BMI160 on the Pi's hardware I2C (SDA pin 3, SCL pin 5, VIN pin 1, GND pin 9, CS to 3.3V pin 17 which selects I2C, SA0 to GND pin 25 which sets address 0x68). HMC5883L on its own pins (SDA pin 37 = GPIO26, SCL pin 35 = GPIO19, GND pin 34, VCC fed from the BMI160 board's 3V3 pad). Pins 35/37 are not hardware I2C, so the compass needs a software I2C bus.
On the Pi:
```bash
sudo raspi-config nonint do_i2c 0
sudo apt install -y i2c-tools python3-smbus2
echo "dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=26,i2c_gpio_scl=19" | sudo tee -a /boot/firmware/config.txt
sudo reboot
```
(If `/boot/firmware/config.txt` does not exist, use `/boot/config.txt`.) After reboot:
```bash
i2cdetect -y 1      # expect 68
i2cdetect -y 3      # expect 1e
python3 -m gabai.spike.pi_imu_check
```
Expected: BMI160 chip id 0xD1, accel about 1.0 g at rest; HMC5883L id "H43" with non-zero field values. If the compass shows `0d` instead of `1e`, the board is actually a QMC5883L clone (different chip, different driver); tell me. If nothing appears on bus 3, recheck the overlay line and pins 35/37.
Notes: the INT pins are unwired, so I will read the IMU by polling, which is fine. Pulling the compass's power from the BMI160 board's 3V3 pad is acceptable for a small load, but if readings are flaky, move it to a Pi 3.3V pin.
Only the chip-ID and one-sample check exist; 200 Hz FIFO reading and calibration are not written and not part of Phase S.

## Test 1: camera identification and controls (5 min)
On the Pi:
```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-ctrls
```
Expected: two EMEET devices (usually /dev/video0 and /dev/video2; use the real numbers below). Report the full output. Failure: only one camera listed, or no `exposure_time_absolute` control.

## Test 2: left/right sync (15 min)
Terminal A (Pi): `python3 -m gabai.spike.pi_led_pulser --duration 100`
Terminal B (Pi): `python3 -m gabai.spike.pi_capture sync --left /dev/video0 --right /dev/video2 --exposure 300 --roi 0,0,64,64 --duration 90 --out sync15`
Tune `--roi x,y,w,h` so the LED sits inside it. Then copy `sync15/frames.csv` to the PC and run:
```
py -m gabai.spike.analyze_sync sync15/frames.csv --exposure-ms 30 --fps 15
```
Expect: `pulses seen` at least 20 per camera (otherwise move the LED/ROI and rerun). Report the offset and verdict. Repeat with `--fps 30` in capture and `--fps 30` in analysis (nearest-frame pairing test). Repeat both while someone shakes the rig by hand (motion). Left/right swapped or wrong device numbers will not change the result.
Note: the estimate saturates at the exposure time; a value near 30 ms means "at least that big".

## Test 3: Pi power, load and memory (12 min)
Terminal A: `bash gabai/spike/pi_health_log.sh health.csv`
Terminal B: `python3 -m gabai.spike.pi_capture sync --duration 600 --out load10`
Report the final `frames=... dropped_grabs=... effective_fps=...` line, and from health.csv the `throttled` column (all `0x0` is healthy; anything else means under-voltage or throttling) and the lowest `mem_available_mb`.

## Test 4: stereo pairs and calibration (30 min)
Print a checkerboard (tell me inner corner cols x rows and square size in mm). On the Pi:
`python3 -m gabai.spike.pi_capture pairs --duration 120 --interval 2 --out calib_pairs`
Hold the board at many angles/distances while it runs. Copy `calib_pairs` to the PC and run:
`py -m gabai.spike.stereo_calib calib_pairs --cols 9 --rows 6 --square-mm 25`
Expect: >= 12 usable pairs and RMS error < 0.5 px. Report the printed lines.

## Test 5: depth scenes (30 min)
Capture 20 pairs (`pairs` mode, no board) of: a wall at 1/2/3 m (tape-measure the distance), a chair, a dark bag, a thin pole, glass, a dim room, a floor step/curb. Write the true distances down. On the PC: `py -m gabai.spike.depth_compare scene_pairs --calib calib.npz --probe 320,240`. Report valid-% and depth versus the tape measure. LiteAnyStereo comparison follows after I clone it (needs your OK).

## How to report (copy this)
```
T1: pass/fail  notes:
T2 15fps: offset=__ ms  verdict=__   30fps: offset=__ ms   moving: offset=__ ms
T3: effective_fps=__ dropped=__ throttled values seen=__ min free MB=__
T4: pairs used=__ RMS=__ baseline=__ cm
T5: scene, true m, SGBM m, valid %
Attach: error text, frames.csv, health.csv, a photo of the rig/LED.
```
Stop a run with Ctrl+C. Restart by rerunning the same command.

## Failure indicators
`cannot open /dev/videoX`; fps far below 15; `throttled` not 0x0; calibration RMS > 1 px; pulses < 20.
