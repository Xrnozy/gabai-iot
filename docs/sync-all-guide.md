# Synchronising the camera, the BMI160 and the HMC5883L

## What "synchronised" means here
The three sensors cannot share a hardware clock. What the software does:
1. **One clock.** The Pi stamps every camera frame, every gyro/accelerometer sample and every compass reading on its own monotonic clock the moment it reads it, and sends the stamps along. The PC never uses its own clock to order them.
2. **Measured delays.** Each stamp is late by a different, unknown amount (camera: exposure and USB transfer; compass: internal averaging and polling). `sync_sensors` measures each sensor's delay against the gyro, which is the reference, from one recording of you tumbling the rig. No board, no light.
3. **Look-ups at the same instant.** The viewer fuses gyro + gravity + compass into one orientation history, then for each camera frame reads the orientation at (frame time - camera delay), and aligns each compass reading by its own delay before using it.
4. **Axis maps.** The same recording finds which gyro axis is which camera axis and which compass axis is which gyro axis (signed permutations only).

## Run
1. Copy the files to the Pi (PowerShell, project folder): `scp -r gabai xrnozy@192.168.1.63:~/`
2. Pi (keep the rig still for the first 1.5 s; the sender must print both `IMU found` and `compass found`):
```bash
cd ~
python3 -m gabai.pi.sender --host 192.168.1.64 --left /dev/video2 --right /dev/video0
```
3. PC, textured scene in front of the cameras (shelves, posters):
```powershell
py -m gabai.tools.sync_sensors
```
When it says "recording", tumble the rig by hand for 30 s through EVERY direction: left, right, up, down, upside down, quickly and smoothly. The compass needs all directions or its result is flagged unreliable.
4. It prints a table like:
```
sensor                  delay vs gyro   match   verdict
gyro (reference)                 0 ms           reference
camera (left image)            +74 ms    0.54   GOOD
compass                        +40 ms    0.98   GOOD
```
and saves `data/imu_sync.json` (camera), `data/mag_cal.json` and `data/mag_sync.json` (compass).
5. Run the viewer:
```powershell
py -m gabai.spike.demo_viewer --source live --model m
```
The panel line reads `SYNC: camera 74 ms, compass 40 ms behind the gyro (calibrated); orientation from gyro + gravity + COMPASS`, and the 3D camera title says `compass`. `--no-mag` turns the compass out of the orientation.

## What to check
- **Camera vs gyro:** turn the rig quickly and stop. The 3D camera should stop at the same moment as the picture.
- **Compass:** hold the rig still for 30 s. The 3D yaw should not creep. Then turn it a full circle and come back to the start: the yaw should return to the same value (without the compass it drifts).
- **Heading:** the 3D world x axis (red) is magnetic north. Point the rig north and the yaw reads about 0.
- Rerun `sync_sensors` if you move the compass or the IMU on the rig; the axis maps are fixed to how they are mounted.

## Verified and not verified
- Verified with simulated sensors (known delays and axes, a 28 s tumble through the real network path): camera delay 74 ms (true 70), compass delay 40 ms (true 40), both axis maps exactly right, compass calibration offset within 5 counts, compass match 0.98, camera match 0.54. 79 automated tests pass across the project, including the estimators, the compass-fused filter and the timed fusion. In simulation, using the compass with the right delay and axes kept orientation within 1 degree after 30 s; with a bad delay (0.25 s off) it was several times worse, and with wrong axes 26 degrees off.
- NOT verified on your real rig: all numbers above come from simulation. The camera delay has a likely error of 5 to 10 ms.
## Known limits
- Indoor magnetic fields are distorted by steel, cables and electronics. If the compass is disturbed the yaw can be pulled the wrong way; use `--no-mag` to compare.
- The compass fusion is a heading reference, not a precise one. Yaw is good to a few degrees at best.
- The axis maps are signed permutations; a sensor mounted at a tilted angle cannot be represented.
- This is software alignment. It cannot remove jitter between the camera's pair of frames (typical 4 ms, worst 22 ms, measured earlier).
