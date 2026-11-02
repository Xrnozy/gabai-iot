# Camera sync: what is done, how to measure it

## What can and cannot be done
The two EMEET webcams are separate USB devices with their own clocks. They cannot be hardware-synchronised. What the software does now:
1. Each camera runs in its own thread at 30 FPS and every frame is timestamped the moment it arrives on the Pi.
2. Each left frame is paired with the right frame nearest in time (before: left and right were grabbed one after the other, which left the right frame up to one frame period, 66 ms at 15 FPS, out of step).
3. Expected result: worst-case offset about 17 ms (half of 33 ms), typically under 10 ms. That is a design estimate, not a measurement on your cameras.
Exposure length also shifts the moment actually captured. For the best sync use the SAME fixed exposure on both cameras. Auto exposure can give them different exposure times.

## What the viewer shows
The LINK line now includes `L/R pairing offset median X ms (worst Y)`. This is the difference between the two cameras' frame-arrival timestamps for the pairs being used. It is a software measure (arrival times), not the true exposure offset; use the LED test below for that.

## Copy the new files to the Pi
```powershell
cd "C:\Users\MY PC\Documents\Hackathon testing\Gabai IOT final"
scp -r gabai xrnozy@192.168.1.63:~/
```

## Measure the real offset with a flashing light (the proper test)
You need a light that flashes: a phone strobe/flashlight app at 3 to 5 flashes per second, or the LED on GPIO17 with `python3 -m gabai.spike.pi_led_pulser --duration 100` (GPIO17 to 330 ohm to LED to GND). Both cameras must see the light inside the ROI box; adjust `--roi x,y,w,h` (default top-left 64x64 of the image).

Pi, in your home folder (the cameras are swapped, left is video2):
```bash
cd ~
python3 -m gabai.spike.pi_capture sync --left /dev/video2 --right /dev/video0 --capture-fps 30 --fps 15 --exposure 300 --roi 0,0,64,64 --duration 90 --out sync30
```
It prints a summary line: pairs recorded, camera FPS (both should be close to 30), and the pairing offset. Then copy the file and analyse on the PC:
```powershell
scp xrnozy@192.168.1.63:~/sync30/frames.csv sync30_frames.csv
py -m gabai.spike.analyze_sync sync30_frames.csv --exposure-ms 30 --fps 15
```
Reading it: the result is the offset between the two exposure windows, and it tops out at the exposure length (30 ms here), so "near 30 ms" means "at least that big". Need at least 20 pulses seen per camera. Compare against the old method by also running with `--capture-fps 15` (pairing is then coarser).

## Pass criteria (from the spec and the plan)
- Spec: 10 ms or less. Realistic outcome: "usable going straight, degraded when turning fast". Report the measured number and I will state the verdict honestly.
- Both camera FPS must be close to 30. If one camera drops far below that, the USB bus or power is the limit.

## Reading the result (updated)
`analyze_sync` now prints one of three things:
- `EDGE METHOD: n light edges seen by both cameras` followed by the median right-minus-left offset and a verdict. This is the main result. It uses how lit the frame is when the light switches on or off during an exposure, so it works with a phone strobe at 2 to 4 flashes per second and gives sub-frame precision (checked on synthetic data to within about 1 ms with 40 or more edges).
- `NO VALID RESULT: ... light was not seen flashing`. The light was not in view of both cameras, or was too dim, or was steady. Aim the strobe straight into both lenses, 30 to 50 cm away, and rerun.
- The older pulse-counting result only appears for very short flashes (such as the GPIO LED), and only from isolated flashes. A bright run of many frames in a row (a hand, a lamp) is ignored.
The first real recording (`sync30b`) was a false positive from a slow brightness change and printed 3.9 ms. That was wrong; the analysis now rejects it.
