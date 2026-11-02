# Gabai Lite: what this is and what it is not

*Draft for the competition write-up. The numbers section is empty on purpose until the Phase M tests are run; fill it from `report.html`.*

## What it is
A tethered proof of concept for warning a blind person about obstacles ahead.
- A Raspberry Pi 3B carries two USB webcams (5.1 cm apart), an IMU and a compass. It sends pictures and sensor readings over the network to a PC.
- The PC measures depth from the two pictures (a learned stereo model, LiteAnyStereo 2), checks the left, ahead and right zones, and plays short pre-recorded spoken warnings ("ahead, about one metre", "drop ahead", "can't see ahead", "tracking delayed").
- It says so when it cannot see, when it is turning too fast to be accurate, and when the link is late or stopped.
- A live telemetry page and a per-run log record frame rate, latency, distances, alerts and the Pi's health. The measured results come from those logs.

## What it is not
- **Not wearable yet.** The Pi is tethered to a PC; speech comes from the PC's speakers.
- **Not safe to rely on.** It is tested only in supervised sessions. Nobody should walk blindfolded or depend on it.
- **Not a mapping or navigation system.** There is no SLAM, no route planning, no object memory.

## Known weaknesses (measured results go next to each in the report)
| Case | Why it is hard |
|---|---|
| Glass, mirrors | the cameras see through or reflect; stereo matches the wrong thing |
| Thin poles and wires | thinner than the matching window; the background wins |
| Black or very dark objects | no texture to match; the picture is noisy |
| Low light | short exposure or high gain; more noise, more rejected pixels |
| Closer than 0.23 m | the depth model's maximum disparity is 192 pixels (an optional second pass reaches about 0.12 m at a cost in speed) |
| Far objects | with a 5.1 cm baseline a half-pixel error costs about 1 cm at 1 m, 4.5 cm at 2 m, 10 cm at 3 m, 28 cm at 5 m |
| Drops (stairs down, curbs) | only reported as working where a real test shows a hit rate; unseen floor becomes "can't see ahead" |
| Fast head turns | the two cameras are not triggered together (software sync only, typical 4 ms, worst about 20 ms), so depth is degraded while turning; the system says "reduced accuracy" |

## Results (fill in from the Phase M report)
| Measure | Result | Spec target |
|---|---|---|
| Capture to PC receipt, p50 / p95 | | |
| Capture to speech, p50 / p95 | | 400 ms (a target, not a result) |
| Depth error at 1 / 2 / 3 m, plain wall | | |
| Targets the system could not see | | |
| False alarms in a clear room, per 3 min | | |
| Missed obstacles, of 10 presentations | | |
| Pi under-voltage during a 10 min run | | |
