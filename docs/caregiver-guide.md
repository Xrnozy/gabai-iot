# Caregiver and family app

Three devices talk to one relay over Tailscale. Nothing here changes the existing navigation server or the Pi-to-PC link in `live-link-guide.md`.

```
Pi  --TCP 5600 (existing)-->  relay (PC, port 8090)  <--HTTP--  phone (GPS, falls, SOS, telemetry)
                                      ^
                                      +--HTTP--  caregiver phone (same app, "Family & safety" > dashboard)
```

## Verified vs not verified
- Verified: the relay (`py -m pytest tests/test_relay.py`, 21 tests), the fall detector against synthetic signals, and that the Android app compiles and its unit tests pass.
- NOT verified (needs hardware or a real phone): GPS on a moving phone, real falls, ARCore tracking and depth on a device, the live MJPEG view with a real Pi, the focus ring with a finger, TalkBack coexistence, battery drain, Tailscale from a phone. The fall thresholds are starting values and need tuning on real data.

## 1. Run the relay (PC)
```powershell
py -m gabai.relay --host <PC_TAILSCALE_IP> --port 8090
```
First start writes `outputs/relay_tokens.json` with three random tokens: `phone`, `caregiver`, `pi`. It owns the Pi receiver on TCP 5600, so stop the navigation viewer/receiver first, or pass `--no-pi-bridge` if something else already holds that port (then the cameras are not relayed).

Add the PC's Tailscale IP to `mobile/app/src/main/res/xml/network_security_config.xml` if it is not there (plain HTTP is only allowed to listed hosts; Tailscale itself encrypts the traffic).

## 2. The blind user's phone
Home > **Family & safety**: enter the relay address and the **phone** token, save, then **Start sharing**. Location goes out every 5 s while moving, 10 s when still, and every 2 s while a caregiver has the dashboard open. Unsent data is queued on disk and replayed in order; the relay ignores duplicates.

- SOS: the red card on Home, or the button in Family & safety. 5 second countdown, cancellable.
- Fall: impact then stillness. 20 second countdown with a big "I'm okay" button, spoken warnings, and a notification action. If the phone dies mid-countdown the relay escalates a stale "pending" alert after 60 s.

## 3. A caregiver's phone
Same app > Family & safety > enter the relay address, the **caregiver** token and your name > **Open caregiver dashboard**. Tabs: Overview, Map, Camera, Safety, Device. "Alerts when the app is closed" keeps an 8 s watch for emergencies and lost connections.

Everything shows its age. Data older than its limit (phone 25 s, Pi 6 s, location 30 s) is labelled "last known", dimmed, and never shown as live. The camera tab plays the Pi's stream; when frames stop it shows the last frame with its real age.

## 4. When the Pi joins Tailscale
Point the Pi sender at the PC's Tailscale IP (`--host <PC_TAILSCALE_IP>`, as in `live-link-guide.md` step 4). No relay or app change is needed.

## 5. ARCore fallback
During navigation, if the nav link has been down for 8 s, or the relay reports a Pi that was online has dropped, the app hands the camera to ARCore. It only warns about close obstacles (needs a phone with ARCore Depth API) and says plainly that it cannot give directions. The caregiver sees mode "ARCore fallback". When the server is reachable again it returns to Pi navigation by itself. If ARCore is missing or unsupported it tells the user to stop and wait.

## 6. Accessible navigation
Home or caregiver header > **Accessible navigation**. Tap to focus and hear an element, double-tap the ringed element to activate, swipe right/left to move through elements, drag vertically to scroll, **Exit** button at top right. **Accessible tabs** limits swiping to the tab bar. It stays off while TalkBack's explore-by-touch is running, so the two never fight.

## Known limits
- Tokens are shared secrets typed into the app; there are no per-person accounts.
- Pi GPU usage and battery level are not reported by the Pi sender, and the dashboard says "Not reported" instead of inventing them.
- The ARCore depth zones assume an upright phone with the rear camera forward.
- The relay stores history in SQLite without pruning; clear `outputs/relay.sqlite3` to reset.
