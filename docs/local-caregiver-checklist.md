# Local caregiver system: implementation checklist

Status: PLAN ONLY. Nothing below is started. Tick a box only after the item is tested (rule 10).
`[ ]` pending, `[x]` done and tested.

## A. What exists today (verified by reading the code)

| Area | Today | Files |
|---|---|---|
| Relay server | FastAPI + SQLite on port 8090. Phone posts GPS/telemetry/events; caregiver polls `/api/state` every 2 s. Person tokens (hashed), shared legacy tokens, invite/respond/revoke links by typed handle. | `gabai/relay/app.py`, `core.py` |
| Video to caregiver | MJPEG from the **Pi stereo cameras** (`FrameStore`, `/stream/{side}.mjpg`). Nothing from the phone camera. | `gabai/relay/frames.py`, `bridge.py` |
| Phone nav video | `WebRTCClient` POSTs an offer to `http://192.168.100.77:8080/offer` and gets nav commands over a data channel. **No server for `/offer` exists in this repo.** It also uses Google STUN. | `WebRTCClient.kt`, `NavigationModule.kt` |
| Network config | Defaults are a Tailscale IP and one LAN IP. Android cleartext HTTP is allowed only for 6 hard-coded IPs. | `SettingsStore.kt:42-43`, `network_security_config.xml` |
| Pairing | Typed handle + invite + Allow, plus a QR screen and tokens tutorial. | `ConnectActivity.kt`, `FamilyLinks.kt`, `safety/PairQr.kt`, `TokensTutorialActivity.kt`, `PairQrCodeTest.kt`, zxing dependency |
| Caregiver UI | 5 tabs: Overview, Map (osmdroid 2D), Camera, Safety, Device, plus an Options dialog. No Color or Read Text tab found; confirm on device. | `caregiver/*.kt` |
| GPS | `TrackingService` foreground service: GPS + fall detection together, 5 s moving / 10 s still. **My last change made it start on its own once paired. That conflicts with item 6 and must be reverted (P3).** | `safety/TrackingService.kt`, `GabaiApp.kt`, `MainActivity.kt` |
| SLAM / perception | Runs on the PC (`gabai/`, `rtabmap_lab/`, `mapper/`). No channel carries detections, nav output or SLAM to the relay. | `gabai/pipeline`, `gabai/slam`, `mapper/` |

## B. Decisions (answered)

- [x] **B1. Server runs on the PC.** The laptop is only the Wi-Fi access point. Devices: laptop (AP), PC (server), patient phone, caregiver phone, Raspberry Pi (sender).
- [x] **B2. Camera is the Raspberry Pi stereo rig.** The Pi sends to the PC over TCP :5600 (already works). The phone camera is not used for video.
- [x] **B3. The `/offer` WebRTC server is not in the repo or its git history.** It is not needed for B2, so the phone's WebRTC path is retired. The phone gets navigation output from the relay instead (P4). *Assumed, please veto if wrong.*
- [x] **B4. Demo location unknown.** I will build the map pipeline for any bounding box and test it on a small area first; you give me the area before P7. 3D data only needs to be downloaded once, with internet, before the demo.
- [x] **B5. LOD1 (extruded footprint) buildings accepted.**
- [x] **B6. Plain HTTP on the private hotspot accepted.**
- [x] **B7. Fall detection stays on when not navigating** (no GPS needed); only the last known location is sent.

Consequences of these answers:
- The Pi 3B has 2.4 GHz Wi-Fi only. The laptop hotspot **must be set to 2.4 GHz** or the Pi cannot join. Add to P0.
- The PC pipeline does not post anything to the relay today (nothing calls `/api/pi/nav`). P4 adds that publisher: nav result, hazards, detections, pose, all stamped with the Pi frame id.
- Basalt/stereo SLAM output exists on the PC, so P8's SLAM layer is feasible without a new sensor.
- The phone's `NavigationModule` now has to take its spoken/haptic guidance from the relay feed instead of WebRTC. This is a real change to existing behavior and needs its own test.

## C. Phases (in order) and dependencies

```
P0 spike -> P1 network -> P2 auth -> P3 GPS -> P4 video -> P5 caregiver UI
                                   \-> P6 nav UI/SOS (independent of P4)
P1 + P3 -> P7 3D map -> P8 camera projection + SLAM layer (needs P4, P7)
P9 integrated test on the hotspot (needs all)
```

### P0. Environment spike (do first, tonight; no app code)
- [ ] Windows Mobile Hotspot works while the laptop's internet comes from the second phone (if not: USB tethering or Ethernet as the source)
- [ ] Record the hotspot gateway IP (Windows usually 192.168.137.1)
- [ ] Hotspot band set to 2.4 GHz; the Pi joins it and reaches the PC on :5600
- [ ] The PC (server) joins the same hotspot and gets a stable IP (reserve or note it)
- [ ] Phone-to-laptop and phone-to-phone reachability over HTTP (no client isolation)
- [ ] Windows Firewall rule for ports 8090 (relay), 8080 (nav server), 5600 (Pi) on the hotspot profile
- [ ] mDNS (multicast) passes between phones and laptop
- [ ] Everything still works with the laptop's internet unplugged

### P1. Local network and discovery
- [~] Remove Tailscale and fixed-IP defaults; no VPN anywhere (defaults blank, visible text fixed; written, NOT compiled. Left: the "Tailscale" telemetry row in `Tabs.kt`/`PhoneTelemetry.kt`, removed in P5)
- [~] Allow cleartext HTTP to private LAN ranges (done as base-config; NOT compiled) (`network_security_config` cannot match CIDR, so use a base-config rule for this LAN-only app and document it)
- [~] Remove the Google STUN server (done; NOT compiled) from `WebRTCClient`; use host candidates only
- [x] ~~mDNS~~ replaced by a UDP query/reply (`gabai/relay/discovery.py`, port 8091): stdlib only, no multicast. 2 server tests pass
- [~] Phone side `RelayDiscovery.kt` (broadcast + DHCP broadcast + gateway) runs at app start; written, NOT compiled, NOT tested on a real hotspot
- [~] Fallbacks: the gateway IP is already one of the query targets; manual entry exists in Family > Advanced
- [x] Server binds `0.0.0.0` and prints its LAN addresses (`py -m gabai.relay`)
- [ ] Apps start, show a clear state, and never block when there is no internet
- [ ] No internet calls anywhere at runtime (map tiles, STUN, fonts)

### P2. Authentication and pairing (QR removed)
- [~] Delete QR (files, test, zxing dependency removed; NOT compiled): `PairQr.kt`, `PairQrCodeTest.kt`, zxing dependency, QR parts of `ConnectActivity` and `TokensTutorialActivity`
- [x] First-claim: the first `user` registration becomes primary; later ones get 409. Simplified from the planned one-time code: any device on the hotspot can claim first, so start the relay and claim your phone straight away (tested)
- [x] New device sends an access request and gets a request id plus a secret (tested)
- [x] 4-digit code returned to the requester and listed to the primary (server tested); screens written, NOT compiled
- [~] Primary polls every 2 s while Connect is open, speaks the request and shows Allow/Decline (NOT compiled). Not done: voice approve/reject, and a notification when Connect is closed
- [x] Per-device token (hashed at rest, handed over once), reject/timeout deny, 3 open requests per IP and 5 overall (tested)
- [~] Uses the existing Linked people list with Stop sharing (NOT compiled)
- [x] Tested: state, history, snapshot and stream return 401 without a token. Known gap: the shared tokens in `relay_tokens.json` still work; they stay on the PC and should be removed in P9
- [x] Several caregivers, each with a separate token (tested)
- [x] Tests: unauthorized device gets 401/403 on every endpoint, replayed or expired requests fail, revoked token stops at once

### P3. GPS tied to Navigation
- [x] Always-on default kept on purpose: the service now runs for fall detection (B7); only GPS is gated, so nothing to revert
- [~] One `TrackingService`, GPS gated by `LiveState.navigating` instead of a split service (fewer moving parts); fall detection always on. NOT compiled
- [~] `NavigationActivity` sets it on in `onResume`, off in `onPause`; no switch for the patient or the caregiver. NOT compiled, not tried on a phone
- [~] Fixed 5 s GPS request and 5 s upload (adaptive 10 s and the 2 s "watched" rate removed). NOT compiled
- [x] Server keeps the last fix and marks it stale with its age; never removed on failure (tested, incl. GPS off after Navigation)
- [~] GPS state sent to the server: off / searching / ok / weak / denied / disabled. NOT shown to caregivers yet: the Outdated / GPS off / Poor accuracy labels are P5
- [~] GPS resumes by itself when the provider is switched back on; uploads already queue in order (`Uplink`). Not tested on a phone
- [x] Caregiver cannot control GPS: no endpoint exists for it
- [ ] Known gap: if the OS kills the service mid-navigation it restarts with GPS off until Navigation is reopened

### P4. Server and live video
- [~] Pi -> PC over TCP :5600 is the existing path; not re-tested on the hotspot (needs P0 and the Pi pointed at the PC's hotspot IP)
- [x] Authenticated MJPEG to caregivers: per-device token required (tested in P2)
- [~] PC pipeline publishes a result per frame: `demo_viewer --source live --relay-port 8090` runs the relay inside the pipeline (one owner of port 5600) and calls `ingest_nav(build_result(...))`. Result builder and endpoint tested; the viewer wiring parses but has NOT been run with a Pi
- [~] Phone reads guidance from `GET /api/guidance` (`RelayGuidance.kt`, polls 400 ms, skips stale or repeated frames). NOT compiled. WebRTC is still started beside it: remove it only after this is verified on a phone
- [~] Result fields: frame_id, hazards (zone, distance), speech, plain summary. `detections` is an empty list: the object detector is not wired into this path
- [x] Result carries `frame_id` = the Pi frame sequence the relay stores the picture under (tested)
- [~] Caregiver Camera tab shows the plain line ("ahead, about two metres", marked old when stale). NOT done: boxes drawn over the video, and the stream does not yet tag each frame with its id
- [x] Stale or lost video keeps the last frame and is labelled (existing behaviour)
- [ ] Target under 500 ms glass-to-glass: not measured. The bridge polls frames every 100 ms, so expect 100 ms or more of that budget before the network

### P5. Caregiver UI and centralized telemetry
- [~] Tabs cut from 5 to 4: Now, Map, Alerts, Details (Camera tab merged into Now; Device renamed Details). Not the 3 first planned: Alerts stays separate. NOT compiled
- [x] Status strip (patient, video, GPS, battery, emergency) is computed on the server and tested; the app only draws it. GPS states: Live / weak signal / Poor accuracy / Outdated (over 15 s) / GPS off, last known / No location yet. Strip drawing NOT compiled
- [~] Event list: Alerts tab lists falls, SOS, off-route, not-moving, each with time and state; Acknowledge exists for the active emergency. No navigation-warning or connection-loss events in the list yet
- [x] SOS/fall alert is a full-screen, max-priority alarm notification (existing `CaregiverAlertService`); not re-tested
- [x] Color and Read Text are not referenced anywhere in the caregiver package (grep). Not checked: whether a caregiver phone can reach them through Settings or voice commands
- [~] "Reset app (for testing)" now appears only in debuggable builds. NOT compiled
- [~] Raw metrics (FPS, temperature, queue) live only on the Details tab; the Tailscale and Internet rows are gone (this system has no internet). Token fields are still in Family > Advanced
- [ ] Palette, spacing, touch-target and TalkBack review: not done, needs a device look. Map tab still loads online map tiles (fixed in P7)

### P6. Navigation UI and SOS (patient)
- [~] Navigation layout trimmed: dropped the decorative handle bar and "voice guidance" label; it already had direction arrow, hazard banner, connection pill and one status line. NOT compiled, not seen on a device
- [~] SOS button on the Navigation screen, bottom-right corner, 72 dp, away from the mic: 1.5 s hold sends; a short tap only says how; a TalkBack double-tap asks "Send SOS?". NOT compiled, not tried on a phone
- [~] Hazard banner shows for 3 s per hazard and the direction text stays; not reviewed for contrast or text size on a device
- [x] SOS keeps the existing cancelable countdown screen (`Emergency.startSos`)
- [ ] Not done: Home screen SOS is unchanged (single tap starts the countdown)

### P7. Real 3D world map (separate from SLAM)
- [ ] Choose stack (section E) after B4/B5
- [ ] Pre-download terrain and building data for the demo area; serve it from the server (offline)
- [ ] Map shown in a WebView on the caregiver phone, loaded from the server
- [ ] Orbit, zoom, tilt; patient marker with last-update time; outdated/poor-GPS styling
- [ ] Legend that distinguishes real 3D data, 2D-only areas, and missing data
- [ ] Performance on a mid-range phone (target 30 fps), memory check

### P8. Camera projection and SLAM layer
- [ ] Camera pose in the 3D map: GPS position + phone orientation (rotation-vector sensor) + calibrated FOV
- [ ] Draw the field-of-view frustum and a video image plane at the far plane (feasible)
- [ ] True projective texturing onto buildings: stretch goal, not promised
- [ ] SLAM layer in a separate local-frame view (point cloud/trajectory from the server), shown as its own toggle
- [ ] Camera movement/orientation readout; timestamps match the video

### P9. Integrated test (no internet)
- [ ] Two phones + PC + laptop AP, internet unplugged, full scenario run
- [ ] Pairing request, approve, reject, revoke
- [ ] Navigation: GPS starts/stops, 5 s updates, dropout and resume
- [ ] Fall and SOS reach the caregiver
- [ ] Unauthorized device is refused everywhere
- [ ] Update this file with results

## D. Risks and limits
1. **Hotspot behavior**: Windows may not share a Wi-Fi connection through the same adapter; mDNS may be blocked. Mitigated by P0 and the fixed-IP fallback.
2. **Firewall profile**: the hotspot may be classed Public, blocking inbound traffic.
3. **Cleartext HTTP**: no TLS, so video can be sniffed by anyone on the hotspot. Per-device tokens stop access through the app, not sniffing.
4. **Missing `/offer` server** (B3): the biggest unknown for P4.
5. **Phone camera vs stereo rig** (B2): changes what SLAM and depth can show.
6. **Battery/thermal** on the patient phone from camera upload plus navigation.
7. **3D buildings** are LOD1 (extruded footprints with heights) for most places; see E.
8. My earlier always-on GPS change must be undone; fall detection stays.

## E. 3D map options (researched)

| Option | Terrain | Buildings | Offline | Cost | Notes |
|---|---|---|---|---|---|
| **CesiumJS (Apache-2.0) in a WebView, self-hosted data** | quantized-mesh built from open DEM (Copernicus GLO-30 / SRTM) with Cesium Terrain Builder | OSM footprints + height tags converted to 3D Tiles (LOD1), or an open LOD2 city dataset where one exists | Yes, served by the server | Free, but you build the data | Real 3D globe, free camera. Cesium's own World Terrain and OSM Buildings are ion-hosted and not free to self-host. A forum report says CTB output can come out flat and needs tuning. |
| **MapLibre GL JS (BSD) in a WebView** | `raster-dem` (Terrarium tiles from open DEM) | `fill-extrusion` from OSM/Overture | Yes with PMTiles on the server | Free | Lighter and simpler, but it is extruded footprints with a tilt, close to what you said you do not want. Same data quality as the Cesium LOD1 option. |
| Google Photorealistic 3D Tiles | Included | Real textured meshes | **No** (terms restrict caching) | Paid beyond free quota | Rejected: offline and cost. |

Honest limit: for most places no free dataset has real textured building meshes. The free ceiling is footprints with heights (OSM, Overture, Microsoft footprints), or true LOD2 in a few countries/cities (for example 3D BAG in the Netherlands, swisstopo in Switzerland). Cesium also gives a real 3D globe and terrain, which MapLibre does not.

Recommendation: **CesiumJS + self-hosted Copernicus-derived terrain + OSM LOD1 building tiles**, with the legend marking LOD1 vs missing areas. Not verified: current Cesium ion free-tier terms, OSM building height coverage in your area, and CTB output quality. These need a test with real data in P0/P7.

Sources: [Cesium offline mode](https://community.cesium.com/t/cesium-completely-offline-mode/36153), [CTB offline terrain](https://community.cesium.com/t/how-to-obtain-pre-built-quantized-mesh-terrain-tiles-for-offline-local-serving-with-cesiumjs/45858), [Protomaps PMTiles + MapLibre](https://docs.protomaps.com/pmtiles/maplibre).
