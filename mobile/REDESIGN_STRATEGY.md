# GabAi redesign: phase 2 strategy

Direction: **Label Tape** (see `.impeccable/surfaces/app-src-main-res-layout-activity-main-xml.md`). Product truth: `PRODUCT.md`.

## 1. Design principles

Each principle has a test a reviewer can run.

1. **Every state has three channels.** Colour, a braille-cell pattern and a spoken/haptic cue. Test: turn off colour, then the screen, and the state is still known.
2. **One skeleton, always.** Same regions in the same order on every screen. Test: blindfold swipe-through lands on the same landmarks in each screen.
3. **SOS is never an accident and never hard.** Isolated by space, reachable in two moves, 5 s countdown with one huge cancel target. Test: nothing within 34 dp of SOS can be mis-hit; SOS reachable from any screen by voice.
4. **Scale is a material.** Layouts survive 200% font scale with no clipped label. Test: font_scale 2.0 on a 360 dp phone, every label fully visible.
5. **Colour names the module, never decorates.** Blue=Navigate, green=Read, amber=Colour, violet=Family, red=SOS/danger only. Test: no tape hue used for a second meaning.
6. **Caregiver view is calm until it is not.** Quiet overview; any alert changes the whole header and cannot be missed. Test: with an active alert, the first thing on screen is the alert and its Acknowledge button.

## 2. Information architecture

### User app (blind / low-vision)
Flat, no tabs. Home is the hub; every module is one level deep; Back always returns to Home.

| Level | Screens |
|---|---|
| Home | Navigate, Read (OCR), Colour, ASL, Family, Settings, SOS |
| Module | Live view (camera or result), result readout, Stop |
| Safety | Emergency countdown (full screen, over lock screen) |
| Setup | Role choice, Onboarding, Pairing/Family, Settings |

Dropped from Home chrome: any control that is not a module, SOS, or the voice strip. Accessible-navigation toggles move to Settings.

### Caregiver app
Keep the 5 tabs (Overview, Map, Camera, Safety, Device). The bottom tab bar stays: Android convention, TalkBack tab semantics already built.

**Problem found:** the header stacks 4 buttons above the content (background alerts, People I look after, two accessible-navigation toggles). That pushes live data down.
**Fix:** header holds title + link chip only. Background alerts, People I look after, and accessibility toggles move to a Settings sheet reached from one overflow icon.

## 3. Screen skeletons

### User skeleton (all user screens)
```
status strip   system-bar inset + listening cell (6-dot braille) + heard phrase
title          one line, 34 sp bold, spoken on enter
work area      scrollable; tape strips or live content
primary action one full-width strip, 92 dp, bottom of work area
footer         Back (left, 64 dp) | SOS (right, hold) on every screen except Emergency
```
Rules: 24 dp page margin, 12 dp between strips, 48 dp minimum target but strips are 92 dp, Back is always bottom-left within thumb reach.

### Caregiver skeleton
```
top bar        "Caregiver" + link chip (Live · ms / Offline / Connecting) + overflow
alert banner   only when an alert is active; red tape, full width, Acknowledge inside it
tab content    one live card stack (Rows.location / navigation / connections reuse)
tab bar        5 tabs, selected tab = filled tape in module colour
```
Every value keeps its age ("12 s ago"); stale values use the warn colour plus a text word, never colour alone.

## 4. SOS and alert escalation path

States from `safety/Emergency.kt`: `IDLE -> COUNTDOWN -> SENT`. Fall countdown 20 s, SOS countdown 5 s.

| Step | User sees / hears / feels | Caregiver sees |
|---|---|---|
| Trigger (hold SOS, voice "help", or fall detected) | Haptic burst, spoken kind of event, Emergency screen over lock screen | Pending alert appears within one poll (2 s) |
| COUNTDOWN | Giant red number, spoken at 10 and 5 s, **I'm okay** = largest target (120 dp+), **Send now** below with gap | Banner: "Possible fall, confirming in N s" |
| SENT | Green-less neutral screen: "Alert sent", who has seen it, I'm okay still available until acknowledged | Confirmed alert, Acknowledge button inside the banner |
| ACKED | Spoken "Maria has seen your alert" + haptic double pulse | Banner turns to "You are responding" |
| Closed | Close button, return to Home | Banner clears after resolve |

Design changes: fall and SOS use the same screen but different headline word; countdown number is the only large red element on the whole app; the **I'm okay** strip is the primary tape (full-width, bottom third, one thumb), **Send now** is outlined until focused (Graphite Console raise).

## 5. Open decisions for phase 3
- Amber tape needs dark text (white on #F5C400 fails contrast). Colour screen strip uses `#111` text.
- Day-light variant: tape on near-black is the default; a high-brightness light scheme is still needed for the system light setting.
- Exact tape hues: verify 4.5:1 for white text on blue/green/violet and 3:1 against the ground.
