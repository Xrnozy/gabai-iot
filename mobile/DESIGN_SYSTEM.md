# GabAi Mobile — Design System

Tesla-inspired minimalist system built for **blind and low-vision users first**.
Every screen is operable without seeing it: TalkBack labels on everything,
voice commands for every action, haptics + TTS for every event. The visual
design is what remains after every non-essential element is removed.

---

## 1. Themes

Two palettes, selected automatically by the system theme — no in-app toggle
needed (one fewer focusable item):

| System setting | Palette | Look |
|---|---|---|
| Dark mode | `values-night/colors.xml` | **Tesla dark** — pure black (default) |
| Light mode | `values/colors.xml` | **High-contrast white** |

Color *names* are identical in both palettes — no layout changes between modes.

## 2. Color palette + measured contrast

### Dark (default) — `values-night/colors.xml`

| Token | Value | On `#000000` | Use |
|---|---|---|---|
| `canvas` | `#000000` | — | App background |
| `surface_card` | `#141414` | — | Cards, secondary buttons |
| `surface_card_elevated` | `#1C1C1C` | — | Pressed state |
| `ink` | `#FFFFFF` | 21.0 : 1 | Headings, primary text |
| `body` | `#E0E0E0` | 15.6 : 1 | Body text |
| `muted` | `#A3A3A3` | 8.4 : 1 | Secondary text (never below this) |
| `primary` | `#3E6AE1` | 4.6 : 1 (white text on fill) | Primary buttons — AAA for ≥20sp text |
| `focus_ring` | `#7C9BFF` | 8.0 : 1 | Focus outlines, accents |
| `status_success` | `#4CAF50` | 7.4 : 1 | OK states |
| `status_warning` | `#FFB340` | 10.5 : 1 | Caution |
| `status_error` | `#FF5B4D` | 7.8 : 1 | Errors, obstacle alerts |

### Light / high-contrast — `values/colors.xml`

| Token | Value | On `#FFFFFF` | Use |
|---|---|---|---|
| `canvas` | `#FFFFFF` | — | App background |
| `surface_card` | `#F2F2F2` | — | Cards |
| `ink` | `#000000` | 21.0 : 1 | Headings |
| `body` | `#1A1A1A` | 17.7 : 1 | Body text |
| `muted` | `#3D3D3D` | 9.7 : 1 | Secondary text |
| `primary` | `#1A4FCC` | 7.6 : 1 | Accent (safe for text AND fill) |
| `focus_ring` | `#1A4FCC` | 7.6 : 1 | Focus outlines |
| `status_success` | `#146B32` | 7.0 : 1 | OK |
| `status_warning` | `#9A4E00` | 7.5 : 1 | Caution |
| `status_error` | `#B3261E` | 7.4 : 1 | Errors |

**Rules:** one accent only (focus + primary actions). No gradients, no
shadows, no borders, no text over imagery, no gray below `muted`.

## 3. Type scale

System sans (Roboto / SF-proportional). All `sp` — scales with the user's
font-size setting to 200%+ (layouts are `wrap_content`, never fixed-height).

| Style | Size | Weight | Use |
|---|---|---|---|
| `text_display` | 36sp | medium | Home greeting |
| `text_headline` | 30sp | medium | Detected results, direction |
| `text_title` | 26sp | medium | Screen titles, nav status |
| `text_body_lg` | 22sp | regular/medium | Emphasized content, row titles |
| `text_body` | 20sp | regular | **Minimum body size** |
| `text_caption` | 16sp | regular | Descriptions |
| `text_small` | 14sp | medium | Section eyebrows only |

## 4. Spacing + layout grid

8dp grid: `4 / 8 / 16 / 24 / 32 / 40 / 48 / 56 / 64`.
Page margin `24dp` everywhere. Huge negative space is the design.

**Every screen follows one skeleton** (muscle memory for blind users):

```
┌─────────────────────────────┐
│ ← Back    TITLE        Help │  top bar, 64dp targets
│                             │
│      content / live result  │  middle — the product
│                             │
│  [ Primary action ]         │  bottom thumb zone
│  [ Secondary action ]       │  full-width, 64dp
└─────────────────────────────┘
```

## 5. Component specs

| Component | Visual | TalkBack | Haptic | States |
|---|---|---|---|---|
| Mode row (home) | Full-width card, 96dp+, `bg_card_focusable` | Combined label: "Navigation. Voice-guided obstacle awareness." + "double-tap to activate" | `click()` | default / focused (4dp accent ring) / pressed |
| Primary button | Accent fill, 64dp, 20sp, `bg_btn_primary_focusable` | Label + "button" | `click()` | default / focused / pressed / disabled (40% alpha) |
| Secondary button | Card fill, 64dp, `bg_btn_secondary_focusable` | Label + "button" | `click()` | same |
| Icon control (Back/Help) | 64dp circle, `bg_icon_btn_focusable` | "Back to home, button" / "Help — hear how this screen works, button" | `click()` | same |
| Status/result text | Ink, 22–30sp | `accessibilityLiveRegion=polite` — auto-announces changes | — | text change = announcement |
| Obstacle alert | Error card, 22sp | `accessibilityLiveRegion=assertive` — interrupts speech | `heavy()` | hidden ↔ shown |
| Headings | 26sp+ | `accessibilityHeading=true` — jumpable | — | — |
| Decorative (icons, glyphs, camera) | — | `importantForAccessibility=no` | — | hidden from a11y tree |

**Rules:** every actionable element has `contentDescription`; every status
has a live region; every icon button gets a text label (never icon-only);
max ~5 focusables per screen; no timed UI, no precision gestures — voice
commands replicate every action.

## 6. Per-screen annotation

### Home (`activity_main`)
Reading order: greeting (heading) → subtitle → voice hint (live) →
4 mode rows. Focusables: **4**. Voice: "navigation" / "O C R" / "A S L" / "color".

### Navigation (`activity_navigation`)
Reading order: Back → title → Help → direction (live) → status card (live)
→ obstacle alert (assertive, when shown). Focusables: **3**.
Non-visual by design: guidance is spoken by `TtsService` and re-announced
by the status live region. Voice: "go back" exits.

### OCR (`activity_ocr`)
Back → title → Help → extracted text (live) → Read aloud → Translate →
Save. Focusables: **6** (one more than ideal — acceptable, all bottom-anchored).

### ASL (`activity_asl`)
Back → title → Help → detected phrase (live) → Speak phrase. Focusables: **4**.

### Color (`activity_color`)
Back → title → Help → detected color + sentence (live) → Announce color.
Focusables: **4**.

**Gesture alternatives:** every action = tap OR voice command. No swipes,
pinch, long-press, or multi-finger gestures anywhere.
