# TalkBack Testing Checklist — GabAi Mobile

Test with **real blind users** where possible. Run each pass twice:
once with screen covered (non-visual use), once watching.

## Setup
- [ ] Enable TalkBack (Settings → Accessibility → TalkBack)
- [ ] Set system font size to **largest** and display size to **largest**
- [ ] Test BOTH themes: system dark (Tesla black) AND system light (high-contrast white)
- [ ] Device volume at a comfortable level; TTS audible

## Global (every screen)
- [ ] Every element TalkBack lands on has a **meaningful label** — no "unlabelled button", no reading raw IDs
- [ ] Reading order is top→bottom, matching the annotated order in `DESIGN_SYSTEM.md` §6
- [ ] ≤ 5 focusable items per screen (≤ 6 on OCR — verify it feels acceptable)
- [ ] Focus ring is clearly visible on the focused element (4dp accent outline)
- [ ] Screen titles are headings (`accessibilityHeading`) — swipe up/down cycles headings
- [ ] No element requires sight: every action works by double-tap OR voice command
- [ ] No content auto-dismisses, times out, or auto-plays

## Home
- [ ] Greeting announced as a heading; subtitle explains what to do
- [ ] Each mode row reads as ONE item: "Navigation. Voice-guided obstacle awareness. Double-tap to activate"
- [ ] Say "navigation" / "O C R" / "A S L" / "color" — each opens the right screen with no touches
- [ ] Mic-permission denial is spoken, not silent

## Navigation (most important)
- [ ] Screen is fully operable **without looking**: put the phone down, walk — guidance is spoken
- [ ] Status line re-announces when guidance changes (live region)
- [ ] Obstacle alert interrupts speech (assertive) and fires a strong haptic
- [ ] "Help" button speaks how the screen works
- [ ] Say "go back" — exits navigation hands-free
- [ ] Back button is the FIRST focusable item

## OCR / ASL / Color
- [ ] Result text re-announces when it changes (live region)
- [ ] Primary action is bottom-anchored, full-width, first button after content
- [ ] "Read aloud" / "Speak phrase" / "Announce color" speaks the current result — confirm it says the CURRENT value, not a stale one
- [ ] Help button speaks the screen instructions

## Haptics & audio
- [ ] Every tap produces a haptic click
- [ ] Errors/alerts produce a distinct heavier haptic
- [ ] No meaning is conveyed by color alone — verify each color-coded element also has text

## Degradation
- [ ] Camera permission denied → spoken error, clear next step
- [ ] Server unreachable → spoken status ("Connecting…" then a failure), not a frozen screen
- [ ] 200% font size → no clipped text, no overlapping elements, all buttons still reachable via scroll
- [ ] Screen rotated → same reading order, nothing lost

## Sign-off
- [ ] A blind user completed: open app → start navigation → receive guidance → go back — **without any visual help**
- [ ] A blind user completed: open app → OCR → hear text → go back — **without any visual help**
- [ ] Total swipes from launch to starting navigation ≤ 3
