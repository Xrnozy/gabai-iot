# Product

<!-- impeccable:product-schema 1 -->

## Platform

android

## Users
- Primary: blind and low-vision people, moving through the world (walking, shopping, reading mail) with a phone plus camera hardware (USB/UVC, WebRTC stream). Voice-first, TalkBack, haptics and TTS are the main channels; the screen is secondary.
- Secondary: sighted family caregivers who monitor the primary user remotely: live location, camera feed, fall and SOS alerts.

## Product Purpose
GabAi gives a blind or low-vision user spoken, hands-free help: navigation with obstacle warnings, OCR reading, color identification, ASL, emergency SOS and fall detection. It gives caregivers peace of mind and fast alerts. Success: the user completes a task without looking at the screen; the caregiver is told the moment something is wrong.

## Capabilities and Constraints
- Two roles chosen on first launch (user / caregiver), paired by a pairing flow.
- Modules: navigation, OCR, color, ASL, settings, family, SOS/emergency, onboarding, voice command and wake word.
- Hard constraints: TalkBack labels on everything, voice command for every action, 200%+ font scaling (wrap_content, no fixed heights), large touch targets, consistent screen skeleton, measured contrast, system light/dark.
- Existing artifacts: `DESIGN_SYSTEM.md` (Tesla black/white/blue) and `res/values/colors.xml` (warm cream + terracotta) disagree. Both are being replaced.

## Product Principles
1. The screen never carries the only copy of information; every state is also spoken, felt or voiced.
2. Same skeleton on every screen: muscle memory beats novelty for non-visual users.
3. Safety first: SOS and alerts are always reachable and unmistakable.
4. Low-vision users are users of the visuals too: contrast and scale are design material, not compliance.
5. Caregiver view is calm and glanceable; it escalates only on real events.

## Accessibility & Inclusion
TalkBack-first (see `TALKBACK_TEST_CHECKLIST.md`), WCAG AAA-leaning contrast, reduce-motion respect, no color-only meaning.
