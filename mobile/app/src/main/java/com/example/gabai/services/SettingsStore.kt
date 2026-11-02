package com.example.gabai.services

import android.content.Context

/**
 * SharedPreferences-backed store for all user-facing settings.
 * Everything lives on-device; nothing sensitive is ever persisted here
 * (health-adjacent vision data lives in [VisionProfileStore]).
 */
object SettingsStore {

    private const val PREFS = "gabai_settings"

    private const val KEY_ONBOARDING_DONE = "onboarding_done"
    private const val KEY_VOICE_COMMANDS = "voice_commands_enabled"
    private const val KEY_WAKE_REQUIRED = "wake_name_required"
    private const val KEY_SCREEN_OFF_LISTENING = "screen_off_listening"
    private const val KEY_SPEECH = "speech_enabled"
    private const val KEY_SPEECH_RATE = "speech_rate"
    private const val KEY_HAPTICS = "haptics_enabled"
    private const val KEY_HIGH_CONTRAST = "high_contrast"
    private const val KEY_LARGE_TEXT = "large_text"
    private const val KEY_FONT_SCALE = "font_scale"
    private const val KEY_CAPTIONS = "captions_enabled"
    private const val KEY_HOME_INTRO_DONE = "home_intro_done"
    private const val KEY_LIVE_TRANSCRIPT = "live_transcript_enabled"
    private const val KEY_SERVER_URL = "server_url"
    private const val KEY_MIC_SOURCE = "mic_source"
    private const val KEY_CAMERA_SOURCE = "camera_source"
    private const val KEY_TOUCH_EXPLORE = "touch_explore_enabled"
    private const val KEY_RELAY_URL = "relay_url"
    private const val KEY_RELAY_TOKEN = "relay_token_"
    private const val KEY_TRACKING = "tracking_enabled"
    private const val KEY_CG_ALERTS = "caregiver_alerts"
    private const val KEY_CG_NAME = "caregiver_name"
    private const val KEY_ACCESS_NAV = "access_nav"
    private const val KEY_ACCESS_TABS = "access_tabs"
    private const val KEY_CRASH = "crash_note"

    /** Fallback server — must stay whitelisted in network_security_config.xml. */
    const val DEFAULT_SERVER_URL = "http://192.168.100.77:8080"
    const val DEFAULT_RELAY_URL = "http://100.66.233.67:8090"
    const val ROLE_PHONE = "phone"
    const val ROLE_CAREGIVER = "caregiver"

    /** Mic input selection: "auto" (system routing), "phone" (built-in
     *  mic), "headset" (wired/BT headset mic). Speech output is separate —
     *  it always follows Android's media routing to connected headphones. */
    const val MIC_AUTO = "auto"
    const val MIC_PHONE = "phone"
    const val MIC_HEADSET = "headset"

    /** Built-in rear/front camera vs USB UVC (OTG webcam). */
    const val CAMERA_PHONE = "phone"
    const val CAMERA_UVC = "uvc"

    const val LARGE_TEXT_SCALE = 1.3f
    const val FONT_SCALE_MIN = 0.8f
    const val FONT_SCALE_MAX = 1.6f
    const val SPEECH_RATE_SLOW = 0.8f
    const val SPEECH_RATE_NORMAL = 1.0f
    const val SPEECH_RATE_FAST = 1.3f

    fun isOnboardingDone(c: Context) = prefs(c).getBoolean(KEY_ONBOARDING_DONE, false)
    fun setOnboardingDone(c: Context, done: Boolean) =
        put(c) { putBoolean(KEY_ONBOARDING_DONE, done) }

    /** The spoken home-screen welcome plays once ever; "help" replays it. */
    fun isHomeIntroDone(c: Context) = prefs(c).getBoolean(KEY_HOME_INTRO_DONE, false)
    fun setHomeIntroDone(c: Context, done: Boolean) =
        put(c) { putBoolean(KEY_HOME_INTRO_DONE, done) }

    fun isVoiceCommandsEnabled(c: Context) = prefs(c).getBoolean(KEY_VOICE_COMMANDS, true)
    fun setVoiceCommandsEnabled(c: Context, on: Boolean) =
        put(c) { putBoolean(KEY_VOICE_COMMANDS, on) }

    /** When true, phrases must start with the wake name ("Hello, walk").
     *  Default OFF: bare command words ("walk", "back", "home") fire
     *  directly — the wake name still works as an optional prefix. */
    fun isWakeNameRequired(c: Context) = prefs(c).getBoolean(KEY_WAKE_REQUIRED, false)
    fun setWakeNameRequired(c: Context, on: Boolean) =
        put(c) { putBoolean(KEY_WAKE_REQUIRED, on) }

    /**
     * Keep listening for "Gabai" while the app is backgrounded or the screen
     * is off (foreground mic service + persistent notification). Default ON
     * for demos; battery cost is real — that's why it's a toggle.
     */
    fun isScreenOffListening(c: Context) =
        prefs(c).getBoolean(KEY_SCREEN_OFF_LISTENING, true)
    fun setScreenOffListening(c: Context, on: Boolean) =
        put(c) { putBoolean(KEY_SCREEN_OFF_LISTENING, on) }

    fun isSpeechEnabled(c: Context) = prefs(c).getBoolean(KEY_SPEECH, true)
    fun setSpeechEnabled(c: Context, on: Boolean) =
        put(c) { putBoolean(KEY_SPEECH, on) }

    fun getSpeechRate(c: Context) = prefs(c).getFloat(KEY_SPEECH_RATE, SPEECH_RATE_NORMAL)
    fun setSpeechRate(c: Context, rate: Float) =
        put(c) { putFloat(KEY_SPEECH_RATE, rate) }

    fun isHapticsEnabled(c: Context) = prefs(c).getBoolean(KEY_HAPTICS, true)
    fun setHapticsEnabled(c: Context, on: Boolean) =
        put(c) { putBoolean(KEY_HAPTICS, on) }

    fun isHighContrast(c: Context) = prefs(c).getBoolean(KEY_HIGH_CONTRAST, false)
    fun setHighContrast(c: Context, on: Boolean) =
        put(c) { putBoolean(KEY_HIGH_CONTRAST, on) }

    /** Continuous text size (0.8–1.6×) via the Settings slider. Migrates
     *  the legacy large_text boolean on first read. */
    fun getFontScale(c: Context): Float {
        val p = prefs(c)
        if (!p.contains(KEY_FONT_SCALE)) {
            return if (p.getBoolean(KEY_LARGE_TEXT, false)) LARGE_TEXT_SCALE else 1.0f
        }
        return p.getFloat(KEY_FONT_SCALE, 1.0f)
    }
    fun setFontScale(c: Context, scale: Float) =
        put(c) { putFloat(KEY_FONT_SCALE, scale.coerceIn(FONT_SCALE_MIN, FONT_SCALE_MAX)) }

    fun isLargeText(c: Context) = getFontScale(c) > 1.0f
    fun setLargeText(c: Context, on: Boolean) =
        setFontScale(c, if (on) LARGE_TEXT_SCALE else 1.0f)

    /** "Gabai said" caption panel — on-screen transcript for helpers. */
    fun isCaptionsEnabled(c: Context) = prefs(c).getBoolean(KEY_CAPTIONS, true)
    fun setCaptionsEnabled(c: Context, on: Boolean) =
        put(c) { putBoolean(KEY_CAPTIONS, on) }

    /**
     * "Show what I say" — rolling Whisper transcription of raw mic audio,
     * shown on the "You said" strip. It bypasses VAD/wake gating, so it
     * keeps captions AND commands alive where Silero never fires — at the
     * cost of running Whisper continuously while listening (battery).
     */
    fun isLiveTranscriptEnabled(c: Context) =
        prefs(c).getBoolean(KEY_LIVE_TRANSCRIPT, true)
    fun setLiveTranscriptEnabled(c: Context, on: Boolean) =
        put(c) { putBoolean(KEY_LIVE_TRANSCRIPT, on) }

    fun getServerUrl(c: Context): String =
        prefs(c).getString(KEY_SERVER_URL, null)?.takeIf { it.isNotBlank() }
            ?: DEFAULT_SERVER_URL

    fun setServerUrl(c: Context, url: String) =
        put(c) { putString(KEY_SERVER_URL, url.trim()) }

    fun getMicSource(c: Context): String =
        prefs(c).getString(KEY_MIC_SOURCE, MIC_AUTO) ?: MIC_AUTO

    fun setMicSource(c: Context, source: String) =
        put(c) { putString(KEY_MIC_SOURCE, source) }

    fun getCameraSource(c: Context): String =
        prefs(c).getString(KEY_CAMERA_SOURCE, CAMERA_PHONE) ?: CAMERA_PHONE

    fun setCameraSource(c: Context, source: String) =
        put(c) { putString(KEY_CAMERA_SOURCE, source) }

    /**
     * Touch explore — finger-track warp effect + dwell announcements: hold
     * a finger over a control for ~1 s to hear its name (TalkBack-style,
     * powered by the app's own TTS). Default ON for the app's audience.
     */
    fun isTouchExploreEnabled(c: Context) =
        prefs(c).getBoolean(KEY_TOUCH_EXPLORE, true)
    fun setTouchExploreEnabled(c: Context, on: Boolean) =
        put(c) { putBoolean(KEY_TOUCH_EXPLORE, on) }

    /** Caregiver relay (gabai/relay): phone uplink + caregiver dashboard. Same laptop as the nav server, port 8090. */
    fun getRelayUrl(c: Context): String =
        prefs(c).getString(KEY_RELAY_URL, null)?.takeIf { it.isNotBlank() } ?: DEFAULT_RELAY_URL
    fun setRelayUrl(c: Context, url: String) = put(c) { putString(KEY_RELAY_URL, url.trim().trimEnd('/')) }

    fun getRelayToken(c: Context, role: String): String =
        prefs(c).getString(KEY_RELAY_TOKEN + role, "") ?: ""
    fun setRelayToken(c: Context, role: String, token: String) =
        put(c) { putString(KEY_RELAY_TOKEN + role, token.trim()) }

    fun isTrackingEnabled(c: Context) = prefs(c).getBoolean(KEY_TRACKING, false)
    fun setTrackingEnabled(c: Context, on: Boolean) = put(c) { putBoolean(KEY_TRACKING, on) }

    fun isCaregiverAlertsEnabled(c: Context) = prefs(c).getBoolean(KEY_CG_ALERTS, false)
    fun setCaregiverAlertsEnabled(c: Context, on: Boolean) = put(c) { putBoolean(KEY_CG_ALERTS, on) }

    fun getCaregiverName(c: Context): String =
        prefs(c).getString(KEY_CG_NAME, null)?.takeIf { it.isNotBlank() } ?: "Caregiver"
    fun setCaregiverName(c: Context, name: String) = put(c) { putString(KEY_CG_NAME, name.trim().take(40)) }

    /** Accessible navigation: tap to focus and hear, double-tap to activate (see accessibility/AccessFocusController). */
    fun isAccessNavEnabled(c: Context) = prefs(c).getBoolean(KEY_ACCESS_NAV, false)
    fun setAccessNavEnabled(c: Context, on: Boolean) = put(c) { putBoolean(KEY_ACCESS_NAV, on) }

    fun isAccessTabsOnly(c: Context) = prefs(c).getBoolean(KEY_ACCESS_TABS, false)
    fun setAccessTabsOnly(c: Context, on: Boolean) = put(c) { putBoolean(KEY_ACCESS_TABS, on) }

    /** Unsent crash note from the previous run; the next uplink batch reports it once. */
    fun takeCrashNote(c: Context): String? =
        prefs(c).getString(KEY_CRASH, null).also { if (it != null) put(c) { remove(KEY_CRASH) } }
    fun setCrashNote(c: Context, note: String) = prefs(c).edit().putString(KEY_CRASH, note).commit()

    /** Clears every GabAi setting (used by "Reset everything" in Settings). */
    fun resetAll(c: Context) {
        prefs(c).edit().clear().apply()
    }

    private fun prefs(c: Context) =
        c.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private inline fun put(c: Context, block: android.content.SharedPreferences.Editor.() -> Unit) {
        prefs(c).edit().apply(block).apply()
    }
}
