package com.example.gabai.services

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat

/**
 * Owns the per-screen [VoiceCommandService] lifecycle and turns raw phrases
 * into [Command]s via [CommandSession] (wake name → ding → command window).
 *
 * The mic only listens while the owning screen is in the foreground — call
 * [start] from onResume and [stop] from onPause. Nothing happens when voice
 * commands are disabled in Settings or mic permission is missing (the caller
 * never crashes or nags; onboarding/MainActivity own the permission ask).
 */
class VoiceCommandController(
    private val activity: Activity,
    private val tts: TtsService,
    onCommand: (Command) -> Unit,
) {

    /** Called on the main thread whenever listening actually starts/stops —
     *  wire it to a mic indicator so the always-on mic state is visible.
     *  true only once the models are loaded and the mic is capturing. */
    var onListeningChanged: ((Boolean) -> Unit)? = null

    /** Called on the main thread while the voice models are loading —
     *  wire it to a progress indicator; "hello" can't be heard yet. */
    var onLoadingChanged: ((Boolean) -> Unit)? = null

    /** Called on the main thread when the "Gab" command window opens/closes —
     *  wire it to the indicator's awaiting-command state. */
    var onAwaitingCommand: ((Boolean) -> Unit)?
        get() = session.onAwaitingCommand
        set(value) { session.onAwaitingCommand = value }

    /** Called on the main thread with the latest text Whisper heard —
     *  wire it to the "You said" live-transcript strip. */
    var onHeard: ((String) -> Unit)? = null

    private val session = CommandSession(
        tts = tts,
        requireWakeName = { SettingsStore.isWakeNameRequired(activity) },
        onCommand = onCommand,
        haptics = Haptics(activity.applicationContext),
    )

    private var service: VoiceCommandService? = null

    /** Mic permission is granted AND voice commands are enabled in Settings. */
    fun canListen(context: Context = activity): Boolean =
        SettingsStore.isVoiceCommandsEnabled(context) &&
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.RECORD_AUDIO,
            ) == PackageManager.PERMISSION_GRANTED

    fun start() {
        if (!canListen()) return
        if (service == null) {
            service = VoiceCommandService(
                context = activity.applicationContext,
                speechGate = { !tts.isSpeaking && !SoundCues.isPlaying },
                onCommand = { utterance -> session.handle(utterance); true },
                onError = { msg ->
                    if (!msg.isNullOrBlank()) tts.speak(msg)
                },
                requireWakeName = { SettingsStore.isWakeNameRequired(activity) },
                commandWindowOpen = { session.isWindowOpen },
                onLoadingChanged = { loading -> onLoadingChanged?.invoke(loading) },
                onListeningChanged = { listening ->
                    onListeningChanged?.invoke(listening)
                },
                onHeard = { text -> onHeard?.invoke(text) },
                liveTranscriptEnabled = {
                    SettingsStore.isLiveTranscriptEnabled(activity)
                },
                selfSpeech = { tts.lastSpoken() },
            )
        }
        service?.start()
    }

    /**
     * Big-mic-button fallback: opens the command window so the next
     * utterance runs without the wake name — the ding + haptic ack match
     * a real "Hello". If voice commands can't run, explains why instead
     * of failing silently.
     */
    fun promptForCommand() {
        if (!canListen()) {
            tts.speak("Voice commands need the microphone — check the mic permission and the voice commands setting.")
            return
        }
        start()
        session.openCommandWindow()
    }

    fun stop() {
        service?.stop()
        session.reset()
        onLoadingChanged?.invoke(false)
        onListeningChanged?.invoke(false)
    }

    fun destroy() {
        service?.destroy()
        service = null
    }
}
