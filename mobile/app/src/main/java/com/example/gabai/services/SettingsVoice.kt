package com.example.gabai.services

import android.content.Context

/**
 * Executes the settings commands ([Command.SET_*] / [Command.RATE_*]) from
 * anywhere in the app, so "Hello, captions off" works without opening the
 * Settings screen.
 *
 * SettingsActivity drives its own switches instead (the listeners already
 * speak + apply side-effects); this is the headless path used by other
 * activities and [VoiceListenerService]. Spoken confirmations reuse the
 * same phrasing the switches use.
 */
object SettingsVoice {

    /** Commands that change a setting — handy for scoping allowedCommands. */
    val commands: Set<Command> = setOf(
        Command.SET_VOICE_ON, Command.SET_VOICE_OFF,
        Command.SET_WAKE_ON, Command.SET_WAKE_OFF,
        Command.SET_TRANSCRIPT_ON, Command.SET_TRANSCRIPT_OFF,
        Command.SET_LISTENING_ON, Command.SET_LISTENING_OFF,
        Command.SET_SPEECH_ON, Command.SET_SPEECH_OFF,
        Command.SET_HAPTICS_ON, Command.SET_HAPTICS_OFF,
        Command.SET_CONTRAST_ON, Command.SET_CONTRAST_OFF,
        Command.SET_BIG_TEXT_ON, Command.SET_BIG_TEXT_OFF,
        Command.SET_CAPTIONS_ON, Command.SET_CAPTIONS_OFF,
        Command.RATE_SLOW, Command.RATE_NORMAL, Command.RATE_FAST,
    )

    fun isSettingsCommand(command: Command) = command in commands

    /** Applies the command; false when it isn't a settings command. */
    fun apply(context: Context, tts: TtsService, command: Command): Boolean {
        val ctx = context.applicationContext
        when (command) {
            Command.SET_VOICE_ON -> {
                SettingsStore.setVoiceCommandsEnabled(ctx, true)
                tts.speak("Voice commands on")
                VoiceListenerService.ensureStarted(ctx)
            }
            Command.SET_VOICE_OFF -> {
                SettingsStore.setVoiceCommandsEnabled(ctx, false)
                tts.speak("Voice commands off")
                VoiceListenerService.stop(ctx)
            }
            Command.SET_WAKE_ON -> {
                SettingsStore.setWakeNameRequired(ctx, true)
                tts.speak("Say Gab first on")
            }
            Command.SET_WAKE_OFF -> {
                SettingsStore.setWakeNameRequired(ctx, false)
                tts.speak("Gab not needed, command words work on their own")
            }
            Command.SET_TRANSCRIPT_ON -> {
                SettingsStore.setLiveTranscriptEnabled(ctx, true)
                tts.speak("Show what I say on")
            }
            Command.SET_TRANSCRIPT_OFF -> {
                SettingsStore.setLiveTranscriptEnabled(ctx, false)
                tts.speak("Show what I say off")
            }
            Command.SET_LISTENING_ON -> {
                SettingsStore.setScreenOffListening(ctx, true)
                VoiceListenerService.ensureStarted(ctx)
                tts.speak("Screen-off listening on. Say Gab off to stop it.")
            }
            Command.SET_LISTENING_OFF -> {
                SettingsStore.setScreenOffListening(ctx, false)
                VoiceListenerService.stop(ctx)
                tts.speak("Screen-off listening off")
            }
            Command.SET_SPEECH_ON -> {
                SettingsStore.setSpeechEnabled(ctx, true)
                tts.speak("Speech on")
            }
            Command.SET_SPEECH_OFF -> {
                // Speak first — TtsService checks the flag at call time, so
                // persisting after the call still lets the farewell play.
                tts.speak("Speech off")
                SettingsStore.setSpeechEnabled(ctx, false)
            }
            Command.SET_HAPTICS_ON -> {
                SettingsStore.setHapticsEnabled(ctx, true)
                tts.speak("Vibration on")
            }
            Command.SET_HAPTICS_OFF -> {
                SettingsStore.setHapticsEnabled(ctx, false)
                tts.speak("Vibration off")
            }
            Command.SET_CONTRAST_ON -> {
                SettingsStore.setHighContrast(ctx, true)
                tts.speak("High contrast on. Applies when the screen reloads")
            }
            Command.SET_CONTRAST_OFF -> {
                SettingsStore.setHighContrast(ctx, false)
                tts.speak("High contrast off. Applies when the screen reloads")
            }
            Command.SET_BIG_TEXT_ON -> {
                SettingsStore.setLargeText(ctx, true)
                tts.speak("Large text on. Applies when the screen reloads")
            }
            Command.SET_BIG_TEXT_OFF -> {
                SettingsStore.setLargeText(ctx, false)
                tts.speak("Large text off. Applies when the screen reloads")
            }
            Command.SET_CAPTIONS_ON -> {
                SettingsStore.setCaptionsEnabled(ctx, true)
                tts.speak("Spoken-text panel on")
            }
            Command.SET_CAPTIONS_OFF -> {
                SettingsStore.setCaptionsEnabled(ctx, false)
                tts.speak("Spoken-text panel off")
            }
            Command.RATE_SLOW -> setRate(ctx, tts,
                SettingsStore.SPEECH_RATE_SLOW, "Slow")
            Command.RATE_NORMAL -> setRate(ctx, tts,
                SettingsStore.SPEECH_RATE_NORMAL, "Normal")
            Command.RATE_FAST -> setRate(ctx, tts,
                SettingsStore.SPEECH_RATE_FAST, "Fast")
            else -> return false
        }
        return true
    }

    private fun setRate(ctx: Context, tts: TtsService, rate: Float, label: String) {
        SettingsStore.setSpeechRate(ctx, rate)
        tts.applySpeechRate(rate)
        tts.speak("Speech speed: $label. This is how I sound now.")
    }
}
