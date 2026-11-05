package com.example.gabai.services

import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.Executors

/**
 * Siri-style turn-taking on top of [CommandParser], shared by
 * [VoiceCommandController] (foreground screens) and [VoiceListenerService]
 * (screen-off background listening).
 *
 * Feed every transcribed utterance to [handle]. A bare wake name ("Gab")
 * acknowledges with a chime and opens a short command window — the next
 * utterance is parsed with the wake name implicit. If nothing actionable
 * arrives before the deadline, a soft closing blip plays. "Gab, walk" said
 * in one breath still works without the window.
 *
 * When the keyword parser can't map a phrase, the utterance is handed to
 * [Needle] — the on-device tool-calling model — but only when it followed
 * the wake name or arrived inside the command window, so ambient speech
 * never reaches the intent model. Its tool call becomes a [Command].
 *
 * Must be called on the main thread (that's where [VoiceCommandService]
 * delivers phrases).
 */
class CommandSession(
    private val tts: TtsService,
    private val requireWakeName: () -> Boolean,
    private val onCommand: (Command) -> Unit,
    private val haptics: Haptics? = null,
    private val config: VoiceConfig = VoiceConfig.DEFAULT,
    /** Commands valid in the current screen/state — the matcher and needle
     *  paths only ever fire these. Default: everything. */
    private val allowedCommands: () -> Set<Command> = { Command.entries.toSet() },
) {

    /** True while the wake word's command window is open — the capture
     *  service uses it to decide which VAD segments Whisper transcribes. */
    val isWindowOpen: Boolean
        get() = SystemClock.elapsedRealtime() < windowDeadlineMs

    /** Fired when the command window opens/closes — drive the "awaiting
     *  command" indicator state with it. */
    var onAwaitingCommand: ((Boolean) -> Unit)? = null

    private val mainHandler = Handler(Looper.getMainLooper())
    // Read from the capture thread via isWindowOpen; written on main.
    @Volatile private var windowDeadlineMs = 0L
    private var windowOpenedAtMs = 0L
    private var lastBareHintAtMs = 0L
    private var utteranceSerial = 0L
    private var accepting = true

    private val needleExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "needle-intent").apply { isDaemon = true }
    }

    private val windowCloser = Runnable { closeWindow(timedOut = true) }

    fun handle(utterance: Utterance) {
        accepting = true
        utteranceSerial++
        val phrase = utterance.text
        val inWindow = isWindowOpen
        val allowed = allowedCommands()
        val result = if (inWindow) {
            CommandParser.parseAfterWake(
                phrase, allowed, strict = config.destructiveCommands)
        } else {
            CommandParser.parse(phrase, requireWakeName = requireWakeName(), allowed)
        }
        Log.d(TAG, "phrase=\"$phrase\" inWindow=$inWindow -> $result")

        when (result) {
            is ParseResult.Parsed -> fire(result.command, inWindow)
            ParseResult.WakeOnly -> {
                // The wake probe and the trailing utterance transcript can
                // deliver the same "hello" a second apart — don't double-ding.
                if (SystemClock.elapsedRealtime() - windowOpenedAtMs >
                    REWAKE_SUPPRESS_MS
                ) {
                    openWindow()
                }
            }
            ParseResult.BareCommand -> {
                // Someone tried a command without the wake name — teach it
                // once in a while instead of staying silent.
                val now = SystemClock.elapsedRealtime()
                if (now - lastBareHintAtMs > BARE_HINT_COOLDOWN_MS) {
                    lastBareHintAtMs = now
                    tts.speak(CommandParser.WAKE_HINT)
                }
            }
            // Keyword matching missed — try the fuzzy/phonetic matcher
            // first (wake-gated only, same as the intent model), then let
            // the intent model reason about what's left.
            ParseResult.NoMatch -> {
                val fuzzy = fuzzyMatch(phrase, inWindow)
                if (fuzzy != null) {
                    fire(fuzzy, inWindow)
                } else if (inWindow || CommandParser.hasWakeName(phrase)) {
                    askNeedle(utterance, inWindow)
                }
            }
        }
    }

    /**
     * A command fired: instant tick + haptic so the user knows it landed,
     * then the screen's own spoken/visual ack follows. The cue is ~90 ms —
     * perceived latency is what the user feels, and TTS takes ~300+ ms to
     * reach its first syllable.
     */
    private fun fire(command: Command, windowWasOpen: Boolean) {
        if (windowWasOpen) closeWindow(timedOut = false)
        SoundCues.commandAck()
        haptics?.click()
        onCommand(command)
    }

    /**
     * Fuzzy/phonetic match of the command text — the wake name stripped
     * when present, the whole phrase when inside the command window or
     * wake-free mode. Below the similarity threshold this returns null:
     * NO ACTION.
     */
    private fun fuzzyMatch(phrase: String, inWindow: Boolean): Command? {
        val candidate = CommandParser.textAfterWake(phrase)?.joinToString(" ")
            ?: if (inWindow || !requireWakeName()) phrase else return null
        // The session classifier wraps the fuzzy matcher: learned phrase
        // and token caches make repeat mishears resolve instantly and
        // consistently for the lifetime of the app.
        val match = CommandClassifier.match(candidate, config, allowedCommands() - Command.SOS)
            ?: return null
        Log.d(TAG, "fuzzy \"$phrase\" -> ${match.command} " +
            "(%.2f via '${match.synonym}')".format(match.score))
        return match.command
    }

    /**
     * Asks Needle to map a phrase the keyword parser couldn't handle onto
     * the app's tools. Runs off the main thread; the result is dropped if a
     * newer utterance arrived in the meantime or the session was stopped.
     * An open command window gets a short grace extension while the model
     * reasons so it doesn't close mid-call.
     */
    private fun askNeedle(utterance: Utterance, windowWasOpen: Boolean) {
        if (!Needle.isAvailable) return
        if (windowWasOpen) extendWindow(NEEDLE_GRACE_MS)
        val serial = utteranceSerial
        needleExecutor.execute {
            // Whisper already transcribed this clip — reason over its text.
            // (The audio path exists for engines that do STT in-model;
            // feeding the transcript keeps one model loaded at a time.)
            val commands = if (utterance.text.isNotBlank()) {
                Needle.commandsFor(utterance.text)
            } else if (utterance.pcm.isNotEmpty()) {
                Needle.commandsForAudio(utterance.pcm, utterance.language)
            } else {
                emptyList()
            }.filter { it in allowedCommands() }
            if (commands.isEmpty()) return@execute
            val command = commands.first()
            if (command in config.destructiveCommands &&
                !destructiveConfirmed(utterance, command)
            ) {
                Log.d(TAG, "needle $command dropped — no explicit phrase in " +
                    "\"${utterance.text}\"")
                return@execute
            }
            mainHandler.post {
                if (!accepting || serial != utteranceSerial) return@post
                fire(command, windowWasOpen)
            }
        }
    }

    /**
     * A destructive needle result only stands when the transcript itself
     * contains the real phrase — an intent-model guess can never stop a
     * session. Checks exact keywords first, then the strict fuzzy bar.
     */
    private fun destructiveConfirmed(utterance: Utterance, command: Command): Boolean {
        val text = utterance.text
        if (text.isBlank()) return false
        val exact = CommandParser.parse(
            text, requireWakeName = false, allowed = setOf(command))
        if (exact is ParseResult.Parsed) return true
        val candidate = CommandParser.textAfterWake(text)?.joinToString(" ") ?: text
        return CommandMatcher.match(candidate, config, setOf(command)) != null
    }

    private fun extendWindow(ms: Long) {
        windowDeadlineMs = SystemClock.elapsedRealtime() + ms
        mainHandler.removeCallbacks(windowCloser)
        mainHandler.postDelayed(windowCloser, ms)
    }

    /** Drops any open command window without a sound — call on stop/destroy. */
    fun reset() {
        accepting = false
        closeWindow(timedOut = false)
    }

    /**
     * Programmatic wake — same as hearing "Hello": ding + haptic + command
     * window, so the next utterance parses without the wake name. Backs the
     * big mic fallback button. No-op while a window is already open.
     */
    fun openCommandWindow() {
        if (!isWindowOpen) openWindow()
    }

    private fun openWindow() {
        windowOpenedAtMs = SystemClock.elapsedRealtime()
        windowDeadlineMs = windowOpenedAtMs + config.commandWindowMs
        mainHandler.removeCallbacks(windowCloser)
        mainHandler.postDelayed(windowCloser, config.commandWindowMs)
        SoundCues.wakeAck()
        haptics?.wakeAck()
        if (config.wakePrompt.isNotBlank()) tts.speak(config.wakePrompt)
        onAwaitingCommand?.invoke(true)
    }

    private fun closeWindow(timedOut: Boolean) {
        if (windowDeadlineMs == 0L) return
        windowDeadlineMs = 0L
        mainHandler.removeCallbacks(windowCloser)
        if (timedOut) {
            SoundCues.windowClosed()
            if (config.windowTimeoutPrompt.isNotBlank()) {
                tts.speak(config.windowTimeoutPrompt)
            }
        }
        onAwaitingCommand?.invoke(false)
    }

    private companion object {
        const val TAG = "GabVoice"
        const val NEEDLE_GRACE_MS = 4_000L
        const val BARE_HINT_COOLDOWN_MS = 8_000L
        // One spoken "hello" lands in ~3 overlapping listen windows (~2 s
        // apart) — suppress only those echoes; a real re-wake later re-dings.
        const val REWAKE_SUPPRESS_MS = 3_500L
    }
}
