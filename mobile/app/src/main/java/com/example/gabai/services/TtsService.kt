package com.example.gabai.services

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale

class TtsService(
    context: Context,
    private val locale: Locale = Locale.US,
) : TextToSpeech.OnInitListener {

    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())

    private var tts: TextToSpeech? = TextToSpeech(appContext, this)
    private var initialized = false
    private var pending: String? = null

    private var lastSpokenNormalized: String = ""
    private var lastSpokenAtMs: Long = 0L

    // One-deep queue for non-interrupting guidance: while an utterance is
    // speaking, routine lines REPLACE this slot instead of chopping the
    // in-flight sentence or piling up stale instructions. Drained by the
    // utterance listener the moment the current line finishes.
    private var queued: String? = null
    private var queuedNormalized: String? = null

    /** True while a queued or speaking utterance is still producing audio. */
    val isSpeaking: Boolean
        get() = tts?.isSpeaking == true || pending != null

    /**
     * Fired on the main thread with the text of every utterance Gabai wants
     * to say — including while speech is muted, so the on-screen caption
     * panel stays a truthful transcript for sighted helpers.
     */
    var onSpoken: ((String) -> Unit)? = null

    override fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) return
        val engine = tts ?: return

        val result = engine.setLanguage(locale)
        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
            return
        }

        engine.setSpeechRate(SettingsStore.getSpeechRate(appContext))
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) { drainQueued() }
            @Deprecated("deprecated in interface")
            override fun onError(utteranceId: String?) { drainQueued() }
            override fun onError(utteranceId: String?, errorCode: Int) {
                drainQueued()
            }
        })
        initialized = true
        pending?.let {
            pending = null
            speak(it)
        }
    }

    /** Utterance finished (or failed) — play the freshest queued line. */
    private fun drainQueued() {
        mainHandler.post {
            val next = queued ?: return@post
            queued = null
            queuedNormalized = null
            tts?.speak(next, TextToSpeech.QUEUE_FLUSH, null, UTTERANCE_ID)
        }
    }

    /** Applies the configured (or a new) speech rate to the live engine. */
    fun applySpeechRate(rate: Float = SettingsStore.getSpeechRate(appContext)) {
        tts?.setSpeechRate(rate)
    }

    /**
     * The last line the engine actually spoke (normalized), if it was said
     * within [withinMs]. The mic keeps capturing through TTS, so the voice
     * pipeline uses this to strip our own echo out of transcripts —
     * "back, please try again" should still read as "back".
     */
    fun lastSpoken(withinMs: Long = 12_000): String? =
        if (lastSpokenNormalized.isNotBlank() &&
            android.os.SystemClock.elapsedRealtime() - lastSpokenAtMs < withinMs
        ) lastSpokenNormalized else null

    /**
     * Speak [text].
     *
     * interrupt = true (default): flush — the line plays immediately,
     * chopping whatever is speaking. Used by interactive prompts and
     * genuine emergencies.
     *
     * interrupt = false: polite queue — if the engine is mid-sentence the
     * line waits in the one-deep [queued] slot (newest wins, so outdated
     * directions are replaced rather than accumulated) and plays as soon
     * as the current utterance finishes. Navigation guidance uses this
     * so sentences are never chopped mid-word.
     */
    fun speak(text: String, interrupt: Boolean = true) {
        val trimmed = text.trim()
        if (trimmed.isBlank()) return

        onSpoken?.let { cb -> mainHandler.post { cb(trimmed) } }

        if (!SettingsStore.isSpeechEnabled(appContext)) return

        if (!initialized) {
            pending = trimmed
            return
        }

        val engine = tts ?: return
        val normalized = trimmed.lowercase(locale)
        val now = android.os.SystemClock.elapsedRealtime()

        // Avoid TTS spam (duplicate focus events / repeated voice command echoes).
        val tooSoon = (now - lastSpokenAtMs) < 700
        if (normalized == lastSpokenNormalized && (engine.isSpeaking || tooSoon)) {
            return
        }
        if (normalized == queuedNormalized) return

        lastSpokenNormalized = normalized
        lastSpokenAtMs = now

        mainHandler.post {
            val t = tts ?: return@post
            if (interrupt) {
                queued = null
                queuedNormalized = null
                t.speak(trimmed, TextToSpeech.QUEUE_FLUSH, null, UTTERANCE_ID)
            } else if (t.isSpeaking) {
                // replace the queued slot — the freshest instruction wins
                queued = trimmed
                queuedNormalized = normalized
            } else {
                t.speak(trimmed, TextToSpeech.QUEUE_FLUSH, null, UTTERANCE_ID)
            }
        }
    }

    /** Stops any in-flight speech without disabling the engine. */
    fun stopSpeaking() {
        pending = null
        queued = null
        queuedNormalized = null
        mainHandler.post { tts?.stop() }
    }

    fun shutdown() {
        val engine = tts ?: return
        tts = null
        mainHandler.post {
            engine.stop()
            engine.shutdown()
        }
    }

    companion object {
        private const val UTTERANCE_ID = "gabai-tts"
    }
}
