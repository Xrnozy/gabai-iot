package com.example.gabai.services

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

/** One finished transcription plus the decoder confidence signals the
 *  rejection filters need. Probabilities are nullable — a recognizer that
 *  cannot expose them reports null and the filters adapt. */
data class Transcription(
    val text: String,
    val language: String?,
    /** P(no speech) reported by the decoder; null when unavailable. */
    val noSpeechProb: Float?,
    /** Mean token log probability; null when unavailable. */
    val avgLogprob: Float?,
    val durationSec: Float,
    val segmentCount: Int = 1,
    /** Speech spans whisper's internal VAD found in this window
     *  ([startMs, endMs] on the input timeline); empty when VAD was off
     *  or found nothing. */
    val vadSegments: List<Pair<Long, Long>> = emptyList(),
    /** Wall time of the whisper_full call — field telemetry. */
    val elapsedMs: Long = 0L,
)

/**
 * Speech-to-text behind an interface so a second implementation (e.g. a
 * grammar-restricted Vosk recognizer) can be swapped in without touching the
 * pipeline. Implementations own their model lifecycle and are NOT required
 * to be thread-safe — callers serialise.
 */
interface Recognizer {
    val isLoaded: Boolean

    /** Loads the model. Throws (with a logged reason) when unavailable. */
    fun load()

    /** Transcribes 16 kHz mono float PCM in [-1, 1]. Never call on the
     *  UI thread. */
    fun transcribe(pcm: FloatArray): Transcription

    /** Frees the model. The one-model-at-a-time rule for low-RAM devices
     *  is enforced by callers loading one recognizer per language. */
    fun release()
}

/**
 * whisper.cpp (Whisper Base Q8) over JNI. The model file comes from
 * [VoiceConfig.resolveModelFile] — external-files override, then the
 * assets/models/ copy. English vs multilingual is a config switch; only ONE
 * model is ever loaded.
 */
class WhisperRecognizer(
    private val context: Context,
    private val config: VoiceConfig = VoiceConfig.DEFAULT,
) : Recognizer {

    private val loaded = AtomicBoolean(false)
    @Volatile private var ctx: Long = 0L

    override val isLoaded: Boolean get() = ctx != 0L

    /** Resolved ggml Silero-VAD model path; null when VAD is off or the
     *  model file is missing (whisper then runs plain full decode). */
    @Volatile private var vadModelPath: String? = null
    /** Standalone VAD context for probability probing (diagnostics). */
    @Volatile private var vadCtx: Long = 0L

    private external fun nativeInit(modelPath: String, nThreads: Int): Long
    private external fun nativeTranscribe(
        handle: Long,
        pcm: FloatArray,
        language: String?,
        initialPrompt: String?,
        nThreads: Int,
        vadModelPath: String?,
        vadThreshold: Float,
        vadMinSpeechMs: Int,
        vadMinSilenceMs: Int,
        vadSpeechPadMs: Int,
        vadMaxSpeechS: Float,
        vadOverlapS: Float,
        audioCtx: Int,
    ): String?
    private external fun nativeVadInit(modelPath: String, nThreads: Int): Long
    private external fun nativeVadMaxProb(handle: Long, pcm: FloatArray): Float
    private external fun nativeVadRelease(handle: Long)
    private external fun nativeRelease(handle: Long)
    private external fun nativeLastError(): String?

    @Synchronized
    override fun load() {
        if (loaded.get()) return
        val file = config.resolveModelFile(
            context, config.whisperModelFile, config.minModelBytes, TAG,
        ) ?: throw IllegalStateException(
            "Whisper model ${config.whisperModelFile} missing — " +
                "put it in assets/models/ or adb push to " +
                context.getExternalFilesDir(config.modelFileDir)
        )
        if (config.vadMode == VadMode.WHISPER_NATIVE) {
            vadModelPath = config.resolveModelFile(
                context, config.whisperVadModelFile, minBytes = 100_000,
                tag = TAG,
            )?.absolutePath
            if (vadModelPath == null) {
                Log.w(TAG, "${config.whisperVadModelFile} missing — " +
                    "whisper-native VAD unavailable; " +
                    "vadMode=${config.vadMode} degrades to plain windows")
            } else {
                Log.i(TAG, "whisper VAD model: ${config.whisperVadModelFile}")
            }
        }
        // #region agent log
        AgentDebugLog.log(
            "Recognizer.kt:load",
            "before System.loadLibrary(whisper_gab)",
            "A",
            mapOf("abi" to android.os.Build.SUPPORTED_ABIS.joinToString()),
        )
        // #endregion
        try {
            System.loadLibrary("whisper_gab")
        } catch (e: UnsatisfiedLinkError) {
            // #region agent log
            AgentDebugLog.log(
                "Recognizer.kt:load",
                "whisper_gab load failed",
                "A",
                mapOf("error" to (e.message ?: e.toString())),
            )
            // #endregion
            throw e
        }
        // #region agent log
        AgentDebugLog.log(
            "Recognizer.kt:load",
            "whisper_gab loaded",
            "A",
            emptyMap(),
        )
        // #endregion
        vadModelPath?.let { path ->
            vadCtx = nativeVadInit(path, 2)
            if (vadCtx == 0L) {
                Log.w(TAG, "vad probe init failed: ${nativeLastError()}")
            }
        }
        val handle = nativeInit(file.absolutePath, config.whisperThreads)
        if (handle == 0L) {
            throw IllegalStateException(nativeLastError() ?: "whisper init failed")
        }
        ctx = handle
        loaded.set(true)
        Log.i(TAG, "whisper.cpp loaded: ${file.name} " +
            "(${file.length() / (1024 * 1024)} MB, lang=${config.whisperLanguage})")
    }

    @Synchronized
    override fun transcribe(pcm: FloatArray): Transcription =
        transcribeInternal(pcm, useVad = true)

    private fun transcribeInternal(pcm: FloatArray, useVad: Boolean): Transcription {
        check(ctx != 0L) { "WhisperRecognizer.load() must succeed first" }
        // The multilingual model always gets an explicit language; the .en
        // model ignores it harmlessly.
        val lang = if (config.useEnglishModel) "en" else config.whisperLanguage
        val json = nativeTranscribe(
            ctx, pcm, lang, config.whisperInitialPrompt(), config.whisperThreads,
            if (useVad) vadModelPath else null,
            config.whisperVadThreshold,
            config.whisperVadMinSpeechMs,
            config.whisperVadMinSilenceMs,
            config.whisperVadSpeechPadMs,
            config.whisperVadMaxSpeechS,
            config.whisperVadOverlapS,
            config.whisperAudioCtx,
        ) ?: throw IllegalStateException(nativeLastError() ?: "transcribe failed")
        val o = JSONObject(json)
        val vadSpans = mutableListOf<Pair<Long, Long>>()
        o.optJSONArray("vad_segments_cs")?.let { arr ->
            for (i in 0 until arr.length()) {
                val span = arr.optJSONArray(i) ?: continue
                // centiseconds -> milliseconds on the input timeline
                vadSpans += span.getLong(0) * 10 to span.getLong(1) * 10
            }
        }
        return Transcription(
            text = o.optString("text").trim(),
            language = o.optString("language").takeIf { it.isNotBlank() },
            noSpeechProb = if (o.has("no_speech_prob"))
                o.getDouble("no_speech_prob").toFloat() else null,
            avgLogprob = if (o.has("avg_logprob"))
                o.getDouble("avg_logprob").toFloat() else null,
            durationSec = pcm.size.toFloat() / config.sampleRate,
            segmentCount = o.optInt("n_segments", 1),
            vadSegments = vadSpans,
            elapsedMs = o.optLong("elapsed_ms", 0L),
        )
    }

    /** True when whisper's internal VAD is armed for transcribe calls. */
    val nativeVadEnabled: Boolean get() = vadModelPath != null

    /** Max Silero speech probability in [pcm] — pure diagnostics so field
     *  logs show how far under the VAD threshold the audio lands. Returns
     *  -1 when the probe context is unavailable. */
    fun vadMaxProb(pcm: FloatArray): Float {
        val c = vadCtx
        return if (c != 0L) nativeVadMaxProb(c, pcm) else -1f
    }

    /** Diagnostic: same as [transcribe] but skips the internal VAD — used
     *  once per session to separate "VAD rejects the audio" from "the PCM
     *  itself carries no speech". */
    fun transcribeNoVad(pcm: FloatArray): Transcription =
        transcribeInternal(pcm, useVad = false)

    @Synchronized
    override fun release() {
        if (vadCtx != 0L) {
            nativeVadRelease(vadCtx)
            vadCtx = 0L
        }
        if (ctx != 0L) {
            nativeRelease(ctx)
            ctx = 0L
        }
        loaded.set(false)
    }

    private companion object {
        const val TAG = "GabVoice"
    }
}
