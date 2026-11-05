package com.example.gabai.services

import android.content.Context
import android.media.MediaRecorder
import android.util.Log
import java.io.File

/**
 * Single source of truth for the GabVoice pipeline — every threshold, model
 * file name and path lives here. There are no magic numbers elsewhere in the
 * voice code; to tune on a real device, change these values.
 *
 * Model files resolve in two places, in order:
 *  1. `<external-files>/models/<name>` — for `adb push` during development
 *     (`adb push x.bin /sdcard/Android/data/com.example.gabai/files/models/`).
 *  2. `assets/models/<name>`, copied to `<files>/models/<name>` on first use.
 * Model binaries are gitignored; never commit them.
 */
/** Which speech detector gates Whisper. WHISPER_NATIVE runs whisper.cpp's
 *  built-in Silero VAD inside each rolling window (needs
 *  [VoiceConfig.whisperVadModelFile]); ONNX_SILERO is the original
 *  per-frame ONNX Silero + [VadSegmenter] pipeline kept as fallback. */
enum class VadMode { WHISPER_NATIVE, ONNX_SILERO }

data class VoiceConfig(

    // ---------- capture ----------
    val sampleRate: Int = 16_000,
    /** AudioRecord sources probed in order; the first that initializes
     *  wins. UNPROCESSED bypasses vendor audio chains (Honor MagicOS's
     *  libhnaudioenhancements mangles speech enough that VAD scores ~0
     *  while amp stays high); MIC is the universal fallback. Some devices
     *  initialize UNPROCESSED but still process — reorder if so. */
    val audioSources: List<Int> = listOf(
        MediaRecorder.AudioSource.UNPROCESSED,
        MediaRecorder.AudioSource.MIC,
    ),
    /** Silero VAD v5 wants 512-sample (32 ms) frames at 16 kHz. */
    val frameSamples: Int = 512,
    /** Software soft-AGC on the float PCM — raw UNPROCESSED audio can be
     *  quiet; normalizing keeps VAD and energy telemetry consistent. */
    val micNormalize: Boolean = true,
    val micTargetRms: Float = 0.05f,
    val micMaxGain: Float = 8f,
    /** Attach AudioEffect processors when the hardware exposes them.
     *  Off while diagnosing: Honor's vendor chain already processes MIC,
     *  and double-processing was flattening VAD scores to ~0. */
    val wantNoiseSuppressor: Boolean = false,
    val wantAcousticEchoCanceler: Boolean = false,
    val wantAutomaticGainControl: Boolean = false,
    /**
     * Reserved: a lightweight denoiser (RNNoise-style) before the VAD.
     * Off by default — no denoiser is bundled yet; enabling logs a warning.
     */
    val enableDenoiser: Boolean = false,

    // ---------- models / recognizer ----------
    /** Whisper Tiny English q5_1 (32 MB): about 3x faster and 2.5x smaller than Base q8 and enough for a short command set.
     *  For Filipino switch [useEnglishModel] off and bundle [modelFileMultilingual]. */
    val modelFileEn: String = "ggml-tiny.en-q5_1.bin",
    /** Whisper large-v3-turbo q5_0 (~570 MB): Filipino FLEURS WER ~11% vs
     *  ~26% for Small q5_1 and ~50%+ for Base, which is unusable for Tagalog.
     *  4-8x heavier on CPU/RAM than Small; fall back to
     *  "ggml-small-q5_1.bin" or "ggml-medium-q5_0.bin" if the device
     *  throttles. */
    val modelFileMultilingual: String = "ggml-large-v3-turbo-q5_0.bin",
    /** Set true for the English-only Base model (English commands only). */
    val useEnglishModel: Boolean = true,
    /** Explicit ISO-639 code passed to whisper.cpp — never auto-detect
     *  (short clips misdetect as Malay/Indonesian). "tl" keeps English words
     *  in Latin script, so Taglish and plain English both transcribe. */
    val whisperLanguage: String = "tl",
    /** Encoder threads. Capped at 4: Small is ~3x Base per decode and more
     *  threads on big.LITTLE mostly add heat and throttling. */
    val whisperThreads: Int =
        Runtime.getRuntime().availableProcessors().coerceIn(2, 4),
    /** Anything smaller than this is not a real ggml model. */
    val minModelBytes: Long = 10L * 1024 * 1024,
    /** assets/<modelAssetDir>/ then files/<modelFileDir>/ */
    val modelAssetDir: String = "models",
    val modelFileDir: String = "models",
    /** Optional encoder-context clamp passed to whisper_full
     *  (EXPERIMENTAL — halves encoder cost for short clips but can hurt
     *  quality). 0 keeps the full 1500-frame context. Reverted to 0 after
     *  field logs showed degraded output; re-enable only after verifying
     *  transcripts stay clean on the demo device. */
    val whisperAudioCtx: Int = 0,

    // ---------- whisper-native VAD ----------
    /** Primary speech detection: whisper.cpp runs its internal Silero VAD
     *  on each window and decodes only the speech spans it finds — when a
     *  window is silent the call is just the tiny VAD pass, so the mic can
     *  listen continuously without Whisper churning on noise. */
    val vadMode: VadMode = VadMode.WHISPER_NATIVE,
    /** ggml-format Silero VAD for whisper.cpp (NOT the .onnx file).
     *  Missing -> WHISPER_NATIVE degrades to ONNX_SILERO with a warning. */
    val whisperVadModelFile: String = "ggml-silero-v6.2.0.bin",
    val whisperVadThreshold: Float = 0.5f,
    val whisperVadMinSpeechMs: Int = 250,
    val whisperVadMinSilenceMs: Int = 300,
    val whisperVadSpeechPadMs: Int = 150,
    val whisperVadMaxSpeechS: Float = 12f,
    val whisperVadOverlapS: Float = 0.1f,
    /** Rolling window sent to whisper_full on every hop. 4.5 s covers a
     *  "hello gab <command>" utterance plus context; overlapping windows
     *  mean one-breath commands land in ~4 consecutive calls. Endpoint
     *  decodes send only the speech burst, so the window length only
     *  affects the hop path — shorter is cheaper per decode. */
    val listenWindowMs: Int = 4_500,
    /** Response latency is bounded by the hop + one decode — an 800 ms hop
     *  answers a command in ~1 s worst case even when endpointing misses.
     *  Costs a heavier decode duty cycle; heat is an accepted trade. */
    val listenHopMs: Int = 1_200,
    /** Early decode ("endpointing"): once voiced audio goes quiet for this
     *  long, the speech burst alone is transcribed (see queueBurst) instead
     *  of waiting for the hop boundary — commands fire the moment speech
     *  ends. Short enough to feel instant, long enough that a pause inside
     *  "go back" doesn't split the phrase. */
    val endpointSilenceMs: Int = 450,
    /** Min voiced audio in the window before an early decode may fire. */
    val endpointMinSpeechMs: Int = 400,
    /** RMS of normalized float PCM treated as "voiced" for endpointing. */
    val endpointVoicedRms: Float = 0.02f,
    /** Padding around the detected speech burst sent to the endpoint
     *  decode — keeps first/last syllables from being clipped. */
    val burstLeadPadMs: Int = 200,
    val burstTailPadMs: Int = 150,

    // ---------- Silero VAD segmentation ----------
    val vadModelFile: String = "silero_vad.onnx",
    /** Speech onset needs a higher bar than speech continuation. */
    val vadStartThreshold: Float = 0.5f,
    val vadContinueThreshold: Float = 0.35f,
    /** Rolling pre-roll so the first syllable is never clipped. */
    val vadLeadInMs: Int = 200,
    /** Audio kept after the model calls the segment over. */
    val vadTrailingPadMs: Int = 250,
    /** This much below-threshold audio closes a segment. */
    val vadSilenceCloseMs: Int = 500,
    /** Shorter segments are noise blips — dropped, never reach Whisper. */
    val vadMinSegmentMs: Int = 500,
    /** Longer segments are split at the quietest point, not mid-word. */
    val vadMaxSegmentMs: Int = 30_000,

    // ---------- wake word ----------
    /** openWakeWord ONNX export for "Hello Gab" — the classifier produced by
     *  openwakeword training ([1,16,96] embedding input, or [1,T,32,1]
     *  melspec input). Supply this file — when it (or the melspec model) is
     *  missing the transcript fallback (below) is used instead. */
    val wakeModelFile: String = "hello_gab.onnx",
    /** Official openwakeword feature models — the detector runs the exact
     *  openwakeword chain melspec -> embedding -> classifier. Required. */
    val wakeMelspecModelFile: String = "melspectrogram.onnx",
    /** Required for embedding-input classifiers (the normal openwakeword
     *  export); optional for melspec-input classifiers. */
    val wakeEmbeddingModelFile: String = "embedding_model.onnx",
    val wakeThreshold: Float = 0.5f,
    val wakeCooldownMs: Long = 2_500,
    /** How often a wake score is computed while audio streams in. The
     *  classifier sees ~2 s of context internally (16 embeddings x 80 ms
     *  steps over a 76-frame mel window). */
    val wakeHopMs: Int = 80,
    /** Fallback only: consecutive wake transcripts required before firing.
     *  1 while the ONNX wake model is absent — a single "hello" opens the
     *  command window; raise back to 2 if noise false-wakes. */
    val fallbackWakeConsecutive: Int = 1,

    // ---------- Whisper rejection filters ----------
    val maxNoSpeechProb: Float = 0.6f,
    val minAvgLogprob: Float = -1.0f,
    /** Blocklisted text still passes when the clip is long AND confident —
     *  a real sentence can legitimately start with "okay". */
    val blocklistMaxClipSec: Float = 4.0f,
    /** Well-known Whisper silence/noise hallucinations. */
    val hallucinationBlocklist: Set<String> = setOf(
        "thank you", "thank you so much", "thanks", "thanks for watching",
        "thank you for watching", "okay", "ok", "o k", "you", "bye",
        "bye bye", "goodbye", "hello", "huh", "mm", "hmm", "mhm",
        "um", "uh", "oh", "ah", "yeah", "yes", "no", "music",
        "applause", "silence", "subtitles", "subtitle", "copyright",
        "so", "the end", "i'm sorry", "sorry",
    ),

    // ---------- command matching ----------
    /** Whole-utterance similarity needed for a fuzzy command match. */
    val commandMatchThreshold: Float = 0.72f,
    /** Session-ending/destructive commands ("stop", "cancel", "exit") need
     *  a near-exact match — a fuzzy near-miss must never end a session. */
    val destructiveMatchThreshold: Float = 0.88f,
    val destructiveCommands: Set<Command> = setOf(Command.STOP),

    // ---------- wake interaction ----------
    /** How long the command window stays open after the wake ding. */
    val commandWindowMs: Long = 10_000,
    /** Short spoken prompt right after the wake ding tells the user the
     *  command window is open. Empty disables it (ding only). */
    val wakePrompt: String = "I'm listening",
    /** Spoken when the command window lapses unanswered — invites a retry
     *  instead of leaving silence after the closing blip. Empty disables. */
    val windowTimeoutPrompt: String = "Please try again.",

    // ---------- debug ----------
    /** Save VAD-rejected and filter-rejected clips under
     *  files/rejected_clips/ for offline replay (see ReplayHarness). */
    val saveRejectedClips: Boolean = false,
    val rejectedClipDir: String = "rejected_clips",
    /** Bring-up aid: dump the first few full windows as WAVs under
     *  <external-files>/debug_windows/ so `adb pull` can answer "is the
     *  PCM actually speech?" when VAD reports 0 with healthy amplitude.
     *  Also runs one no-VAD whisper probe per session (see
     *  [WhisperRecognizer.transcribeNoVad]). Disable once audio is good. */
    val dumpDebugWindows: Boolean = false,
    val debugWindowsDir: String = "debug_windows",
    val dumpWindowCount: Int = 3,
    /** Live transcript ("Show what I say"): rolling window of raw mic
     *  audio transcribed on a fixed hop, independent of VAD/wake gating —
     *  keeps captions and commands alive on devices where Silero never
     *  crosses the start threshold. */
    val liveCaptionWindowMs: Int = 4_000,
    val liveCaptionHopMs: Int = 1_000,
) {

    val whisperModelFile: String
        get() = if (useEnglishModel) modelFileEn else modelFileMultilingual

    /** Bias prompt handed to whisper.cpp — wake-word variants ONLY. Biasing
     *  toward the full command synonym list made the decoder echo those
     *  exact words out of noise (see [CommandParser.WAKE_KEYWORDS]); field
     *  logs showed "fast speech"/"go"/"back" loops on silence. */
    fun whisperInitialPrompt(): String =
        "GabAi voice commands. " + CommandParser.WAKE_KEYWORDS.joinToString(", ")

    /**
     * Resolves [fileName] to a readable on-disk model file:
     * external-files override first (dev push), else the filesDir copy,
     * else a one-time copy out of assets. Returns null and logs when the
     * file is missing or implausibly small.
     */
    fun resolveModelFile(
        context: Context,
        fileName: String,
        minBytes: Long = 1L,
        tag: String = "GabVoice",
        /** Expected-absent models (e.g. the optional ONNX wake classifier)
         *  log quietly instead of an error every service start. */
        optional: Boolean = false,
    ): File? {
        val app = context.applicationContext
        val external = app.getExternalFilesDir(modelFileDir)?.let { File(it, fileName) }
        if (external != null && isUsableModel(external, minBytes, tag)) return external

        val local = File(app.filesDir, "$modelFileDir/$fileName")
        if (isUsableModel(local, minBytes, tag)) return local

        // First launch: copy out of assets (noCompress → mmap-friendly read).
        val assetPath = "$modelAssetDir/$fileName"
        try {
            app.assets.open(assetPath).use { input ->
                local.parentFile?.mkdirs()
                local.outputStream().use { output -> input.copyTo(output) }
            }
            Log.i(tag, "copied asset $assetPath -> ${local.absolutePath}")
        } catch (e: Exception) {
            val msg = "model $fileName not found in assets/models or " +
                "external files dir: ${e.message}"
            if (optional) Log.d(tag, msg) else Log.e(tag, msg)
            return null
        }
        return if (isUsableModel(local, minBytes, tag)) local else null
    }

    private fun isUsableModel(f: File, minBytes: Long, tag: String): Boolean {
        if (!f.isFile) return false
        if (f.length() < minBytes) {
            Log.e(tag, "model ${f.absolutePath} is only ${f.length()} bytes " +
                "(need >= $minBytes) — refusing to load")
            return false
        }
        return true
    }

    companion object {
        val DEFAULT = VoiceConfig()
    }
}
