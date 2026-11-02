package com.example.gabai.services

import android.content.Context
import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.util.Locale

/**
 * Post-Whisper rejection gate — every transcript passes through here before
 * it can touch the command matcher. Thresholds live in [VoiceConfig];
 * [Decision.reason] is logged for every result so field debugging shows why
 * a clip was dropped.
 */
object TranscriptFilter {

    data class Decision(val accepted: Boolean, val reason: String)

    /**
     * [skipBlocklist] — the transcript-fallback wake path calls this with
     * true for wake-only transcripts, because "hello" is itself on the
     * blocklist and would otherwise eat every fallback wake attempt. Only
     * the blocklist is skipped; every other rule still applies.
     */
    fun evaluate(
        t: Transcription,
        vadConfirmed: Boolean,
        cfg: VoiceConfig,
        skipBlocklist: Boolean = false,
    ): Decision {
        if (!vadConfirmed) return Decision(false, "vad_no_speech")

        t.noSpeechProb?.let {
            if (it >= cfg.maxNoSpeechProb)
                return Decision(false, "no_speech_prob=%.2f".format(it))
        }
        t.avgLogprob?.let {
            if (it < cfg.minAvgLogprob)
                return Decision(false, "avg_logprob=%.2f".format(it))
        }

        val norm = normalize(t.text)
        if (norm.isEmpty()) return Decision(false, "empty")

        if (isRepeatedGarbage(norm)) return Decision(false, "repeated_tokens")
        if (isRepetitionLoop(norm)) return Decision(false, "repetition_loop")

        if ((cfg.whisperLanguage == "en" || cfg.whisperLanguage == "tl") && hasNonLatinLetter(t.text))
            return Decision(false, "non_latin")

        if (!skipBlocklist && norm in cfg.hallucinationBlocklist) {
            val short = t.durationSec <= cfg.blocklistMaxClipSec
            val lowConf = (t.noSpeechProb ?: 0f) > 0.35f ||
                (t.avgLogprob ?: 0f) < -0.6f
            if (short || lowConf) {
                return Decision(false, "blocklist_hallucination")
            }
        }
        return Decision(true, "accepted")
    }

    /** Canonical log line for one transcription — keep it parseable. */
    fun log(tag: String, t: Transcription, vadConfirmed: Boolean, d: Decision) {
        Log.d(tag, ("whisper text=\"${t.text}\" dur=%.1fs lang=${t.language} " +
            "no_speech=%s avg_logprob=%s vad=$vadConfirmed -> %s").format(
                t.durationSec,
                t.noSpeechProb?.let { "%.2f".format(it) } ?: "n/a",
                t.avgLogprob?.let { "%.2f".format(it) } ?: "n/a",
                d.reason,
            ))
    }

    /** Lowercased, punctuation-stripped, whitespace-collapsed. */
    fun normalize(raw: String): String =
        raw.lowercase(Locale.ROOT)
            .replace(Regex("[^\\p{L}\\p{N} ]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()

    /** One token repeated, or fewer than 2 distinct letters — the classic
     *  noise decode ("you you you", "!!!!"). */
    private fun isRepeatedGarbage(norm: String): Boolean {
        val tokens = norm.split(' ')
        if (tokens.size >= 3 && tokens.all { it == tokens[0] }) return true
        return norm.replace(" ", "").toSet().size < 2 && norm.length >= 3
    }

    /** One token that is itself a short unit tiled 4+ times —
     *  "gabababababab…" (whisper loop inside a single word). */
    private val TILED_TOKEN = Regex("^(.{1,4})\\1{3,}$")

    /**
     * Whisper's decoder loop on noise: an n-gram repeated over and over.
     * Field logs: "fast speech" x60, "go" x100, "back" x200, alternating
     * "fast speech slow down slower" cycles, lead-ins like "no go go go…".
     * Rejects when any 1–4-token phrase appears ≥6 times total, ≥4 times
     * consecutively, or a single long token is a tiled unit. Deliberate
     * repeats ("back back", "say again") stay far under the bars.
     */
    private fun isRepetitionLoop(norm: String): Boolean {
        val tokens = norm.split(' ').filter { it.isNotBlank() }
        // A single long token that is a short unit tiled ("gabababab…").
        if (tokens.any { it.length >= 12 && TILED_TOKEN.matches(it) })
            return true
        if (tokens.size < 6) return false

        // Total-count rule — catches alternating patterns ("a b c a b c…")
        // that the consecutive rule misses by one token.
        val counts = HashMap<List<String>, Int>()
        for (i in tokens.indices) {
            for (n in 1..4) {
                if (i + n > tokens.size) break
                if ((counts.merge(tokens.subList(i, i + n), 1, Int::plus)
                        ?: 0) >= 6
                ) return true
            }
        }

        // Consecutive-run rule — a phrase looped back-to-back ≥4 times.
        for (n in 1..4) {
            var run = 1
            var j = n
            while (j + n <= tokens.size) {
                if (tokens.subList(j, j + n) == tokens.subList(j - n, j)) {
                    if (++run >= 4) return true
                    j += n
                } else {
                    run = 1
                    j++
                }
            }
        }
        return false
    }

    private fun hasNonLatinLetter(text: String): Boolean =
        text.any { ch ->
            Character.isLetter(ch) &&
                Character.UnicodeScript.of(ch.code).let {
                    it != Character.UnicodeScript.LATIN &&
                        it != Character.UnicodeScript.COMMON &&
                        it != Character.UnicodeScript.INHERITED
                }
        }
}

/**
 * Debug helper — writes rejected clips as 16 kHz mono WAV under
 * files/rejected_clips/ so they can be replayed offline (see ReplayHarness).
 * Enabled only via [VoiceConfig.saveRejectedClips].
 */
class ClipStore(private val context: Context, private val config: VoiceConfig) {

    /** Bring-up aid: dumps a raw window WAV under
     *  `<external-files>/debug_windows/` so it can be pulled with a plain
     *  `adb pull` — no run-as/root needed. */
    fun saveProbe(pcm: FloatArray, name: String) {
        try {
            val dir = File(
                context.getExternalFilesDir(null), config.debugWindowsDir
            ).apply { mkdirs() }
            val f = File(dir, name)
            writeWav(f, pcm, config.sampleRate)
            Log.i("GabVoice", "probe wav -> ${f.absolutePath}")
        } catch (e: Exception) {
            Log.w("GabVoice", "probe wav failed: ${e.message}")
        }
    }

    fun save(pcm: FloatArray, reason: String) {
        if (!config.saveRejectedClips) return
        try {
            val dir = File(context.filesDir, config.rejectedClipDir).apply { mkdirs() }
            val name = "reject_${System.currentTimeMillis()}_${reason.take(24)}.wav"
            writeWav(File(dir, name), pcm, config.sampleRate)
        } catch (e: Exception) {
            Log.w("GabVoice", "clip save failed: ${e.message}")
        }
    }

    /** Minimal PCM16 WAV writer. */
    private fun writeWav(f: File, pcm: FloatArray, sampleRate: Int) {
        val dataSize = pcm.size * 2
        RandomAccessFile(f, "rw").use { raf ->
            raf.writeBytes("RIFF")
            raf.writeInt(Integer.reverseBytes(36 + dataSize))
            raf.writeBytes("WAVEfmt ")
            raf.writeInt(Integer.reverseBytes(16))
            raf.writeShort(java.lang.Short.reverseBytes(1.toShort()).toInt())
            raf.writeShort(java.lang.Short.reverseBytes(1.toShort()).toInt())
            raf.writeInt(Integer.reverseBytes(sampleRate))
            raf.writeInt(Integer.reverseBytes(sampleRate * 2))
            raf.writeShort(java.lang.Short.reverseBytes(2.toShort()).toInt())
            raf.writeShort(java.lang.Short.reverseBytes(16.toShort()).toInt())
            raf.writeBytes("data")
            raf.writeInt(Integer.reverseBytes(dataSize))
            for (s in pcm) {
                val v = (s.coerceIn(-1f, 1f) * 32767f).toInt().toShort()
                raf.writeShort(java.lang.Short.reverseBytes(v).toInt())
            }
        }
    }
}
