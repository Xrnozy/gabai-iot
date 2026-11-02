package com.example.gabai.services

import android.content.Context
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

/**
 * On-device speech-to-text backed by Cactus Compute's Whistle model running on
 * the Needle engine. The engine is process-global and not thread-safe, so all
 * calls are serialised through this object.
 */
object Whistle {

    private const val MODEL_ASSET = "whistle.cact"
    private val loaded = AtomicBoolean(false)

    init {
        System.loadLibrary("whistle_jni")
    }

    private external fun nativeLoad(cact: ByteArray): Int
    private external fun nativeModels(): Int
    private external fun nativeTranscribe(
        pcm: FloatArray,
        language: String?,
        keywords: String?,
    ): String?
    private external fun nativeLastError(): String?

    /** Loads whistle.cact from assets. Safe to call repeatedly; loads once. */
    @Synchronized
    fun load(context: Context) {
        if (loaded.get()) return
        val bytes = context.applicationContext.assets.open(MODEL_ASSET).use { it.readBytes() }
        if (nativeLoad(bytes) < 0) {
            throw IllegalStateException(nativeLastError() ?: "whistle failed to load")
        }
        loaded.set(true)
    }

    data class Result(
        val text: String,
        val language: String?,
        val words: List<Word>,
    )

    data class Word(val word: String, val start: Double, val end: Double, val probability: Double)

    /**
     * Transcribes 16 kHz mono float PCM in [-1, 1]. [language] is an ISO 639-1
     * code from Whistle's supported set, or null to auto-detect. [keywords] are
     * words/phrases to favour during decoding.
     */
    @Synchronized
    fun transcribe(
        pcm: FloatArray,
        language: String? = null,
        keywords: List<String> = emptyList(),
    ): Result {
        check(loaded.get()) { "Whistle.load() must succeed before transcribe()" }
        val keywordsArg = keywords.takeIf { it.isNotEmpty() }?.joinToString("\n")
        val json = nativeTranscribe(pcm, language, keywordsArg)
            ?: throw IllegalStateException(nativeLastError() ?: "whistle transcription failed")
        val parsed = JSONObject(json)
        val wordsJson = parsed.optJSONArray("words")
        val words = buildList {
            if (wordsJson != null) {
                for (i in 0 until wordsJson.length()) {
                    val w = wordsJson.getJSONObject(i)
                    add(Word(w.getString("word"), w.getDouble("start"),
                             w.getDouble("end"), w.getDouble("probability")))
                }
            }
        }
        return Result(
            text = parsed.optString("text"),
            language = parsed.optString("language").takeIf { it.isNotBlank() },
            words = words,
        )
    }
}
