package com.example.gabai.services

import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.content.Context
import android.os.SystemClock
import android.util.Log
import java.io.Closeable
import java.nio.FloatBuffer

/**
 * Always-on wake word spotter for "Hello Gab". Runs on the raw mic stream —
 * cheap enough to evaluate every [VoiceConfig.wakeHopMs] — while Whisper
 * stays asleep. Implementations feed fixed 32 ms frames through [accept]
 * and return true once per detection ([VoiceConfig.wakeCooldownMs] applies).
 */
interface WakeWordDetector : Closeable {
    fun accept(frame: FloatArray): Boolean
    fun reset()
    override fun close() {}
}

/**
 * openWakeWord ONNX pipeline — replicates the inference chain of the
 * openwakeword python runtime (v0.5/0.6) exactly:
 *
 *   pcm -> melspectrogram.onnx ([1,N] raw int16-as-float -> (T,32),
 *          transform x/10+2)
 *       -> embedding_model.onnx ([1,76,32,1] -> 96-dim row; one embedding
 *          per 8 mel frames, produced once per 1280-sample block)
 *       -> <wakeword classifier>.onnx ([1,T,96] -> score)
 *
 * Classifiers exported against raw melspec input instead of embeddings
 * (rank-4 input ending in 32 mel bins) are detected by input rank and fed
 * the mel buffer directly — embedding_model.onnx is then not required.
 *
 * All model files come from config and resolve through
 * [VoiceConfig.resolveModelFile]; when the classifier or melspec model is
 * missing the factory returns null and the transcript fallback is used.
 */
class OnnxWakeWordDetector(
    private val config: VoiceConfig,
    private val clsSession: OrtSession,
    private val melSession: OrtSession,
    private val embSession: OrtSession?,
    private val clsInputName: String,
    private val melInputName: String,
    private val embInputName: String?,
    private val clsIsEmbedding: Boolean,
    private val clsFrames: Int,
) : WakeWordDetector {

    // Raw int16-scale audio history; melspec needs MEL_LOOKBACK context
    // samples before each block, so blocks are not independent windows.
    private val raw = FloatArray(RAW_CAPACITY)
    private var rawLen = 0

    private var pending = 0                    // unprocessed samples
    private var sinceLastScore = 0
    private var lastTriggerAtMs = 0L

    private val melRows = ArrayDeque<FloatArray>()   // (32,), transformed
    private val featRows = ArrayDeque<FloatArray>()  // (96,)

    override fun accept(frame: FloatArray): Boolean {
        for (s in frame) pushRaw(s * 32768f)
        pending += frame.size
        processBlocks()

        sinceLastScore += frame.size
        val hopSamples = config.sampleRate * config.wakeHopMs / 1000
        if (sinceLastScore < hopSamples) return false
        sinceLastScore = 0

        // Warmup: the classifier needs clsFrames embeddings (~2 s of audio).
        if (clsIsEmbedding && featRows.size < clsFrames) return false
        if (!clsIsEmbedding && melRows.size < clsFrames) return false

        val now = SystemClock.elapsedRealtime()
        if (now - lastTriggerAtMs < config.wakeCooldownMs) return false

        val score = try { classify() } catch (e: Exception) {
            Log.w(TAG, "wake inference failed: ${e.message}")
            return false
        }
        Log.v(TAG, "wake score %.3f".format(score))
        if (score >= config.wakeThreshold) {
            lastTriggerAtMs = now
            Log.i(TAG, "wake word fired (score=%.3f)".format(score))
            return true
        }
        return false
    }

    /** Convert all complete 1280-sample blocks into mel rows + embeddings. */
    private fun processBlocks() {
        val n = pending - pending % BLOCK
        if (n < BLOCK) return
        pending -= n

        melRows.addAll(runMel(lastRaw(n + MEL_LOOKBACK)))
        while (melRows.size > MEL_KEEP) melRows.removeFirst()

        if (embSession != null) {
            // i counts back from the oldest block: windows run oldest->newest
            for (i in n / BLOCK - 1 downTo 0) {
                val end = melRows.size - MEL_STRIDE * i
                val start = end - EMB_WINDOW
                if (start >= 0) {
                    val w = FloatArray(EMB_WINDOW * MEL_BINS)
                    var p = 0
                    for (r in start until end) {
                        System.arraycopy(melRows.elementAt(r), 0, w, p, MEL_BINS)
                        p += MEL_BINS
                    }
                    featRows.addLast(runEmbedding(w))
                }
            }
            while (featRows.size > FEAT_KEEP) featRows.removeFirst()
        }
    }

    private fun pushRaw(v: Float) {
        if (rawLen == RAW_CAPACITY) {
            System.arraycopy(raw, 1, raw, 0, RAW_CAPACITY - 1)
            rawLen = RAW_CAPACITY - 1
        }
        raw[rawLen++] = v
    }

    /** Newest [count] samples as int16-scale floats, oldest first. */
    private fun lastRaw(count: Int): FloatArray {
        val c = minOf(count, rawLen)
        val out = FloatArray(c)
        System.arraycopy(raw, rawLen - c, out, 0, c)
        return out
    }

    /** melspectrogram.onnx: [1, N] raw int16-as-float -> rows of 32, x/10+2. */
    private fun runMel(audio: FloatArray): List<FloatArray> {
        OnnxTensor.createTensor(
            Onnx.environment, FloatBuffer.wrap(audio),
            longArrayOf(1, audio.size.toLong()),
        ).use { t ->
            melSession.run(mapOf(melInputName to t)).use { out ->
                val fb = (out.get(0).value as OnnxTensor).floatBuffer
                val rows = ArrayList<FloatArray>(fb.remaining() / MEL_BINS)
                while (fb.remaining() >= MEL_BINS) {
                    val row = FloatArray(MEL_BINS)
                    for (b in 0 until MEL_BINS) row[b] = fb.get() / 10f + 2f
                    rows.add(row)
                }
                return rows
            }
        }
    }

    /** embedding_model.onnx: [1,76,32,1] -> 96-dim row. */
    private fun runEmbedding(window: FloatArray): FloatArray {
        OnnxTensor.createTensor(
            Onnx.environment, FloatBuffer.wrap(window),
            longArrayOf(1, EMB_WINDOW.toLong(), MEL_BINS.toLong(), 1),
        ).use { t ->
            embSession!!.run(mapOf(embInputName!! to t)).use { out ->
                val fb = (out.get(0).value as OnnxTensor).floatBuffer
                val row = FloatArray(EMB_DIM)
                fb.get(row)
                return row
            }
        }
    }

    private fun classify(): Float {
        val tensor = if (clsIsEmbedding) {
            val data = FloatArray(clsFrames * EMB_DIM)
            val have = minOf(clsFrames, featRows.size)
            for (k in 0 until have) {
                val row = featRows.elementAt(featRows.size - have + k)
                System.arraycopy(
                    row, 0, data, (clsFrames - have + k) * EMB_DIM, EMB_DIM,
                )
            }
            OnnxTensor.createTensor(
                Onnx.environment, FloatBuffer.wrap(data),
                longArrayOf(1, clsFrames.toLong(), EMB_DIM.toLong()),
            )
        } else {
            val data = FloatArray(clsFrames * MEL_BINS)
            val have = minOf(clsFrames, melRows.size)
            for (k in 0 until have) {
                val row = melRows.elementAt(melRows.size - have + k)
                System.arraycopy(
                    row, 0, data, (clsFrames - have + k) * MEL_BINS, MEL_BINS,
                )
            }
            OnnxTensor.createTensor(
                Onnx.environment, FloatBuffer.wrap(data),
                longArrayOf(1, clsFrames.toLong(), MEL_BINS.toLong(), 1),
            )
        }
        tensor.use { t ->
            clsSession.run(mapOf(clsInputName to t)).use { out ->
                val fb = (out.get(0).value as OnnxTensor).floatBuffer
                return if (fb.hasRemaining()) fb.get() else 0f
            }
        }
    }

    override fun reset() {
        rawLen = 0
        pending = 0
        sinceLastScore = 0
        melRows.clear()
        featRows.clear()
    }

    override fun close() {
        try { clsSession.close() } catch (_: Exception) {}
        try { melSession.close() } catch (_: Exception) {}
        try { embSession?.close() } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "GabVoice"
        private const val BLOCK = 1280          // 80 ms of new audio per step
        private const val MEL_LOOKBACK = 480    // context prepended to melspec
        private const val RAW_CAPACITY = 16_000 * 4
        private const val MEL_BINS = 32
        private const val MEL_STRIDE = 8        // mel frames per 80 ms block
        private const val MEL_KEEP = 160
        private const val EMB_WINDOW = 76
        private const val EMB_DIM = 96
        private const val FEAT_KEEP = 120
        private const val DEFAULT_CLS_FRAMES = 16

        /** Loads the wakeword classifier + openwakeword feature models; null
         *  when a required file is missing (caller then uses the transcript
         *  fallback) or unreadable. */
        fun create(context: Context, config: VoiceConfig): OnnxWakeWordDetector? {
            // Optional by design — the transcript-fallback wake path is the
            // normal one until a trained hello_gab.onnx is shipped.
            val clsFile = config.resolveModelFile(
                context, config.wakeModelFile, minBytes = 1_000, tag = TAG,
                optional = true,
            ) ?: return null
            val melFile = config.resolveModelFile(
                context, config.wakeMelspecModelFile, minBytes = 100_000, tag = TAG,
            )
            if (melFile == null) {
                Log.e(TAG, "${config.wakeMelspecModelFile} missing — cannot " +
                    "run ONNX wakeword, falling back to transcript wake")
                return null
            }
            return try {
                val cls = Onnx.environment.createSession(clsFile.absolutePath)
                val mel = Onnx.environment.createSession(melFile.absolutePath)
                val inputName = cls.inputNames.first()
                val dims = (cls.inputInfo[inputName]?.info
                    as? ai.onnxruntime.TensorInfo)?.shape
                // Embedding classifiers: rank-3 [1,T,96]. Melspec
                // classifiers: rank-4 [1,T,32,1].
                val isEmbedding = (dims?.size ?: 3) == 3
                val frames = if (dims != null && dims.size >= 2 && dims[1] > 0)
                    dims[1].toInt() else DEFAULT_CLS_FRAMES
                var emb: OrtSession? = null
                var embName: String? = null
                if (isEmbedding) {
                    val embFile = config.resolveModelFile(
                        context, config.wakeEmbeddingModelFile,
                        minBytes = 100_000, tag = TAG,
                    )
                    if (embFile == null) {
                        Log.e(TAG, "${config.wakeEmbeddingModelFile} missing " +
                            "— cannot run ONNX wakeword")
                        cls.close(); mel.close()
                        return null
                    }
                    emb = Onnx.environment.createSession(embFile.absolutePath)
                    embName = emb.inputNames.first()
                }
                Log.i(TAG, "wake model loaded: ${clsFile.name} input=$inputName " +
                    "shape=${dims?.contentToString()} " +
                    "kind=${if (isEmbedding) "embedding" else "melspec"} " +
                    "frames=$frames")
                OnnxWakeWordDetector(
                    config, cls, mel, emb,
                    inputName, mel.inputNames.first(), embName,
                    isEmbedding, frames,
                )
            } catch (e: Exception) {
                Log.e(TAG, "wake model failed to load: ${e.message}")
                null
            }
        }
    }
}
