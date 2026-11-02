package com.example.gabai.services

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.Closeable
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * Shared ONNX Runtime environment — Silero VAD and the openWakeWord model
 * both run on it, so sessions share the thread pool and allocators.
 */
object Onnx {
    val environment: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
}

/** Per-frame speech probability source — [VadSegmenter] consumes this.
 *  Implementations take one [VoiceConfig.frameSamples] chunk of 16 kHz
 *  float PCM and return P(speech) in [0, 1]. */
interface FrameVad : Closeable {
    fun speechProb(frame: FloatArray): Float
    fun reset()
}

/**
 * Silero VAD (v5 ONNX) over ONNX Runtime, fully on-device.
 *
 * The model takes 512-sample (32 ms) frames at 16 kHz plus a recurrent
 * `state` tensor carried between calls, so probabilities are contextual,
 * not per-frame energy. Falls back to a safe "no speech" answer on session
 * errors so a broken VAD can never spam the recognizer.
 */
class SileroVad(
    private val modelFile: java.io.File,
    private val config: VoiceConfig = VoiceConfig.DEFAULT,
) : FrameVad {

    private var session: OrtSession? = null
    private var state = FloatArray(2 * 128)   // [2, 1, 128] h/c
    private val stateShape = longArrayOf(2, 1, 128)

    /** Opens the ONNX session on the shared environment. */
    fun load() {
        session = Onnx.environment.createSession(
            modelFile.absolutePath,
            OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
            },
        )
        Log.i(TAG, "silero vad loaded: ${modelFile.name} " +
            "inputs=${session?.inputNames} outputs=${session?.outputNames}")
    }

    companion object {
        private const val TAG = "GabVoice"
        private const val SAMPLE_RATE = 16_000L

        /** Resolves silero_vad.onnx via [VoiceConfig.resolveModelFile] and
         *  loads it. Throws when the model file is absent. */
        fun create(context: Context, config: VoiceConfig): SileroVad {
            val file = config.resolveModelFile(
                context, config.vadModelFile, minBytes = 10_000, tag = TAG,
            ) ?: throw IllegalStateException(
                "${config.vadModelFile} missing — expected in assets/models/ " +
                    "or external files dir"
            )
            return SileroVad(file, config).apply { load() }
        }
    }

    override fun speechProb(frame: FloatArray): Float {
        val s = session ?: return 0f
        try {
            OnnxTensor.createTensor(
                Onnx.environment, FloatBuffer.wrap(frame),
                longArrayOf(1, frame.size.toLong()),
            ).use { inputTensor ->
                OnnxTensor.createTensor(
                    Onnx.environment, FloatBuffer.wrap(state), stateShape,
                ).use { stateTensor ->
                    OnnxTensor.createTensor(Onnx.environment, SAMPLE_RATE).use { srTensor ->
                        val inputs = mutableMapOf<String, OnnxTensor>()
                        // Feed only what this export actually declares —
                        // keeps v4 ("sr" absent? no, still there) and v5
                        // layouts working.
                        for (name in s.inputNames) {
                            when (name) {
                                "input" -> inputs[name] = inputTensor
                                "state" -> inputs[name] = stateTensor
                                "sr" -> inputs[name] = srTensor
                            }
                        }
                        s.run(inputs).use { result ->
                            val prob = ((result.get("output").orElseGet { result.get(0) })
                                .value as Array<FloatArray>)[0][0]
                            @Suppress("UNCHECKED_CAST")
                            val newState = (result.get("stateN").orElseGet { result.get(1) })
                                .value as Array<Array<FloatArray>>
                            var i = 0
                            for (a in newState) for (b in a) for (v in b) state[i++] = v
                            return prob
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "vad inference failed, reporting no speech: ${e.message}")
            return 0f
        }
    }

    override fun reset() {
        state = FloatArray(2 * 128)
    }

    override fun close() {
        try { session?.close() } catch (_: Exception) {}
        session = null
    }
}

/**
 * ONNX-free fallback [FrameVad] used when silero_vad.onnx isn't on the
 * device (temporary, for exercising the Whisper path without the ONNX
 * models). Maps smoothed frame RMS energy onto a pseudo speech
 * probability so [VadSegmenter] still opens/closes segments — crude next
 * to Silero, but it keeps Whisper fed with speech-sized chunks.
 */
class EnergyVad : FrameVad {

    private var smoothed = 0f

    override fun speechProb(frame: FloatArray): Float {
        var sum = 0.0
        for (s in frame) sum += s * s
        val rms = sqrt(sum / frame.size).toFloat()
        // Phone-mic speech typically sits at RMS 0.02-0.1 while room noise
        // stays under ~0.005. Mapping [RMS_FLOOR, RMS_CEIL] onto [0, 1]
        // makes the segmenter's 0.5/0.35 thresholds act like a ~0.02 RMS
        // gate with hangover from the smoothing.
        val raw = ((rms - RMS_FLOOR) / (RMS_CEIL - RMS_FLOOR)).coerceIn(0f, 1f)
        smoothed = 0.5f * smoothed + 0.5f * raw
        return smoothed
    }

    override fun reset() {
        smoothed = 0f
    }

    override fun close() {}

    private companion object {
        const val RMS_FLOOR = 0.008f
        const val RMS_CEIL = 0.04f
    }
}
