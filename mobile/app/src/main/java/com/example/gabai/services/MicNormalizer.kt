package com.example.gabai.services

import kotlin.math.abs
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Streaming soft-AGC for float mic frames. UNPROCESSED audio can arrive
 * far below the levels Silero VAD and the whisper VAD were trained on;
 * this normalizes toward [VoiceConfig.micTargetRms] with gain capped at
 * [VoiceConfig.micMaxGain]. Gain falls fast on loud input (no clipping)
 * and recovers slowly, so bursts don't pump the noise floor.
 */
class MicNormalizer(private val config: VoiceConfig = VoiceConfig.DEFAULT) {

    private var gain = 1f

    /** Applies the current gain to [frame] in place, then updates it from
     *  the frame's RMS for the next call. */
    fun process(frame: FloatArray) {
        val g = gain
        if (g != 1f) {
            for (i in frame.indices) {
                frame[i] = (frame[i] * g).coerceIn(-1f, 1f)
            }
        }
        var sum = 0.0
        var peak = 0f
        for (s in frame) {
            sum += s * s
            val a = abs(s)
            if (a > peak) peak = a
        }
        val rms = sqrt(sum / frame.size).toFloat()
        val target = if (rms > 1e-4f)
            min(config.micMaxGain, config.micTargetRms / rms)
        else config.micMaxGain
        // Never amplify a frame into clipping — loud frames drop gain fast.
        val clipSafe = if (peak * g > 0.9f) 0.9f / peak else Float.MAX_VALUE
        val wanted = min(target, clipSafe)
        gain = if (wanted < gain)
            gain * 0.5f + wanted * 0.5f       // fast attack: protect loud speech
        else
            gain * 0.98f + wanted * 0.02f     // slow release up
        gain = gain.coerceIn(1f, config.micMaxGain)
    }

    fun reset() {
        gain = 1f
    }
}
