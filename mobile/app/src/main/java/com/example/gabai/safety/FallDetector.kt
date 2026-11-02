package com.example.gabai.safety

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.sqrt

/** A possible fall, scored 0..1. Only a candidate: the user gets a countdown to cancel before anyone is alerted. */
data class FallCandidate(
    val confidence: Float,
    val peakG: Float,
    val freefallMs: Long,
    val gyroPeakDps: Float,
    val stillSeconds: Float,
    val orientationChangeDeg: Float,
    val wasMovingBefore: Boolean?,
    val atMs: Long,
) {
    fun evidence(): Map<String, Any?> = mapOf(
        "peak_g" to peakG, "freefall_ms" to freefallMs, "gyro_peak_dps" to gyroPeakDps,
        "still_s" to stillSeconds, "orientation_change_deg" to orientationChangeDeg, "was_moving" to wasMovingBefore,
    )
}

/**
 * Impact-first fall detector on accelerometer (+ optional gyro) samples. A bump alone never qualifies: a fall needs a hard
 * impact AND the phone staying still afterwards, and an impact on a phone that was lying at rest (dropped off a table)
 * is penalised. Time comes from the samples, so it is unit-tested with synthetic signals.
 */
class FallDetector(private val onCandidate: (FallCandidate) -> Unit) {

    private class Sample(val t: Long, val g: Float, val gyro: Float, val ux: Float, val uy: Float, val uz: Float)

    private val buf = ArrayDeque<Sample>()
    private var impactAt = -1L
    private var peak = 0f
    private var quietUntil = 0L

    /** [gyroDps] is NaN when the device has no gyroscope. */
    fun onSample(tMs: Long, ax: Float, ay: Float, az: Float, gyroDps: Float) {
        val mag = sqrt(ax * ax + ay * ay + az * az)
        if (mag < 1e-3f) return
        buf.addLast(Sample(tMs, mag / G, gyroDps, ax / mag, ay / mag, az / mag))
        while (buf.isNotEmpty() && tMs - buf.first().t > KEEP_MS) buf.removeFirst()

        val g = mag / G
        if (impactAt < 0) {
            if (g >= IMPACT_G && tMs >= quietUntil) {
                impactAt = tMs
                peak = g
            }
            return
        }
        if (tMs - impactAt <= PEAK_WINDOW_MS && g > peak) peak = g
        if (tMs - impactAt >= ASSESS_AFTER_MS) {
            assess(impactAt, peak)?.let(onCandidate)
            quietUntil = tMs + COOLDOWN_MS
            impactAt = -1
        }
    }

    private fun window(from: Long, to: Long) = buf.filter { it.t in from..to }

    private fun assess(at: Long, peakG: Float): FallCandidate? {
        val post = window(at + 700, at + ASSESS_AFTER_MS)
        if (post.size < 10) return null
        val still = post.count { abs(it.g - 1f) < 0.12f && (it.gyro.isNaN() || it.gyro < 40f) }
        val stillFrac = still.toFloat() / post.size
        val stillSeconds = (ASSESS_AFTER_MS - 700) / 1000f * stillFrac

        var freefall = 0L
        var runStart = -1L
        for (s in window(at - 1000, at)) {
            if (s.g < FREEFALL_G) {
                if (runStart < 0) runStart = s.t
                freefall = maxOf(freefall, s.t - runStart)
            } else runStart = -1
        }

        val gyroPeak = window(at - 300, at + 300).map { it.gyro }.filter { !it.isNaN() }.maxOrNull() ?: 0f

        val before = window(at - 4000, at - 1000)
        val movingBefore: Boolean? = if (before.size >= 30) {
            val m = before.map { it.g }.average()
            val sd = sqrt(before.map { (it.g - m) * (it.g - m) }.average()).toFloat()
            if (sd < 0.04f) false else if (sd >= 0.15f) true else null
        } else null

        val pre = window(at - 1500, at - 500)
        val tail = window(at + ASSESS_AFTER_MS - 1000, at + ASSESS_AFTER_MS)
        val turn = if (pre.isEmpty() || tail.isEmpty()) 0f else angleDeg(mean(pre), mean(tail))

        var c = 0f
        c += if (peakG >= 5f) 0.4f else if (peakG >= 3.5f) 0.35f else 0.25f
        c += if (freefall >= 200) 0.15f else if (freefall >= 80) 0.1f else 0f
        c += if (gyroPeak >= 400f) 0.15f else if (gyroPeak >= 200f) 0.1f else 0f
        c += if (stillFrac >= 0.8f) 0.3f else if (stillFrac >= 0.6f) 0.15f else 0f
        if (turn >= 40f) c += 0.1f
        if (movingBefore == true) c += 0.1f
        if (movingBefore == false) c -= 0.5f
        c = c.coerceIn(0f, 1f)
        if (c < MIN_CONFIDENCE || stillFrac < 0.6f) return null
        return FallCandidate(c, peakG, freefall, gyroPeak, stillSeconds, turn, movingBefore, at)
    }

    private fun mean(w: List<Sample>) = floatArrayOf(
        w.map { it.ux }.average().toFloat(), w.map { it.uy }.average().toFloat(), w.map { it.uz }.average().toFloat())

    private fun angleDeg(a: FloatArray, b: FloatArray): Float {
        val dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        val na = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        val nb = sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
        if (na < 1e-3f || nb < 1e-3f) return 0f
        return Math.toDegrees(acos((dot / (na * nb)).coerceIn(-1f, 1f).toDouble())).toFloat()
    }

    companion object {
        const val G = 9.81f
        const val IMPACT_G = 2.5f
        const val FREEFALL_G = 0.5f
        const val MIN_CONFIDENCE = 0.6f
        private const val PEAK_WINDOW_MS = 400L
        private const val ASSESS_AFTER_MS = 3_500L
        private const val COOLDOWN_MS = 30_000L
        private const val KEEP_MS = 10_000L
    }
}
