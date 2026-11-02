package com.example.gabai.safety

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class FallDetectorTest {

    /** Feeds a scripted 50 Hz signal; each phase gives the accel magnitude in g over time, the gyro rate and the gravity axis. */
    private class Script(val det: FallDetector) {
        var t = 0L
        fun phase(sec: Float, gyro: Float = 5f, axis: Int = 2, g: (Float) -> Float) {
            val n = (sec * 50).toInt()
            for (i in 0 until n) {
                val a = FloatArray(3)
                a[axis] = g(i / 50f) * FallDetector.G
                det.onSample(t, a[0], a[1], a[2], gyro)
                t += 20
            }
        }
    }

    private fun run(block: (Script) -> Unit): List<FallCandidate> {
        val out = mutableListOf<FallCandidate>()
        block(Script(FallDetector { out += it }))
        return out
    }

    private fun walk(s: Script, sec: Float) =
        s.phase(sec, gyro = 60f) { t -> 1f + 0.35f * sin(2 * PI * 2 * t).toFloat() }

    @Test
    fun walkingAloneIsNotAFall() {
        assertTrue(run { walk(it, 20f) }.isEmpty())
    }

    @Test
    fun realFallWhileWalkingIsDetected() {
        val c = run { s ->
            walk(s, 8f)
            s.phase(0.25f, gyro = 300f) { 0.2f }
            s.phase(0.06f, gyro = 500f) { 4.5f }
            s.phase(6f, axis = 0) { 1f }
        }
        assertEquals(1, c.size)
        assertTrue(c[0].confidence >= 0.8f)
        assertTrue(c[0].orientationChangeDeg > 60f)
        assertEquals(true, c[0].wasMovingBefore)
    }

    @Test
    fun phoneDroppedFromRestingTableIsNotAFall() {
        val c = run { s ->
            s.phase(6f) { 1f }
            s.phase(0.3f, gyro = 300f) { 0.2f }
            s.phase(0.06f, gyro = 500f) { 4.5f }
            s.phase(6f, axis = 0) { 1f }
        }
        assertTrue(c.isEmpty())
    }

    @Test
    fun hardBumpThenKeepWalkingIsNotAFall() {
        val c = run { s ->
            walk(s, 8f)
            s.phase(0.06f, gyro = 200f) { 3.2f }
            walk(s, 8f)
        }
        assertTrue(c.isEmpty())
    }

    @Test
    fun sittingDownHardBelowImpactThresholdIsIgnored() {
        val c = run { s ->
            walk(s, 8f)
            s.phase(0.06f) { 2.2f }
            s.phase(6f) { 1f }
        }
        assertTrue(c.isEmpty())
    }

    @Test
    fun slowCollapseWithoutFreeFallStillCounts() {
        val c = run { s ->
            walk(s, 8f)
            s.phase(0.06f, gyro = 100f) { 3.0f }
            s.phase(6f, axis = 1) { 1f }
        }
        assertEquals(1, c.size)
        assertTrue(c[0].confidence in 0.6f..1f)
    }

    @Test
    fun cooldownPreventsDuplicateAlerts() {
        val c = run { s ->
            walk(s, 8f)
            s.phase(0.06f, gyro = 300f) { 4f }
            s.phase(4f, axis = 0) { 1f }
            s.phase(0.06f, gyro = 300f) { 4f }
            s.phase(4f, axis = 0) { 1f }
        }
        assertEquals(1, c.size)
    }
}
