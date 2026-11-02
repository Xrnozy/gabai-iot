package com.example.gabai

import com.example.gabai.services.VadSegmenter
import com.example.gabai.services.VoiceConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class VadSegmenterTest {

    /** Small time constants so tests stay readable: 32 ms frames. */
    private fun cfg(
        leadInMs: Int = 96,        // 3 frames
        trailMs: Int = 64,         // 2 frames
        silenceMs: Int = 128,      // 4 frames
        minMs: Int = 160,          // 5 frames
        maxMs: Int = 960,          // 30 frames
    ) = VoiceConfig(
        vadLeadInMs = leadInMs, vadTrailingPadMs = trailMs,
        vadSilenceCloseMs = silenceMs, vadMinSegmentMs = minMs,
        vadMaxSegmentMs = maxMs,
    )

    private class Sink {
        val out = mutableListOf<com.example.gabai.services.VadSegment>()
    }

    private fun run(
        config: VoiceConfig,
        probs: List<Float>,
    ): List<com.example.gabai.services.VadSegment> {
        val seg = VadSegmenter(config)
        val sink = Sink()
        seg.onSegment = { sink.out += it }
        probs.forEach { seg.accept(FloatArray(512) { 0.01f }, it) }
        seg.flush()
        return sink.out
    }

    @Test
    fun leadInPrecedesSpeechOnset() {
        // 10 quiet, 20 speech, 20 quiet. Speech onset at frame 10;
        // lead-in (3 frames) must pull the start back to frame 7.
        val probs = List(10) { 0f } + List(20) { 0.9f } + List(20) { 0f }
        val segs = run(cfg(), probs)
        assertEquals(1, segs.size)
        assertEquals(7 * 32L, segs[0].startMs)
        // Kept: frames 7..33 minus the 2 silence frames past the pad → 25.
        assertEquals(25 * 512, segs[0].pcm.size)
    }

    @Test
    fun shortBlipsAreDropped() {
        // 2 speech frames then quiet — lead-in(3)+speech(2)+pad(2)=7 frames,
        // under a 10-frame (320 ms) minimum.
        val probs = List(5) { 0f } + List(2) { 0.9f } + List(10) { 0f }
        assertTrue(run(cfg(minMs = 320), probs).isEmpty())
    }

    @Test
    fun silenceNeverProducesSegments() {
        val segs = run(cfg(), List(100) { 0f })
        assertTrue(segs.isEmpty())
    }

    @Test
    fun longSegmentSplitsAtQuietestPoint() {
        // 45 speech frames; a quiet dip at frame 20. Max = 30 frames → the
        // segment must split AT the dip, not mid-word at the cap.
        val probs = MutableList(45) { 0.9f }
        probs[20] = 0.05f                       // the quiet point
        probs += List(10) { 0f }                // trailing silence closes it
        val segs = run(cfg(minMs = 0), probs)
        assertEquals(2, segs.size)
        // Head ends where the dip started (frame 20 → 640 ms, with lead-in
        // the start pulls back to frame -... first frame at 0).
        assertEquals(0L, segs[0].startMs)
        assertEquals(20 * 32L, segs[0].endMs)
        assertEquals(20 * 32L, segs[1].startMs)
    }

    @Test
    fun trailingPadKeepsWordTail() {
        // Speech 30..44, then silence — pad keeps 2 quiet frames on the end.
        val probs = List(30) { 0f } + List(15) { 0.9f } + List(20) { 0f }
        val segs = run(cfg(), probs)
        assertEquals(1, segs.size)
        // 3 lead-in + 15 speech + 2 pad = 20 frames.
        assertEquals(20 * 512, segs[0].pcm.size)
        assertEquals(27 * 32L, segs[0].startMs)
    }
}
