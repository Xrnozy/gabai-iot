package com.example.gabai.services

import kotlin.math.min

/** A finished speech segment: PCM plus where it sat in the stream. */
data class VadSegment(
    val pcm: FloatArray,
    /** Stream time (ms) of the first sample, lead-in included. */
    val startMs: Long,
    val endMs: Long,
    /** Peak and mean speech probability inside the segment. */
    val maxProb: Float,
    val meanProb: Float,
)

/**
 * Turns a stream of fixed-size frames + per-frame speech probabilities into
 * speech segments (the Silero-VAD-article technique):
 *
 *  - a rolling pre-roll keeps [VoiceConfig.vadLeadInMs] of audio before the
 *    detected onset, so the first syllable is never clipped;
 *  - a segment stays open until [VoiceConfig.vadSilenceCloseMs] of
 *    below-threshold audio, then closes keeping [VoiceConfig.vadTrailingPadMs]
 *    of trailing padding;
 *  - segments shorter than [VoiceConfig.vadMinSegmentMs] are dropped;
 *  - segments longer than [VoiceConfig.vadMaxSegmentMs] are split at the
 *    quietest frame, never mid-word at a hard cap.
 *
 * Memory stays flat: frames older than the lead-in are only kept while a
 * segment is open, and the cap bounds an open segment. Pure Kotlin — unit
 * tests drive it with fake probabilities, no ONNX needed.
 */
class VadSegmenter(private val config: VoiceConfig = VoiceConfig.DEFAULT) {

    /** Called (on the feeding thread) with each completed, kept segment. */
    var onSegment: ((VadSegment) -> Unit)? = null

    /** Called (on the feeding thread) with each segment dropped for being
     *  shorter than [VoiceConfig.vadMinSegmentMs] — noise blips are silent
     *  without this, which makes field debugging painful. */
    var onTooShortSegment: ((pcm: FloatArray, startMs: Long) -> Unit)? = null

    private val frameMs = config.frameSamples * 1000L / config.sampleRate
    private val leadInFrames = (config.vadLeadInMs / frameMs).toInt().coerceAtLeast(1)
    private val padFrames = (config.vadTrailingPadMs / frameMs).toInt().coerceAtLeast(0)
    private val silenceCloseFrames =
        (config.vadSilenceCloseMs / frameMs).toInt().coerceAtLeast(1)
    private val minFrames = (config.vadMinSegmentMs / frameMs).toInt().coerceAtLeast(1)
    private val maxFrames = (config.vadMaxSegmentMs / frameMs).toInt().coerceAtLeast(1)

    private class Slot(val frame: FloatArray, val prob: Float)

    private val preRoll = ArrayDeque<Slot>()
    private val segment = ArrayList<Slot>()
    private var silenceFrames = 0
    private var streamPos = 0L          // samples accepted so far
    private var segStart = 0L           // stream sample where the open segment began

    var inSpeech = false
        private set

    /** Stream sample index where the open segment started (lead-in
     *  included), or -1 when no segment is open. */
    val currentSegmentStartSample: Long
        get() = if (inSpeech) segStart else -1L

    /** Feed one frame. [frame] is [VoiceConfig.frameSamples] float samples;
     *  [speechProb] is the VAD probability for it. */
    fun accept(frame: FloatArray, speechProb: Float) {
        val slot = Slot(frame, speechProb)
        if (!inSpeech) {
            if (speechProb >= config.vadStartThreshold) {
                inSpeech = true
                silenceFrames = 0
                // preRoll holds the frames BEFORE this onset — the segment
                // starts lead-in early so the first syllable is kept.
                segStart = streamPos - preRoll.sumOf { it.frame.size }
                segment.addAll(preRoll)
                preRoll.clear()
                segment.add(slot)
            } else {
                preRoll.addLast(slot)
                while (preRoll.size > leadInFrames) preRoll.removeFirst()
            }
        } else {
            segment.add(slot)
            if (speechProb < config.vadContinueThreshold) silenceFrames++ else silenceFrames = 0
            when {
                silenceFrames >= silenceCloseFrames -> closeSegment()
                segment.size >= maxFrames -> splitAtQuietest()
            }
        }
        streamPos += frame.size
    }

    /** Closes an open segment (e.g. mic stopping). Safe to call anytime. */
    fun flush() {
        if (inSpeech) closeSegment()
        preRoll.clear()
    }

    fun reset() {
        preRoll.clear()
        segment.clear()
        silenceFrames = 0
        inSpeech = false
    }

    // ---------- internals ----------

    private fun closeSegment() {
        // Drop trailing below-threshold frames beyond the trailing pad.
        val drop = min(silenceFrames - padFrames, segment.size).coerceAtLeast(0)
        repeat(drop) { segment.removeAt(segment.size - 1) }
        inSpeech = false
        silenceFrames = 0
        if (segment.size >= minFrames) {
            emit(segment, segStart)
        } else if (onTooShortSegment != null && segment.isNotEmpty()) {
            val n = segment.sumOf { it.frame.size }
            val pcm = FloatArray(n)
            var off = 0
            for (s in segment) {
                s.frame.copyInto(pcm, off)
                off += s.frame.size
            }
            onTooShortSegment?.invoke(pcm, segStart * 1000L / config.sampleRate)
        }
        segment.clear()
    }

    /**
     * Over-long segment: find the quietest frame strictly inside the usable
     * middle (so neither half is shorter than the minimum) and cut there.
     * The tail stays in [segment] as the still-open segment.
     */
    private fun splitAtQuietest() {
        var best = -1
        var bestProb = Float.MAX_VALUE
        for (i in minFrames until segment.size - minFrames) {
            if (segment[i].prob < bestProb) {
                bestProb = segment[i].prob
                best = i
            }
        }
        if (best <= 0) best = segment.size / 2  // flat probs: split in half
        val head = ArrayList(segment.subList(0, best))
        emit(head, segStart)
        segStart += head.sumOf { it.frame.size }
        repeat(best) { segment.removeAt(0) }
        // The tail remains open; silence still has to close it.
    }

    private fun emit(frames: List<Slot>, startSample: Long) {
        if (frames.isEmpty()) return
        val n = frames.sumOf { it.frame.size }
        val pcm = FloatArray(n)
        var off = 0
        var maxP = 0f
        var sumP = 0.0
        for (s in frames) {
            s.frame.copyInto(pcm, off)
            off += s.frame.size
            if (s.prob > maxP) maxP = s.prob
            sumP += s.prob
        }
        onSegment?.invoke(
            VadSegment(
                pcm = pcm,
                startMs = startSample * 1000L / config.sampleRate,
                endMs = (startSample + n) * 1000L / config.sampleRate,
                maxProb = maxP,
                meanProb = (sumP / frames.size).toFloat(),
            )
        )
    }
}
