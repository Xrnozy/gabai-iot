package com.example.gabai

import com.example.gabai.services.SileroVad
import com.example.gabai.services.TranscriptFilter
import com.example.gabai.services.Transcription
import com.example.gabai.services.VadSegmenter
import com.example.gabai.services.VoiceConfig
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.io.RandomAccessFile

/**
 * Offline replay harness: runs saved 16 kHz mono WAV clips through the real
 * Silero VAD + [VadSegmenter] + [TranscriptFilter] and prints the decision
 * for each, so noisy recordings from the field can be audited without a
 * device.
 *
 * Clips live in `src/test/resources/clips/` (or a dir passed via
 * `-DGAB_REPLAY_DIR=...`, e.g. a pull of files/rejected_clips). Optional
 * sidecar `<clip>.txt` supplies the transcript Whisper produced so the
 * rejection filters can be replayed too.
 *
 * The test passes vacuously when the directory or the Silero model is
 * absent — it is a debugging tool, not a gate.
 */
class ReplayHarnessTest {

    @Test
    fun replayClips() {
        val cfg = VoiceConfig.DEFAULT
        val dir = System.getProperty("GAB_REPLAY_DIR")?.let { File(it) }
            ?: File("src/test/resources/clips")
        val model = File("src/main/assets/models/${cfg.vadModelFile}")
            .takeIf { it.isFile }
        assumeTrue("no clips dir and/or no silero model — nothing to replay",
            dir.isDirectory && model != null)

        val clips = dir.listFiles { f -> f.extension.equals("wav", true) }
            ?.sorted() ?: emptyList()
        assumeTrue("no .wav clips in $dir", clips.isNotEmpty())

        val vad = SileroVad(model!!, cfg).apply { load() }
        val segmenter = VadSegmenter(cfg)
        var total = 0
        var kept = 0
        for (clip in clips) {
            val pcm = readWav(clip)
            val segments = mutableListOf<com.example.gabai.services.VadSegment>()
            segmenter.onSegment = { segments += it }
            var pos = 0
            while (pos + cfg.frameSamples <= pcm.size) {
                val frame = pcm.copyOfRange(pos, pos + cfg.frameSamples)
                segmenter.accept(frame, vad.speechProb(frame))
                pos += cfg.frameSamples
            }
            segmenter.flush()
            segmenter.reset()

            total++
            println("[replay] ${clip.name}: ${pcm.size / cfg.sampleRate}s audio, " +
                "${segments.size} segment(s) kept")
            if (segments.isEmpty()) {
                println("         -> never reaches Whisper")
            }
            for (s in segments) {
                kept++
                val dur = s.pcm.size.toFloat() / cfg.sampleRate
                println("         segment ${s.startMs}-${s.endMs}ms " +
                    "dur=%.2fs maxVad=%.2f meanVad=%.2f".format(dur, s.maxProb, s.meanProb))
                // Sidecar transcript replays the rejection filter too.
                val sidecar = File(clip.parent, clip.nameWithoutExtension + ".txt")
                if (sidecar.isFile) {
                    val t = Transcription(
                        text = sidecar.readText().trim(), language = "en",
                        noSpeechProb = null, avgLogprob = null,
                        durationSec = dur,
                    )
                    val d = TranscriptFilter.evaluate(t, vadConfirmed = true, cfg)
                    println("         transcript=\"${t.text}\" -> ${d.reason}")
                }
            }
        }
        println("[replay] done: $kept/$total clips produced speech segments")
        vad.close()
    }

    /** Reads a RIFF PCM16 WAV into float samples in [-1, 1]. */
    private fun readWav(f: File): FloatArray {
        RandomAccessFile(f, "r").use { raf ->
            val riff = ByteArray(12).also { raf.readFully(it) }
            require(String(riff, 0, 4) == "RIFF" && String(riff, 8, 4) == "WAVE") {
                "$f is not a WAV file"
            }
            var dataOff = -1L
            var dataLen = 0
            // Walk chunks until "data".
            while (true) {
                val hdr = ByteArray(8)
                if (raf.read(hdr) < 8) break
                val id = String(hdr, 0, 4)
                val len = Integer.reverseBytes(
                    ((hdr[7].toInt() and 0xFF) shl 24) or
                        ((hdr[6].toInt() and 0xFF) shl 16) or
                        ((hdr[5].toInt() and 0xFF) shl 8) or
                        (hdr[4].toInt() and 0xFF))
                if (id == "data") {
                    dataOff = raf.filePointer
                    dataLen = len
                    break
                }
                raf.skipBytes(len)
            }
            require(dataOff >= 0) { "no data chunk in $f" }
            val raw = ByteArray(dataLen).also { raf.readFully(it) }
            return FloatArray(dataLen / 2) { i ->
                val lo = raw[i * 2].toInt() and 0xFF
                val hi = raw[i * 2 + 1].toInt()
                ((hi shl 8) or lo).toShort() / 32768f
            }
        }
    }
}
