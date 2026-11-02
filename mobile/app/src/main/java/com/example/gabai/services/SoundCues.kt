package com.example.gabai.services

import android.content.Context
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaPlayer
import android.util.Log
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.sin

/**
 * Assistant-style audio cues, synthesized in code — no .wav assets needed.
 *
 * [wakeAck] is the Siri-like "ding" played when the wake name ("Gab") is
 * heard, signalling the app is now waiting for a command. [windowClosed] is
 * a softer descending blip played when that command window lapses unused.
 *
 * Playback goes out on STREAM_MUSIC at moderate volume so it sits under the
 * user's media volume control; calls are non-blocking (a single worker
 * serializes overlapping cues).
 */
object SoundCues {

    private val player = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "sound-cues").apply { isDaemon = true }
    }

    /** True while a cue is playing — the mic gate uses it so the chime
     *  isn't transcribed back as a command. */
    @Volatile
    var isPlaying: Boolean = false
        private set

    @Volatile private var appContext: Context? = null
    @Volatile private var wakeResId = 0

    /**
     * Call once with any context — picks up an optional bundled sound at
     * `res/raw/wake_ding.*` (mp4/m4a/wav/ogg all work). Without it the
     * synthesized chime plays.
     */
    fun configure(context: Context) {
        appContext = context.applicationContext
        if (wakeResId == 0) {
            wakeResId = context.resources
                .getIdentifier("wake_ding", "raw", context.packageName)
            if (wakeResId != 0) Log.d(TAG, "wake cue: bundled res/raw/wake_ding")
        }
    }

    /** Ascending two-note chime — "wake word heard, say a command". */
    fun wakeAck() {
        if (wakeResId != 0) playResource(wakeResId)
        else play(chime(Notes.WAKE, volume = 0.55f))
    }

    /** Gentle descending blip — the command window closed unanswered. */
    fun windowClosed() = play(chime(Notes.CLOSE, volume = 0.30f))

    /** Single bright tick — "command heard". Fires the instant a phrase
     *  parses so the user gets feedback while TTS is still spinning up. */
    fun commandAck() = play(chime(Notes.ACK, volume = 0.30f, noteMs = 90))

    private object Notes {
        // B5 → E6: bright, friendly "ding-ding" (like a doorbell up-chime).
        val WAKE = doubleArrayOf(987.77, 1318.51)
        // E6 → B5 reversed + lower fifth: a soft "never mind" fall.
        val CLOSE = doubleArrayOf(987.77, 659.25)
        // G6: one short tick — acknowledgment, not a sentence.
        val ACK = doubleArrayOf(1567.98)
    }

    private const val TAG = "GabVoice"
    private const val SAMPLE_RATE = 22050
    private const val NOTE_MS = 170
    private const val GAP_MS = 50
    private const val RESOURCE_VOLUME = 0.5f // bundled wake_ding clip, 50%

    /**
     * Renders [freqs] as sequential sine notes with an exponential decay
     * envelope (struck-bell feel) into 16-bit mono PCM.
     */
    private fun chime(
        freqs: DoubleArray,
        volume: Float,
        noteMs: Int = NOTE_MS,
        gapMs: Int = GAP_MS,
    ): ShortArray {
        val noteSamples = SAMPLE_RATE * noteMs / 1000
        val gapSamples = SAMPLE_RATE * gapMs / 1000
        val total = freqs.size * noteSamples + (freqs.size - 1) * gapSamples
        val pcm = ShortArray(total)

        val tau = noteSamples / 4.0 // decay time constant → rings, then fades
        for ((n, freq) in freqs.withIndex()) {
            val start = n * (noteSamples + gapSamples)
            for (i in 0 until noteSamples) {
                val t = i.toDouble() / SAMPLE_RATE
                val envelope = exp(-i / tau)
                val s = sin(2.0 * Math.PI * freq * t) * envelope * volume
                pcm[start + i] = (s * Short.MAX_VALUE).toInt().toShort()
            }
        }
        return pcm
    }

    /** Plays a bundled res/raw clip (the custom wake sound). */
    private fun playResource(resId: Int) {
        val ctx = appContext ?: return
        player.execute {
            isPlaying = true
            var mp: MediaPlayer? = null
            try {
                mp = MediaPlayer.create(ctx, resId)
                    ?: throw IllegalStateException("MediaPlayer.create failed")
                mp.setVolume(RESOURCE_VOLUME, RESOURCE_VOLUME)
                mp.start()
                Thread.sleep(mp.duration + 150L)
            } catch (e: Exception) {
                Log.e(TAG, "wake cue playback failed", e)
            } finally {
                isPlaying = false
                try { mp?.release() } catch (_: Exception) {}
            }
        }
    }

    private fun play(pcm: ShortArray) {
        player.execute {
            val track = try {
                AudioTrack(
                    AudioManager.STREAM_MUSIC,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    pcm.size * 2,
                    AudioTrack.MODE_STATIC,
                )
            } catch (e: Exception) {
                Log.e(TAG, "could not create AudioTrack for cue", e)
                return@execute
            }
            isPlaying = true
            try {
                track.write(pcm, 0, pcm.size)
                track.play()
                // Static mode: wait out the clip, then free the native buffer.
                Thread.sleep(pcm.size * 1000L / SAMPLE_RATE + 50)
            } catch (e: Exception) {
                Log.e(TAG, "audio cue playback failed", e)
            } finally {
                isPlaying = false
                try { track.stop() } catch (_: Exception) {}
                track.release()
            }
        }
    }
}
