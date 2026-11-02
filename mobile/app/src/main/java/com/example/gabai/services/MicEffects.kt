package com.example.gabai.services

import android.media.AudioRecord
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.util.Log

/**
 * Attaches the platform audio pre-processors to a live [AudioRecord] session.
 * All three are optional hardware features — [isAvailable] is checked and
 * whatever actually engaged is logged so field reports show which cleaners
 * were active. Nothing here throws on failure; worst case is plain mic audio.
 */
object MicEffects {

    // AudioEffect instances are released on GC — keep them referenced for
    // the life of the capture session.
    private val active = java.util.concurrent.CopyOnWriteArrayList<android.media.audiofx.AudioEffect>()

    fun attach(record: AudioRecord, config: VoiceConfig, tag: String = "GabVoice") {
        release()
        val session = record.audioSessionId
        if (session == 0) {
            Log.w(tag, "audio session id 0 — cannot attach mic effects")
            return
        }
        if (config.wantNoiseSuppressor) {
            runEffect(tag, "NoiseSuppressor", NoiseSuppressor.isAvailable()) {
                NoiseSuppressor.create(session)
            }
        }
        if (config.wantAcousticEchoCanceler) {
            runEffect(tag, "AcousticEchoCanceler", AcousticEchoCanceler.isAvailable()) {
                AcousticEchoCanceler.create(session)
            }
        }
        if (config.wantAutomaticGainControl) {
            runEffect(tag, "AutomaticGainControl", AutomaticGainControl.isAvailable()) {
                AutomaticGainControl.create(session)
            }
        }
    }

    private inline fun <reified T> runEffect(
        tag: String,
        name: String,
        available: Boolean,
        create: () -> T?,
    ) {
        if (!available) {
            Log.i(tag, "$name: not available on this device")
            return
        }
        try {
            val fx = create()
            if (fx is android.media.audiofx.AudioEffect) active.add(fx)
            Log.i(tag, "$name: " +
                if (fx != null) "enabled" else "unavailable (create returned null)")
        } catch (e: Exception) {
            Log.w(tag, "$name failed to attach: ${e.message}")
        }
    }

    /** Releases effects from a previous capture session. */
    fun release() {
        for (fx in active) {
            try { fx.release() } catch (_: Exception) {}
        }
        active.clear()
    }
}
