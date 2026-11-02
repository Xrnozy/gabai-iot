package com.example.gabai.services

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

class Haptics(context: Context) {

    private val appContext = context.applicationContext

    private val vibrator: Vibrator? = run {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vm = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
            vm?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }
    }

    private fun enabled() = SettingsStore.isHapticsEnabled(appContext)

    fun click() {
        vibrateOneShot(30, VibrationEffect.DEFAULT_AMPLITUDE)
    }

    /** Feather-light tick — fired as the finger crosses onto each new
     *  readable element during touch explore. */
    fun hoverTick() {
        vibrateOneShot(15, 120)
    }

    fun heavy() {
        vibrateOneShot(60, 200)
    }

    /** Wake-word ack — two quick ticks paired with the audio ding. */
    fun wakeAck() {
        vibratePattern(longArrayOf(0, 35, 70, 35), intArrayOf(0, 160, 0, 160))
    }

    /** Advisory cue — hazard on the left: one short tick. */
    fun hazardLeft() {
        vibratePattern(longArrayOf(0, 50), intArrayOf(0, 180))
    }

    /** Advisory cue — hazard on the right: two short ticks. */
    fun hazardRight() {
        vibratePattern(longArrayOf(0, 50, 90, 50), intArrayOf(0, 180, 0, 180))
    }

    /** Advisory cue — hazard ahead: one long pulse. */
    fun hazardAhead() {
        vibrateOneShot(220, 220)
    }

    private fun vibratePattern(timings: LongArray, amplitudes: IntArray) {
        if (!enabled()) return
        val v = vibrator ?: return
        if (!v.hasVibrator()) return
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                v.vibrate(VibrationEffect.createWaveform(timings, amplitudes, -1))
            } else {
                @Suppress("DEPRECATION")
                v.vibrate(timings.sum())
            }
        } catch (_: SecurityException) {
            // If VIBRATE permission is missing, fail silently (never crash).
        }
    }

    private fun vibrateOneShot(durationMs: Long, amplitude: Int) {
        if (!enabled()) return
        val v = vibrator ?: return
        if (!v.hasVibrator()) return

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                v.vibrate(VibrationEffect.createOneShot(durationMs, amplitude))
            } else {
                @Suppress("DEPRECATION")
                v.vibrate(durationMs)
            }
        } catch (_: SecurityException) {
            // If VIBRATE permission is missing, fail silently (never crash).
        }
    }
}

