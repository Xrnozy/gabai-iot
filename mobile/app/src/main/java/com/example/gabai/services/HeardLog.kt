package com.example.gabai.services

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Persistent log of everything Whisper heard — one timestamped line per
 * accepted transcript, appended to
 * `<external-files>/heard_transcripts.txt` (plain `adb pull` works).
 *
 * Purpose: collecting real mishears on-device — the "close enough" words
 * Whisper actually produces for "hello gab" and the commands — so the
 * wake-variant and synonym tables can be grown from field data instead of
 * guesses.
 *
 * Writes are queued on a daemon thread; failures are logged and dropped,
 * never thrown into the audio path. The file is capped — past
 * [MAX_BYTES] the oldest half is dropped so the log can't grow forever.
 */
object HeardLog {

    private val io = Executors.newSingleThreadExecutor { r ->
        Thread(r, "heard-log").apply { isDaemon = true }
    }

    private const val TAG = "GabVoice"
    private const val FILE_NAME = "heard_transcripts.txt"
    private const val MAX_BYTES = 512L * 1024

    private val stamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    fun append(context: Context, text: String) {
        val dir = context.applicationContext.getExternalFilesDir(null) ?: return
        val line = buildString {
            synchronized(stamp) { append(stamp.format(Date())) }
            append('\t')
            append(text.replace('\n', ' '))
            append('\n')
        }
        io.execute {
            try {
                val file = File(dir, FILE_NAME)
                if (file.length() > MAX_BYTES) {
                    // Rotate: keep the newest half so the log stays useful
                    // without ever ballooning.
                    val keep = file.readBytes()
                        .drop(MAX_BYTES.toInt() / 2)
                        .toByteArray()
                    file.writeBytes(keep)
                }
                file.appendText(line)
            } catch (e: Exception) {
                Log.w(TAG, "heard log write failed", e)
            }
        }
    }
}
