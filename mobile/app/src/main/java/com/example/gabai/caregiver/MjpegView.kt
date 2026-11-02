package com.example.gabai.caregiver

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import com.example.gabai.safety.MjpegReader
import com.example.gabai.safety.RelayApi
import com.example.gabai.services.SettingsStore
import com.example.gabai.ui.Level
import com.example.gabai.ui.Ui

/**
 * One Raspberry Pi camera, streamed from the relay. While the Pi is sending, frames play live; when the stream ends it shows
 * the last frame the relay holds, dimmed and labelled with its real age, so an old picture is never mistaken for a live one.
 */
class MjpegView(context: Context, private val side: String) : FrameLayout(context) {

    private val image = ImageView(context).apply { scaleType = ImageView.ScaleType.FIT_CENTER; setBackgroundColor(Color.BLACK) }
    private val badge = TextView(context).apply {
        setTextColor(Color.WHITE); setBackgroundColor(0xCC000000.toInt()); textSize = 14f
        setPadding(Ui.dp(context, 10), Ui.dp(context, 6), Ui.dp(context, 10), Ui.dp(context, 6))
    }
    private val main = Handler(Looper.getMainLooper())
    private val api = RelayApi(context, SettingsStore.ROLE_CAREGIVER)
    @Volatile private var running = false
    private var worker: Thread? = null

    init {
        addView(image, LayoutParams(LayoutParams.MATCH_PARENT, Ui.dp(context, 200)))
        addView(badge, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT, Gravity.TOP or Gravity.START))
        setState("No $side frame yet", Level.OFF)
    }

    fun start() {
        if (running) return
        running = true
        worker = Thread({ loop() }, "mjpeg-$side").also { it.start() }
    }

    fun stop() {
        running = false
        worker?.interrupt()
        worker = null
    }

    private fun loop() {
        var backoff = 1_000L
        while (running) {
            val conn = api.openStream(side)
            if (conn != null) {
                backoff = 1_000L
                try {
                    val reader = MjpegReader(conn.inputStream)
                    var frames = 0
                    val t0 = System.currentTimeMillis()
                    while (running) {
                        val jpeg = reader.next() ?: break
                        val bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: continue
                        frames++
                        val fps = frames * 1000.0 / (System.currentTimeMillis() - t0).coerceAtLeast(1)
                        showLive(bmp, fps)
                    }
                } catch (_: Exception) {
                } finally {
                    conn.disconnect()
                }
            }
            if (!running) break
            showLastKnown()
            try { Thread.sleep(backoff) } catch (_: InterruptedException) { break }
            backoff = (backoff * 2).coerceAtMost(8_000L)
        }
    }

    private fun showLive(bmp: Bitmap, fps: Double) = main.post {
        image.setImageBitmap(bmp)
        image.alpha = 1f
        setState("LIVE · ${"%.0f".format(fps)} fps", Level.OK)
    }

    private fun showLastKnown() {
        val snap = api.snapshot(side)
        main.post {
            if (snap == null) {
                setState("No $side frame yet", Level.OFF)
            } else {
                image.setImageBitmap(snap.first)
                image.alpha = 0.45f
                setState("STALE · last frame ${Fmt.age(snap.second)} ago", Level.WARN)
            }
        }
    }

    private fun setState(text: String, level: Level) {
        badge.text = "${level.symbol} ${side.replaceFirstChar { it.uppercase() }} camera · $text"
        contentDescription = "${side.replaceFirstChar { it.uppercase() }} camera. $text"
        importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
    }
}
