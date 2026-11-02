package com.example.gabai.services

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.view.View
import androidx.core.content.ContextCompat
import com.example.gabai.R
import kotlin.math.cos
import kotlin.math.sin

/**
 * Pass-through touch overlay for the touch-explore system: while a finger
 * is held down, draws a soft ring of "warping" dots around the touch point
 * that expand as the dwell timer builds toward the announcement; fades out
 * on release. Subtle by design — it locates the finger, it never obscures.
 *
 * The view never consumes events: [BaseActivity] feeds it coordinates from
 * dispatchTouchEvent via [onFingerDown]/[onFingerMove]/[onFingerUp].
 */
class TouchExploreOverlay(context: Context) : View(context) {

    private val dotColor = ContextCompat.getColor(context, R.color.primary)
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val density = resources.displayMetrics.density

    private var fx = 0f
    private var fy = 0f
    private var downAt = 0L
    private var upAt = 0L
    private var pressed = false

    fun onFingerDown(x: Float, y: Float) {
        fx = x; fy = y
        downAt = android.os.SystemClock.uptimeMillis()
        pressed = true
        postInvalidateOnAnimation()
    }

    fun onFingerMove(x: Float, y: Float) {
        fx = x; fy = y
        if (pressed) postInvalidateOnAnimation()
    }

    fun onFingerUp() {
        pressed = false
        upAt = android.os.SystemClock.uptimeMillis()
        postInvalidateOnAnimation()
    }

    private fun alphaFraction(now: Long): Float = when {
        pressed -> 0.7f
        else -> (1f - (now - upAt) / FADE_MS).coerceIn(0f, 1f) * 0.7f
    }

    override fun onDraw(canvas: Canvas) {
        val now = android.os.SystemClock.uptimeMillis()
        val a = alphaFraction(now)
        if (a <= 0f) return

        // Dwell progress 0..1 — dots warp outward as the announcement nears.
        val progress = if (pressed)
            ((now - downAt).toFloat() / DWELL_MS).coerceIn(0f, 1f) else 1f

        // Soft center marker.
        paint.style = Paint.Style.FILL
        paint.color = dotColor
        paint.alpha = (0.25f * a * 255).toInt()
        canvas.drawCircle(fx, fy, 14f * density, paint)

        // Two concentric rings of warping dots — radii breathe with dwell
        // progress and the inner ring counter-rotates, so the whole reads
        // as a vortex gathering toward the announcement.
        var outerR = 0f
        for (ring in 0..1) {
            val dots = if (ring == 0) 8 else 16
            val baseR = (if (ring == 0) 20f + 14f * progress
                else 36f + 30f * progress) * density
            if (ring == 1) outerR = baseR
            val dir = if (ring == 0) -1.0 else 1.0
            for (i in 0 until dots) {
                val angle = (i.toDouble() / dots) * Math.PI * 2 +
                    dir * (now % 2000L) / 2000.0 * 0.35 + ring * 0.4
                val wobble = 1f + 0.12f * sin(now / 180.0 + i + ring * 2).toFloat()
                val r = baseR * wobble
                val dx = fx + (cos(angle) * r).toFloat()
                val dy = fy + (sin(angle) * r).toFloat()
                paint.alpha = ((0.15f + 0.45f * progress) * a * 255)
                    .toInt().coerceAtMost(255)
                val dotR = (if (ring == 0) 2f + 1.5f * progress
                    else 2.5f + 2f * progress) * density
                canvas.drawCircle(dx, dy, dotR, paint)
            }
        }

        // Outer ring completes as dwell fills — the "about to announce" cue.
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2f * density
        paint.alpha = (0.5f * a * 255).toInt()
        canvas.drawCircle(fx, fy, outerR, paint)

        if (pressed || now - upAt < FADE_MS) postInvalidateOnAnimation()
    }

    private companion object {
        const val DWELL_MS = 1000f
        const val FADE_MS = 350f
    }
}
