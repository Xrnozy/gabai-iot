package com.example.gabai.services

import android.animation.ValueAnimator
import android.content.res.ColorStateList
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.ImageView
import android.widget.ProgressBar
import androidx.core.content.ContextCompat
import com.example.gabai.R

/**
 * Turns a screen's mic ImageView into an assistant-style state light:
 *
 *  - not listening        → hidden (honest: the mic is off)
 *  - models loading       → [progress] spinner if one was supplied, else a
 *    fast alpha pulse on the mic — "hello" can't be heard yet
 *  - listening            → green mic, slow alpha "breathing" pulse
 *  - awaiting a command   → mic on a filled accent disc with a quick scale
 *    pulse — the visual half of the wake ding ("Gab heard you, talk now")
 *
 * Call [setListening] / [setLoading] / [setAwaitingCommand] on the main
 * thread. Loading wins over listening: the pipeline is warming up, not
 * capturing.
 */
class ListeningIndicator(
    private val view: ImageView,
    private val progress: ProgressBar? = null,
) {

    private var listening = false
    private var loading = false
    private var awaiting = false
    private var pulse: ValueAnimator? = null

    fun setListening(value: Boolean) {
        if (listening == value) return
        listening = value
        render()
    }

    fun setLoading(value: Boolean) {
        if (loading == value) return
        loading = value
        render()
    }

    fun setAwaitingCommand(value: Boolean) {
        if (awaiting == value) return
        awaiting = value
        render()
    }

    private fun render() {
        pulse?.cancel()
        view.alpha = 1f
        view.scaleX = 1f
        view.scaleY = 1f
        progress?.visibility = View.GONE

        when {
            loading -> {
                if (progress != null) {
                    view.visibility = View.GONE
                    progress.visibility = View.VISIBLE
                } else {
                    view.visibility = View.VISIBLE
                    view.background = null
                    view.imageTintList = tintOf(R.color.on_dark_muted)
                    startPulse(scale = false, fast = true)
                }
                view.contentDescription =
                    view.context.getString(R.string.voice_loading_desc)
            }

            !listening -> view.visibility = View.GONE

            awaiting -> {
                view.visibility = View.VISIBLE
                view.setBackgroundResource(R.drawable.bg_fab_mic)
                view.imageTintList = tintOf(R.color.on_primary)
                view.contentDescription =
                    view.context.getString(R.string.voice_awaiting_desc)
                startPulse(scale = true)
            }

            else -> {
                view.visibility = View.VISIBLE
                view.background = null
                view.imageTintList = tintOf(R.color.ai_listening)
                view.contentDescription =
                    view.context.getString(R.string.voice_listening_desc)
                startPulse(scale = false)
            }
        }
    }

    private fun startPulse(scale: Boolean, fast: Boolean = false) {
        val animator = if (scale) {
            ValueAnimator.ofFloat(1f, 1.2f).apply {
                duration = 400
                addUpdateListener {
                    val v = it.animatedValue as Float
                    view.scaleX = v
                    view.scaleY = v
                }
            }
        } else {
            ValueAnimator.ofFloat(1f, 0.5f).apply {
                duration = if (fast) 500 else 1500
                addUpdateListener { view.alpha = it.animatedValue as Float }
            }
        }
        animator.repeatCount = ValueAnimator.INFINITE
        animator.repeatMode = ValueAnimator.REVERSE
        animator.interpolator = AccelerateDecelerateInterpolator()
        animator.start()
        pulse = animator
    }

    private fun tintOf(colorRes: Int): ColorStateList =
        ColorStateList.valueOf(ContextCompat.getColor(view.context, colorRes))
}
