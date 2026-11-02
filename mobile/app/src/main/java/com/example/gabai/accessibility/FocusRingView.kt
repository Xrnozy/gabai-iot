package com.example.gabai.accessibility

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.view.View
import com.example.gabai.R
import com.example.gabai.ui.Ui

/**
 * The visible accessibility focus: a two-tone rounded rectangle (accent outside, canvas colour inside) that stays readable on
 * any background, light or dark. It slides between elements so a sighted helper can follow what a blind user is hearing.
 */
class FocusRingView(context: Context) : View(context) {

    private val outer = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; color = Ui.color(context, R.color.focus_ring); strokeWidth = Ui.dp(context, 5).toFloat() }
    private val inner = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; color = Ui.color(context, R.color.canvas); strokeWidth = Ui.dp(context, 2).toFloat() }
    private val radius = Ui.dp(context, 12).toFloat()
    private val pad = Ui.dp(context, 3).toFloat()
    private val shown = RectF()
    private var from = RectF()
    private var to = RectF()
    private var visible = false
    private val anim = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 160
        addUpdateListener {
            val t = it.animatedValue as Float
            shown.set(from.left + (to.left - from.left) * t, from.top + (to.top - from.top) * t,
                from.right + (to.right - from.right) * t, from.bottom + (to.bottom - from.bottom) * t)
            invalidate()
        }
    }

    init {
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
        isClickable = false
    }

    fun moveTo(r: RectF, animate: Boolean) {
        to = RectF(r).apply { inset(-pad, -pad) }
        if (!visible || !animate) {
            visible = true
            from = RectF(to)
            shown.set(to)
            invalidate()
        } else {
            from = RectF(shown)
            anim.cancel(); anim.start()
        }
    }

    fun clear() {
        anim.cancel()
        visible = false
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        if (!visible) return
        canvas.drawRoundRect(shown, radius, radius, outer)
        val i = RectF(shown).apply { inset(outer.strokeWidth / 2 + inner.strokeWidth / 2, outer.strokeWidth / 2 + inner.strokeWidth / 2) }
        canvas.drawRoundRect(i, radius - 2, radius - 2, inner)
    }
}
