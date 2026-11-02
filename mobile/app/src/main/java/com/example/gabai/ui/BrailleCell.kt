package com.example.gabai.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.example.gabai.R

/**
 * Six-dot braille cell. Raised dots use [dotColor]; unraised ones the faint braille_off. Purely visual, so it is hidden
 * from accessibility: every state it shows is also spoken or labelled elsewhere.
 */
class BrailleCell @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    private val dot = resources.getDimension(R.dimen.braille_dot)
    private val gap = resources.getDimension(R.dimen.braille_dot_gap)
    private val on = Paint(Paint.ANTI_ALIAS_FLAG)
    private val off = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = ContextCompat.getColor(context, R.color.braille_off) }

    /** Dim the whole cell when the thing it represents is idle. */
    var active: Boolean = true
        set(v) { field = v; alpha = if (v) 1f else 0.45f }

    var mask: Int = 0
        set(v) { field = v; invalidate() }

    init {
        on.color = ContextCompat.getColor(context, R.color.braille_on)
        context.obtainStyledAttributes(attrs, R.styleable.BrailleCell).apply {
            mask = getInt(R.styleable.BrailleCell_cellMask, 0)
            if (hasValue(R.styleable.BrailleCell_dotColor)) on.color = getColor(R.styleable.BrailleCell_dotColor, on.color)
            recycle()
        }
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    override fun onMeasure(w: Int, h: Int) {
        setMeasuredDimension((dot * 2 + gap * 1.2f).toInt(), (dot * 3 + gap * 2).toInt())
    }

    override fun onDraw(canvas: Canvas) {
        val r = dot / 2f
        for (i in 0 until 6) {
            val col = i / 3
            val row = i % 3
            val cx = r + col * (dot + gap * 1.2f)
            val cy = r + row * (dot + gap)
            canvas.drawCircle(cx, cy, r, if (mask and (1 shl i) != 0) on else off)
        }
    }
}
