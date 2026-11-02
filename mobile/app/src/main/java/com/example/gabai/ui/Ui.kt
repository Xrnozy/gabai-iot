package com.example.gabai.ui

import android.content.Context
import android.graphics.Typeface
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import com.example.gabai.R

/** Health of a thing, shown as colour AND a symbol AND a word, so no status depends on colour alone. */
enum class Level(val symbol: String, val word: String, val color: Int) {
    OK("✓", "OK", R.color.status_success),
    WARN("!", "Check", R.color.status_warning),
    BAD("✕", "Problem", R.color.status_error),
    OFF("–", "Off", R.color.muted),
}

/**
 * Code-built widgets in the app design system (cream canvas, white cards, terracotta accent, 8 dp grid, 48 dp+ targets).
 * Building them here keeps the caregiver screens and the blind user's screens on the same tokens.
 */
object Ui {

    fun dp(c: Context, v: Int) = (v * c.resources.displayMetrics.density).toInt()

    fun color(c: Context, res: Int) = ContextCompat.getColor(c, res)

    fun column(c: Context, pad: Int = 0) = LinearLayout(c).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(c, pad), dp(c, pad), dp(c, pad), dp(c, pad))
    }

    fun row(c: Context) = LinearLayout(c).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
    }

    fun lp(w: Int = ViewGroup.LayoutParams.MATCH_PARENT, h: Int = ViewGroup.LayoutParams.WRAP_CONTENT, top: Int = 0, ctx: Context? = null, weight: Float = 0f) =
        LinearLayout.LayoutParams(w, h, weight).apply { if (ctx != null) topMargin = dp(ctx, top) }

    fun text(c: Context, s: CharSequence, sizeSp: Float, colorRes: Int = R.color.ink, bold: Boolean = false) = TextView(c).apply {
        text = s
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sizeSp)
        setTextColor(color(c, colorRes))
        typeface = if (bold) Typeface.create("sans-serif-medium", Typeface.NORMAL) else Typeface.SANS_SERIF
        setLineSpacing(0f, 1.2f)
    }

    fun title(c: Context, s: CharSequence) = text(c, s, 26f, bold = true).also { ViewCompat.setAccessibilityHeading(it, true) }
    fun heading(c: Context, s: CharSequence) = text(c, s, 18f, bold = true).also { ViewCompat.setAccessibilityHeading(it, true) }
    fun body(c: Context, s: CharSequence) = text(c, s, 16f, R.color.body)
    fun caption(c: Context, s: CharSequence) = text(c, s, 14f, R.color.muted)

    /** White rounded card with a heading; add children to the returned column. */
    fun card(c: Context, headingText: String? = null): LinearLayout = column(c, 16).apply {
        setBackgroundResource(R.drawable.bg_card_cream)
        elevation = dp(c, 2).toFloat()
        if (headingText != null) addView(heading(c, headingText), lp(ctx = c))
    }

    /** Label on the left, value on the right; the pair is one accessibility unit that reads "label, value". */
    fun kv(c: Context, label: String, value: String, level: Level? = null): View = row(c).apply {
        minimumHeight = dp(c, 40)
        val l = caption(c, label)
        val v = text(c, value, 16f, R.color.ink, bold = true).apply { gravity = Gravity.END }
        addView(l, lp(0, ViewGroup.LayoutParams.WRAP_CONTENT, weight = 1f))
        if (level != null) addView(chip(c, level), lp(ViewGroup.LayoutParams.WRAP_CONTENT).also { it.marginEnd = dp(c, 8) })
        addView(v, lp(ViewGroup.LayoutParams.WRAP_CONTENT))
        contentDescription = "$label: $value" + (level?.let { ", ${it.word}" } ?: "")
        importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
        (getChildAt(0) as TextView).importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        v.importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    fun chip(c: Context, level: Level, label: String = level.word) = TextView(c).apply {
        text = "${level.symbol} $label"
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
        setTextColor(color(c, level.color))
        typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        setBackgroundResource(R.drawable.bg_chip)
        importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    enum class Kind { PRIMARY, SECONDARY, DANGER }

    fun button(c: Context, label: String, kind: Kind = Kind.PRIMARY, description: String = label, onClick: () -> Unit) = TextView(c).apply {
        text = label
        gravity = Gravity.CENTER
        minHeight = dp(c, 64)
        setPadding(dp(c, 16), dp(c, 12), dp(c, 16), dp(c, 12))
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
        typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        setTextColor(color(c, if (kind == Kind.SECONDARY) R.color.ink else R.color.on_primary))
        setBackgroundResource(when (kind) {
            Kind.PRIMARY -> R.drawable.bg_btn_primary_focusable
            Kind.SECONDARY -> R.drawable.bg_btn_secondary_focusable
            Kind.DANGER -> R.drawable.bg_btn_danger_focusable
        })
        isClickable = true
        isFocusable = true
        contentDescription = description
        setOnClickListener { onClick() }
    }

    fun field(c: Context, hint: String, value: String, secret: Boolean = false) = android.widget.EditText(c).apply {
        setText(value)
        this.hint = hint
        contentDescription = hint
        minHeight = dp(c, 56)
        setBackgroundResource(R.drawable.bg_field)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
        setTextColor(color(c, R.color.ink))
        setHintTextColor(color(c, R.color.muted))
        setSingleLine()
        inputType = android.text.InputType.TYPE_CLASS_TEXT or
            (if (secret) android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD else android.text.InputType.TYPE_TEXT_VARIATION_URI)
    }

    fun space(c: Context, h: Int) = View(c).apply { layoutParams = LinearLayout.LayoutParams(1, dp(c, h)) }
}
