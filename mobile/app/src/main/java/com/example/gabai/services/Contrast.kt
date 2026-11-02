package com.example.gabai.services

import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.example.gabai.R

/**
 * Runtime high-contrast pass: secondary text tokens (muted, body and their onboarding/on-dark twins) are pushed to the
 * full-strength ink token of the current theme, so text lands at the top of the contrast range in both dark (white)
 * and light (black) schemes. Tape, accent and semantic colours are left untouched.
 */
object Contrast {

    private val SECONDARY_TOKENS = intArrayOf(
        R.color.muted, R.color.body, R.color.onb_muted, R.color.on_dark_muted, R.color.on_dark_soft,
    )

    fun apply(root: View) {
        val c = root.context
        walk(root, SECONDARY_TOKENS.map { ContextCompat.getColor(c, it) }.toSet(), ContextCompat.getColor(c, R.color.ink))
    }

    private fun walk(v: View, secondary: Set<Int>, ink: Int) {
        if (v is TextView && v.currentTextColor in secondary) v.setTextColor(ink)
        if (v is ViewGroup) {
            for (i in 0 until v.childCount) walk(v.getChildAt(i), secondary, ink)
        }
    }
}
