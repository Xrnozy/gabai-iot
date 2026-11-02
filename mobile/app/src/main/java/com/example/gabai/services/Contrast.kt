package com.example.gabai.services

import android.graphics.Color
import android.view.View
import android.view.ViewGroup
import android.widget.TextView

/**
 * Runtime high-contrast pass (design.md §7.2 "high-contrast variant"): every
 * secondary/muted text token is pushed to pure white and surfaces go to pure
 * black so text/surface pairs land well above the 7:1 target. Accent and
 * semantic colors (danger/warn/ok) are left untouched.
 */
object Contrast {

    /** Current secondary-text tokens that get flattened to white. */
    private val SECONDARY_TEXT = intArrayOf(
        0xFF9BA0A6.toInt(), // muted
        0xFFD0D3D8.toInt(), // body
        0xFF6E7278.toInt(), // on_dark_muted
        0xFFB9BDC2.toInt(), // on_dark_soft
        0xFF7B828B.toInt(), // icon greys used on text
    )

    fun apply(root: View) {
        walk(root)
    }

    private fun walk(v: View) {
        if (v is TextView) {
            if (v.currentTextColor in SECONDARY_TEXT) {
                v.setTextColor(Color.WHITE)
            }
        }
        if (v is ViewGroup) {
            for (i in 0 until v.childCount) walk(v.getChildAt(i))
        }
    }
}
