package com.example.gabai.ui

import android.content.Context
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import com.example.gabai.R

/**
 * A card of label/value rows that updates in place. Rows are created once and only their text changes, so a screen reader
 * that is focused on a row keeps its place while the numbers refresh every couple of seconds.
 */
class LiveCard(private val ctx: Context, heading: String) {

    private class Row(val box: LinearLayout, val value: TextView, val chip: TextView)

    val view: LinearLayout = Ui.card(ctx, heading)
    private val body = Ui.column(ctx).also { view.addView(it, Ui.lp(top = 4, ctx = ctx)) }
    private val rows = LinkedHashMap<String, Row>()
    private val touched = HashSet<String>()

    fun begin() = touched.clear()

    fun set(label: String, value: String, level: Level? = null) {
        touched += label
        val row = rows.getOrPut(label) { newRow(label) }
        if (row.value.text != value) row.value.text = value
        if (level == null) {
            row.chip.visibility = View.GONE
        } else {
            row.chip.visibility = View.VISIBLE
            val t = "${level.symbol} ${level.word}"
            if (row.chip.text != t) {
                row.chip.text = t
                row.chip.setTextColor(Ui.color(ctx, level.color))
            }
        }
        val d = "$label: $value" + (level?.let { ", ${it.word}" } ?: "")
        if (row.box.contentDescription != d) row.box.contentDescription = d
    }

    /** Removes rows that were not set since [begin]. */
    fun end() {
        val gone = rows.keys.filter { it !in touched }
        gone.forEach { body.removeView(rows.remove(it)!!.box) }
    }

    fun add(v: View) = view.addView(v, Ui.lp(top = 12, ctx = ctx))

    private fun newRow(label: String): Row {
        val box = Ui.row(ctx).apply { minimumHeight = Ui.dp(ctx, 40); importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES }
        val l = Ui.caption(ctx, label).apply { importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO }
        val chip = Ui.chip(ctx, Level.OFF).apply { visibility = View.GONE }
        val v = Ui.text(ctx, "", 16f, R.color.ink, bold = true).apply { gravity = Gravity.END; importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO }
        box.addView(l, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        box.addView(chip, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).also { it.marginEnd = Ui.dp(ctx, 8) })
        box.addView(v, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.4f))
        body.addView(box)
        return Row(box, v, chip)
    }
}
