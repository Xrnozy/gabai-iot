package com.example.gabai

import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.example.gabai.ui.Ui

/**
 * Three-card primer on the design tokens the app is built from: colour, type, spacing.
 * Each card shows the real tokens, so it can never drift from the design system.
 */
class TokensTutorialActivity : BaseActivity() {

    private class Card(val title: String, val intro: String, val build: (LinearLayout) -> Unit)

    private lateinit var cards: List<Card>
    private lateinit var body: LinearLayout
    private lateinit var step: TextView
    private lateinit var next: TextView
    private lateinit var back: TextView
    private var index = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val c = this
        cards = listOf(
            Card("Colour", "Colours have names, not hex codes. Use the name and the app follows light and dark mode for you.", ::colourCard),
            Card("Type", "Text sizes come in sp, so they grow with the user's font setting. Body text is never below 20sp.", ::typeCard),
            Card("Spacing", "Everything sits on an 8dp grid. Pick a step from the scale; never invent a number.", ::spacingCard),
        )

        body = Ui.column(c)
        step = Ui.caption(c, "")
        next = Ui.button(c, "", Ui.Kind.PRIMARY) { go(index + 1) }
        back = Ui.button(c, "Back", Ui.Kind.SECONDARY) { go(index - 1) }

        val page = Ui.column(c, 24).apply {
            addView(step, Ui.lp(ctx = c))
            addView(body, Ui.lp(top = 8, ctx = c))
            addView(next, Ui.lp(top = 24, ctx = c))
            addView(back, Ui.lp(top = 12, ctx = c))
        }
        setContentView(ScrollView(c).apply {
            setBackgroundColor(Ui.color(c, R.color.canvas))
            isFillViewport = true
            addView(page)
        })
        render()
    }

    private fun go(to: Int) {
        if (to >= cards.size) { finish(); return }
        if (to < 0) return
        index = to
        render()
    }

    private fun render() {
        val c = this
        val card = cards[index]
        step.text = "Design tokens, ${index + 1} of ${cards.size}"
        body.removeAllViews()
        body.addView(Ui.title(c, card.title), Ui.lp(ctx = c))
        body.addView(Ui.body(c, card.intro), Ui.lp(top = 8, ctx = c))
        val demo = Ui.card(c).also { body.addView(it, Ui.lp(top = 16, ctx = c)) }
        card.build(demo)
        next.text = if (index == cards.size - 1) "Done" else "Next"
        back.visibility = if (index == 0) View.GONE else View.VISIBLE
        step.announceForAccessibility("${card.title}. Step ${index + 1} of ${cards.size}.")
    }

    private fun swatch(parent: LinearLayout, token: String, res: Int) {
        val c = this
        parent.addView(Ui.row(c).apply {
            minimumHeight = Ui.dp(c, 48)
            addView(View(c).apply { setBackgroundColor(Ui.color(c, res)) }, LinearLayout.LayoutParams(Ui.dp(c, 40), Ui.dp(c, 40)))
            addView(Ui.text(c, token, 16f, bold = true), LinearLayout.LayoutParams(0, -2, 1f).also { it.marginStart = Ui.dp(c, 16) })
            contentDescription = "$token colour"
        }, Ui.lp(top = 8, ctx = c))
    }

    private fun colourCard(into: LinearLayout) {
        swatch(into, "canvas", R.color.canvas)
        swatch(into, "ink", R.color.ink)
        swatch(into, "tape_navigate", R.color.tape_navigate)
        swatch(into, "tape_read", R.color.tape_read)
        swatch(into, "status_error", R.color.status_error)
    }

    private fun typeCard(into: LinearLayout) {
        val c = this
        listOf("text_title" to 26f, "text_body_lg" to 22f, "text_body" to 20f, "text_caption" to 16f).forEach { (name, sp) ->
            into.addView(Ui.text(c, "$name  ${sp.toInt()}sp", sp), Ui.lp(top = 8, ctx = c))
        }
    }

    private fun spacingCard(into: LinearLayout) {
        val c = this
        listOf("space_sm" to 8, "space_lg" to 16, "space_2xl" to 24, "space_3xl" to 32).forEach { (name, dp) ->
            into.addView(Ui.row(c).apply {
                addView(View(c).apply { setBackgroundColor(Ui.color(c, R.color.tape_navigate)) }, LinearLayout.LayoutParams(Ui.dp(c, dp), Ui.dp(c, 16)))
                addView(Ui.text(c, "$name  ${dp}dp", 16f), LinearLayout.LayoutParams(-2, -2).also { it.marginStart = Ui.dp(c, 16) })
            }, Ui.lp(top = 8, ctx = c))
        }
    }
}
