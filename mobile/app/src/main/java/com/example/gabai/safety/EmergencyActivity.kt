package com.example.gabai.safety

import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.TextView
import com.example.gabai.BaseActivity
import com.example.gabai.R
import com.example.gabai.ui.Ui

/** Full-screen countdown. One huge "I'm okay" target, because the person pressing it may have just fallen. */
class EmergencyActivity : BaseActivity() {

    private lateinit var headline: TextView
    private lateinit var count: TextView
    private lateinit var detail: TextView
    private lateinit var cancel: TextView
    private lateinit var sendNow: TextView
    private lateinit var close: TextView
    private val onChange: () -> Unit = { runOnUiThread { render() } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (Build.VERSION.SDK_INT >= 27) { setShowWhenLocked(true); setTurnScreenOn(true) }
        else window.addFlags(WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val c = this
        headline = Ui.text(c, "", 32f, R.color.ink, bold = true).apply { gravity = Gravity.CENTER }
        // The countdown number is the only large red element in the app. It is not a live region: TTS speaks at 10 s and 5 s.
        count = Ui.text(c, "", 128f, R.color.status_error, bold = true).apply {
            gravity = Gravity.CENTER
            setLineSpacing(0f, 1f)
            includeFontPadding = false
        }
        detail = Ui.body(c, "").apply { gravity = Gravity.CENTER; textSize = 20f }
        cancel = Ui.button(c, "I'm okay — cancel", Ui.Kind.PRIMARY, "I'm okay. Cancel the alert. Button.") { Emergency.cancel(c) }
        sendNow = Ui.button(c, "Send alert now", Ui.Kind.DANGER, "Send the alert now. Button.") { Emergency.sendNow(c) }.apply {
            // Outline until focused, so the loud action is never the easy one to hit by accident.
            setBackgroundResource(R.drawable.bg_sos_outline)
            setTextColor(Ui.color(c, R.color.ink))
            setOnFocusChangeListener { v, focused -> (v as TextView).setTextColor(Ui.color(c, if (focused) R.color.on_tape else R.color.ink)) }
        }
        close = Ui.button(c, "Close", Ui.Kind.SECONDARY, "Close this screen. Button.") { Emergency.dismiss(); finish() }
        headline.accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_ASSERTIVE

        val col = LinearLayout(c).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(Ui.color(c, R.color.canvas))
            setPadding(Ui.dp(c, 24), Ui.dp(c, 24), Ui.dp(c, 24), Ui.dp(c, 24))
            addView(headline, Ui.lp(ctx = c))
            addView(count, Ui.lp(top = 8, ctx = c))
            addView(detail, Ui.lp(top = 8, ctx = c))
            addView(cancel, Ui.lp(h = Ui.dp(c, 120), top = 32, ctx = c))
            addView(sendNow, Ui.lp(top = 34, ctx = c))
            addView(close, Ui.lp(top = 16, ctx = c))
        }
        setContentView(col)
    }

    override fun onStart() { super.onStart(); Emergency.listen(onChange); render() }
    override fun onStop() { Emergency.unlisten(onChange); super.onStop() }

    private fun render() {
        when (Emergency.phase) {
            Emergency.Phase.COUNTDOWN -> {
                val what = if (Emergency.kind == "fall") "Possible fall" else "Emergency"
                headline.text = what
                count.text = "${Emergency.secondsLeft()}"
                count.visibility = View.VISIBLE
                detail.text = "Your caregivers will be alerted when the countdown ends."
                cancel.visibility = View.VISIBLE; sendNow.visibility = View.VISIBLE; close.visibility = View.GONE
            }
            Emergency.Phase.SENT -> {
                headline.text = "Alert sent"
                count.visibility = View.GONE
                val by = Emergency.ackedBy
                detail.text = if (by != null) "$by has seen your alert." else "Waiting for a caregiver to respond."
                cancel.text = "I'm okay — cancel the alert"
                cancel.visibility = if (by == null) View.VISIBLE else View.GONE
                sendNow.visibility = View.GONE; close.visibility = View.VISIBLE
            }
            Emergency.Phase.IDLE -> finish()
        }
    }
}
