package com.example.gabai

import android.content.Intent
import android.os.Bundle
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import androidx.appcompat.app.AlertDialog
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.TtsService
import com.example.gabai.ui.Level
import com.example.gabai.ui.Ui

/**
 * The one place to pair a user with a caregiver. It always shows what to do next for this phone's role: numbered steps,
 * the relay's health with a fix when it is down, and the access request with its 4-digit check code. Nothing hides while the
 * relay is unreachable, so the way to connect is never something to find or guess.
 */
class ConnectActivity : BaseActivity() {

    private val user get() = SettingsStore.getAppRole(this) != SettingsStore.APP_CAREGIVER
    private lateinit var links: FamilyLinks
    private lateinit var status: LinearLayout
    private var tts: TtsService? = null
    private var checking = true

    private lateinit var access: AccessPairing

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val c = this
        links = FamilyLinks(c, if (user) "user" else "caregiver") { }
        links.onLoaded = { checking = false; renderStatus() }
        access = AccessPairing(c, user) {
            links.refresh()
            if (!user) { startActivity(Intent(c, com.example.gabai.caregiver.CaregiverActivity::class.java)); finish() }   // approved: show the dashboard
        }

        val col = Ui.column(c, 24)
        col.addView(Ui.button(c, "← Back", Ui.Kind.SECONDARY, "Back. Button.") { leave() }, Ui.lp(ctx = c))
        col.addView(Ui.title(c, if (user) "Connect a caregiver" else "Connect to someone"), Ui.lp(top = 16, ctx = c))
        col.addView(Ui.button(c, "Skip for now", Ui.Kind.SECONDARY, "Skip connecting. Use the app on this phone without a second phone. Button.") { skip() }, Ui.lp(top = 12, ctx = c))

        val steps = Ui.card(c, "How to connect").also { col.addView(it, Ui.lp(top = 16, ctx = c)) }
        stepsFor().forEachIndexed { i, line -> steps.addView(Ui.text(c, "${i + 1}.  $line", 18f, R.color.ink), Ui.lp(top = 12, ctx = c)) }

        status = Ui.column(c).apply { accessibilityLiveRegion = android.view.View.ACCESSIBILITY_LIVE_REGION_POLITE }
        col.addView(Ui.card(c, "Connection").also { it.addView(status, Ui.lp(top = 8, ctx = c)) }, Ui.lp(top = 16, ctx = c))

        col.addView(Ui.card(c, if (user) "Requests" else "Connect").also { it.addView(access.view, Ui.lp(top = 8, ctx = c)) }, Ui.lp(top = 16, ctx = c))
        col.addView(Ui.card(c, "Linked people").also { it.addView(links.view, Ui.lp(top = 8, ctx = c)) }, Ui.lp(top = 16, ctx = c))
        col.addView(Ui.space(c, 48))

        setContentView(ScrollView(c).apply { setBackgroundColor(Ui.color(c, R.color.canvas)); addView(col); isFillViewport = true })
        renderStatus()

        tts = TtsService(this, java.util.Locale.US).also {
            it.speak(if (user) "Connect a caregiver. When they ask, a code appears here. Check it matches their phone, then tap Allow."
            else "Connect to someone. Tap Ask to connect, then read the code to them.")
        }
    }

    override fun onStart() { super.onStart(); links.refresh(); access.start() }
    override fun onStop() { access.stop(); super.onStop() }
    override fun onDestroy() { tts?.shutdown(); super.onDestroy() }

    private fun stepsFor() = if (user) listOf(
        "Ask your caregiver to open GabAi and tap Connect.",
        "They tap Ask to connect. A 4-digit code appears on both phones.",
        "If the codes match, tap Allow.",
    ) else listOf(
        "Ask the person to open GabAi and tap Connect.",
        "Tap Ask to connect. A 4-digit code appears.",
        "They check it matches their phone, then tap Allow.",
    )

    /** One phone only: go straight into the app. A caregiver sees the dashboard empty, without being sent back to a link screen. */
    private fun skip() {
        startActivity(if (user) Intent(this, MainActivity::class.java)
        else Intent(this, com.example.gabai.caregiver.CaregiverActivity::class.java).putExtra(com.example.gabai.caregiver.CaregiverActivity.EXTRA_DEBUG_PREVIEW, true))
        finish()
    }

    private fun leave() {
        if (isTaskRoot) startActivity(Intent(this, MainActivity::class.java))
        finish()
    }

    private fun toast(s: String) = android.widget.Toast.makeText(this, s, android.widget.Toast.LENGTH_LONG).show()

    private fun renderStatus() {
        val c = this
        status.removeAllViews()
        val relay = SettingsStore.getRelayUrl(c)
        when {
            checking -> status.addView(Ui.kv(c, "Relay", "Checking…", Level.OFF))
            links.unreachable -> {
                status.addView(Ui.kv(c, "Relay", "Not reachable", Level.BAD))
                status.addView(Ui.body(c, "This phone cannot reach $relay. Start the relay on your computer, make sure both phones are on the PC hotspot, or change the address if it moved."), Ui.lp(top = 12, ctx = c))
                status.addView(Ui.button(c, "Try again", Ui.Kind.PRIMARY, "Try the relay again. Button.") { checking = true; renderStatus(); links.retry() }, Ui.lp(top = 12, ctx = c))
                status.addView(Ui.button(c, "Change relay address", Ui.Kind.SECONDARY, "Change the relay address. Button.") { changeRelay() }, Ui.lp(top = 12, ctx = c))
            }
            links.connected -> {
                status.addView(Ui.kv(c, "Relay", "Connected", Level.OK))
                status.addView(Ui.caption(c, relay), Ui.lp(top = 4, ctx = c))
            }
            else -> status.addView(Ui.kv(c, "Relay", "Setting up", Level.WARN))
        }
    }

    private fun changeRelay() {
        val input = EditText(this).apply {
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_URI
            setText(SettingsStore.getRelayUrl(this@ConnectActivity)); setSingleLine()
        }
        AlertDialog.Builder(this)
            .setTitle("Relay address")
            .setMessage("A new relay needs a new profile, so this phone signs up there again.")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                val url = input.text.toString().trim()
                if (!(url.startsWith("http://") || url.startsWith("https://"))) { toast("Start with http:// or https://"); return@setPositiveButton }
                SettingsStore.setRelayUrl(this, url)
                SettingsStore.setRelayToken(this, if (user) SettingsStore.ROLE_PHONE else SettingsStore.ROLE_CAREGIVER, "")
                checking = true; renderStatus(); links.retry()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
}
