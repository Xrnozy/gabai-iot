package com.example.gabai

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import androidx.appcompat.app.AlertDialog
import com.example.gabai.safety.PairQr
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.TtsService
import com.example.gabai.ui.Level
import com.example.gabai.ui.Ui
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanOptions

/**
 * The one place to pair a user with a caregiver. It always shows what to do next for this phone's role: numbered steps,
 * the relay's health with a fix when it is down, and the pairing code as both a QR and typed text. Nothing hides while the
 * relay is unreachable, so the way to connect is never something to find or guess.
 */
class ConnectActivity : BaseActivity() {

    private val user get() = SettingsStore.getAppRole(this) != SettingsStore.APP_CAREGIVER
    private lateinit var links: FamilyLinks
    private lateinit var status: LinearLayout
    private lateinit var pair: LinearLayout
    private var tts: TtsService? = null
    private var checking = true

    private val scanner = registerForActivityResult(ScanContract()) { result ->
        val text = result.contents ?: return@registerForActivityResult
        val scanned = PairQr.decode(text)
        if (scanned == null) { toast("That is not a GabAi pairing code."); return@registerForActivityResult }
        if (scanned.relay == SettingsStore.getRelayUrl(this)) { links.invite(scanned.handle, null); return@registerForActivityResult }
        // A code can name any server, so the caregiver sees the address before the app switches to it.
        AlertDialog.Builder(this)
            .setTitle("Use this relay?")
            .setMessage("This code connects you through ${scanned.relay}. Only continue if you trust the person who showed it.")
            .setPositiveButton("Use it") { _, _ -> links.invite(scanned.handle, scanned.relay) }
            .setNegativeButton("Cancel", null)
            .show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val c = this
        links = FamilyLinks(c, if (user) "user" else "caregiver") { }
        links.onLoaded = { checking = false; renderStatus(); renderPair() }

        val col = Ui.column(c, 24)
        col.addView(Ui.button(c, "← Back", Ui.Kind.SECONDARY, "Back. Button.") { leave() }, Ui.lp(ctx = c))
        col.addView(Ui.title(c, if (user) "Connect a caregiver" else "Connect to someone"), Ui.lp(top = 16, ctx = c))

        val steps = Ui.card(c, "How to connect").also { col.addView(it, Ui.lp(top = 16, ctx = c)) }
        stepsFor().forEachIndexed { i, line -> steps.addView(Ui.text(c, "${i + 1}.  $line", 18f, R.color.ink), Ui.lp(top = 12, ctx = c)) }

        status = Ui.column(c).apply { accessibilityLiveRegion = android.view.View.ACCESSIBILITY_LIVE_REGION_POLITE }
        col.addView(Ui.card(c, "Connection").also { it.addView(status, Ui.lp(top = 8, ctx = c)) }, Ui.lp(top = 16, ctx = c))

        pair = Ui.card(c, if (user) "Your code" else "Connect").also { col.addView(it, Ui.lp(top = 16, ctx = c)) }
        col.addView(Ui.card(c, "Linked people").also { it.addView(links.view, Ui.lp(top = 8, ctx = c)) }, Ui.lp(top = 16, ctx = c))
        col.addView(Ui.space(c, 48))

        setContentView(ScrollView(c).apply { setBackgroundColor(Ui.color(c, R.color.canvas)); addView(col); isFillViewport = true })
        renderStatus()
        renderPair()

        tts = TtsService(this, java.util.Locale.US).also {
            it.speak(if (user) "Connect a caregiver. Show your code to them, or read it aloud. When they ask, tap Allow."
            else "Connect to someone. Tap Scan QR code and point at their phone, or type their code.")
        }
    }

    override fun onStart() { super.onStart(); links.refresh() }
    override fun onDestroy() { tts?.shutdown(); super.onDestroy() }

    private fun stepsFor() = if (user) listOf(
        "Ask your caregiver to open GabAi and tap Connect.",
        "Show them the code below, or read it aloud to them.",
        "When their name appears under Linked people, tap Allow.",
    ) else listOf(
        "Ask the person to open GabAi and tap Connect.",
        "Tap Scan QR code and point your camera at their screen.",
        "No camera? Type the code under their QR code instead.",
        "Wait for them to tap Allow on their phone.",
    )

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
                status.addView(Ui.body(c, "This phone cannot reach $relay. Start the relay on your computer, make sure both phones are on the same Wi-Fi or Tailscale, or change the address if it moved."), Ui.lp(top = 12, ctx = c))
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

    private fun renderPair() {
        val c = this
        pair.removeAllViews()
        pair.addView(Ui.caption(c, if (user) "Show this to your caregiver." else "Scan their code, or type it."), Ui.lp(top = 4, ctx = c))
        if (user) renderMyCode() else renderScan()
    }

    private fun renderMyCode() {
        val c = this
        val handle = links.handle
        if (handle == null) {
            pair.addView(Ui.body(c, if (links.unreachable) "Your code needs the relay. Fix the connection above first." else "Your code appears here in a moment."), Ui.lp(top = 12, ctx = c))
            return
        }
        val size = Ui.dp(c, 280)
        pair.addView(ImageView(c).apply {
            setImageBitmap(PairQr.bitmap(PairQr.encode(SettingsStore.getRelayUrl(c), handle)))
            setBackgroundColor(Color.WHITE)
            setPadding(Ui.dp(c, 12), Ui.dp(c, 12), Ui.dp(c, 12), Ui.dp(c, 12))
            contentDescription = "QR code for ${PairQr.formatCode(handle)}. Your caregiver scans this with their phone."
        }, LinearLayout.LayoutParams(size, size).also { it.gravity = Gravity.CENTER_HORIZONTAL; it.topMargin = Ui.dp(c, 12) })
        pair.addView(Ui.caption(c, "Or give them this code"), Ui.lp(top = 16, ctx = c))
        pair.addView(Ui.text(c, PairQr.formatCode(handle), 30f, bold = true).apply {
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
            gravity = Gravity.CENTER
            setTextIsSelectable(true)
        }, Ui.lp(top = 4, ctx = c))
        pair.addView(Ui.button(c, "Read code aloud", Ui.Kind.PRIMARY, "Read my code aloud, letter by letter. Button.") {
            tts?.speak("Your code is " + PairQr.formatCode(handle).map { if (it == '-') "dash" else it.toString() }.joinToString(", "))
        }, Ui.lp(top = 16, ctx = c))
        pair.addView(Ui.button(c, "Copy code", Ui.Kind.SECONDARY, "Copy my code. Button.") {
            (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("GabAi code", PairQr.formatCode(handle)))
            toast("Code copied")
        }, Ui.lp(top = 12, ctx = c))
    }

    private fun renderScan() {
        val c = this
        pair.addView(Ui.button(c, "Scan QR code", Ui.Kind.PRIMARY, "Scan the QR code on the person's phone. Button.") {
            scanner.launch(ScanOptions().setDesiredBarcodeFormats(ScanOptions.QR_CODE).setPrompt("Point at the code on their phone").setBeepEnabled(false).setOrientationLocked(false))
        }, Ui.lp(top = 12, ctx = c))
        pair.addView(Ui.caption(c, "No camera? Type their code instead"), Ui.lp(top = 20, ctx = c))
        val code = Ui.field(c, "Their code, like GAB-K3M7X2", "")
        code.inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS
        pair.addView(code, Ui.lp(top = 8, ctx = c))
        pair.addView(Ui.button(c, "Connect", Ui.Kind.SECONDARY, "Send an invite to the code typed above. Button.") {
            val handle = PairQr.parseCode(code.text.toString())
            if (handle == null) { code.error = "Enter their code, like GAB-K3M7X2"; code.requestFocus() } else links.invite(handle, null)
        }, Ui.lp(top = 12, ctx = c))
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
