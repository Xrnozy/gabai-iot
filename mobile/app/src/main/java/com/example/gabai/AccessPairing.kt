package com.example.gabai

import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.LinearLayout
import com.example.gabai.safety.RelayApi
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.TtsService
import com.example.gabai.ui.Level
import com.example.gabai.ui.Ui

/**
 * Pairing without QR codes. A new caregiver phone asks; the primary phone shows who asked with a 4-digit code that must match
 * the code on the asking phone, and approves or declines. Approval hands the asking phone its own token once.
 * Call [start]/[stop] with the screen so nothing polls in the background.
 */
class AccessPairing(private val act: BaseActivity, private val isUser: Boolean, private val onPaired: () -> Unit) {
    private val api = RelayApi(act, if (isUser) SettingsStore.ROLE_PHONE else SettingsStore.ROLE_CAREGIVER)
    private val body: LinearLayout = Ui.column(act)
    private val ui = Handler(Looper.getMainLooper())
    private val announced = HashSet<String>()
    private var tts: TtsService? = null
    private var running = false
    private var requestId: String? = null           // caregiver side: the open request
    private var secret = ""
    private var code = ""
    private var note: String? = null
    val view: View get() = body

    fun start() {
        running = true
        if (!isUser && SettingsStore.getRelayToken(act, SettingsStore.ROLE_CAREGIVER).isBlank() && requestId == null) ask()   // no button to find: asking is the first step
        tick()
    }
    fun stop() { running = false; ui.removeCallbacksAndMessages(null); tts?.shutdown(); tts = null }

    private fun tick() {
        if (!running) return
        Thread {
            if (isUser) { val r = api.pendingAccess(); post { renderPending(r.body?.optJSONArray("requests"), r.error) } }
            else if (SettingsStore.getRelayToken(act, SettingsStore.ROLE_CAREGIVER).isNotBlank()) post { requestId = null; onPaired() }   // approved: the token is stored
            else if (requestId != null) pollRequest()
            else post { renderAsk() }
            post { ui.postDelayed(::tick, 2000) }
        }.start()
    }

    private fun post(f: () -> Unit) = act.runOnUiThread { if (running && !act.isFinishing && !act.isDestroyed) f() }

    // ---- primary phone ----
    private fun renderPending(list: org.json.JSONArray?, error: String?) {
        body.removeAllViews()
        if (list == null) { body.addView(Ui.kv(act, "Requests", error ?: "Unavailable", Level.WARN)); return }
        if (list.length() == 0) { body.addView(Ui.caption(act, "No one is asking to connect.")); return }
        for (i in 0 until list.length()) {
            val r = list.getJSONObject(i); val id = r.getString("id"); val name = r.getString("name"); val c = r.getString("code")
            body.addView(Ui.kv(act, name, "Wants to connect. Code $c", Level.WARN), Ui.lp(top = 12, ctx = act))
            if (announced.add(id)) {
                if (tts == null) tts = TtsService(act.applicationContext)
                tts?.speak("$name wants to connect. The code is ${c.toList().joinToString(", ")}. Check it matches their phone, then tap Allow.", interrupt = false)
            }
            body.addView(Ui.button(act, "Allow $name", Ui.Kind.PRIMARY, "Allow $name. Check that code $c is on their phone. Button.") { decide(id, true) }, Ui.lp(top = 8, ctx = act))
            body.addView(Ui.button(act, "Decline", Ui.Kind.SECONDARY, "Decline $name. Button.") { decide(id, false) }, Ui.lp(top = 8, ctx = act))
        }
    }

    private fun decide(id: String, approve: Boolean) = Thread { api.decideAccess(id, approve); post { tick() } }.start()

    // ---- caregiver phone ----
    private fun renderAsk() {
        body.removeAllViews()
        note?.let { body.addView(Ui.kv(act, "Note", it, Level.WARN), Ui.lp(top = 8, ctx = act)) }
        body.addView(Ui.body(act, "Ask the person to open GabAi on their phone. Then tap the button below."), Ui.lp(top = 8, ctx = act))
        body.addView(Ui.button(act, "Ask to connect", Ui.Kind.PRIMARY, "Ask the person to let you connect. Button.") { ask() }, Ui.lp(top = 12, ctx = act))
    }

    private fun ask() = Thread {
        val r = api.requestAccess(SettingsStore.getCaregiverName(act))
        post {
            if (r.ok) { requestId = r.body!!.getString("request_id"); secret = r.body!!.getString("secret"); code = r.body!!.getString("code"); note = null }
            else note = r.error
            renderWaiting(); tick()
        }
    }.start()

    private fun renderWaiting() {
        body.removeAllViews()
        if (requestId == null) { renderAsk(); return }
        body.addView(Ui.body(act, "Waiting for them to tap Allow. Their phone must show this code:"), Ui.lp(top = 8, ctx = act))
        body.addView(Ui.text(act, code.toList().joinToString(" "), 40f, bold = true).apply { gravity = android.view.Gravity.CENTER }, Ui.lp(top = 12, ctx = act))
        body.addView(Ui.caption(act, "The request ends after 2 minutes."), Ui.lp(top = 8, ctx = act))
    }

    private fun pollRequest() {
        val id = requestId ?: return
        val r = api.pollAccess(id, secret)
        val status = r.body?.optString("status")
        val token = r.body?.optString("token").orEmpty()
        post {
            when {
                status == "approved" && token.isNotBlank() -> {
                    SettingsStore.setRelayToken(act, SettingsStore.ROLE_CAREGIVER, token); requestId = null; onPaired()
                }
                status == "pending" -> renderWaiting()
                r.ok || r.code == 404 -> { note = if (status == "rejected") "They declined." else "The request ended. Try again."; requestId = null; renderAsk() }
                else -> { note = r.error; renderWaiting() }   // relay unreachable: keep the request, keep trying
            }
        }
    }
}
