package com.example.gabai

import android.view.View
import android.widget.LinearLayout
import com.example.gabai.safety.RelayApi
import com.example.gabai.safety.Reply
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.TtsService
import com.example.gabai.ui.Level
import com.example.gabai.ui.Ui
import org.json.JSONObject

/**
 * Caregiver links for one side of the relationship. A caregiver invites a user by handle; nothing is shared until the user
 * allows it, and either side can end the link at any time. [personRole] is "user" (the blind user's phone) or "caregiver".
 * Registering stores a personal token in the same slot the shared tokens use, so the rest of the app keeps working unchanged.
 */
class FamilyLinks(
    private val act: BaseActivity,
    private val personRole: String,
    private val onToken: (String) -> Unit,
) {
    private val user = personRole == "user"
    private val api = RelayApi(act, if (user) SettingsStore.ROLE_PHONE else SettingsStore.ROLE_CAREGIVER)
    private val body: LinearLayout = Ui.column(act)
    private val announced = HashSet<String>()
    private var tts: TtsService? = null
    private var message: String? = null
    val view: View get() = body

    /** Set after every load, for the Connect screen: the user's own handle, and whether the relay could not be reached at all. */
    var handle: String? = null
        private set
    var unreachable = false
        private set
    var connected = false
        private set
    var onLoaded: (() -> Unit)? = null

    fun retry() { failed = false; refresh() }

    fun refresh() {
        Thread {
            val r = api.links()
            act.runOnUiThread { if (!act.isFinishing && !act.isDestroyed) render(r) }
        }.start()
    }

    private fun render(r: Reply) {
        body.removeAllViews()
        connected = r.ok
        unreachable = r.error?.startsWith("Cannot reach") == true || message?.startsWith("Cannot reach") == true
        handle = r.body?.optJSONObject("person")?.optString("handle")?.takeIf { it.isNotBlank() }
        act.runOnUiThread { onLoaded?.invoke() }
        message?.let { body.addView(Ui.kv(act, "Note", it, Level.WARN), Ui.lp(top = 8, ctx = act)) }
        val json = r.body
        val blank = r.code == 0 && r.error?.startsWith("No relay token") == true
        when {
            blank && !user -> {}                                   // a caregiver gets a token by access request, never by registering
            blank && !failed -> { body.addView(Ui.kv(act, "Setting up", "Creating your profile", Level.OFF), Ui.lp(top = 8, ctx = act)); register() }
            blank -> body.addView(Ui.button(act, "Try again", Ui.Kind.SECONDARY) { failed = false; refresh() }, Ui.lp(top = 12, ctx = act))
            r.code == 401 -> {                                   // a typed-in shared token is in use: replacing it is the user's call
                body.addView(Ui.caption(act, "This phone uses a shared token. Switch to a personal profile to use caregiver links. This replaces the token typed above."), Ui.lp(top = 8, ctx = act))
                body.addView(Ui.button(act, "Switch to personal profile", Ui.Kind.SECONDARY) { register() }, Ui.lp(top = 12, ctx = act))
            }
            !r.ok || json == null -> {
                body.addView(Ui.kv(act, "Link status", r.error ?: "Unavailable", Level.BAD), Ui.lp(top = 8, ctx = act))
                body.addView(Ui.button(act, "Try again", Ui.Kind.SECONDARY) { refresh() }, Ui.lp(top = 12, ctx = act))
            }
            else -> renderLinks(json)
        }
    }

    private var registering = false
    private var failed = false                                   // after one failed attempt, wait for the user instead of looping

    /** No typing: the handle is generated, and the name is the one given at role choice (a caregiver) or a plain default (a user). */
    private fun register() {
        if (registering) return
        registering = true
        val name = if (user) "GabAi user" else SettingsStore.getCaregiverName(act)
        Thread {
            var reply = api.register(personRole, name, if (user) newHandle() else null)
            var tries = 1
            while (reply.code == 409 && tries++ < 3) reply = api.register(personRole, name, newHandle())   // handle collision: draw another
            act.runOnUiThread {
                registering = false
                val token = reply.body?.optString("token").orEmpty()
                if (reply.ok && token.isNotBlank()) {
                    SettingsStore.setRelayToken(act, if (user) SettingsStore.ROLE_PHONE else SettingsStore.ROLE_CAREGIVER, token)
                    onToken(token)
                    message = null
                } else { message = reply.error ?: "Could not create the profile."; failed = true }
                if (!act.isFinishing && !act.isDestroyed) refresh()
            }
        }.start()
    }

    private fun newHandle(): String {
        val letters = "abcdefghjkmnpqrstuvwxyz23456789"                      // no 0/o/1/l/i: it is read aloud and typed by someone else
        return "gab-" + (1..6).map { letters.random() }.joinToString("")
    }

    private fun renderLinks(json: JSONObject) {
        val links = json.getJSONArray("links")
        if (user) {
            body.addView(Ui.kv(act, "Your handle", json.getJSONObject("person").optString("handle")), Ui.lp(top = 8, ctx = act))
            body.addView(Ui.caption(act, "Nobody sees your location until you allow them. To add a caregiver, use Connect."), Ui.lp(top = 4, ctx = act))
        }
        if (links.length() == 0) {
            body.addView(Ui.caption(act, if (user) "No caregivers yet." else "No one has allowed you yet."), Ui.lp(top = 12, ctx = act))
        }
        for (i in 0 until links.length()) {
            val l = links.getJSONObject(i)
            val id = l.getString("id")
            val who = l.optString("name")
            if (l.getString("status") == "pending") {
                body.addView(Ui.kv(act, who, "Wants to be your caregiver", Level.WARN), Ui.lp(top = 12, ctx = act))
                if (announced.add(id)) {
                    if (tts == null) tts = TtsService(act.applicationContext)
                    tts?.speak("$who wants to be your caregiver. Allow or decline in Family and safety.", interrupt = false)
                }
                body.addView(Ui.button(act, "Allow $who", Ui.Kind.PRIMARY, "Allow $who to see my location. Button.") { answer(id, "accept") }, Ui.lp(top = 8, ctx = act))
                body.addView(Ui.button(act, "Decline", Ui.Kind.SECONDARY, "Decline $who. Button.") { answer(id, "decline") }, Ui.lp(top = 8, ctx = act))
                body.addView(Ui.button(act, "Block", Ui.Kind.SECONDARY, "Block $who from inviting me again. Button.") { answer(id, "block") }, Ui.lp(top = 8, ctx = act))
            } else {
                body.addView(Ui.kv(act, who, if (user) "Can see your location" else "You can see their location", Level.OK), Ui.lp(top = 12, ctx = act))
                body.addView(Ui.button(act, if (user) "Stop sharing with $who" else "Disconnect from $who", Ui.Kind.SECONDARY,
                    if (user) "Stop sharing my location with $who. Button." else "Disconnect from $who. Button.") { answer(id, "revoke") }, Ui.lp(top = 8, ctx = act))
            }
        }
    }

    private fun answer(linkId: String, action: String) {
        Thread {
            val r = if (action == "revoke") api.revoke(linkId) else api.respond(linkId, action)
            act.runOnUiThread { message = if (r.ok) null else r.error; refresh() }
        }.start()
    }
}
