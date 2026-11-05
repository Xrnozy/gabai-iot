package com.example.gabai.modules

import android.content.Context
import android.os.SystemClock
import com.example.gabai.services.Lexicon
import com.example.gabai.safety.RelayApi
import com.example.gabai.services.SettingsStore
import org.json.JSONObject

/**
 * Pulls the PC's per-frame result from the relay (GET /api/guidance) and hands each new frame to [onResult], the same call
 * the WebRTC data channel made. Polling every [everyMs] over the hotspot is cheap; a result already handled (same frame_id)
 * or marked stale is skipped, so a dead PC never replays old advice.
 */
class RelayGuidance(
    context: Context,
    private val onResult: (summary: String, full: JSONObject) -> Unit,
    private val onAround: (List<String>) -> Unit = {},
    private val everyMs: Long = 400,
) {
    private val api = RelayApi(context, SettingsStore.ROLE_PHONE)
    @Volatile private var running = false
    private var lastFrame = -1L
    private val lexicon by lazy { Lexicon(context.applicationContext) }      // read on first use, inside runCatching: a missing asset must not stop navigation
    private val saidAt = HashMap<String, Long>()                      // sentence -> when it was last handed to speech
    private var thread: Thread? = null

    /** true while the relay answers; the module uses it for its "connected" state. */
    @Volatile var reachable = false
        private set

    fun start() {
        if (running) return
        running = true
        thread = Thread({
            while (running) {
                val r = api.guidance()
                reachable = r.ok
                val g = r.body
                if (g != null && g.optBoolean("available") && !g.optBoolean("stale") && g.optLong("frame_id", -1) != lastFrame) {
                    lastFrame = g.optLong("frame_id", -1)
                    onResult(g.optString("summary"), g)
                    g.optJSONObject("ego")?.let { around(it) }
                }
                try { Thread.sleep(everyMs) } catch (_: InterruptedException) { return@Thread }
            }
        }, "relay-guidance").also { it.start() }
    }

    /** What is around the user, decoded from the server's numeric message. A sentence is spoken at most once per 15 s and two per update, so a steady scene is not read out again and again. */
    private fun around(ego: JSONObject) {
        val now = SystemClock.elapsedRealtime()
        val fresh = runCatching { lexicon.decode(ego) }.getOrDefault(emptyList())
            .filter { now - (saidAt[it] ?: -SAY_AGAIN_MS) >= SAY_AGAIN_MS }.take(2)
        if (fresh.isEmpty()) return
        fresh.forEach { saidAt[it] = now }
        onAround(fresh)
    }

    fun stop() { running = false; thread?.interrupt() }

    private companion object { const val SAY_AGAIN_MS = 15_000L }
}
