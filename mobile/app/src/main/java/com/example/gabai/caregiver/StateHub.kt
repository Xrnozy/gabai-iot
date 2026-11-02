package com.example.gabai.caregiver

import android.content.Context
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import com.example.gabai.safety.RelayApi
import com.example.gabai.services.SettingsStore
import org.json.JSONObject
import java.util.concurrent.CopyOnWriteArrayList

/** How this caregiver app is doing talking to the relay; separate from what the relay says about the phone and the Pi. */
class HubLink(val ok: Boolean, val rttMs: Long, val lastSuccessMs: Long, val error: String?, val downSinceMs: Long?)

/**
 * The one poller behind both the dashboard and the background alert service. It polls fast (2 s) while the dashboard is on
 * screen, slowly (8 s) when only alerts are wanted, and not at all when nobody needs it, so an idle caregiver phone sends nothing.
 */
object StateHub {
    private const val FAST_MS = 2_000L
    private const val SLOW_MS = 8_000L

    @Volatile var snapshot: JSONObject? = null
        private set
    @Volatile var link = HubLink(false, 0, 0, null, null)
        private set

    private val owners = HashSet<String>()
    private val listeners = CopyOnWriteArrayList<() -> Unit>()
    private val main = Handler(Looper.getMainLooper())
    private var thread: HandlerThread? = null
    private var h: Handler? = null
    private var api: RelayApi? = null
    private var appCtx: Context? = null
    private var lastOk = 0L
    private var downSince: Long? = null

    private val loop = object : Runnable {
        override fun run() {
            val a = api ?: return
            val r = a.state()
            val now = System.currentTimeMillis()
            if (r.ok) {
                snapshot = r.body
                lastOk = now
                downSince = null
                link = HubLink(true, r.rttMs, now, null, null)
            } else {
                if (downSince == null) downSince = now
                link = HubLink(false, r.rttMs, lastOk, r.error, downSince)
            }
            main.post { listeners.forEach { it() } }
            val fast = synchronized(owners) { "ui" in owners }
            h?.postDelayed(this, if (fast) FAST_MS else SLOW_MS)
        }
    }

    fun listen(l: () -> Unit) { listeners.add(l) }
    fun unlisten(l: () -> Unit) { listeners.remove(l) }

    /** [owner] "ui" polls fast; any other owner (the alert service) polls slowly. */
    @Synchronized
    fun acquire(ctx: Context, owner: String) {
        synchronized(owners) { owners += owner }
        appCtx = ctx.applicationContext
        if (thread == null) {
            thread = HandlerThread("state-hub").also { it.start() }
            h = Handler(thread!!.looper)
            api = RelayApi(appCtx!!, SettingsStore.ROLE_CAREGIVER)
            h!!.post(loop)
        } else if (owner == "ui") {
            h?.removeCallbacks(loop)
            h?.post(loop)
        }
    }

    @Synchronized
    fun release(owner: String) {
        val empty = synchronized(owners) { owners -= owner; owners.isEmpty() }
        if (empty) {
            h?.removeCallbacksAndMessages(null)
            thread?.quitSafely()
            thread = null; h = null; api = null
        }
    }

    fun ageSince(ms: Long): Long = if (ms <= 0) -1 else (System.currentTimeMillis() - ms) / 1000
}

internal fun JSONObject?.num(key: String): Double? =
    this?.takeIf { it.has(key) && !it.isNull(key) }?.optDouble(key)?.takeIf { !it.isNaN() }

internal fun JSONObject?.obj(key: String): JSONObject? = this?.optJSONObject(key)
internal fun JSONObject?.str(key: String): String? = this?.takeIf { it.has(key) && !it.isNull(key) }?.optString(key)?.takeIf { it.isNotBlank() }
internal fun JSONObject?.flag(key: String): Boolean? = this?.takeIf { it.has(key) && !it.isNull(key) }?.optBoolean(key)
