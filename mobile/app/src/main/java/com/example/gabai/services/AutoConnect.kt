package com.example.gabai.services

import android.content.Context
import com.example.gabai.safety.RelayApi
import com.example.gabai.safety.TrackingService

/**
 * Connects this phone to the PC without any typing. Runs for as long as the app process lives: it looks for the relay while
 * there is none, and once found, the navigation-help phone claims its place as the primary device by itself (the first phone to
 * ask wins; the server refuses later ones). A caregiver phone is never linked here: it needs the owner's approval (AccessPairing).
 */
object AutoConnect {
    private var started = false
    private var claimedElsewhere = false

    fun start(c: Context) {
        val app = c.applicationContext
        synchronized(this) { if (started) return; started = true }
        Thread({
            while (true) {
                val healthy = runCatching { step(app) }.getOrDefault(false)
                try { Thread.sleep(if (healthy) 30_000 else 8_000) } catch (_: InterruptedException) { return@Thread }
            }
        }, "auto-connect").apply { isDaemon = true }.start()
    }

    /** One pass. Returns true when the relay answers and nothing is left to set up. */
    private fun step(c: Context): Boolean {
        if (!reachable(c) && !RelayDiscovery.refresh(c)) return false
        val store = SettingsStore
        if (store.getAppRole(c) == store.APP_USER && store.getRelayToken(c, store.ROLE_PHONE).isBlank() && !claimedElsewhere) register(c)
        return store.getAppRole(c) != store.APP_USER || store.getRelayToken(c, store.ROLE_PHONE).isNotBlank() || claimedElsewhere
    }

    private fun reachable(c: Context): Boolean = runCatching {
        val base = SettingsStore.getRelayUrl(c)
        if (base.isBlank()) return false
        val conn = java.net.URL("$base/api/health").openConnection() as java.net.HttpURLConnection
        conn.connectTimeout = 1500; conn.readTimeout = 1500
        (conn.responseCode == 200).also { conn.disconnect() }
    }.getOrDefault(false)

    private fun register(c: Context) {
        val api = RelayApi(c, SettingsStore.ROLE_PHONE)
        val letters = "abcdefghjkmnpqrstuvwxyz23456789"
        var reply = api.register("user", "GabAi user", "gab-" + (1..6).map { letters.random() }.joinToString(""))
        var tries = 1
        while (reply.code == 409 && reply.error?.contains("primary") != true && tries++ < 3)
            reply = api.register("user", "GabAi user", "gab-" + (1..6).map { letters.random() }.joinToString(""))
        val token = reply.body?.optString("token").orEmpty()
        if (reply.ok && token.isNotBlank()) {
            SettingsStore.setRelayToken(c, SettingsStore.ROLE_PHONE, token)
            if (SettingsStore.isTrackingEnabled(c)) runCatching { TrackingService.start(c) }
        } else if (reply.code == 409) claimedElsewhere = true      // another phone is already the primary: stop asking
    }
}
