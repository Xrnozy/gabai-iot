package com.example.gabai.modules

import android.content.Context
import com.example.gabai.WebRTCClient
import java.util.concurrent.atomic.AtomicReference

class ServerManager(private val context: Context) {
    private val webRtcRef = AtomicReference<WebRTCClient?>(null)
    private var currentMode: String = "navigation"

    fun getOrCreateWebRtcClient(mode: String): WebRTCClient {
        val existing = webRtcRef.get()
        if (existing != null && currentMode == mode) return existing

        // Replace if the mode changed (prevents mixing navigation vs other pipelines).
        webRtcRef.getAndSet(null)?.release()

        currentMode = mode
        val created = WebRTCClient(context.applicationContext, mode = mode)
        webRtcRef.set(created)
        return created
    }

    fun releaseWebRtc() {
        val client = webRtcRef.getAndSet(null) ?: return
        client.release()
    }
}

