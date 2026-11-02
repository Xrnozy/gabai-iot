package com.example.gabai.modules

import android.content.Context
import com.example.gabai.WebRTCClient
import com.example.gabai.services.TtsService
import java.util.concurrent.atomic.AtomicBoolean

class AslModule(
    context: Context,
    private val tts: TtsService,
) : WebRTCClient.NavigationListener {
    private val modelManager = ModelManager()
    private val serverManager = ServerManager(context)
    private val started = AtomicBoolean(false)

    var onStatusText: ((String) -> Unit)? = null

    fun start() {
        if (!started.compareAndSet(false, true)) return

        // Lazy init: ASL-only models/pipeline.
        modelManager.ensureAslModelsInitialized()

        val rtc = serverManager.getOrCreateWebRtcClient(mode = "asl")
        rtc.navigationListener = this
        rtc.startCamera()
        rtc.createOffer()
    }

    fun stop() {
        if (!started.compareAndSet(true, false)) return
        modelManager.disposeAslModels()
        serverManager.releaseWebRtc()
    }

    override fun onNavigationCommand(command: String, fullData: org.json.JSONObject) {
        val shouldSpeak = fullData.optBoolean("should_speak", false)
        if (shouldSpeak) {
            onStatusText?.invoke(command)
            tts.speak(command)
        }
    }
}

