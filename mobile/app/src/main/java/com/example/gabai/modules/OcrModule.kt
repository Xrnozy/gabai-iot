package com.example.gabai.modules

import android.content.Context
import com.example.gabai.WebRTCClient
import com.example.gabai.services.TtsService
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

class OcrModule(
    context: Context,
    private val tts: TtsService,
) : WebRTCClient.NavigationListener {
    private val modelManager = ModelManager()
    private val serverManager = ServerManager(context)
    private val started = AtomicBoolean(false)

    var onStatusText: ((String) -> Unit)? = null

    /** Recognizer confidence in [0,1] — drives the confidence bar. */
    var onConfidence: ((Float) -> Unit)? = null

    /** Fired when the peer connection is live — hides "connecting" UI. */
    var onConnected: (() -> Unit)? = null

    /** Fired when the server connection fails — honest degraded state. */
    var onConnectionLost: (() -> Unit)? = null

    fun start() {
        if (!started.compareAndSet(false, true)) return

        // Lazy init: OCR-only pipeline.
        modelManager.ensureOcrModelsInitialized()

        val rtc = serverManager.getOrCreateWebRtcClient(mode = "ocr")
        rtc.navigationListener = this
        rtc.startCamera()
        rtc.createOffer()
    }

    fun stop() {
        if (!started.compareAndSet(true, false)) return
        modelManager.disposeOcrModels()
        serverManager.releaseWebRtc()
    }

    override fun onNavigationCommand(command: String, fullData: JSONObject) {
        onStatusText?.invoke(command)
        val conf = fullData.optDouble("confidence", Double.NaN)
        if (!conf.isNaN()) onConfidence?.invoke(conf.toFloat())
        val shouldSpeak = fullData.optBoolean("should_speak", false)
        if (shouldSpeak) {
            tts.speak(command)
        }
    }

    override fun onConnected() {
        tts.speak("Connected")
        onConnected?.invoke()
    }

    override fun onConnectionFailed() {
        tts.speak("Connection lost. Cannot read text right now.")
        onStatusText?.invoke("Connection lost")
        onConnectionLost?.invoke()
    }
}
