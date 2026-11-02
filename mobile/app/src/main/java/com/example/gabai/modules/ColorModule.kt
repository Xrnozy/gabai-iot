package com.example.gabai.modules

import android.content.Context
import com.example.gabai.WebRTCClient
import com.example.gabai.services.TtsService
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

class ColorModule(
    context: Context,
    private val tts: TtsService,
) : WebRTCClient.NavigationListener {
    private val modelManager = ModelManager()
    private val serverManager = ServerManager(context)
    private val started = AtomicBoolean(false)

    var onStatusText: ((String) -> Unit)? = null

    /** Fired with the detected RGB so the UI can paint the swatch. */
    var onColorDetected: ((r: Int, g: Int, b: Int) -> Unit)? = null

    /** Fired when the peer connection is live — hides "connecting" UI. */
    var onConnected: (() -> Unit)? = null

    /** Fired when the server connection fails — honest degraded state. */
    var onConnectionLost: (() -> Unit)? = null

    fun start() {
        if (!started.compareAndSet(false, true)) return

        modelManager.ensureColorModelsInitialized()

        val rtc = serverManager.getOrCreateWebRtcClient(mode = "color")
        rtc.navigationListener = this
        rtc.startCamera()
        rtc.createOffer()
    }

    fun stop() {
        if (!started.compareAndSet(true, false)) return
        modelManager.disposeColorModels()
        serverManager.releaseWebRtc()
    }

    override fun onNavigationCommand(command: String, fullData: JSONObject) {
        val rgbArr = fullData.optJSONArray("rgb")
        val rgbText = if (rgbArr != null && rgbArr.length() == 3) {
            onColorDetected?.invoke(
                rgbArr.optInt(0), rgbArr.optInt(1), rgbArr.optInt(2))
            "RGB: (${rgbArr.optInt(0)}, ${rgbArr.optInt(1)}, ${rgbArr.optInt(2)})"
        } else {
            ""
        }
        val shown = if (rgbText.isNotBlank()) "${command}\n$rgbText" else command
        onStatusText?.invoke(shown)

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
        tts.speak("Connection lost. Cannot detect colors right now.")
        onStatusText?.invoke("Connection lost")
        onConnectionLost?.invoke()
    }
}

