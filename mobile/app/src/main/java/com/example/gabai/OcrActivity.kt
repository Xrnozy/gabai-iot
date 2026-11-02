package com.example.gabai

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.example.gabai.modules.OcrModule
import com.example.gabai.services.Command
import com.example.gabai.services.CommandParser
import com.example.gabai.services.Haptics
import com.example.gabai.services.HeardPanel
import com.example.gabai.services.ListeningIndicator
import com.example.gabai.services.SettingsVoice
import com.example.gabai.services.TtsService
import com.example.gabai.services.VoiceCommandController
import java.util.Locale

class OcrActivity : BaseActivity() {

    private lateinit var tts: TtsService
    private lateinit var haptics: Haptics
    private lateinit var module: OcrModule
    private lateinit var voice: VoiceCommandController
    private lateinit var micIndicator: ListeningIndicator
    private lateinit var heardPanel: HeardPanel

    private lateinit var tvStatus: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var confidenceFill: android.view.View
    private lateinit var tvAiStatus: TextView
    private lateinit var tvScanning: TextView

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraGranted = permissions[Manifest.permission.CAMERA] == true
        if (cameraGranted) {
            startOcr()
        } else {
            tts.speak("Camera permission denied. Cannot read text.")
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ocr)

        tts = TtsService(this, Locale.US)
        haptics = Haptics(this)
        attachCaptions(tts)

        tvStatus = findViewById(R.id.tv_ocr_status)
        tvConfidence = findViewById(R.id.tv_ocr_confidence)
        confidenceFill = findViewById(R.id.confidence_fill_ocr)
        tvAiStatus = findViewById(R.id.tv_ocr_ai_status)
        tvScanning = findViewById(R.id.tv_ocr_scanning)

        module = OcrModule(this, tts).apply {
            onStatusText = { text ->
                runOnUiThread { tvStatus.text = text }
            }
            onConfidence = { conf ->
                runOnUiThread { showConfidence(conf) }
            }
            onConnected = { runOnUiThread { showConnected() } }
            onConnectionLost = { runOnUiThread { showConnectionFailed() } }
        }

        micIndicator = ListeningIndicator(
            findViewById(R.id.mic_indicator),
            findViewById(R.id.mic_loading),
        )
        heardPanel = HeardPanel(findViewById(R.id.heard_panel))

        voice = VoiceCommandController(this, tts, onCommand = ::handleCommand).apply {
            onListeningChanged = { listening ->
                runOnUiThread {
                    micIndicator.setListening(listening)
                    heardPanel.setListening(listening)
                }
            }
            onLoadingChanged = { loading ->
                runOnUiThread {
                    micIndicator.setLoading(loading)
                    heardPanel.setLoading(loading)
                }
            }
            onAwaitingCommand = { awaiting ->
                runOnUiThread { micIndicator.setAwaitingCommand(awaiting) }
            }
            onHeard = { text ->
                runOnUiThread { heardPanel.show(text) }
            }
        }

        // Big mic fallback — tap instead of saying the wake word.
        findViewById<android.view.View>(R.id.btn_mic_fab)?.setOnClickListener {
            haptics.click()
            voice.promptForCommand()
        }

        // Back button is now an ImageView
        findViewById<ImageView>(R.id.btn_ocr_back).setOnClickListener {
            haptics.click()
            tts.speak("Back")
            finish()
        }

        // Read Aloud button
        findViewById<android.widget.Button>(R.id.btn_ocr_read).setOnClickListener {
            haptics.click()
            readAloud()
        }

    }

    override fun onResume() {
        super.onResume()
        tts.speak("Read text mode. Say read to hear it out loud.")
        ensurePermissionsAndStart()
        voice.start()
    }

    override fun onPause() {
        super.onPause()
        voice.stop()
        module.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        voice.destroy()
        module.stop()
        tts.shutdown()
    }

    private fun handleCommand(command: Command) {
        when (command) {
            Command.BACK, Command.STOP -> {
                tts.speak("Back")
                finish()
            }
            Command.NAVIGATION -> {
                tts.speak("Opening Navigation")
                startActivity(Intent(this, NavigationActivity::class.java))
            }
            Command.COLOR -> {
                tts.speak("Opening Color")
                startActivity(Intent(this, ColorActivity::class.java))
            }
            Command.SETTINGS -> {
                tts.speak("Opening Settings")
                startActivity(Intent(this, SettingsActivity::class.java))
            }
            Command.READ_TEXT, Command.REPEAT -> readAloud()
            Command.HELP -> tts.speak(CommandParser.HELP_HINT)
            else -> SettingsVoice.apply(this, tts, command)
        }
    }

    private fun showConnected() {
        findViewById<android.view.View>(R.id.conn_pill).visibility = android.view.View.GONE
        tvScanning.visibility = android.view.View.VISIBLE
        tvAiStatus.setText(R.string.ocr_connected_status)
    }

    private fun showConnectionFailed() {
        findViewById<android.view.View>(R.id.conn_progress).visibility = android.view.View.GONE
        tvScanning.visibility = android.view.View.GONE
        tvAiStatus.setText(R.string.ocr_offline_status)
        findViewById<TextView>(R.id.conn_text).apply {
            text = getString(R.string.conn_offline)
            setTextColor(getColor(R.color.status_error))
        }
    }

    private fun showConfidence(conf: Float) {
        val pct = (conf.coerceIn(0f, 1f) * 100).toInt()
        tvConfidence.text = "$pct%"
        val bar = confidenceFill.parent as android.view.View
        if (bar.width > 0) {
            confidenceFill.layoutParams.width =
                (bar.width * conf.coerceIn(0f, 1f)).toInt()
            confidenceFill.requestLayout()
        }
    }

    private fun readAloud() {
        val text = tvStatus.text?.toString().orEmpty()
        if (text.isNotBlank() && text != getString(R.string.ocr_no_text)) {
            tts.speak(text)
        } else {
            tts.speak("No text detected yet. Point the camera at something to read.")
        }
    }

    private fun ensurePermissionsAndStart() {
        val cameraGranted =
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

        if (cameraGranted) {
            startOcr()
        } else {
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
        }
    }

    private fun startOcr() {
        module.start()
    }
}
