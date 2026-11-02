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
import com.example.gabai.modules.ColorModule
import com.example.gabai.services.Command
import com.example.gabai.services.CommandParser
import com.example.gabai.services.Haptics
import com.example.gabai.services.HeardPanel
import com.example.gabai.services.ListeningIndicator
import com.example.gabai.services.SettingsVoice
import com.example.gabai.services.TtsService
import com.example.gabai.services.VoiceCommandController
import java.util.Locale

class ColorActivity : BaseActivity() {

    private lateinit var tts: TtsService
    private lateinit var haptics: Haptics
    private lateinit var module: ColorModule
    private lateinit var voice: VoiceCommandController
    private lateinit var micIndicator: ListeningIndicator
    private lateinit var heardPanel: HeardPanel

    private lateinit var tvStatus: TextView
    private lateinit var swatch: android.view.View

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraGranted = permissions[Manifest.permission.CAMERA] == true
        if (cameraGranted) {
            startColor()
        } else {
            tts.speak("Camera permission denied. Cannot start color mode.")
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_color)

        tts = TtsService(this, Locale.US)
        haptics = Haptics(this)
        attachCaptions(tts)

        tvStatus = findViewById(R.id.tv_color_status)
        swatch = findViewById(R.id.view_color_swatch)

        module = ColorModule(this, tts).apply {
            onStatusText = { text ->
                runOnUiThread { tvStatus.text = text }
            }
            onColorDetected = { r, g, b ->
                runOnUiThread {
                    swatch.background.mutate()
                        .setTint(android.graphics.Color.rgb(r, g, b))
                }
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
        findViewById<ImageView>(R.id.btn_color_back).setOnClickListener {
            haptics.click()
            tts.speak("Back")
            finish()
        }

        // Announce color button
        findViewById<android.widget.Button>(R.id.btn_color_announce).setOnClickListener {
            haptics.click()
            announceColor()
        }
    }

    override fun onResume() {
        super.onResume()
        tts.speak("Color mode. Say color to hear it out loud.")
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
            Command.READ_TEXT -> {
                tts.speak("Opening Read text")
                startActivity(Intent(this, OcrActivity::class.java))
            }
            Command.SETTINGS -> {
                tts.speak("Opening Settings")
                startActivity(Intent(this, SettingsActivity::class.java))
            }
            Command.COLOR, Command.REPEAT -> announceColor()
            Command.HELP -> tts.speak(CommandParser.HELP_HINT)
            else -> SettingsVoice.apply(this, tts, command)
        }
    }

    private fun showConnected() {
        findViewById<android.view.View>(R.id.conn_pill).visibility = android.view.View.GONE
    }

    private fun showConnectionFailed() {
        findViewById<android.view.View>(R.id.conn_progress).visibility = android.view.View.GONE
        findViewById<TextView>(R.id.conn_text).apply {
            text = getString(R.string.conn_offline)
            setTextColor(getColor(R.color.status_error))
        }
    }

    private fun announceColor() {
        val color = tvStatus.text?.toString().orEmpty()
        if (color.isNotBlank() && color != getString(R.string.color_no_detection)) {
            tts.speak(color)
        } else {
            tts.speak("Nothing detected yet. Point the camera at an object.")
        }
    }

    private fun ensurePermissionsAndStart() {
        val cameraGranted =
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

        if (cameraGranted) {
            startColor()
        } else {
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
        }
    }

    private fun startColor() {
        module.start()
    }
}
