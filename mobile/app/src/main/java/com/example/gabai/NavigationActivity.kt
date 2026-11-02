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
import com.example.gabai.modules.NavigationModule
import com.example.gabai.services.Command
import com.example.gabai.services.CommandParser
import com.example.gabai.services.Haptics
import com.example.gabai.services.HeardPanel
import com.example.gabai.services.ListeningIndicator
import com.example.gabai.services.SettingsVoice
import com.example.gabai.services.TtsService
import com.example.gabai.services.VisionProfile
import com.example.gabai.services.VisionProfilePicker
import com.example.gabai.services.VisionProfileStore
import com.example.gabai.services.VisionProfiles
import com.example.gabai.services.VoiceCommandController
import com.example.gabai.services.Zone
import java.util.Locale

class NavigationActivity : BaseActivity() {

    private lateinit var tts: TtsService
    private lateinit var haptics: Haptics
    private lateinit var module: NavigationModule
    private lateinit var voice: VoiceCommandController
    private lateinit var micIndicator: ListeningIndicator
    private lateinit var heardPanel: HeardPanel

    private lateinit var tvStatus: TextView
    private lateinit var btnProfile: TextView
    private lateinit var ivArrow: ImageView
    private lateinit var tvDirection: TextView
    private lateinit var obstacleBanner: android.view.View
    private lateinit var tvAlertText: TextView
    private var profile: VisionProfile = VisionProfiles.byId("all")

    private val hideBanner = Runnable { obstacleBanner.visibility = android.view.View.GONE }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraGranted = permissions[Manifest.permission.CAMERA] == true
        if (cameraGranted) {
            startNavigation()
        } else {
            tts.speak("Camera permission denied. Cannot start navigation.")
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_navigation)

        // Navigation must never let the phone doze — screen-off throttles
        // the WebRTC socket and camera, ICE consent fails, and the feed
        // silently dies mid-navigation.
        window.addFlags(
            android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Navigation page keeps a dark chrome over the camera feed — dark
        // bar + light icons (the theme default is dark icons for cream).
        window.statusBarColor = getColor(R.color.surface_dark)
        window.navigationBarColor = getColor(R.color.surface_dark)
        androidx.core.view.WindowInsetsControllerCompat(window, window.decorView)
            .isAppearanceLightStatusBars = false

        tvStatus = findViewById(R.id.tv_nav_status)
        btnProfile = findViewById(R.id.btn_nav_profile)
        ivArrow = findViewById(R.id.iv_direction_arrow)
        tvDirection = findViewById(R.id.tv_nav_direction)
        obstacleBanner = findViewById(R.id.nav_obstacle_alert)
        tvAlertText = findViewById(R.id.tv_nav_alert_text)

        tts = TtsService(this, Locale.US)
        haptics = Haptics(this)
        attachCaptions(tts)
        profile = VisionProfileStore.load(this)
        updateProfileChip()

        module = NavigationModule(this, tts).apply {
            onStatusText = { text ->
                runOnUiThread { tvStatus.text = text }
            }
            onHazard = { zone ->
                runOnUiThread {
                    when (zone) {
                        Zone.LEFT, Zone.SLIGHT_LEFT -> haptics.hazardLeft()
                        Zone.RIGHT, Zone.SLIGHT_RIGHT -> haptics.hazardRight()
                        else -> haptics.hazardAhead()
                    }
                    showHazard(zone)
                }
            }
            onConnected = { runOnUiThread { showConnected() } }
            onConnectionLost = {
                runOnUiThread {
                    tvStatus.text = getString(R.string.nav_connection_lost)
                    showConnectionFailed()
                }
            }
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
        findViewById<ImageView>(R.id.btn_nav_back).setOnClickListener {
            haptics.click()
            tts.speak("Back")
            finish()
        }

        btnProfile.setOnClickListener {
            haptics.click()
            VisionProfilePicker.show(this, profile) { chosen ->
                applyProfile(chosen)
                tts.speak(getString(R.string.nav_profile_set, chosen.label))
            }
        }
    }

    override fun onResume() {
        super.onResume()
        tts.speak("Navigation mode. Say back to leave.")
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
        window.decorView.removeCallbacks(hideBanner)
        voice.destroy()
        module.stop()
        tts.shutdown()
    }

    /**
     * Point the direction arrow at the hazard (arrow asset faces right at
     * 0°) and flash the obstacle banner — it auto-hides after 3s since the
     * server sends no explicit "all clear".
     */
    private fun showHazard(zone: Zone) {
        val (rotation, label) = when (zone) {
            Zone.LEFT -> -180f to "Obstacle on your left"
            Zone.SLIGHT_LEFT -> -135f to "Obstacle slightly left"
            Zone.FORWARD -> -90f to "Obstacle ahead"
            Zone.SLIGHT_RIGHT -> -45f to "Obstacle slightly right"
            Zone.RIGHT -> 0f to "Obstacle on your right"
        }
        ivArrow.animate().rotation(rotation).setDuration(200).start()
        tvDirection.text = label
        tvAlertText.text = label
        if (obstacleBanner.visibility != android.view.View.VISIBLE) {
            obstacleBanner.alpha = 0f
            obstacleBanner.visibility = android.view.View.VISIBLE
            obstacleBanner.animate().alpha(1f).setDuration(180).start()
        }
        window.decorView.removeCallbacks(hideBanner)
        window.decorView.postDelayed(hideBanner, 3000)
    }

    private fun handleCommand(command: Command) {
        when (command) {
            Command.BACK, Command.STOP -> {
                tts.speak("Back")
                finish()
            }
            Command.READ_TEXT -> {
                tts.speak("Opening Read text")
                startActivity(Intent(this, OcrActivity::class.java))
            }
            Command.COLOR -> {
                tts.speak("Opening Color")
                startActivity(Intent(this, ColorActivity::class.java))
            }
            Command.SETTINGS -> {
                tts.speak("Opening Settings")
                startActivity(Intent(this, SettingsActivity::class.java))
            }
            Command.NAVIGATION -> tts.speak("You are already navigating")
            Command.REPEAT ->
                tts.speak(tvStatus.text?.toString().orEmpty().ifBlank {
                    getString(R.string.nav_status_ready)
                })
            Command.HELP -> tts.speak(CommandParser.HELP_HINT)
            else -> SettingsVoice.apply(this, tts, command)
        }
    }

    private fun ensurePermissionsAndStart() {
        val cameraGranted =
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

        if (cameraGranted) {
            startNavigation()
        } else {
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
        }
    }

    private fun startNavigation() {
        tvStatus.text = getString(R.string.nav_status_ready)
        module.start()
    }

    // ---------- connection indicator ----------

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

    // ---------- vision profile ----------

    private fun updateProfileChip() {
        btnProfile.text = "Profile: ${profile.label}"
        btnProfile.contentDescription = getString(R.string.nav_profile_chip_desc)
    }

    private fun applyProfile(p: VisionProfile) {
        profile = p
        module.updateProfile(p)
        updateProfileChip()
    }
}
