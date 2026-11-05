package com.example.gabai

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import com.example.gabai.caregiver.CaregiverActivity
import com.example.gabai.safety.Emergency
import com.example.gabai.services.Command
import com.example.gabai.services.CommandParser
import com.example.gabai.services.Haptics
import com.example.gabai.services.HeardPanel
import com.example.gabai.services.ListeningIndicator
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.SettingsVoice
import com.example.gabai.services.TtsService
import com.example.gabai.services.VoiceCommandController
import com.example.gabai.services.VoiceListenerService
import java.util.Calendar
import java.util.Locale
import java.util.concurrent.atomic.AtomicReference

class MainActivity : BaseActivity() {

    private lateinit var tts: TtsService
    private lateinit var haptics: Haptics
    private lateinit var voice: VoiceCommandController
    private lateinit var micIndicator: ListeningIndicator
    private lateinit var heardPanel: HeardPanel
    private var voiceAwaiting = false
    private var isListening = false
    private var isLoading = false
    private lateinit var btnAccess: TextView

    private val openingMode = AtomicReference<Mode?>(null)

    /** True while this instance only exists to forward to onboarding —
     *  lifecycle methods must not touch the lateinit fields. */
    private var routingToOnboarding = false

    private enum class Mode { NAVIGATION, SETTINGS }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val micGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        startSafetyIfReady()
        if (!SettingsStore.isVoiceCommandsEnabled(this)) return@registerForActivityResult
        if (micGranted) {
            voice.start()
        } else {
            tts.speak("Microphone permission denied. You can still use the buttons.")
            Toast.makeText(this, "Microphone permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // First launch asks who uses this phone; caregivers never see the navigation home screen.
        val appRole = SettingsStore.getAppRole(this)
        if (appRole.isEmpty() || appRole == SettingsStore.APP_CAREGIVER) {
            routingToOnboarding = true
            startActivity(Intent(this, if (appRole.isEmpty()) RoleActivity::class.java else CaregiverActivity::class.java))
            finish()
            return
        }

        // First run: route through onboarding (blindness type + tutorial).
        if (!SettingsStore.isOnboardingDone(this)) {
            routingToOnboarding = true
            startActivity(Intent(this, OnboardingActivity::class.java))
            finish()
            return
        }

        setContentView(R.layout.activity_main)

        tts = TtsService(this, Locale.US)
        haptics = Haptics(this)
        attachCaptions(tts)

        // Keep "Gab" listening alive for the screen-off demo if enabled.
        VoiceListenerService.ensureStarted(this)

        val btnNavigation = findViewById<LinearLayout>(R.id.btn_navigation)
        val btnSettings = findViewById<LinearLayout>(R.id.btn_settings)
        val btnFamily = findViewById<LinearLayout>(R.id.btn_family)
        val btnSos = findViewById<LinearLayout>(R.id.btn_sos)
        btnAccess = findViewById(R.id.btn_access_nav)

        val tvTitle = findViewById<TextView>(R.id.tv_title)
        tvTitle.text = getGreeting()

        // Ensure TalkBack focuses the primary actions early.
        ViewCompat.setAccessibilityHeading(tvTitle, true)
        btnNavigation.requestFocus()

        val speakOnFocus: (android.view.View, Boolean, String) -> Unit =
            { _, hasFocus, label -> if (hasFocus) tts.speak(label) }

        btnNavigation.setOnFocusChangeListener { v, f -> speakOnFocus(v, f, "Navigation") }
        btnSettings.setOnFocusChangeListener { v, f -> speakOnFocus(v, f, "Settings") }

        btnNavigation.setOnClickListener {
            haptics.click()
            openMode(Mode.NAVIGATION)
        }
        btnSettings.setOnClickListener {
            haptics.click()
            openMode(Mode.SETTINGS)
        }
        btnFamily.setOnClickListener {
            haptics.click()
            startActivity(Intent(this, FamilyActivity::class.java))
        }
        btnSos.setOnClickListener {
            Emergency.startSos(this)
            Emergency.openScreen()
        }
        btnAccess.setOnClickListener {
            haptics.click()
            setAccessNav(!isAccessNavOn())
            btnAccess.postDelayed({ refreshAccessLabel() }, 200)
        }

        // Staggered entrance — cards rise in sequence, mic button pops last.
        val cards = listOfNotNull(
            findViewById<android.view.View>(R.id.card_voice_status),
            btnNavigation, btnSettings,
        )
        val d = resources.displayMetrics.density
        cards.forEachIndexed { i, card ->
            card.alpha = 0f
            card.translationY = 24f * d
            card.animate().alpha(1f).translationY(0f)
                .setDuration(280).setStartDelay(60L + i * 60L).start()
        }
        findViewById<android.view.View>(R.id.btn_mic_fab)?.let { fab ->
            fab.scaleX = 0f
            fab.scaleY = 0f
            fab.animate().scaleX(1f).scaleY(1f)
                .setDuration(300).setStartDelay(60L + cards.size * 60L).start()
        }

        micIndicator = ListeningIndicator(
            findViewById(R.id.mic_indicator),
            findViewById(R.id.mic_loading),
        )
        heardPanel = HeardPanel(findViewById(R.id.heard_panel))
        renderVoiceState()

        voice = VoiceCommandController(this, tts, onCommand = ::handleCommand).apply {
            onListeningChanged = { listening ->
                runOnUiThread {
                    micIndicator.setListening(listening)
                    heardPanel.setListening(listening)
                    isListening = listening
                    renderVoiceState()
                }
            }
            onLoadingChanged = { loading ->
                runOnUiThread {
                    micIndicator.setLoading(loading)
                    heardPanel.setLoading(loading)
                    isLoading = loading
                    renderVoiceState()
                }
            }
            onAwaitingCommand = { awaiting ->
                voiceAwaiting = awaiting
                runOnUiThread {
                    micIndicator.setAwaitingCommand(awaiting)
                }
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
    }

    override fun onResume() {
        super.onResume()
        if (routingToOnboarding) return
        openingMode.set(null)

        findViewById<TextView>(R.id.tv_title).text = getGreeting()
        refreshAccessLabel()

        // Voice guidance plays once ever — returning home shouldn't nag.
        // "Hello gab, help" replays it (see handleCommand).
        if (!SettingsStore.isHomeIntroDone(this)) {
            SettingsStore.setHomeIntroDone(this, true)
            tts.speak("Welcome. " + CommandParser.HELP_HINT)
        }

        ensureMicPermissionAndStartListening()
    }

    override fun onPause() {
        super.onPause()
        if (!routingToOnboarding) voice.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (!routingToOnboarding) {
            voice.destroy()
            tts.shutdown()
        }
    }

    /** Panel text mirrors the real mic state; loading wins over listening, as in ListeningIndicator. */
    private fun renderVoiceState() {
        findViewById<TextView>(R.id.tv_voice_state).setText(
            when {
                isLoading -> R.string.voice_state_loading
                isListening -> R.string.dashboard_ai_listening
                else -> R.string.voice_state_off
            }
        )
    }

    private fun refreshAccessLabel() {
        val on = isAccessNavOn()
        btnAccess.text = "Accessible navigation: ${if (on) "On" else "Off"}"
        btnAccess.contentDescription = "Accessible navigation is ${if (on) "on" else "off"}. " +
            "When on, tap to hear an item and double tap to open it. Button."
    }

    private fun getGreeting(): String {
        val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
        return when {
            hour < 12 -> "Good morning"
            hour < 17 -> "Good afternoon"
            else -> "Good evening"
        }
    }

    private fun granted(p: String) = ContextCompat.checkSelfPermission(this, p) == PackageManager.PERMISSION_GRANTED

    /** Paired phones share location and watch for falls without any switch; this just makes sure the service is running. */
    private fun safetyWanted() = SettingsStore.isTrackingEnabled(this) &&
        SettingsStore.getRelayToken(this, SettingsStore.ROLE_PHONE).isNotBlank()

    private fun startSafetyIfReady() {
        if (safetyWanted() && granted(Manifest.permission.ACCESS_FINE_LOCATION)) {
            runCatching { com.example.gabai.safety.TrackingService.start(this) }
        }
    }

    private fun ensureMicPermissionAndStartListening() {
        // One permission prompt for everything the home screen needs: mic, plus location when sharing is wanted.
        val missing = buildList {
            if (SettingsStore.isVoiceCommandsEnabled(this@MainActivity) && !granted(Manifest.permission.RECORD_AUDIO)) add(Manifest.permission.RECORD_AUDIO)
            if (safetyWanted() && !granted(Manifest.permission.ACCESS_FINE_LOCATION)) {
                add(Manifest.permission.ACCESS_FINE_LOCATION); add(Manifest.permission.ACCESS_COARSE_LOCATION)
                if (android.os.Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        startSafetyIfReady()
        if (missing.isNotEmpty()) requestPermissionLauncher.launch(missing.toTypedArray())
        else if (SettingsStore.isVoiceCommandsEnabled(this)) voice.start()
    }

    private fun handleCommand(command: Command) {
        when (command) {
            Command.NAVIGATION -> openMode(Mode.NAVIGATION)
            Command.SETTINGS -> openMode(Mode.SETTINGS)
            Command.SOS -> Emergency.startSos(this)
            // "help" replays the same welcome line the one-time intro used.
            Command.HELP -> tts.speak("Welcome. " + CommandParser.HELP_HINT)
            Command.REPEAT -> tts.speak(
                "You are on the home screen. " + CommandParser.HELP_HINT
            )
            Command.BACK, Command.STOP ->
                tts.speak("You are on the home screen.")
            else -> SettingsVoice.apply(this, tts, command)
        }
    }

    private fun openMode(mode: Mode) {
        val existing = openingMode.get()
        if (existing == mode) return
        if (!openingMode.compareAndSet(existing, mode)) return

        voice.stop()

        when (mode) {
            Mode.NAVIGATION -> {
                tts.speak("Opening Navigation")
                startActivity(Intent(this, NavigationActivity::class.java))
            }
            Mode.SETTINGS -> {
                tts.speak("Opening Settings")
                startActivity(Intent(this, SettingsActivity::class.java))
            }
        }
    }
}
