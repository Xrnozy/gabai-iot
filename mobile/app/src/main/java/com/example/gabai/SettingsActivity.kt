package com.example.gabai

import android.content.Context
import android.content.Intent
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.text.InputType
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import com.google.android.material.slider.Slider
import com.example.gabai.services.CameraSourceResolver
import com.example.gabai.services.Command
import com.example.gabai.services.CommandParser
import com.example.gabai.services.Contrast
import com.example.gabai.services.Haptics
import com.example.gabai.services.HeardPanel
import com.example.gabai.services.ListeningIndicator
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.SettingsVoice
import com.example.gabai.services.TtsService
import com.example.gabai.services.VisionProfilePicker
import com.example.gabai.services.VisionProfileStore
import com.example.gabai.services.VoiceCommandController
import com.example.gabai.services.VoiceListenerService
import com.google.android.material.switchmaterial.SwitchMaterial
import java.util.Locale

/**
 * Plain-language settings, grouped for blind users (Voice/Speech/Touch/
 * Vision profile) and sighted helpers (Display/For helpers). Every change
 * is spoken back so it can be operated entirely by ear.
 */
class SettingsActivity : BaseActivity() {

    private lateinit var tts: TtsService
    private lateinit var haptics: Haptics
    private lateinit var voice: VoiceCommandController
    private lateinit var micIndicator: ListeningIndicator
    private lateinit var heardPanel: HeardPanel

    private lateinit var swVoiceCommands: SwitchMaterial
    private lateinit var swWakeName: SwitchMaterial
    private lateinit var swLiveTranscript: SwitchMaterial
    private lateinit var swScreenOff: SwitchMaterial
    private lateinit var swSpeech: SwitchMaterial
    private lateinit var swHaptics: SwitchMaterial
    private lateinit var swHighContrast: SwitchMaterial
    private lateinit var sliderFontScale: Slider
    private lateinit var tvFontScaleValue: TextView
    private lateinit var swTouchExplore: SwitchMaterial
    private lateinit var btnRateSlow: Button
    private lateinit var btnRateNormal: Button
    private lateinit var btnRateFast: Button
    private lateinit var tvVisionProfileValue: TextView
    private lateinit var tvServerUrlValue: TextView
    private lateinit var tvRelayIpValue: TextView
    private lateinit var tvMicSourceValue: TextView
    private lateinit var tvCameraSourceValue: TextView
    private lateinit var tvOutputValue: TextView

    /** Suppresses switch listeners while [refreshValues] syncs the UI —
     *  otherwise every onResume would speak "Voice commands on" etc. */
    private var refreshing = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        tts = TtsService(this, Locale.US)
        haptics = Haptics(this)
        attachCaptions(tts)

        swVoiceCommands = findViewById(R.id.sw_voice_commands)
        swWakeName = findViewById(R.id.sw_wake_name)
        swLiveTranscript = findViewById(R.id.sw_live_transcript)
        swScreenOff = findViewById(R.id.sw_screen_off)
        swSpeech = findViewById(R.id.sw_speech)
        swHaptics = findViewById(R.id.sw_haptics)
        swHighContrast = findViewById(R.id.sw_high_contrast)
        sliderFontScale = findViewById(R.id.slider_font_scale)
        tvFontScaleValue = findViewById(R.id.tv_font_scale_value)
        swTouchExplore = findViewById(R.id.sw_touch_explore)
        btnRateSlow = findViewById(R.id.btn_rate_slow)
        btnRateNormal = findViewById(R.id.btn_rate_normal)
        btnRateFast = findViewById(R.id.btn_rate_fast)
        tvVisionProfileValue = findViewById(R.id.tv_vision_profile_value)
        tvServerUrlValue = findViewById(R.id.tv_server_url_value)
        tvRelayIpValue = findViewById(R.id.tv_relay_ip_value)
        tvMicSourceValue = findViewById(R.id.tv_mic_source_value)
        tvCameraSourceValue = findViewById(R.id.tv_camera_source_value)
        tvOutputValue = findViewById(R.id.tv_output_value)

        findViewById<ImageView>(R.id.btn_settings_back).setOnClickListener {
            haptics.click()
            tts.speak("Back")
            finish()
        }

        bindToggles()
        bindRows()
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
    }

    override fun onResume() {
        super.onResume()
        refreshValues()
        voice.start()
    }

    override fun onPause() {
        super.onPause()
        voice.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        voice.destroy()
        tts.shutdown()
    }

    // ---------- toggles ----------

    private fun bindToggles() {
        swVoiceCommands.setOnCheckedChangeListener { _, checked ->
            if (refreshing) return@setOnCheckedChangeListener
            SettingsStore.setVoiceCommandsEnabled(this, checked)
            haptics.click()
            tts.speak(if (checked) "Voice commands on" else "Voice commands off")
            if (!checked) {
                voice.stop()
                VoiceListenerService.stop(this)
            } else {
                voice.start()
                VoiceListenerService.ensureStarted(this)
            }
        }
        swWakeName.setOnCheckedChangeListener { _, checked ->
            if (refreshing) return@setOnCheckedChangeListener
            SettingsStore.setWakeNameRequired(this, checked)
            haptics.click()
            tts.speak(
                if (checked) "Say Gab first on"
                else "Gab not needed, command words work on their own"
            )
        }
        swLiveTranscript.setOnCheckedChangeListener { _, checked ->
            if (refreshing) return@setOnCheckedChangeListener
            SettingsStore.setLiveTranscriptEnabled(this, checked)
            haptics.click()
            heardPanel.refresh()
            tts.speak(if (checked) "Show what I say on" else "Show what I say off")
        }
        swScreenOff.setOnCheckedChangeListener { _, checked ->
            if (refreshing) return@setOnCheckedChangeListener
            SettingsStore.setScreenOffListening(this, checked)
            haptics.click()
            if (checked) {
                VoiceListenerService.ensureStarted(this)
                tts.speak("Screen-off listening on. Say Gab off to stop it.")
            } else {
                VoiceListenerService.stop(this)
                tts.speak("Screen-off listening off")
            }
        }
        swSpeech.setOnCheckedChangeListener { _, checked ->
            if (refreshing) return@setOnCheckedChangeListener
            SettingsStore.setSpeechEnabled(this, checked)
            haptics.click()
            if (checked) tts.speak("Speech on")
        }
        swHaptics.setOnCheckedChangeListener { _, checked ->
            if (refreshing) return@setOnCheckedChangeListener
            SettingsStore.setHapticsEnabled(this, checked)
            haptics.click()
            tts.speak(if (checked) "Vibration on" else "Vibration off")
        }
        swHighContrast.setOnCheckedChangeListener { _, checked ->
            if (refreshing) return@setOnCheckedChangeListener
            SettingsStore.setHighContrast(this, checked)
            tts.speak(if (checked) "High contrast on" else "High contrast off")
            if (checked) {
                // No recreate needed — recolor the live view tree directly.
                Contrast.apply(window.decorView)
                syncDisplaySettings()
            } else {
                // Restoring originals needs a fresh view tree; delay so the
                // switch animation + spoken feedback aren't cut off.
                delayedRecreate()
            }
        }
        sliderFontScale.addOnChangeListener { _, value, fromUser ->
            tvFontScaleValue.text = "${(value * 100).toInt()}%"
            if (fromUser && !refreshing) SettingsStore.setFontScale(this, value)
        }
        sliderFontScale.addOnSliderTouchListener(object : Slider.OnSliderTouchListener {
            override fun onStartTrackingTouch(slider: Slider) {}
            override fun onStopTrackingTouch(slider: Slider) {
                if (refreshing) return
                haptics.click()
                tts.speak("Font size ${(slider.value * 100).toInt()} percent")
                // fontScale only applies via attachBaseContext — a recreate
                // is unavoidable; delaying it avoids the hard flicker.
                delayedRecreate()
            }
        })
        swTouchExplore.setOnCheckedChangeListener { _, checked ->
            if (refreshing) return@setOnCheckedChangeListener
            SettingsStore.setTouchExploreEnabled(this, checked)
            haptics.click()
            tts.speak(if (checked) "Touch explore on. Hold your finger on a button to hear its name"
                else "Touch explore off")
        }
        btnRateSlow.setOnClickListener { setRate(SettingsStore.SPEECH_RATE_SLOW, "Slow") }
        btnRateNormal.setOnClickListener { setRate(SettingsStore.SPEECH_RATE_NORMAL, "Normal") }
        btnRateFast.setOnClickListener { setRate(SettingsStore.SPEECH_RATE_FAST, "Fast") }
    }

    /**
     * Android doesn't let apps pick the media output sink directly — the
     * sanctioned route is the system media-output switcher (the same kind
     * of device picker Windows shows), opened here as a settings panel.
     * Pre-Android 11 there's no panel, so fall back to Bluetooth settings.
     */
    private fun openOutputPicker() {
        val intent = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // Settings.Panel.ACTION_MEDIA_OUTPUT — literal since the
            // constant isn't in this compileSdk.
            Intent("android.settings.panel.action.MEDIA_OUTPUT")
        } else {
            Intent(android.provider.Settings.ACTION_BLUETOOTH_SETTINGS)
        }
        try {
            startActivity(intent)
            tts.speak("Choose where GabAi's voice plays")
        } catch (e: Exception) {
            tts.speak("Couldn't open the output picker")
        }
    }

    /** Best guess at where media currently plays — first non-embedded sink
     *  (headset/BT/USB wins when connected), else the phone speaker. */
    private fun outputLabel(): String {
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val external = am.getDevices(AudioManager.GET_DEVICES_OUTPUTS).firstOrNull {
            it.type != AudioDeviceInfo.TYPE_BUILTIN_SPEAKER &&
                it.type != AudioDeviceInfo.TYPE_BUILTIN_EARPIECE
        }
        return external?.productName?.toString()?.takeIf { it.isNotBlank() }
            ?: getString(R.string.output_phone_speaker)
    }

    private fun delayedRecreate() {
        window.decorView.postDelayed({ recreate() }, 400)
    }

    private fun setFontScaleFromVoice(large: Boolean) {
        val scale = if (large) SettingsStore.LARGE_TEXT_SCALE else 1.0f
        SettingsStore.setFontScale(this, scale)
        sliderFontScale.value = scale
        tvFontScaleValue.text = "${(scale * 100).toInt()}%"
        tts.speak(if (large) "Large text on" else "Normal text size")
        delayedRecreate()
    }

    private fun setRate(rate: Float, label: String) {
        SettingsStore.setSpeechRate(this, rate)
        tts.applySpeechRate(rate)
        haptics.click()
        refreshRateButtons()
        tts.speak("Speech speed: $label. This is how I sound now.")
    }

    // ---------- rows ----------

    private fun bindRows() {
        findViewById<android.view.View>(R.id.row_vision_profile).setOnClickListener {
            haptics.click()
            val current = VisionProfileStore.load(this)
            VisionProfilePicker.show(this, current) { chosen ->
                refreshVisionProfile()
                tts.speak(getString(R.string.nav_profile_set, chosen.label))
            }
        }

        findViewById<android.view.View>(R.id.row_server_url).setOnClickListener {
            haptics.click()
            showServerUrlDialog()
        }

        findViewById<android.view.View>(R.id.row_relay_ip).setOnClickListener {
            haptics.click()
            showRelayIpDialog()
        }

        findViewById<android.view.View>(R.id.row_replay_tutorial).setOnClickListener {
            haptics.click()
            tts.speak("Replaying the tutorial")
            startActivity(Intent(this, OnboardingActivity::class.java))
        }

        findViewById<android.view.View>(R.id.row_tokens_tutorial).setOnClickListener {
            haptics.click()
            startActivity(Intent(this, TokensTutorialActivity::class.java))
        }

        findViewById<android.view.View>(R.id.row_mic_source).setOnClickListener {
            haptics.click()
            showMicSourceDialog()
        }

        findViewById<android.view.View>(R.id.row_camera_source).setOnClickListener {
            haptics.click()
            showCameraSourceDialog()
        }

        findViewById<android.view.View>(R.id.row_output_device).setOnClickListener {
            haptics.click()
            openOutputPicker()
        }

        findViewById<android.view.View>(R.id.row_debug_get_started).setOnClickListener {
            haptics.click()
            // True first-run replay: clear the flags so the flow behaves
            // exactly like a fresh install, then launch it.
            SettingsStore.setOnboardingDone(this, false)
            SettingsStore.setHomeIntroDone(this, false)
            tts.speak("Showing get started")
            startActivity(Intent(this, OnboardingActivity::class.java))
        }

        val caregiverNow = SettingsStore.getAppRole(this) == SettingsStore.APP_CAREGIVER
        findViewById<android.widget.TextView>(R.id.tv_role_desc).text =
            if (caregiverNow) "You are a caregiver. Tap to use this phone for navigation help instead."
            else "You use navigation help. Tap to make this phone a caregiver phone instead."
        findViewById<android.view.View>(R.id.row_role).setOnClickListener {
            haptics.click()
            confirmSwitchRole(caregiverNow)
        }

        findViewById<android.view.View>(R.id.row_reset).setOnClickListener {
            haptics.click()
            confirmReset()
        }
    }

    private fun showServerUrlDialog() {
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setText(SettingsStore.getServerUrl(this@SettingsActivity))
            setSingleLine()
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.settings_server_url)
            .setView(input)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                val url = input.text.toString().trim()
                if (url.startsWith("http://") || url.startsWith("https://")) {
                    SettingsStore.setServerUrl(this, url)
                    tvServerUrlValue.text = url
                    tts.speak("Server address saved. It applies next time a mode starts.")
                } else {
                    tts.speak("That doesn't look like a server address. Nothing saved.")
                }
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    /** Accepts "192.168.1.5", "192.168.1.5:8090" or a full http:// address; empty goes back to automatic discovery. AutoConnect picks the change up within seconds. */
    private fun showRelayIpDialog() {
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setText(SettingsStore.getRelayUrl(this@SettingsActivity))
            setSingleLine()
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.settings_relay_ip)
            .setView(input)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                val raw = input.text.toString().trim()
                val url = when {
                    raw.isEmpty() -> ""
                    raw.startsWith("http://") || raw.startsWith("https://") -> raw
                    else -> "http://" + (if (':' in raw) raw else "$raw:8090")
                }
                SettingsStore.setRelayUrl(this, url)
                tvRelayIpValue.text = url.ifEmpty { getString(R.string.settings_relay_ip_desc) }
                tts.speak(if (url.isEmpty()) "PC address cleared. Finding it automatically." else "PC address saved. Connecting.")            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    // ---------- audio devices ----------

    /** Every mic Android exposes right now (built-in, wired, BT, USB). */
    private fun inputDevices(): List<AudioDeviceInfo> {
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        return am.getDevices(AudioManager.GET_DEVICES_INPUTS).filter {
            it.type in MIC_DEVICE_TYPES
        }
    }

    private fun deviceLabel(d: AudioDeviceInfo): String =
        if (d.type == AudioDeviceInfo.TYPE_BUILTIN_MIC) {
            getString(R.string.mic_source_phone)
        } else {
            d.productName?.toString()?.takeIf { it.isNotBlank() } ?: "Device ${d.id}"
        }

    private fun micSourceLabel(source: String): String = when (source) {
        SettingsStore.MIC_PHONE -> getString(R.string.mic_source_phone)
        SettingsStore.MIC_HEADSET -> getString(R.string.mic_source_headset)
        SettingsStore.MIC_AUTO -> getString(R.string.mic_source_auto)
        else -> inputDevices().firstOrNull { it.id.toString() == source }
            ?.let(::deviceLabel) ?: getString(R.string.mic_source_auto)
    }

    /** Live device picker — "Automatic" plus every mic currently present. */
    private fun showMicSourceDialog() {
        val devices = inputDevices()
        val keys = mutableListOf(SettingsStore.MIC_AUTO)
        val labels = mutableListOf(getString(R.string.mic_source_auto))
        devices.forEach {
            keys.add(it.id.toString())
            labels.add(deviceLabel(it))
        }
        val current = SettingsStore.getMicSource(this)
        val checked = keys.indexOf(current).coerceAtLeast(0)
        AlertDialog.Builder(this)
            .setTitle(R.string.settings_mic_source)
            .setSingleChoiceItems(labels.toTypedArray(), checked) { dialog, which ->
                SettingsStore.setMicSource(this, keys[which])
                tvMicSourceValue.text =
                    "${labels[which]}. ${getString(R.string.settings_mic_source_desc)}"
                tts.speak("Microphone: ${labels[which]}")
                dialog.dismiss()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun showCameraSourceDialog() {
        val keys = arrayOf(SettingsStore.CAMERA_PHONE, SettingsStore.CAMERA_UVC)
        val labels = arrayOf(
            getString(R.string.camera_source_phone),
            getString(R.string.camera_source_uvc),
        )
        val current = SettingsStore.getCameraSource(this)
        val checked = keys.indexOf(current).coerceAtLeast(0)
        AlertDialog.Builder(this)
            .setTitle(R.string.settings_camera_source)
            .setSingleChoiceItems(labels, checked) { dialog, which ->
                SettingsStore.setCameraSource(this, keys[which])
                val label = CameraSourceResolver.label(this, keys[which])
                tvCameraSourceValue.text =
                    "$label. ${getString(R.string.settings_camera_source_desc)}"
                tts.speak(
                    "Camera: $label. ${getString(R.string.settings_camera_source_desc)}"
                )
                dialog.dismiss()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    /** Keeps both roles' tokens: switching back later needs no new pairing. */
    private fun confirmSwitchRole(caregiverNow: Boolean) {
        AlertDialog.Builder(this)
            .setTitle(if (caregiverNow) "Switch to navigation help?" else "Switch to caregiver?")
            .setMessage(if (caregiverNow) "This phone will show the navigation home screen and share location and fall alerts."
            else "This phone will show the caregiver dashboard. Location sharing and fall detection on this phone will stop.")
            .setPositiveButton(android.R.string.ok) { _, _ ->
                if (caregiverNow) {
                    SettingsStore.setAppRole(this, SettingsStore.APP_USER)
                    SettingsStore.setTrackingEnabled(this, true)       // leaving the caregiver role must not leave sharing switched off
                } else {
                    SettingsStore.setAppRole(this, SettingsStore.APP_CAREGIVER)
                    com.example.gabai.safety.TrackingService.stop(this)
                }
                tts.speak(if (caregiverNow) "Now using navigation help." else "Now a caregiver phone.")
                tvServerUrlValue.postDelayed({
                    startActivity(Intent(this, MainActivity::class.java))
                    finishAffinity()
                }, 800)
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun confirmReset() {
        AlertDialog.Builder(this)
            .setTitle(R.string.settings_reset)
            .setMessage(R.string.settings_reset_desc)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                SettingsStore.resetAll(this)
                VisionProfileStore.reset(this)
                tts.speak("Settings reset. Starting setup again.")
                tvServerUrlValue.postDelayed({
                    startActivity(Intent(this, MainActivity::class.java))
                    finishAffinity()
                }, 800)
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    // ---------- values ----------

    private fun refreshValues() {
        refreshing = true
        swVoiceCommands.isChecked = SettingsStore.isVoiceCommandsEnabled(this)
        swWakeName.isChecked = SettingsStore.isWakeNameRequired(this)
        swLiveTranscript.isChecked = SettingsStore.isLiveTranscriptEnabled(this)
        swScreenOff.isChecked = SettingsStore.isScreenOffListening(this)
        swSpeech.isChecked = SettingsStore.isSpeechEnabled(this)
        swHaptics.isChecked = SettingsStore.isHapticsEnabled(this)
        swHighContrast.isChecked = SettingsStore.isHighContrast(this)
        val fontScale = SettingsStore.getFontScale(this)
        sliderFontScale.value = fontScale
        tvFontScaleValue.text = "${(fontScale * 100).toInt()}%"
        swTouchExplore.isChecked = SettingsStore.isTouchExploreEnabled(this)
        tvServerUrlValue.text = SettingsStore.getServerUrl(this)
        tvRelayIpValue.text = SettingsStore.getRelayUrl(this).ifEmpty { getString(R.string.settings_relay_ip_desc) }
        tvMicSourceValue.text =
            "${micSourceLabel(SettingsStore.getMicSource(this))}. " +
            getString(R.string.settings_mic_source_desc)
        tvCameraSourceValue.text =
            "${CameraSourceResolver.label(this, SettingsStore.getCameraSource(this))}. " +
            getString(R.string.settings_camera_source_desc)
        tvOutputValue.text =
            "${outputLabel()}. ${getString(R.string.settings_output_device_desc)}"
        refreshVisionProfile()
        refreshRateButtons()
        refreshing = false
    }

    private fun refreshVisionProfile() {
        val profile = VisionProfileStore.load(this)
        tvVisionProfileValue.text =
            "${profile.label}. ${getString(R.string.settings_vision_profile_desc)}"
    }

    private fun refreshRateButtons() {
        val current = SettingsStore.getSpeechRate(this)
        styleRateButton(btnRateSlow, current == SettingsStore.SPEECH_RATE_SLOW,
            R.drawable.bg_rate_slow_sel, R.drawable.bg_rate_slow)
        styleRateButton(btnRateNormal, current == SettingsStore.SPEECH_RATE_NORMAL,
            R.drawable.bg_rate_normal_sel, R.drawable.bg_rate_normal)
        styleRateButton(btnRateFast, current == SettingsStore.SPEECH_RATE_FAST,
            R.drawable.bg_rate_fast_sel, R.drawable.bg_rate_fast)
    }

    private fun styleRateButton(
        button: Button, selected: Boolean,
        selectedBg: Int, offBg: Int,
    ) {
        button.setBackgroundResource(if (selected) selectedBg else offBg)
        button.setTextColor(ContextCompat.getColor(this, R.color.on_primary))
    }

    // ---------- voice ----------

    private fun handleCommand(command: Command) {
        when (command) {
            Command.BACK, Command.STOP -> {
                tts.speak("Back")
                finish()
            }
            Command.NAVIGATION -> if (SettingsStore.getAppRole(this) == SettingsStore.APP_CAREGIVER) tts.speak("That is not part of the caregiver app.") else {
                tts.speak("Opening Navigation")
                startActivity(Intent(this, NavigationActivity::class.java))
            }
            Command.READ_TEXT, Command.COLOR -> tts.speak("That feature has been removed.")
            Command.SOS -> com.example.gabai.safety.Emergency.startSos(this)
            Command.SETTINGS -> tts.speak("You are in settings")
            Command.REPEAT -> tts.speak("Settings. " + CommandParser.HELP_HINT)
            Command.HELP -> tts.speak(CommandParser.HELP_HINT)

            // Settings commands drive the switches themselves — the
            // listeners already persist, speak and run the side-effects.
            Command.SET_VOICE_ON -> setSwitch(swVoiceCommands, true, "Voice commands")
            Command.SET_VOICE_OFF -> setSwitch(swVoiceCommands, false, "Voice commands")
            Command.SET_WAKE_ON -> setSwitch(swWakeName, true, "Say Gab first")
            Command.SET_WAKE_OFF -> setSwitch(swWakeName, false, "Say Gab first")
            Command.SET_TRANSCRIPT_ON -> setSwitch(swLiveTranscript, true, "Show what I say")
            Command.SET_TRANSCRIPT_OFF -> setSwitch(swLiveTranscript, false, "Show what I say")
            Command.SET_LISTENING_ON -> setSwitch(swScreenOff, true, "Screen-off listening")
            Command.SET_LISTENING_OFF -> setSwitch(swScreenOff, false, "Screen-off listening")
            Command.SET_SPEECH_ON -> setSwitch(swSpeech, true, "Speech")
            Command.SET_SPEECH_OFF -> setSwitch(swSpeech, false, "Speech")
            Command.SET_HAPTICS_ON -> setSwitch(swHaptics, true, "Vibration")
            Command.SET_HAPTICS_OFF -> setSwitch(swHaptics, false, "Vibration")
            Command.SET_CONTRAST_ON -> setSwitch(swHighContrast, true, "High contrast")
            Command.SET_CONTRAST_OFF -> setSwitch(swHighContrast, false, "High contrast")
            Command.SET_BIG_TEXT_ON -> setFontScaleFromVoice(true)
            Command.SET_BIG_TEXT_OFF -> setFontScaleFromVoice(false)
            Command.SET_CAPTIONS_ON -> tts.speak("Spoken-text panel removed")
            Command.SET_CAPTIONS_OFF -> tts.speak("Spoken-text panel removed")
            Command.RATE_SLOW -> setRate(SettingsStore.SPEECH_RATE_SLOW, "Slow")
            Command.RATE_NORMAL -> setRate(SettingsStore.SPEECH_RATE_NORMAL, "Normal")
            Command.RATE_FAST -> setRate(SettingsStore.SPEECH_RATE_FAST, "Fast")
        }
    }

    /** Mirrors a voice "X on/off" command onto the real switch — already
     *  in the target state just says so instead of toggling. */
    private fun setSwitch(sw: SwitchMaterial, on: Boolean, label: String) {
        if (sw.isChecked == on) {
            tts.speak("$label is already ${if (on) "on" else "off"}")
        } else {
            sw.isChecked = on
        }
    }

    private companion object {
        /** Mic types worth offering — TYPE_BLE_HEADSET is API 31+; the
         *  constant inlines at compile time so it's safe on older phones. */
        val MIC_DEVICE_TYPES = setOf(
            AudioDeviceInfo.TYPE_BUILTIN_MIC,
            AudioDeviceInfo.TYPE_WIRED_HEADSET,
            AudioDeviceInfo.TYPE_BLUETOOTH_SCO,
            AudioDeviceInfo.TYPE_USB_HEADSET,
            AudioDeviceInfo.TYPE_USB_DEVICE,
            AudioDeviceInfo.TYPE_BLE_HEADSET,
        )
    }
}
