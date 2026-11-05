package com.example.gabai.services

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.example.gabai.GabaiApp
import com.example.gabai.MainActivity
import com.example.gabai.NavigationActivity
import com.example.gabai.R
import com.example.gabai.SettingsActivity
import java.util.Locale

/**
 * Foreground service that keeps "Gabai" voice commands alive while the app
 * is backgrounded or the screen is off (demo feature — toggle in Settings).
 *
 * Mic ownership: activities listen while in the foreground; this service
 * listens only while the app is backgrounded, coordinated through
 * [GabaiApp.isInForeground] so the two AudioRecord streams never contend.
 *
 * The persistent notification is the honest mic indicator — the mic is on.
 */
class VoiceListenerService : Service() {

    private lateinit var tts: TtsService
    private var voice: VoiceCommandService? = null
    private lateinit var session: CommandSession
    private var voiceLoading = false
    private var voiceAwaiting = false
    private val foregroundListener: (Boolean) -> Unit = { foreground ->
        if (foreground) {
            voice?.stop()
            session.reset()
        } else {
            startListening()
        }
    }

    override fun onCreate() {
        super.onCreate()
        tts = TtsService(this, Locale.US)
        session = CommandSession(
            tts = tts,
            // Screen-off stays wake-gated even in keyword-first mode:
            // a pocketed phone can't warn before ambient "off"/"back"
            // fires — the greeting is the price of always-on safety.
            requireWakeName = { true },
            onCommand = ::execute,
            haptics = Haptics(this),
        ).apply {
            onAwaitingCommand = { awaiting ->
                voiceAwaiting = awaiting
                refreshNotification()
            }
        }
        createChannel()
        startForegroundCompat()
        GabaiApp.addForegroundListener(foregroundListener)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (!GabaiApp.isInForeground) startListening()
        return START_STICKY
    }

    override fun onDestroy() {
        GabaiApp.removeForegroundListener(foregroundListener)
        session.reset()
        voice?.destroy()
        voice = null
        tts.shutdown()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // ---------- listening ----------

    private fun startListening() {
        if (!SettingsStore.isVoiceCommandsEnabled(this) ||
            !SettingsStore.isScreenOffListening(this)
        ) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) return

        if (voice == null) {
            voice = VoiceCommandService(
                context = applicationContext,
                speechGate = { !tts.isSpeaking && !SoundCues.isPlaying },
                onCommand = { utterance -> session.handle(utterance); true },
                onError = {},
                requireWakeName = { true },
                selfSpeech = { tts.lastSpoken() },
                commandWindowOpen = { session.isWindowOpen },
                onLoadingChanged = { loading ->
                    voiceLoading = loading
                    refreshNotification()
                },
            )
        }
        voice?.start()
    }

    private fun execute(command: Command) {
        when (command) {
            Command.NAVIGATION -> openScreen(NavigationActivity::class.java, "Opening Navigation")
            Command.SETTINGS -> openScreen(SettingsActivity::class.java, "Opening Settings")
            Command.SOS -> com.example.gabai.safety.Emergency.startSos(this)
            Command.HELP -> tts.speak(CommandParser.HELP_HINT)
            Command.REPEAT -> tts.speak(CommandParser.HELP_HINT)
            Command.BACK -> tts.speak("Nothing to go back from")
            Command.STOP -> {
                // "Gabai, off" — switch background listening off for good.
                SettingsStore.setScreenOffListening(this, false)
                tts.speak("Voice listening off")
                voice?.stop()
                // Let the farewell finish before the service (and TTS) dies.
                android.os.Handler(android.os.Looper.getMainLooper())
                    .postDelayed({ stopSelf() }, 1_500)
            }
            // "captions off", "speak faster", ... while backgrounded.
            else -> SettingsVoice.apply(this, tts, command)
        }
    }

    private fun openScreen(cls: Class<*>, ack: String) {
        wakeScreen()
        tts.speak(ack)
        val intent = Intent(this, cls).addFlags(
            Intent.FLAG_ACTIVITY_NEW_TASK or
                Intent.FLAG_ACTIVITY_CLEAR_TOP or
                Intent.FLAG_ACTIVITY_SINGLE_TOP
        )
        try {
            // Allowed here on the demo path: the app was recently foreground
            // or the OEM permits FGS launches. If blocked, the persistent
            // notification still opens the app on tap.
            startActivity(intent)
        } catch (_: Exception) {
        }
    }

    /** Turns the screen on so the launched screen (and its own mic) takes over. */
    private fun wakeScreen() {
        val pm = getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return
        @Suppress("DEPRECATION") // Deprecated but still the simplest demo path.
        val lock = pm.newWakeLock(
            PowerManager.FULL_WAKE_LOCK or
                PowerManager.ACQUIRE_CAUSES_WAKEUP or
                PowerManager.ON_AFTER_RELEASE,
            "gabai:voice_wake",
        )
        // Held for the 10 s timeout — the woken screen lets the launched
        // activity (and its own mic) take over.
        lock.acquire(10_000)
    }

    // ---------- notification ----------

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Gabai voice listener",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "Shows while Gabai is listening for voice commands"
                setShowBadge(false)
            }
            getSystemService(NotificationManager::class.java)
                ?.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(
        statusText: String = "Say: Gab, walk. Or Gab, off to stop",
    ): Notification {
        val openApp = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_mic)
            .setContentTitle("Gab is listening")
            .setContentText(statusText)
            .setContentIntent(openApp)
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    /** Reflects model loading + the wake-word command window in the
     *  persistent notification. */
    private fun refreshNotification() {
        val statusText = when {
            voiceLoading -> "Loading voice model…"
            voiceAwaiting -> "Listening. Say a command"
            else -> "Say: Gab, walk. Or Gab, off to stop"
        }
        getSystemService(NotificationManager::class.java)
            ?.notify(NOTIFICATION_ID, buildNotification(statusText))
    }

    private fun startForegroundCompat() {
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID, notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    companion object {
        private const val CHANNEL_ID = "gabai_voice_listener"
        private const val NOTIFICATION_ID = 7

        /** Starts the listener if the user enabled it; safe to call repeatedly. */
        fun ensureStarted(context: Context) {
            if (!SettingsStore.isScreenOffListening(context) ||
                !SettingsStore.isVoiceCommandsEnabled(context)
            ) return
            val intent = Intent(context, VoiceListenerService::class.java)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, VoiceListenerService::class.java))
        }
    }
}
