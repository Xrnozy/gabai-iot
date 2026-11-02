package com.example.gabai.modules

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.example.gabai.WebRTCClient
import com.example.gabai.safety.ArFallback
import com.example.gabai.safety.LiveState
import com.example.gabai.safety.NavReport
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.TtsService
import com.example.gabai.services.VisionProfile
import com.example.gabai.services.VisionProfileStore
import com.example.gabai.services.Zone
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

class NavigationModule(
    context: Context,
    private val tts: TtsService,
) : WebRTCClient.NavigationListener {

    private val appContext = context.applicationContext
    private val modelManager = ModelManager()
    private val serverManager = ServerManager(context)
    private val started = AtomicBoolean(false)

    private var profile: VisionProfile = VisionProfileStore.load(appContext)
    private val advisoryCooldowns = mutableMapOf<String, Long>()
    private val hapticCooldowns = mutableMapOf<String, Long>()
    private val rawCommandCooldowns = mutableMapOf<String, Long>()

    // Reconnect machinery — a dropped peer used to be a dead end: the
    // app announced "connection lost" once and never recovered, leaving
    // the user walking blind until they restarted the screen.
    private val reconnectHandler = Handler(Looper.getMainLooper())
    private var reconnectAttempts = 0
    private var disconnectWatchdog: Runnable? = null
    private var pendingReconnect: Runnable? = null
    private var everConnected = false

    // ARCore fallback: entered when the nav link is down for FALLBACK_AFTER_MS or the relay says the Pi dropped off; left once both are fine.
    private val fallback = ArFallback(appContext, tts)
    private var inFallback = false
    private var unhealthySince = 0L
    private val watchdog = object : Runnable {
        override fun run() {
            if (!started.get()) return
            evaluateFallback()
            reconnectHandler.postDelayed(this, WATCH_MS)
        }
    }

    var onStatusText: ((String) -> Unit)? = null

    /** Fired when navigation switches between the Pi pipeline and the ARCore fallback. */
    var onFallbackChanged: ((Boolean) -> Unit)? = null

    /** Fired with the zone of each spoken advisory — used for haptic cues. */
    var onHazard: ((Zone) -> Unit)? = null

    /** Fired when the peer connection is live — hides "connecting" UI. */
    var onConnected: (() -> Unit)? = null

    /** Fired when the server connection fails — honest degraded state. */
    var onConnectionLost: (() -> Unit)? = null

    fun start() {
        if (!started.compareAndSet(false, true)) return
        profile = VisionProfileStore.load(appContext)

        // Lazy init: only now load navigation pipeline & connect.
        modelManager.ensureNavigationModelsInitialized()

        val rtc = serverManager.getOrCreateWebRtcClient(mode = "navigation")
        rtc.navigationListener = this
        rtc.startCamera()
        rtc.createOffer()
        reconnectHandler.postDelayed(watchdog, WATCH_MS)
    }

    fun stop() {
        if (!started.compareAndSet(true, false)) return
        cancelDisconnectWatchdog()
        reconnectHandler.removeCallbacksAndMessages(null)
        pendingReconnect = null
        fallback.stop()
        inFallback = false
        LiveState.navMode = "none"
        everConnected = false
        reconnectAttempts = 0
        serverManager.releaseWebRtc()
        modelManager.disposeNavigationModels()
    }

    /** Persist + apply a new vision profile (profile picker). */
    fun updateProfile(p: VisionProfile) {
        profile = p
        VisionProfileStore.saveProfile(appContext, p)
    }

    override fun onNavigationCommand(command: String, fullData: JSONObject) {
        onStatusText?.invoke(command)
        NavReport.publish("pi", command, fullData)
        val now = SystemClock.elapsedRealtime()

        // Haptic cue regardless of which speech path is active.
        fireHazardHaptic(fullData)

        val speech = fullData.optJSONObject("speech")
        if (speech != null) {
            // Server-composed natural guidance — already deduplicated,
            // prioritized and cooldown-managed by the backend. Speak it
            // verbatim; only steering/comfort lines honour the steering
            // toggle. interrupt controls whether it chops the in-flight
            // utterance (fresh emergencies) or queues behind it.
            val text = speech.optString("text", "")
            val kind = speech.optString("kind", "")
            val allowed = when (kind) {
                "steer", "clear" -> profile.steeringEnabled
                else -> true
            }
            if (allowed && text.isNotBlank()) {
                tts.speak(text, interrupt = speech.optBoolean(
                    "interrupt", kind == "emergency"))
            }
            return
        }

        // Legacy path: older backends send no "speech" object.
        if (fullData.optBoolean("should_speak", false)) {
            // "stop" (collision) and "caution" (degraded perception) are
            // safety notices — never gated by the steering toggle.
            // "veer left/right"/"proceed" are steering guidance and are.
            val spoken = when (command) {
                "stop", "caution" -> true
                else -> profile.steeringEnabled
            }
            if (spoken) speakRawCommand(command)
        }

        // While STOP is active every payload carries command=="stop" —
        // suppress advisories entirely: side warnings would chop the
        // critical utterance mid-word (QUEUE_FLUSH) and add chatter on
        // top of an already-issued collision warning.
        if (command != "stop") {
            speakTopAdvisory(fullData, now)
        }
    }

    /**
     * Last-resort path for backends that still push bare `should_speak`
     * + command words instead of composed `speech`. Never trust their
     * cadence: an old server emits "stop" on EVERY frame while a stop
     * persists, which machine-guns the word into the speaker. Enforce a
     * per-command floor here so no backend can spam raw words, and only
     * let a genuinely fresh stop interrupt — repeats queue politely.
     */
    private fun speakRawCommand(command: String) {
        val now = SystemClock.elapsedRealtime()
        val floor = when (command) {
            "stop" -> RAW_COMMAND_FLOORS_STOP_MS
            "caution" -> RAW_COMMAND_FLOOR_MS
            else -> RAW_COMMAND_FLOOR_MS
        }
        val last = rawCommandCooldowns[command] ?: Long.MIN_VALUE
        if (now - last < floor) return
        rawCommandCooldowns[command] = now
        // Interrupt only when this is a fresh stop after a real gap —
        // a stop that repeats within the episode window just queues
        // behind the current sentence instead of chopping it.
        val interrupt = command == "stop" && now - last > RAW_STOP_INTERRUPT_GAP_MS
        tts.speak(command, interrupt = interrupt)
    }

    /**
     * Haptic cue for the top hazard inside the user's blind zones —
     * independent of which speech path produced the audible line.
     * Throttled per side: hazards[] arrives every frame, and a constant
     * buzz is as useless as constant speech.
     */
    private fun fireHazardHaptic(fullData: JSONObject) {
        val hazards = fullData.optJSONArray("hazards") ?: return
        val now = SystemClock.elapsedRealtime()
        for (i in 0 until hazards.length()) {
            val h = hazards.optJSONObject(i) ?: continue
            val zone = Zone.fromWire(h.optString("zone", "")) ?: continue
            if (zone !in profile.blindZones && !h.optBoolean("priority", false)) continue
            val key = when (zone) {
                Zone.LEFT, Zone.SLIGHT_LEFT -> "hl"
                Zone.RIGHT, Zone.SLIGHT_RIGHT -> "hr"
                else -> "hf"
            }
            if (now - (hapticCooldowns[key] ?: 0L) < HAPTIC_COOLDOWN_MS) return
            hapticCooldowns[key] = now
            onHazard?.invoke(zone)
            return
        }
    }

    /**
     * Advisory tier: hazards[] is sorted by severity server-side. We speak
     * the first one inside the user's blind zones, throttled PER ZONE —
     * keying on the zone rather than (kind, zone) means the same physical
     * object flickering between pothole/obstacle/uneven_ground shares one
     * cooldown instead of cycling phrases back-to-back.
     * Only one advisory per payload — TtsService flushes the queue, so a
     * second speak() would chop the first mid-word.
     */
    private fun speakTopAdvisory(fullData: JSONObject, now: Long) {
        val hazards = fullData.optJSONArray("hazards") ?: return
        for (i in 0 until hazards.length()) {
            val h = hazards.optJSONObject(i) ?: continue
            val zone = Zone.fromWire(h.optString("zone", "")) ?: continue
            // Priority hazards (potholes, drop-offs, uneven terrain,
            // parked wheels) announce regardless of the blind-zone
            // profile — they are trip hazards, not just blind spots.
            if (zone !in profile.blindZones && !h.optBoolean("priority", false)) continue

            val key = when (zone) {
                Zone.LEFT, Zone.SLIGHT_LEFT -> "left"
                Zone.RIGHT, Zone.SLIGHT_RIGHT -> "right"
                else -> "forward"
            }
            val last = advisoryCooldowns[key] ?: 0L
            if (now - last < ADVISORY_COOLDOWN_MS) continue

            val phrase = phraseFor(h) ?: continue
            advisoryCooldowns[key] = now
            tts.speak(phrase, interrupt = false)
            onHazard?.invoke(zone)
            return
        }
    }

    private fun phraseFor(h: JSONObject): String? {
        var noun = when (h.optString("kind", "")) {
            "pothole" -> "Pothole"
            "dropoff" -> "Drop-off"
            "uneven_ground" -> "Uneven ground"
            "obstacle" -> "Obstacle"
            "person" -> "Person"
            "dog" -> "Dog"
            "animal" -> "Animal"
            "wall" -> "Wall"
            "doorway" -> "Doorway"
            "chair" -> "Chair"
            "bench" -> "Bench"
            "furniture" -> "Furniture"
            "stairs" -> "Stairs"
            "curb" -> "Curb"
            "hydrant" -> "Hydrant"
            "post" -> "Post"
            "sign" -> "Sign"
            "tricycle" -> "Tricycle"
            "pedicab" -> "Pedicab"
            "bicycle" -> "Bicycle"
            "motorcycle" -> "Motorcycle"
            "vehicle" -> "Vehicle"
            "vendor_stall" -> "Vendor stall"
            "plant_pot" -> "Plant pot"
            "debris" -> "Debris"
            else -> return null
        }
        // Static wheeled things read as "parked" — a bicycle that isn't
        // approaching is a parked bicycle for advisory purposes.
        if (h.optBoolean("parked", false) && noun in WHEELED_NOUNS) {
            noun = "Parked " + noun.lowercase()
        }
        val dir = when (h.optString("zone", "")) {
            "left", "slight_left" -> "on your left"
            "right", "slight_right" -> "on your right"
            else -> "ahead"
        }
        val approaching = if (h.optBoolean("approaching", false)) " approaching" else ""
        return "$noun$approaching, $dir"
    }

    override fun onConnected() {
        unhealthySince = 0L
        cancelDisconnectWatchdog()
        pendingReconnect?.let { reconnectHandler.removeCallbacks(it) }
        pendingReconnect = null
        reconnectAttempts = 0
        if (everConnected) {
            // Reconnected after a drop — say so; the user knows the feed
            // (and guidance) is live again rather than wondering.
            tts.speak("Connection restored.")
        } else {
            // Version marker spoken aloud — if the phone just says
            // "Connected", the installed APK predates the natural-
            // guidance build and was compiled from a different copy.
            tts.speak("Connected. Gabai guidance version two.")
        }
        everConnected = true
        onConnected?.invoke()
    }

    override fun onDisconnected() {
        // Transient ICE gap — usually self-heals back to CONNECTED.
        // If it doesn't within the grace window, the peer is effectively
        // dead: rebuild the session instead of waiting forever.
        cancelDisconnectWatchdog()
        val r = Runnable {
            if (started.get()) {
                Log.w(TAG, "[RTC] Disconnected $DISCONNECT_GRACE_MS ms —" +
                        " rebuilding session")
                scheduleReconnect()
            }
        }
        disconnectWatchdog = r
        reconnectHandler.postDelayed(r, DISCONNECT_GRACE_MS)
    }

    private fun cancelDisconnectWatchdog() {
        disconnectWatchdog?.let { reconnectHandler.removeCallbacks(it) }
        disconnectWatchdog = null
    }

    /** Rebuild the whole peer connection after FAILED (or a stuck
     *  DISCONNECT). Backoff so a dead network doesn't hot-loop, but the
     *  first retry is fast — this is a navigation aid, not a chat app. */
    private fun scheduleReconnect() {
        if (!started.get() || pendingReconnect != null) return
        val exp = reconnectAttempts.coerceAtMost(4)
        val delay = RECONNECT_BASE_MS * (1L shl exp)   // 2s 4s 8s 16s 32s
        reconnectAttempts++
        val r = Runnable {
            pendingReconnect = null
            reconnect()
        }
        pendingReconnect = r
        reconnectHandler.postDelayed(r, delay)
    }

    private fun reconnect() {
        if (!started.get()) return
        serverManager.releaseWebRtc()
        val rtc = serverManager.getOrCreateWebRtcClient(mode = "navigation")
        rtc.navigationListener = this
        rtc.startCamera()
        rtc.createOffer()
    }

    override fun onCameraError(message: String) {
        // A blind user cannot see a black preview — say it out loud.
        tts.speak(message, interrupt = false)
        onStatusText?.invoke(message)
    }

    override fun onConnectionFailed() {
        // Honest degradation + self-heal: tell the user and rebuild —
        // waiting for them to restart the screen is not an option when
        // they cannot see the screen.
        cancelDisconnectWatchdog()
        tts.speak("Connection lost. Trying to reconnect.")
        onConnectionLost?.invoke()
        scheduleReconnect()
    }

    // ---------- ARCore fallback ----------

    private fun serverReachable(): Boolean = runCatching {
        val c = java.net.URL(SettingsStore.getServerUrl(appContext)).openConnection() as java.net.HttpURLConnection
        c.connectTimeout = 2_000; c.readTimeout = 2_000
        c.responseCode; c.disconnect(); true
    }.getOrDefault(false)

    private fun piDroppedPerRelay() = LiveState.relayOk && LiveState.piEverOnline && LiveState.piConnected == false

    private fun evaluateFallback() {
        val now = SystemClock.elapsedRealtime()
        val linkDown = pendingReconnect != null || disconnectWatchdog != null || reconnectAttempts > 0
        val unhealthy = linkDown || piDroppedPerRelay()
        if (!inFallback) {
            if (!unhealthy) { unhealthySince = 0L; return }
            if (unhealthySince == 0L) unhealthySince = now
            if (now - unhealthySince >= FALLBACK_AFTER_MS) enterFallback()
            return
        }
        // In fallback the camera belongs to ARCore, so probe the server over plain HTTP instead of trying to stream.
        Thread {
            val up = serverReachable() && !piDroppedPerRelay()
            reconnectHandler.post { if (up && inFallback && started.get()) exitFallback() }
        }.start()
    }

    private fun enterFallback() {
        if (inFallback) return
        if (!fallback.available()) {
            if (unhealthySince > 0) { unhealthySince = 0L; tts.speak("Connection lost. Backup navigation is not available on this phone. Please stop and wait.", interrupt = true) }
            return
        }
        inFallback = true
        pendingReconnect?.let { reconnectHandler.removeCallbacks(it) }
        pendingReconnect = null
        cancelDisconnectWatchdog()
        serverManager.releaseWebRtc()
        if (fallback.start()) onFallbackChanged?.invoke(true) else { inFallback = false; scheduleReconnect() }
    }

    private fun exitFallback() {
        inFallback = false
        fallback.stop()
        unhealthySince = 0L
        reconnectAttempts = 0
        onFallbackChanged?.invoke(false)
        // Give ARCore a moment to release the camera before WebRTC takes it back.
        reconnectHandler.postDelayed({ if (started.get()) reconnect() }, 800)
    }

    companion object {
        private const val WATCH_MS = 3_000L
        private const val FALLBACK_AFTER_MS = 8_000L
        private const val TAG = "NavigationModule"
        private const val ADVISORY_COOLDOWN_MS = 5_000L
        private const val HAPTIC_COOLDOWN_MS = 1_500L
        // Minimum interval between raw-command utterances on the legacy
        // path — the backend sends these per-frame, so this floor is the
        // only thing keeping a persistent "stop" from stuttering.
        private const val RAW_COMMAND_FLOORS_STOP_MS = 4_000L
        private const val RAW_COMMAND_FLOOR_MS = 6_000L
        private const val RAW_STOP_INTERRUPT_GAP_MS = 12_000L
        // Reconnect tuning: a DISCONNECTED peer gets this long to
        // self-heal before we tear it down and re-offer.
        private const val DISCONNECT_GRACE_MS = 15_000L
        private const val RECONNECT_BASE_MS = 2_000L
        private val WHEELED_NOUNS =
            setOf("Bicycle", "Motorcycle", "Vehicle", "Tricycle", "Pedicab")
    }
}
