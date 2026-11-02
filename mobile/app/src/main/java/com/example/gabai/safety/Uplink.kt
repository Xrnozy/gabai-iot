package com.example.gabai.safety

import android.content.Context
import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors

/** What the rest of the app may read about the live link and navigation mode; written by the uplink, the navigator and the AR fallback. */
object LiveState {
    @Volatile var navMode = "none"            // "pi" | "arcore" | "none"
    @Volatile var arState = "off"             // off | tracking | limited | unavailable
    @Volatile var piConnected: Boolean? = null
    @Volatile var piEverOnline = false
    @Volatile var relayOk = false
    @Volatile var relayRttMs = 0L
    @Volatile var relayError: String? = null
    @Volatile var lastSuccessMs = 0L
    @Volatile var gpsState = "off"            // off | searching | ok | weak | denied | disabled
    @Volatile var trackingRunning = false
    @Volatile var latestEmergency: JSONObject? = null
    @Volatile var watchers = 0
    private val listeners = CopyOnWriteArrayList<() -> Unit>()

    fun listen(l: () -> Unit) { listeners.add(l) }
    fun unlisten(l: () -> Unit) { listeners.remove(l) }
    fun changed() = listeners.forEach { it() }
}

/**
 * Everything the phone tells the relay goes through here: queued on disk while the relay is unreachable, replayed in order,
 * and safe to replay because every location and event has its own id (the relay ignores duplicates).
 */
object Uplink {
    private const val MAX_LOCATIONS = 1500
    private const val MAX_EVENTS = 200
    private const val BATCH_LOCATIONS = 100
    private const val TELEMETRY_EVERY_MS = 30_000L

    private val io = Executors.newSingleThreadExecutor { Thread(it, "uplink") }
    private val locations = ArrayDeque<JSONObject>()
    private val events = ArrayDeque<JSONObject>()
    private var nav: JSONObject? = null
    private var navDirty = false
    private var telemetry: (() -> JSONObject)? = null
    private var lastTelemetryAt = 0L
    private var file: File? = null
    private var loaded = false
    private val flushLock = Any()
    @Volatile private var pendingCount = 0

    @Volatile var lastFix: JSONObject? = null
        private set

    @Synchronized
    fun init(ctx: Context) {
        if (loaded) return
        loaded = true
        file = File(ctx.applicationContext.filesDir, "uplink_queue.json")
        runCatching {
            val root = JSONObject(file!!.readText())
            root.optJSONArray("loc")?.let { for (i in 0 until it.length()) locations.addLast(it.getJSONObject(i)) }
            root.optJSONArray("ev")?.let { for (i in 0 until it.length()) events.addLast(it.getJSONObject(i)) }
        }
        pendingCount = locations.size + events.size
    }

    @Synchronized
    fun addLocation(loc: JSONObject) {
        lastFix = loc
        locations.addLast(loc)
        while (locations.size > MAX_LOCATIONS) locations.removeFirst()
        pendingCount = locations.size + events.size
    }

    @Synchronized
    fun addEvent(ev: JSONObject) {
        events.addLast(ev)
        while (events.size > MAX_EVENTS) events.removeFirst()
        pendingCount = locations.size + events.size
        persist()
    }

    @Synchronized
    fun setNav(n: JSONObject) {
        nav = n
        navDirty = true
    }

    fun setTelemetrySource(src: () -> JSONObject) { telemetry = src }

    @Synchronized
    fun persist() {
        val f = file ?: return
        runCatching {
            val tmp = File(f.path + ".tmp")
            tmp.writeText(JSONObject().put("loc", JSONArray(locations.toList())).put("ev", JSONArray(events.toList())).toString())
            tmp.renameTo(f)
        }
    }

    fun flushAsync(ctx: Context) { io.execute { flush(ctx) } }

    /** Sends one batch. Blocking: never on the main thread. Anything not acknowledged stays queued. */
    fun flush(ctx: Context): Reply = synchronized(flushLock) {
        init(ctx)
        val now = SystemClock.elapsedRealtime()
        val batch = JSONObject().put("device_t", System.currentTimeMillis() / 1000.0)
        val sentLocs: List<JSONObject>
        val sentEvents: List<JSONObject>
        synchronized(this) {
            sentLocs = locations.take(BATCH_LOCATIONS)
            sentEvents = events.toList()
            if (navDirty) nav?.let { batch.put("nav", it) }
        }
        if (sentLocs.isNotEmpty()) batch.put("locations", JSONArray(sentLocs))
        if (sentEvents.isNotEmpty()) batch.put("events", JSONArray(sentEvents))
        if (now - lastTelemetryAt >= TELEMETRY_EVERY_MS || sentEvents.isNotEmpty()) telemetry?.let { batch.put("telemetry", it()) }

        // The queue lock is NOT held across the network call: an SOS press must never wait on a slow upload.
        val r = RelayApi(ctx, com.example.gabai.services.SettingsStore.ROLE_PHONE).ingest(batch)
        if (r.ok) {
            synchronized(this) {
                val gone = java.util.Collections.newSetFromMap(java.util.IdentityHashMap<JSONObject, Boolean>())
                gone.addAll(sentLocs); gone.addAll(sentEvents)
                locations.removeAll { it in gone }
                events.removeAll { it in gone }
                if (batch.has("nav")) navDirty = false
                pendingCount = locations.size + events.size
            }
            if (batch.has("telemetry")) lastTelemetryAt = now
            if (sentEvents.isNotEmpty() || sentLocs.isNotEmpty()) persist()
            LiveState.relayOk = true
            LiveState.relayError = null
            LiveState.lastSuccessMs = System.currentTimeMillis()
            LiveState.piConnected = r.body?.optBoolean("pi_connected")
            LiveState.piEverOnline = r.body?.optBoolean("pi_ever_online") ?: false
            LiveState.latestEmergency = r.body?.optJSONObject("latest_emergency")
            LiveState.watchers = r.body?.optInt("watchers") ?: 0
        } else {
            LiveState.relayOk = false
            LiveState.relayError = r.error
        }
        LiveState.relayRttMs = r.rttMs
        LiveState.changed()
        r
    }

    fun pending() = pendingCount
}
