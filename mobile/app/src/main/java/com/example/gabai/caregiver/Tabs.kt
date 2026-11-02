package com.example.gabai.caregiver

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.view.View
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.example.gabai.R
import com.example.gabai.safety.RelayApi
import com.example.gabai.services.SettingsStore
import com.example.gabai.ui.LiveCard
import com.example.gabai.ui.Level
import com.example.gabai.ui.Ui
import org.json.JSONObject

interface CaregiverTab {
    val id: String
    val label: String
    val icon: Int
    /** Spoken after the tab name by accessible navigation. */
    val purpose: String
    val view: View
    fun update(s: JSONObject?, link: HubLink)
    fun onShow() {}
    fun onHide() {}
}

/** Fills the cards shared by several tabs; every value carries the age it was measured at. */
internal object Rows {

    fun ago(s: Double?) = Fmt.age(s) + " ago"

    fun location(card: LiveCard, s: JSONObject?) {
        val loc = s.obj("location")
        val phone = s.obj("connections").obj("phone")
        card.begin()
        if (loc == null || loc.flag("available") != true) {
            card.set("Position", "No location received yet", Level.OFF)
            card.end(); return
        }
        val stale = loc.flag("stale") == true
        val acc = loc.num("accuracy_m")
        card.set("Status", if (stale) "Last known location, not live" else "Live", if (stale) Level.WARN else Level.OK)
        card.set("Updated", "${ago(loc.num("age_s"))} (${Fmt.clock(loc.num("fix_t"))})", if (stale) Level.WARN else null)
        card.set("Position", "%.5f, %.5f".format(loc.num("lat"), loc.num("lon")))
        card.set("Accuracy", if (acc == null) "unknown" else "±${acc.toInt()} m", when (loc.str("quality")) {
            "good" -> Level.OK; "weak" -> Level.WARN; "poor" -> Level.BAD; else -> Level.OFF })
        card.set("Speed", Fmt.kmh(loc.num("speed_mps")) + if (loc.flag("moving") == true) "" else " (standing still)")
        card.set("Moving toward", Fmt.compass(loc.num("bearing_deg")))
        card.set("Phone facing", Fmt.compass(loc.num("heading_deg")))
        card.set("Distance walked", Fmt.metres(loc.num("odometer_m")))
        loc.num("off_route_m")?.let { card.set("From saved route", "${it.toInt()} m away", if (it > 30) Level.BAD else Level.OK) }
        if (phone?.str("state") == "offline") card.set("Phone", "Offline since ${Fmt.clock(phone.num("offline_since"))}", Level.WARN)
        card.end()
    }

    fun navigation(card: LiveCard, s: JSONObject?, detailed: Boolean) {
        val nav = s.obj("navigation")
        card.begin()
        if (nav == null || nav.flag("available") != true) {
            card.set("Navigation", "No navigation data yet", Level.OFF)
            card.end(); return
        }
        val stale = nav.flag("stale") == true
        val mode = nav.str("mode")
        card.set("Mode", when (mode) {
            "pi" -> "Raspberry Pi navigation"
            "arcore" -> "ARCore fallback (phone only, limited)"
            else -> "Unknown" }, when (mode) { "pi" -> Level.OK; "arcore" -> Level.WARN; else -> Level.OFF })
        card.set("Data", if (stale) "Last known, ${ago(nav.num("age_s"))}" else "Live", if (stale) Level.WARN else Level.OK)
        nav.str("instruction")?.let { card.set("Instruction", it) }
        nav.str("direction")?.let { card.set("Direction", it) }
        nav.num("distance_m")?.let { card.set("Next instruction in", Fmt.metres(it)) }
        val safe = nav.flag("forward_safe")
        if (safe != null) card.set("Path ahead", if (safe) "Clear to continue" else "Stop or take care", if (safe) Level.OK else Level.BAD)
        nav.num("path_confidence")?.let { card.set("Walkable path confidence", Fmt.pct(if (it <= 1.0) it * 100 else it)) }
        if (detailed) {
            nav.num("clearance_left_m")?.let { card.set("Clearance on left", Fmt.metres(it)) }
            nav.num("clearance_right_m")?.let { card.set("Clearance on right", Fmt.metres(it)) }
            val obs = nav.optJSONArray("obstacles")
            if (obs == null || obs.length() == 0) card.set("Obstacles", "None reported", Level.OK)
            else for (i in 0 until minOf(obs.length(), 5)) {
                val o = obs.getJSONObject(i)
                val kind = o.str("kind")?.replace('_', ' ')?.replaceFirstChar { it.uppercase() } ?: "Obstacle"
                val d = o.num("distance_m")?.let { " · ${Fmt.metres(it)}" } ?: ""
                card.set("Obstacle ${i + 1}", "$kind ${Fmt.zoneWords(o.str("zone"))}$d" + if (o.flag("approaching") == true) " · approaching" else "", Level.WARN)
            }
        }
        card.end()
    }

    fun connections(card: LiveCard, s: JSONObject?, link: HubLink) {
        card.begin()
        val c = s.obj("connections")
        for ((key, label) in listOf("phone" to "Phone to server", "pi" to "Raspberry Pi to server")) {
            val d = c.obj(key)
            val online = d?.str("state") == "online"
            val v = if (d == null) "Unknown"
            else if (online) "Connected · last data ${ago(d.num("last_seen_age_s"))}"
            else "Offline" + (d.num("offline_since")?.let { " since ${Fmt.clock(it)}" } ?: "") + (d.str("offline_reason")?.let { " ($it)" } ?: "")
            card.set(label, v, if (d == null) Level.OFF else if (online) Level.OK else Level.BAD)
        }
        card.set("This app to server", if (link.ok) "Connected · ${link.rttMs} ms" else (link.error ?: "Not connected") +
            (link.downSinceMs?.let { " since ${Fmt.clock(it / 1000.0)}" } ?: ""), if (link.ok) Level.OK else Level.BAD)
        if (link.lastSuccessMs > 0) card.set("Last successful update", Fmt.clock(link.lastSuccessMs / 1000.0))
        card.set("Caregivers watching", "${c?.optInt("caregivers") ?: 0}")
        card.end()
    }
}

private fun scroll(ctx: Context, content: View) = ScrollView(ctx).apply {
    setBackgroundColor(Ui.color(ctx, R.color.canvas)); isFillViewport = true; addView(content)
}

private fun page(ctx: Context) = Ui.column(ctx, 16)

private fun LinearLayout.card(card: LiveCard, top: Int = 12) = addView(card.view, Ui.lp(top = top, ctx = context))

/** Status banner. An emergency is solid red tape with on-tape text; other levels are tinted panels with a 2dp edge. */
private fun banner(ctx: Context, level: Level, text: String, tv: TextView, solid: Boolean = false) {
    val c = Ui.color(ctx, level.color)
    tv.text = "${level.symbol}  $text"
    tv.background = GradientDrawable().apply {
        if (solid) setColor(Ui.color(ctx, R.color.tape_sos))
        else { setColor(Color.argb(28, Color.red(c), Color.green(c), Color.blue(c))); setStroke(Ui.dp(ctx, 2), c) }
        cornerRadius = Ui.dp(ctx, 6).toFloat()
    }
    tv.setTextColor(if (solid) Ui.color(ctx, R.color.on_tape) else c)
}

class OverviewTab(private val ctx: Context, private val open: (String) -> Unit, private val ack: (String) -> Unit) : CaregiverTab {
    override val id = "overview"
    override val label = "Overview"
    override val icon = R.drawable.ic_tab_overview
    override val purpose = "Everything at a glance: status, location, navigation and connections."

    private val status = Ui.text(ctx, "Waiting for the first update…", 20f, R.color.ink, bold = true).apply {
        setPadding(Ui.dp(ctx, 16), Ui.dp(ctx, 16), Ui.dp(ctx, 16), Ui.dp(ctx, 16))
        accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
    }
    private val ackBtn = Ui.button(ctx, "Acknowledge — I'm responding", Ui.Kind.DANGER, "Acknowledge the emergency. Tells the user you are responding. Button.") {
        activeId?.let(ack)
    }.apply { visibility = View.GONE }
    private var activeId: String? = null
    private val loc = LiveCard(ctx, "Location")
    private val nav = LiveCard(ctx, "Navigation")
    private val conn = LiveCard(ctx, "Connections")

    override val view = scroll(ctx, page(ctx).apply {
        addView(status, Ui.lp(ctx = ctx))
        addView(ackBtn, Ui.lp(top = 12, ctx = ctx))
        card(loc); card(nav); card(conn)
        loc.add(Ui.button(ctx, "Open map", Ui.Kind.SECONDARY, "Open the map tab. Button.") { open("map") })
        nav.add(Ui.button(ctx, "Open live cameras", Ui.Kind.SECONDARY, "Open the Raspberry Pi camera tab. Button.") { open("camera") })
        conn.add(Ui.button(ctx, "Open device health", Ui.Kind.SECONDARY, "Open the device health tab. Button.") { open("device") })
    })

    override fun update(s: JSONObject?, link: HubLink) {
        val active = s.obj("safety").obj("active")
        val phone = s.obj("connections").obj("phone")?.str("state")
        val pi = s.obj("connections").obj("pi")?.str("state")
        activeId = active?.takeIf { it.str("status") != "acknowledged" }?.str("id")
        ackBtn.visibility = if (activeId != null) View.VISIBLE else View.GONE
        when {
            s == null -> banner(ctx, if (link.error != null) Level.BAD else Level.OFF,
                link.error?.let { "Cannot reach the server. $it" } ?: "Waiting for the first update…", status)
            active != null -> banner(ctx, Level.BAD, (if (active.str("kind") == "sos") "EMERGENCY SOS" else "POSSIBLE FALL") +
                " · ${active.str("status")} · ${Fmt.clock(active.num("t"))}", status, solid = true)
            !link.ok -> banner(ctx, Level.WARN, "This app lost the server. Showing last known information from ${Fmt.clock(link.lastSuccessMs / 1000.0)}.", status)
            phone != "online" -> banner(ctx, Level.WARN, "Phone is offline. Showing last known information.", status)
            pi != "online" -> banner(ctx, Level.WARN, "Raspberry Pi is offline. The phone may be using the ARCore fallback.", status)
            else -> banner(ctx, Level.OK, "All good. Everything is connected.", status)
        }
        Rows.location(loc, s); Rows.navigation(nav, s, detailed = false); Rows.connections(conn, s, link)
    }
}

class SafetyTab(private val ctx: Context, private val ack: (String) -> Unit) : CaregiverTab {
    override val id = "safety"
    override val label = "Safety"
    override val icon = R.drawable.ic_tab_safety
    override val purpose = "Emergencies, falls, route deviations and the alert history."

    private val active = LiveCard(ctx, "Active alert")
    private val history = LiveCard(ctx, "Alert history")
    private var activeId: String? = null
    private val ackBtn = Ui.button(ctx, "Acknowledge — I'm responding", Ui.Kind.DANGER, "Acknowledge the emergency. Button.") { activeId?.let(ack) }
    private val api = RelayApi(ctx, SettingsStore.ROLE_CAREGIVER)

    override val view = scroll(ctx, page(ctx).apply { card(active, 0); card(history); active.add(ackBtn) })

    override fun update(s: JSONObject?, link: HubLink) {
        val a = s.obj("safety").obj("active")
        activeId = a?.takeIf { it.str("status") != "acknowledged" }?.str("id")
        ackBtn.visibility = if (activeId != null) View.VISIBLE else View.GONE
        active.begin()
        if (a == null) active.set("Status", "No active emergency", Level.OK)
        else {
            val ev = a.obj("evidence")
            active.set("Type", if (a.str("kind") == "sos") "SOS pressed" else "Possible fall", Level.BAD)
            active.set("State", a.str("status") ?: "unknown", if (a.str("status") == "pending") Level.WARN else Level.BAD)
            active.set("When", Fmt.dateTime(a.num("t")))
            a.num("confidence")?.let { active.set("Fall confidence", Fmt.pct(it * 100), if (it >= 0.8) Level.BAD else Level.WARN) }
            ev.num("peak_g")?.let { active.set("Impact", "%.1f g".format(it)) }
            ev.num("freefall_ms")?.let { if (it > 0) active.set("Free-fall", "${it.toInt()} ms") }
            ev.num("still_s")?.let { active.set("Stillness afterwards", "%.1f s".format(it)) }
            ev.num("gyro_peak_dps")?.let { active.set("Rotation", "${it.toInt()}°/s") }
            ev.num("orientation_change_deg")?.let { active.set("Phone orientation change", "${it.toInt()}°") }
            a.num("lat")?.let { active.set("Location", "%.5f, %.5f".format(it, a.num("lon") ?: 0.0)) }
            ev.num("fix_age_s")?.let { active.set("Location age", Fmt.age(it)) }
        }
        active.end()

        val recent = s.obj("safety")?.optJSONArray("recent")
        history.begin()
        if (recent == null || recent.length() == 0) history.set("Events", "None yet", Level.OK)
        else for (i in 0 until recent.length()) {
            val e = recent.getJSONObject(i)
            val status = e.str("status") ?: ""
            val who = if (status == "acknowledged") " by ${e.str("ack_by") ?: "a caregiver"}" else ""
            history.set("${kindName(e.str("kind"))} · ${Fmt.dateTime(e.num("t"))}", status + who, when (status) {
                "acknowledged", "resolved" -> Level.OK; "cancelled" -> Level.OFF; "pending" -> Level.WARN; else -> Level.BAD })
        }
        history.end()
    }

    private fun kindName(k: String?) = when (k) {
        "fall" -> "Fall"; "sos" -> "SOS"; "deviation" -> "Off route"; "inactivity" -> "Not moving"; else -> k ?: "Event" }
}

class DeviceTab(private val ctx: Context) : CaregiverTab {
    override val id = "device"
    override val label = "Device"
    override val icon = R.drawable.ic_tab_device
    override val purpose = "Raspberry Pi and phone health, cameras and connection status."

    private val pi = LiveCard(ctx, "Raspberry Pi")
    private val cams = LiveCard(ctx, "Pi cameras")
    private val phone = LiveCard(ctx, "Phone")
    private val conn = LiveCard(ctx, "Connections")
    private var lastBytes: Pair<Double, Double>? = null

    override val view = scroll(ctx, page(ctx).apply { card(pi, 0); card(cams); card(phone); card(conn) })

    override fun update(s: JSONObject?, link: HubLink) {
        val p = s.obj("pi")
        val h = p.obj("health")
        val stale = p == null || p.flag("stale") != false
        val lvl = { l: Level -> if (stale && l == Level.OK) Level.WARN else l }
        pi.begin()
        if (p == null || p.flag("available") != true) pi.set("Raspberry Pi", "Has not reported yet", Level.OFF)
        else {
            pi.set("Data", if (stale) "Last known, ${Rows.ago(p.num("age_s"))}" else "Live", if (stale) Level.WARN else Level.OK)
            h.num("cpu_pct")?.let { pi.set("CPU usage", Fmt.pct(it), lvl(if (it > 90) Level.WARN else Level.OK)) } ?: pi.set("CPU usage", "Not reported", Level.OFF)
            pi.set("GPU usage", "Not reported by this Pi", Level.OFF)
            val total = h.num("mem_total_mb"); val avail = h.num("mem_available_mb")
            if (total != null && avail != null) pi.set("RAM used", "${(total - avail).toInt()} of ${total.toInt()} MB (${Fmt.pct((total - avail) / total * 100)})",
                lvl(if (avail / total < 0.1) Level.WARN else Level.OK))
            else avail?.let { pi.set("RAM free", "${it.toInt()} MB") }
            h.num("temp_c")?.let { pi.set("CPU temperature", "%.1f °C".format(it), lvl(if (it >= 80) Level.BAD else if (it >= 70) Level.WARN else Level.OK)) }
            pi.set("Battery / power", "Mains or power bank; battery level not reported", Level.OFF)
            val (now, past) = Fmt.powerFlags(h.str("throttled"))
            if (h.str("throttled") == null) pi.set("Power and throttling", "Not reported", Level.OFF)
            else pi.set("Power and throttling", if (now.isEmpty() && past.isEmpty()) "Normal" else
                (if (now.isNotEmpty()) "Now: ${now.joinToString()}. " else "") + (if (past.isNotEmpty()) "Since boot: ${past.joinToString()}" else ""),
                if (now.isNotEmpty()) Level.BAD else if (past.isNotEmpty()) Level.WARN else Level.OK)
            h.num("disk_free_gb")?.let { pi.set("Storage free", "%.1f GB".format(it), lvl(if (it < 1) Level.BAD else Level.OK)) } ?: pi.set("Storage free", "Not reported", Level.OFF)
            h.num("uptime_s")?.let { pi.set("Uptime", Fmt.age(it)) } ?: pi.set("Uptime", "Not reported", Level.OFF)
        }
        pi.end()

        cams.begin()
        val cam = p.obj("camera")
        val connected = p != null && !stale && ((cam.num("fps_left") ?: 0.0) > 0 || (cam.num("fps_right") ?: 0.0) > 0)
        cams.set("Cameras", if (connected) "Connected" else if (cam == null) "Unknown" else "Not sending", if (connected) Level.OK else if (cam == null) Level.OFF else Level.BAD)
        cams.set("Left camera", cam.num("fps_left")?.let { "%.1f fps".format(it) } ?: "no data", lvl(if ((cam.num("fps_left") ?: 0.0) >= 10) Level.OK else Level.WARN))
        cams.set("Right camera", cam.num("fps_right")?.let { "%.1f fps".format(it) } ?: "no data", lvl(if ((cam.num("fps_right") ?: 0.0) >= 10) Level.OK else Level.WARN))
        cams.set("Dropped frames", cam.num("dropped")?.toInt()?.toString() ?: "Not reported")
        cams.set("Left/right sync difference", cam.num("pair_dt_ms")?.let { "%.1f ms".format(it) } ?: "Not reported",
            cam.num("pair_dt_ms")?.let { lvl(if (kotlin.math.abs(it) > 20) Level.WARN else Level.OK) })
        cams.set("Resolution", if (cam.num("width") != null) "${cam.num("width")!!.toInt()} × ${cam.num("height")!!.toInt()}" else "Not reported")
        cams.set("Latency to server", p.num("latency_ms")?.let { "${it.toInt()} ms" } ?: "Not reported")
        val bytes = p.obj("link").num("bytes")
        val now = System.currentTimeMillis() / 1000.0
        val prev = lastBytes
        if (bytes != null && prev != null && now > prev.second && bytes >= prev.first && !stale)
            cams.set("Data rate", "%.2f Mbit/s".format((bytes - prev.first) * 8 / 1e6 / (now - prev.second)))
        else cams.set("Data rate", if (stale) "No data" else "Measuring…")
        if (bytes != null) lastBytes = bytes to now
        cams.end()

        val ph = s.obj("phone")
        phone.begin()
        if (ph == null || ph.flag("available") != true) phone.set("Phone", "Has not reported yet", Level.OFF)
        else {
            val pst = ph.flag("stale") == true
            val pl = { l: Level -> if (pst && l == Level.OK) Level.WARN else l }
            phone.set("Data", if (pst) "Last known, ${Rows.ago(ph.num("age_s"))}" else "Live", if (pst) Level.WARN else Level.OK)
            ph.num("battery_pct")?.let { phone.set("Battery", Fmt.pct(it) + if (ph.flag("charging") == true) " · charging" else "", pl(if (it < 15 && ph.flag("charging") != true) Level.BAD else if (it < 30) Level.WARN else Level.OK)) }
            ph.num("battery_temp_c")?.let { phone.set("Battery temperature", "%.1f °C".format(it), pl(if (it >= 45) Level.BAD else if (it >= 40) Level.WARN else Level.OK)) }
            phone.set("Location permission", ph.str("location_permission") ?: "unknown", when (ph.str("location_permission")) { "precise" -> pl(Level.OK); "approximate" -> Level.WARN; else -> Level.BAD })
            phone.set("GPS", (if (ph.flag("gps_enabled") == true) "On" else "Off") + " · ${ph.str("gps_state") ?: "unknown"}", pl(if (ph.flag("gps_enabled") == true && ph.str("gps_state") == "ok") Level.OK else Level.WARN))
            phone.set("Motion sensors", "Accelerometer ${if (ph.flag("accelerometer") == true) "yes" else "no"} · gyroscope ${if (ph.flag("gyroscope") == true) "yes" else "no"}",
                pl(if (ph.flag("accelerometer") == true) Level.OK else Level.BAD))
            phone.set("ARCore", ph.str("arcore") ?: "unknown", Level.OFF)
            phone.set("Internet", if (ph.flag("internet") == true) "Connected" else "No internet", pl(if (ph.flag("internet") == true) Level.OK else Level.BAD))
            phone.set("Tailscale", if (ph.flag("tailscale") == true) "Connected" else "Not connected", pl(if (ph.flag("tailscale") == true) Level.OK else Level.BAD))
            phone.set("Background tracking", if (ph.flag("tracking_service") == true) "Running" else "Stopped", pl(if (ph.flag("tracking_service") == true) Level.OK else Level.BAD))
            ph.num("queued")?.let { if (it > 0) phone.set("Waiting to send", "${it.toInt()}", Level.WARN) }
            ph.str("restart_note")?.let { phone.set("Last restart", it, Level.WARN) }
        }
        phone.end()
        Rows.connections(conn, s, link)
    }
}
