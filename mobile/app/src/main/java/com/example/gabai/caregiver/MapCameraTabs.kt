package com.example.gabai.caregiver

import android.content.Context
import android.graphics.Color
import android.graphics.Paint
import android.view.MotionEvent
import android.view.View
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Toast
import com.example.gabai.R
import com.example.gabai.safety.RelayApi
import com.example.gabai.services.SettingsStore
import com.example.gabai.ui.Level
import com.example.gabai.ui.LiveCard
import com.example.gabai.ui.Ui
import org.json.JSONObject
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polygon
import org.osmdroid.views.overlay.Polyline

class MapTab(private val ctx: Context) : CaregiverTab {
    override val id = "map"
    override val label = "Map"
    override val icon = R.drawable.ic_tab_map
    override val purpose = "Live location on a map, recent route history and the saved route."

    private val api = RelayApi(ctx, SettingsStore.ROLE_CAREGIVER)
    private val map: MapView
    private val marker: Marker
    private val accuracy: Polygon
    private val trail = Polyline()
    private val route = Polyline()
    private val info = LiveCard(ctx, "Where they are")
    private var hours = 1
    private var centred = false
    private var lastHistoryAt = 0L
    private var shown = false
    private var lastPoints: List<GeoPoint> = emptyList()

    init {
        Configuration.getInstance().apply {
            load(ctx, ctx.getSharedPreferences("osmdroid", Context.MODE_PRIVATE))
            userAgentValue = ctx.packageName
        }
        map = MapView(ctx).apply {
            setTileSource(TileSourceFactory.MAPNIK)
            setMultiTouchControls(true)
            controller.setZoom(17.0)
            // Inside a ScrollView the map must keep its own pan and zoom gestures.
            setOnTouchListener { v, e -> if (e.action == MotionEvent.ACTION_DOWN) v.parent.requestDisallowInterceptTouchEvent(true); false }
            contentDescription = "Map showing the user's location"
        }
        accuracy = Polygon(map).apply {
            fillPaint.color = Color.argb(40, 40, 100, 220)
            outlinePaint.color = Color.argb(160, 40, 100, 220)
            outlinePaint.strokeWidth = 2f
        }
        trail.outlinePaint.apply { color = Color.rgb(176, 82, 47); strokeWidth = 8f }
        route.outlinePaint.apply { color = Color.rgb(58, 125, 68); strokeWidth = 8f; pathEffect = android.graphics.DashPathEffect(floatArrayOf(24f, 16f), 0f); style = Paint.Style.STROKE }
        marker = Marker(map).apply { setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM) }
        map.overlays.addAll(listOf(route, trail, accuracy, marker))
        marker.setVisible(false)
    }

    private fun rangeButton(h: Int, label: String) =
        Ui.button(ctx, label, Ui.Kind.SECONDARY, "Show the last $label of movement. Button.") { hours = h; loadHistory(true) }

    override val view: View = ScrollView(ctx).apply {
        setBackgroundColor(Ui.color(ctx, R.color.canvas)); isFillViewport = true
        addView(Ui.column(ctx, 16).apply {
            addView(map, Ui.lp(h = Ui.dp(ctx, 340), ctx = ctx))
            addView(Ui.row(ctx).apply {
                addView(rangeButton(1, "1 hour"), LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
                addView(rangeButton(6, "6 hours"), LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).also { it.marginStart = Ui.dp(ctx, 8) })
                addView(rangeButton(24, "24 hours"), LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).also { it.marginStart = Ui.dp(ctx, 8) })
            }, Ui.lp(top = 12, ctx = ctx))
            addView(Ui.button(ctx, "Centre on their location", Ui.Kind.PRIMARY, "Centre the map on the user's latest location. Button.") { lastPoints.firstOrNull()?.let { map.controller.animateTo(it) } },
                Ui.lp(top = 12, ctx = ctx))
            addView(info.view, Ui.lp(top = 12, ctx = ctx))
            info.add(Ui.button(ctx, "Use the last hour as the saved route", Ui.Kind.SECONDARY, "Save the last hour of movement as the expected route. Leaving it raises an alert. Button.") { saveRoute() })
            info.add(Ui.button(ctx, "Clear saved route", Ui.Kind.SECONDARY, "Remove the saved route. Button.") { clearRoute() })
        })
    }

    override fun onShow() { shown = true; map.onResume(); loadHistory(true) }
    override fun onHide() { shown = false; map.onPause() }

    override fun update(s: JSONObject?, link: HubLink) {
        val loc = s.obj("location")
        Rows.location(info, s)
        if (loc?.flag("available") == true) {
            val p = GeoPoint(loc.num("lat")!!, loc.num("lon")!!)
            val stale = loc.flag("stale") == true
            lastPoints = listOf(p)
            marker.position = p
            marker.title = (if (stale) "Last known location · " else "Current location · ") + Rows.ago(loc.num("age_s"))
            marker.alpha = if (stale) 0.45f else 1f
            marker.setVisible(true)
            val acc = loc.num("accuracy_m") ?: 0.0
            accuracy.points = if (acc > 0) Polygon.pointsAsCircle(p, acc) else emptyList()
            if (!centred) { map.controller.setCenter(p); centred = true }
        }
        val r = s?.optJSONObject("route")?.optJSONArray("points")
        route.setPoints(if (r == null) emptyList() else (0 until r.length()).map { GeoPoint(r.getJSONArray(it).getDouble(0), r.getJSONArray(it).getDouble(1)) })
        if (shown && System.currentTimeMillis() - lastHistoryAt > 30_000) loadHistory(false)
        map.invalidate()
    }

    private fun loadHistory(force: Boolean) {
        lastHistoryAt = System.currentTimeMillis()
        Thread {
            val since = System.currentTimeMillis() / 1000.0 - hours * 3600
            val pts = api.history(since).body?.optJSONArray("points") ?: return@Thread
            val geo = (0 until pts.length()).map { pts.getJSONObject(it) }.filter { (it.num("acc") ?: 0.0) <= 100 }.map { GeoPoint(it.getDouble("lat"), it.getDouble("lon")) }
            map.post { trail.setPoints(geo); map.invalidate(); if (force && geo.size > 1) map.zoomToBoundingBox(org.osmdroid.util.BoundingBox.fromGeoPoints(geo).increaseByScale(1.3f), true) }
        }.start()
    }

    private fun saveRoute() = Thread {
        val pts = trailPoints()
        val msg = if (pts.size < 2) "Not enough movement in the last hour" else
            if (api.setRoute("Last hour", pts.map { it.latitude to it.longitude }.thinned()).ok) "Route saved" else "Could not save the route"
        map.post { Toast.makeText(ctx, msg, Toast.LENGTH_SHORT).show() }
    }.start()

    private fun clearRoute() = Thread {
        val ok = api.clearRoute().ok
        map.post { Toast.makeText(ctx, if (ok) "Route cleared" else "Could not clear the route", Toast.LENGTH_SHORT).show() }
    }.start()

    private fun trailPoints(): List<GeoPoint> = trail.actualPoints.toList()

    private fun List<Pair<Double, Double>>.thinned(max: Int = 300): List<Pair<Double, Double>> =
        if (size <= max) this else filterIndexed { i, _ -> i % ((size + max - 1) / max) == 0 }
}

class CameraTab(private val ctx: Context) : CaregiverTab {
    override val id = "camera"
    override val label = "Camera"
    override val icon = R.drawable.ic_tab_camera
    override val purpose = "Live left and right camera feeds from the Raspberry Pi."

    private val status = LiveCard(ctx, "What the Raspberry Pi sees")
    private val leftCam = MjpegView(ctx, "left")
    private val rightCam = MjpegView(ctx, "right")

    override val view: View = ScrollView(ctx).apply {
        setBackgroundColor(Ui.color(ctx, R.color.canvas)); isFillViewport = true
        addView(Ui.column(ctx, 16).apply {
            addView(status.view, Ui.lp(ctx = ctx))
            addView(Ui.heading(ctx, "Left camera"), Ui.lp(top = 16, ctx = ctx))
            addView(leftCam, Ui.lp(top = 8, ctx = ctx))
            addView(Ui.heading(ctx, "Right camera"), Ui.lp(top = 16, ctx = ctx))
            addView(rightCam, Ui.lp(top = 8, ctx = ctx))
            addView(Ui.caption(ctx, "These are the Raspberry Pi's own cameras, relayed over Tailscale. The phone camera is never used here."), Ui.lp(top = 12, ctx = ctx))
        })
    }

    override fun onShow() { leftCam.start(); rightCam.start() }
    override fun onHide() { leftCam.stop(); rightCam.stop() }

    override fun update(s: JSONObject?, link: HubLink) {
        val pi = s.obj("pi")
        val state = s.obj("connections").obj("pi")
        val online = state?.str("state") == "online"
        status.begin()
        status.set("Raspberry Pi", if (online) "Connected" else "Offline" + (state.num("offline_since")?.let { " since ${Fmt.clock(it)}" } ?: ""), if (online) Level.OK else Level.BAD)
        pi.obj("link").num("fps")?.let { if (online) status.set("Stream", "%.1f frames per second".format(it)) }
        if (!online) status.set("Showing", "Last frame the server received", Level.WARN)
        status.end()
    }
}
