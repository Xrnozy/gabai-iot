package com.example.gabai.safety

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
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.example.gabai.FamilyActivity
import com.example.gabai.R
import com.example.gabai.services.SettingsStore
import org.json.JSONObject
import kotlin.math.sqrt

/**
 * Foreground service for the blind user's phone: GPS every 5 s while moving and 10 s while still, fall detection on the
 * accelerometer, and the uplink that sends it all to the relay. One thread, one timer, batched sensor delivery: it wakes
 * the CPU as little as the job allows.
 */
class TrackingService : Service(), LocationListener, SensorEventListener {

    private lateinit var thread: HandlerThread
    private lateinit var h: Handler
    private lateinit var lm: LocationManager
    private lateinit var sm: SensorManager

    private val detector = FallDetector { c -> Emergency.startFall(this, c) }
    private var gyroDps = Float.NaN
    private var headingDeg: Float? = null
    private var requestedMs = 0L
    private var lastGpsAt = 0L
    private var lastMovingAt = System.currentTimeMillis()
    private val rot = FloatArray(9)
    private val remapped = FloatArray(9)
    private val orient = FloatArray(3)

    private val loop = object : Runnable {
        override fun run() {
            updateGpsState()
            Uplink.flush(this@TrackingService)
            Uplink.persist()
            adaptLocationRate()
            h.postDelayed(this, uplinkIntervalMs())
        }
    }

    override fun onCreate() {
        super.onCreate()
        thread = HandlerThread("tracking").also { it.start() }
        h = Handler(thread.looper)
        lm = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        sm = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        Uplink.init(this)
        Uplink.setTelemetrySource { PhoneTelemetry.collect(this) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            SettingsStore.setTrackingEnabled(this, false)
            stopSelf()
            return START_NOT_STICKY
        }
        startInForeground()
        LiveState.trackingRunning = true
        h.removeCallbacksAndMessages(null)
        h.post { startLocation(); registerSensors(); loop.run() }
        return START_STICKY
    }

    private fun startInForeground() {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Location sharing", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(this, 0, Intent(this, FamilyActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val stop = PendingIntent.getService(this, 1, Intent(this, TrackingService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE)
        val n: Notification = NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_navigation)
            .setContentTitle("Sharing location with your caregivers")
            .setContentText("Fall detection is on.")
            .setOngoing(true)
            .setContentIntent(open)
            .addAction(0, "Stop sharing", stop)
            .build()
        val type = if (Build.VERSION.SDK_INT >= 29) ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION else 0
        ServiceCompat.startForeground(this, NOTIF_ID, n, type)
    }

    // ---------- GPS ----------
    private fun hasLocationPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED

    @android.annotation.SuppressLint("MissingPermission")
    private fun startLocation() {
        lm.removeUpdates(this)
        if (!hasLocationPermission()) { LiveState.gpsState = "denied"; return }
        val every = if (stationary()) STILL_MS else MOVING_MS
        requestedMs = every
        runCatching { lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, every, 0f, this, thread.looper) }
        // The network provider is nearly free and covers the seconds before a GPS fix and indoor gaps.
        runCatching { lm.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, NETWORK_MS, 0f, this, thread.looper) }
    }

    private fun stationary() = System.currentTimeMillis() - lastMovingAt > STILL_AFTER_MS

    private fun adaptLocationRate() {
        val want = if (stationary()) STILL_MS else MOVING_MS
        if (want != requestedMs) startLocation()
    }

    // A caregiver is looking right now: report every 2 s. Nobody watching: 5 s moving, 10 s still.
    private fun uplinkIntervalMs() = if (LiveState.watchers > 0) WATCHED_MS else if (stationary()) STILL_MS else MOVING_MS

    private fun updateGpsState() {
        LiveState.gpsState = when {
            !hasLocationPermission() -> "denied"
            !runCatching { lm.isProviderEnabled(LocationManager.GPS_PROVIDER) }.getOrDefault(false) -> "disabled"
            System.currentTimeMillis() - lastGpsAt > NO_FIX_MS -> "searching"
            else -> LiveState.gpsState
        }
    }

    override fun onLocationChanged(loc: Location) {
        val now = System.currentTimeMillis()
        val isGps = loc.provider == LocationManager.GPS_PROVIDER
        if (isGps) lastGpsAt = now
        else if (now - lastGpsAt < NETWORK_OK_AFTER_MS) return           // GPS is flowing: ignore the coarser fix
        if (loc.hasSpeed() && loc.speed > MOVING_SPEED) lastMovingAt = now
        LiveState.gpsState = if (loc.hasAccuracy() && loc.accuracy > WEAK_ACC_M) "weak" else "ok"
        val j = JSONObject()
            .put("id", "${loc.time}-${loc.provider}")
            .put("t", loc.time / 1000.0)
            .put("lat", loc.latitude).put("lon", loc.longitude)
            .put("provider", loc.provider)
        if (loc.hasAccuracy()) j.put("acc", loc.accuracy.toDouble())
        if (loc.hasSpeed()) j.put("speed", loc.speed.toDouble())
        if (loc.hasBearing() && (loc.hasSpeed() && loc.speed > MOVING_SPEED)) j.put("bearing", loc.bearing.toDouble())
        headingDeg?.let { j.put("heading", it.toDouble()) }
        Uplink.addLocation(j)
    }

    @Deprecated("required below API 30")
    override fun onStatusChanged(provider: String?, status: Int, extras: android.os.Bundle?) {}
    override fun onProviderEnabled(provider: String) { updateGpsState() }
    override fun onProviderDisabled(provider: String) { updateGpsState() }

    // ---------- sensors ----------
    private fun registerSensors() {
        sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let { sm.registerListener(this, it, ACCEL_PERIOD_US, BATCH_US, h) }
        sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { sm.registerListener(this, it, ACCEL_PERIOD_US, BATCH_US, h) }
        sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL, h) }
    }

    override fun onSensorChanged(e: SensorEvent) {
        when (e.sensor.type) {
            Sensor.TYPE_ACCELEROMETER ->
                detector.onSample(e.timestamp / 1_000_000L, e.values[0], e.values[1], e.values[2], gyroDps)
            Sensor.TYPE_GYROSCOPE -> {
                val x = e.values[0]; val y = e.values[1]; val z = e.values[2]
                gyroDps = Math.toDegrees(sqrt((x * x + y * y + z * z).toDouble())).toFloat()
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                SensorManager.getRotationMatrixFromVector(rot, e.values)
                // Camera-forward azimuth: right for a phone held or worn upright, as it is during navigation.
                SensorManager.remapCoordinateSystem(rot, SensorManager.AXIS_X, SensorManager.AXIS_Z, remapped)
                SensorManager.getOrientation(remapped, orient)
                headingDeg = ((Math.toDegrees(orient[0].toDouble()) + 360) % 360).toFloat()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        h.removeCallbacksAndMessages(null)
        runCatching { lm.removeUpdates(this) }
        sm.unregisterListener(this)
        LiveState.trackingRunning = false
        LiveState.gpsState = "off"
        Uplink.persist()
        Uplink.flushAsync(applicationContext)
        thread.quitSafely()
        super.onDestroy()
    }

    companion object {
        const val ACTION_STOP = "com.example.gabai.TRACKING_STOP"
        private const val CHANNEL = "gabai_tracking"
        private const val NOTIF_ID = 4400
        private const val MOVING_MS = 5_000L
        private const val STILL_MS = 10_000L
        private const val WATCHED_MS = 2_000L
        private const val NETWORK_MS = 15_000L
        private const val STILL_AFTER_MS = 60_000L
        private const val NO_FIX_MS = 20_000L
        private const val NETWORK_OK_AFTER_MS = 15_000L
        private const val MOVING_SPEED = 0.5f
        private const val WEAK_ACC_M = 50f
        private const val ACCEL_PERIOD_US = 40_000
        private const val BATCH_US = 1_000_000

        fun start(ctx: Context) {
            SettingsStore.setTrackingEnabled(ctx, true)
            ContextCompat.startForegroundService(ctx, Intent(ctx, TrackingService::class.java))
        }

        fun stop(ctx: Context) {
            SettingsStore.setTrackingEnabled(ctx, false)
            ctx.stopService(Intent(ctx, TrackingService::class.java))
        }
    }
}
