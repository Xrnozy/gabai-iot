package com.example.gabai.safety

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorManager
import android.location.LocationManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import androidx.core.content.ContextCompat
import com.example.gabai.services.SettingsStore
import org.json.JSONObject
import java.net.Inet4Address
import java.net.NetworkInterface

/** The phone-side health report shown on the caregiver Device tab. Every field is a measurement, never a guess. */
object PhoneTelemetry {

    fun collect(ctx: Context): JSONObject {
        val app = ctx.applicationContext
        val battery = app.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val temp = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        val status = battery?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        val sm = app.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val lm = app.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val fine = granted(app, Manifest.permission.ACCESS_FINE_LOCATION)
        val coarse = granted(app, Manifest.permission.ACCESS_COARSE_LOCATION)

        return JSONObject()
            .put("t", System.currentTimeMillis() / 1000.0)
            .put("battery_pct", if (level >= 0) level * 100 / scale else JSONObject.NULL)
            .put("charging", status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL)
            .put("battery_temp_c", if (temp != Int.MIN_VALUE) temp / 10.0 else JSONObject.NULL)
            .put("location_permission", if (fine) "precise" else if (coarse) "approximate" else "denied")
            .put("gps_enabled", runCatching { lm.isProviderEnabled(LocationManager.GPS_PROVIDER) }.getOrDefault(false))
            .put("gps_state", LiveState.gpsState)
            .put("accelerometer", sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null)
            .put("gyroscope", sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null)
            .put("arcore", LiveState.arState)
            .put("internet", hasInternet(app))
            .put("tailscale", tailscaleUp())
            .put("tracking_service", LiveState.trackingRunning)
            .put("relay_rtt_ms", LiveState.relayRttMs)
            .put("queued", Uplink.pending())
            .put("app_version", runCatching { app.packageManager.getPackageInfo(app.packageName, 0).versionName }.getOrNull())
            .also { j -> SettingsStore.takeCrashNote(app)?.let { j.put("restart_note", it) } }
    }

    private fun granted(c: Context, p: String) = ContextCompat.checkSelfPermission(c, p) == PackageManager.PERMISSION_GRANTED

    private fun hasInternet(c: Context): Boolean {
        val cm = c.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
            caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }

    /** Tailscale hands every device an address in 100.64.0.0/10; an interface holding one means the tailnet is up. */
    private fun tailscaleUp(): Boolean = runCatching {
        NetworkInterface.getNetworkInterfaces().asSequence().filter { it.isUp }.flatMap { it.inetAddresses.asSequence() }
            .filterIsInstance<Inet4Address>().any { val b = it.address; (b[0].toInt() and 0xFF) == 100 && (b[1].toInt() and 0xC0) == 64 }
    }.getOrDefault(false)
}
