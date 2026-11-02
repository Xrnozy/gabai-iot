package com.example.gabai.caregiver

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.example.gabai.GabaiApp
import com.example.gabai.R
import com.example.gabai.services.SettingsStore

/**
 * Keeps a slow (8 s) watch on the relay so a caregiver whose app is closed still hears about emergencies, route deviations
 * and lost connections. While the dashboard is open it stays quiet: the dashboard shows the same alerts itself.
 */
class CaregiverAlertService : Service() {

    private val seen = HashMap<String, String>()
    private val states = HashMap<String, String>()
    private val onHub: () -> Unit = { check() }

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(NotificationChannel(WATCH, "Watching for alerts", NotificationManager.IMPORTANCE_MIN))
        nm.createNotificationChannel(NotificationChannel(ALERT, "Caregiver alerts", NotificationManager.IMPORTANCE_HIGH))
        val open = PendingIntent.getActivity(this, 0, Intent(this, CaregiverActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val n = NotificationCompat.Builder(this, WATCH).setSmallIcon(R.drawable.ic_family)
            .setContentTitle("Watching for alerts").setOngoing(true).setContentIntent(open).build()
        val type = if (Build.VERSION.SDK_INT >= 29) ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC else 0
        ServiceCompat.startForeground(this, 4410, n, type)
        StateHub.listen(onHub)
        StateHub.acquire(this, "alerts")
    }

    private fun check() {
        val s = StateHub.snapshot ?: return
        if (GabaiApp.isInForeground) return
        val active = s.obj("safety").obj("active")
        if (active != null && active.str("status") != "acknowledged") {
            val key = active.str("id") + active.str("status")
            if (seen["emergency"] != key) {
                seen["emergency"] = key
                val what = if (active.str("kind") == "sos") "SOS pressed" else "Possible fall"
                alert(1, "$what · ${active.str("status")}", "Open the dashboard to see where they are and acknowledge.")
            }
        }
        val recent = s.obj("safety")?.optJSONArray("recent")
        if (recent != null) for (i in 0 until recent.length()) {
            val e = recent.getJSONObject(i)
            val kind = e.str("kind")
            if ((kind == "deviation" || kind == "inactivity") && seen[e.str("id") ?: ""] == null) {
                seen[e.str("id") ?: ""] = "1"
                alert(2, if (kind == "deviation") "Left the saved route" else "Not moving for a long time", "Open the dashboard for their location.")
            }
        }
        for (dev in listOf("phone", "pi")) {
            val st = s.obj("connections").obj(dev)?.str("state") ?: continue
            if (states[dev] == "online" && st == "offline") {
                alert(if (dev == "phone") 3 else 4, if (dev == "phone") "Phone went offline" else "Raspberry Pi went offline",
                    "Last known information is on the dashboard, marked as not live.")
            }
            states[dev] = st
        }
    }

    private fun alert(id: Int, title: String, text: String) {
        val open = PendingIntent.getActivity(this, 0, Intent(this, CaregiverActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).notify(4420 + id,
            NotificationCompat.Builder(this, ALERT).setSmallIcon(R.drawable.ic_sos).setContentTitle(title).setContentText(text)
                .setPriority(NotificationCompat.PRIORITY_MAX).setCategory(NotificationCompat.CATEGORY_ALARM)
                .setAutoCancel(true).setContentIntent(open).setFullScreenIntent(open, id == 1).build())
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) { stopSelf(); return START_NOT_STICKY }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        StateHub.unlisten(onHub)
        StateHub.release("alerts")
        super.onDestroy()
    }

    companion object {
        private const val ACTION_STOP = "com.example.gabai.ALERTS_STOP"
        private const val WATCH = "gabai_watch"
        private const val ALERT = "gabai_cg_alert"

        fun start(ctx: Context) = ContextCompat.startForegroundService(ctx, Intent(ctx, CaregiverAlertService::class.java))
        fun stop(ctx: Context) { ctx.stopService(Intent(ctx, CaregiverAlertService::class.java)) }
    }
}
