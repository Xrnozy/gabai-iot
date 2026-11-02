package com.example.gabai.safety

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper
import androidx.core.app.NotificationCompat
import com.example.gabai.GabaiApp
import com.example.gabai.R
import com.example.gabai.services.Haptics
import com.example.gabai.services.TtsService
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Fall and SOS alerts. A possible fall or an SOS press starts a countdown the user can cancel; the caregiver sees a
 * "pending" alert right away and a "confirmed" one when the countdown ends, so a dead phone mid-countdown still escalates
 * (the relay confirms stale pending alerts on its own).
 */
object Emergency {
    enum class Phase { IDLE, COUNTDOWN, SENT }

    const val FALL_COUNTDOWN_S = 20
    const val SOS_COUNTDOWN_S = 5
    const val ACTION_CANCEL = "com.example.gabai.EMERGENCY_CANCEL"
    private const val CHANNEL = "gabai_emergency"
    private const val NOTIF_ID = 4401

    @Volatile var phase = Phase.IDLE
        private set
    @Volatile var kind = ""
        private set
    @Volatile var deadlineMs = 0L
        private set
    @Volatile var confidence: Float? = null
        private set
    @Volatile var ackedBy: String? = null
        private set

    private var id = ""
    private var evidence: Map<String, Any?> = emptyMap()
    private var tts: TtsService? = null
    private val main = Handler(Looper.getMainLooper())
    private val listeners = CopyOnWriteArrayList<() -> Unit>()
    private var spokenAck = false

    private val tick = object : Runnable {
        override fun run() {
            if (phase != Phase.COUNTDOWN) return
            val left = secondsLeft()
            if (left <= 0) { confirm(appCtx); return }
            if (left == 10 || left == 5) tts?.speak("$left seconds. Tap I'm okay to cancel.", interrupt = true)
            changed()
            main.postDelayed(this, 1000)
        }
    }
    private lateinit var appCtx: Context

    fun listen(l: () -> Unit) { listeners.add(l) }
    fun unlisten(l: () -> Unit) { listeners.remove(l) }
    private fun changed() = main.post { listeners.forEach { it() } }

    fun secondsLeft(): Int = ((deadlineMs - System.currentTimeMillis() + 999) / 1000).toInt().coerceAtLeast(0)

    fun startFall(ctx: Context, c: FallCandidate) {
        main.post { start(ctx, "fall", FALL_COUNTDOWN_S, c.confidence, c.evidence(),
            "Possible fall detected. Tap I'm okay to cancel. Alerting your caregivers in $FALL_COUNTDOWN_S seconds.") }
    }

    fun startSos(ctx: Context) {
        main.post { start(ctx, "sos", SOS_COUNTDOWN_S, null, emptyMap(),
            "Emergency. Alerting your caregivers in $SOS_COUNTDOWN_S seconds. Tap I'm okay to cancel.") }
    }

    private fun start(ctx: Context, k: String, seconds: Int, conf: Float?, ev: Map<String, Any?>, prompt: String) {
        if (phase != Phase.IDLE) return
        appCtx = ctx.applicationContext
        Uplink.init(appCtx)
        id = "$k-${UUID.randomUUID().toString().take(8)}"
        kind = k
        confidence = conf
        evidence = ev
        ackedBy = null
        spokenAck = false
        deadlineMs = System.currentTimeMillis() + seconds * 1000L
        phase = Phase.COUNTDOWN
        if (tts == null) tts = TtsService(appCtx)
        tts?.speak(prompt, interrupt = true)
        Haptics(appCtx).heavy()
        post("pending")
        notifyUser(prompt)
        if (GabaiApp.isInForeground) openScreen()
        LiveState.listen(ackWatcher)
        main.postDelayed(tick, 1000)
        changed()
    }

    fun openScreen() {
        appCtx.startActivity(Intent(appCtx, EmergencyActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP))
    }

    fun cancel(ctx: Context) {
        main.post {
            if (phase == Phase.IDLE) return@post
            main.removeCallbacks(tick)
            post("cancelled")
            tts?.speak("Alert cancelled.", interrupt = true)
            finish()
        }
    }

    fun sendNow(ctx: Context) {
        main.post { if (phase == Phase.COUNTDOWN) confirm(ctx.applicationContext) }
    }

    private fun confirm(ctx: Context) {
        main.removeCallbacks(tick)
        phase = Phase.SENT
        post("confirmed")
        tts?.speak("Alert sent to your caregivers.", interrupt = true)
        notifyUser("Alert sent to your caregivers.")
        changed()
    }

    /** Closes the emergency screen state once the user has seen the outcome. */
    fun dismiss() = main.post { finish() }

    private fun finish() {
        phase = Phase.IDLE
        LiveState.unlisten(ackWatcher)
        (appCtx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).cancel(NOTIF_ID)
        changed()
    }

    private val ackWatcher: () -> Unit = {
        val e = LiveState.latestEmergency
        if (e != null && e.optString("id") == id && e.optString("status") == "acknowledged" && !spokenAck) {
            spokenAck = true
            ackedBy = e.optString("ack_by").ifBlank { "A caregiver" }
            main.post { tts?.speak("${ackedBy} has seen your alert and is responding.", interrupt = true); changed() }
        }
    }

    private fun post(status: String) {
        val fix = Uplink.lastFix
        val ev = JSONObject()
            .put("id", id).put("kind", kind).put("status", status).put("t", System.currentTimeMillis() / 1000.0)
            .put("evidence", JSONObject(evidence.filterValues { it != null }))
        confidence?.let { ev.put("confidence", it.toDouble()) }
        if (fix != null) {
            ev.put("lat", fix.optDouble("lat")).put("lon", fix.optDouble("lon"))
            ev.getJSONObject("evidence").put("fix_age_s", System.currentTimeMillis() / 1000.0 - fix.optDouble("t"))
                .put("fix_accuracy_m", fix.optDouble("acc"))
        }
        Uplink.addEvent(ev)
        Uplink.flushAsync(appCtx)
    }

    private fun notifyUser(text: String) {
        val nm = appCtx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Emergency alerts", NotificationManager.IMPORTANCE_HIGH))
        val open = PendingIntent.getActivity(appCtx, 0, Intent(appCtx, EmergencyActivity::class.java), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val cancel = PendingIntent.getBroadcast(appCtx, 1, Intent(ACTION_CANCEL).setPackage(appCtx.packageName), PendingIntent.FLAG_IMMUTABLE)
        nm.notify(NOTIF_ID, NotificationCompat.Builder(appCtx, CHANNEL)
            .setSmallIcon(R.drawable.ic_obstacle)
            .setContentTitle(if (kind == "fall") "Possible fall" else "Emergency")
            .setContentText(text)
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setOngoing(phase == Phase.COUNTDOWN)
            .setFullScreenIntent(open, true)
            .setContentIntent(open)
            .addAction(0, "I'm okay — cancel", cancel)
            .build())
    }
}

/** Lets the "I'm okay" notification action cancel without opening the app. */
class EmergencyReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Emergency.ACTION_CANCEL) Emergency.cancel(context)
    }
}
