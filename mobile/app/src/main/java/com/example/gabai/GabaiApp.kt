package com.example.gabai

import android.app.Activity
import android.app.Application
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import com.serenegiant.utils.UVCUtils
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Tracks whether any activity is in the foreground so the background voice
 * listener (VoiceListenerService) knows when to hold and release the mic —
 * activities own the mic while visible; the service owns it when the app is
 * backgrounded or the screen is off.
 *
 * The background→foreground transition is debounced: moving between screens
 * briefly dips to zero resumed activities, and we don't want the service's
 * AudioRecord to flap open/closed on every screen change.
 */
class GabaiApp : Application() {

    companion object {
        @Volatile
        var isInForeground: Boolean = false
            private set

        private val listeners = CopyOnWriteArrayList<(Boolean) -> Unit>()
        private val handler = Handler(Looper.getMainLooper())
        private var pendingChange: Runnable? = null
        private const val TRANSITION_DEBOUNCE_MS = 600L

        fun addForegroundListener(listener: (Boolean) -> Unit) {
            listeners.add(listener)
        }

        fun removeForegroundListener(listener: (Boolean) -> Unit) {
            listeners.remove(listener)
        }

        private fun setForeground(foreground: Boolean) {
            if (isInForeground == foreground) return
            pendingChange?.let { handler.removeCallbacks(it) }
            if (foreground) {
                // Foreground wins immediately — pause the service mic fast.
                isInForeground = true
                listeners.forEach { it(true) }
            } else {
                // Confirm it's a real background, not an activity transition.
                val r = Runnable {
                    if (isInForeground) return@Runnable
                    pendingChange = null
                    isInForeground = false
                    listeners.forEach { it(false) }
                }
                pendingChange = r
                handler.postDelayed(r, TRANSITION_DEBOUNCE_MS)
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        UVCUtils.init(this)
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            // Reported to caregivers by the next uplink so a crash loop is visible, not silent.
            com.example.gabai.services.SettingsStore.setCrashNote(this, "App restarted after ${e.javaClass.simpleName} in ${t.name}")
            previous?.uncaughtException(t, e)
        }
        // Starts by itself once this phone is paired; caregiver phones and unpaired phones have nothing to share.
        val store = com.example.gabai.services.SettingsStore
        if (store.getAppRole(this) == store.APP_USER && store.isTrackingEnabled(this) &&
            store.getRelayToken(this, store.ROLE_PHONE).isNotBlank()) {
            runCatching { com.example.gabai.safety.TrackingService.start(this) }
        }
        com.example.gabai.services.AutoConnect.start(this)
        var resumedCount = 0
        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            override fun onActivityResumed(activity: Activity) {
                resumedCount++
                setForeground(true)
            }

            override fun onActivityPaused(activity: Activity) {
                resumedCount--
                if (resumedCount <= 0) {
                    resumedCount = 0
                    setForeground(false)
                }
            }

            override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) {}
            override fun onActivityStarted(activity: Activity) {}
            override fun onActivityStopped(activity: Activity) {}
            override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) {}
            override fun onActivityDestroyed(activity: Activity) {}
        })
    }
}
