package com.example.gabai.safety

import android.content.Context
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.os.Handler
import android.os.HandlerThread
import com.example.gabai.services.TtsService
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteOrder

/**
 * Offline navigation fallback. Used only while the server or the Raspberry Pi is unreachable: ARCore motion tracking plus, where
 * the phone supports it, the ARCore Depth API give coarse "something is close on the left / ahead / right" warnings.
 * It is NOT a replacement for the Pi's stereo pipeline: it has no route guidance, no object classification, and its depth is
 * unreliable closer than ~0.5 m and farther than ~5 m. Every announcement says so, and tracking trouble is spoken, not hidden.
 *
 * Assumes the phone is carried upright with the rear camera facing forward.
 */
class ArFallback(private val ctx: Context, private val tts: TtsService) {

    private val thread = HandlerThread("ar-fallback").also { it.start() }
    private val h = Handler(thread.looper)
    @Volatile private var running = false
    private var session: Session? = null
    private var egl: Array<Any?> = arrayOfNulls(3)
    private var depthOk = false
    private var lastWarn = 0L
    private var lastTrackingWarn = 0L
    private var lastReport = 0L
    private var announcedStart = false

    /** True when ARCore is installed and supported; callers skip the fallback otherwise and say so. */
    fun available(): Boolean = ArCoreApk.getInstance().checkAvailability(ctx).isSupported

    fun start(): Boolean {
        if (running) return true
        if (!available()) { LiveState.arState = "unavailable"; LiveState.changed(); return false }
        running = true
        announcedStart = false
        h.post { open() }
        return true
    }

    fun stop() {
        if (!running) return
        running = false
        h.post { close() }
    }

    private fun open() {
        try {
            val disp = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            val ver = IntArray(2)
            EGL14.eglInitialize(disp, ver, 0, ver, 1)
            val cfgs = arrayOfNulls<EGLConfig>(1)
            val n = IntArray(1)
            EGL14.eglChooseConfig(disp, intArrayOf(EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT, EGL14.EGL_NONE), 0, cfgs, 0, 1, n, 0)
            val ectx = EGL14.eglCreateContext(disp, cfgs[0], EGL14.EGL_NO_CONTEXT, intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE), 0)
            val surf = EGL14.eglCreatePbufferSurface(disp, cfgs[0], intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0)
            EGL14.eglMakeCurrent(disp, surf, surf, ectx)
            egl = arrayOf(disp, ectx, surf)

            val tex = IntArray(1)
            GLES20.glGenTextures(1, tex, 0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, tex[0])

            val s = Session(ctx)
            s.setCameraTextureName(tex[0])
            val cfg = Config(s).apply {
                updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                depthMode = if (s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) Config.DepthMode.AUTOMATIC else Config.DepthMode.DISABLED
            }
            depthOk = cfg.depthMode == Config.DepthMode.AUTOMATIC
            s.configure(cfg)
            s.resume()
            session = s
            LiveState.arState = "limited"
            LiveState.changed()
            tts.speak(
                if (depthOk) "Connection lost. Backup navigation is on. I can warn about close obstacles, but I can not give directions. Walk slowly."
                else "Connection lost. Backup navigation is on, but this phone can not sense obstacles. I can not give directions. Please stop and wait for the connection.",
                interrupt = true,
            )
            announcedStart = true
            h.post(frameLoop)
        } catch (e: Exception) {
            LiveState.arState = "unavailable"
            LiveState.changed()
            running = false
            tts.speak("Backup navigation could not start. Please stop and wait for the connection.", interrupt = true)
            close()
        }
    }

    private val frameLoop = object : Runnable {
        override fun run() {
            if (!running) return
            val s = session ?: return
            try {
                val frame = s.update()
                val tracking = frame.camera.trackingState == TrackingState.TRACKING
                LiveState.arState = if (tracking) "tracking" else "limited"
                val now = System.currentTimeMillis()
                var zones: DoubleArray? = null
                if (tracking && depthOk) zones = nearestPerZone(frame)
                if (!tracking && now - lastTrackingWarn > 10_000) {
                    lastTrackingWarn = now
                    tts.speak("Backup tracking is unreliable. Stop and hold the phone steady.", interrupt = false)
                }
                if (zones != null) warn(zones, now)
                if (now - lastReport > 2_000) { lastReport = now; report(tracking, zones) }
            } catch (_: Exception) {
                LiveState.arState = "limited"
            }
            h.postDelayed(this, 100)
        }
    }

    /** Nearest 10th-percentile distance in metres for [left, ahead, right]; NaN where depth is missing. */
    private fun nearestPerZone(frame: com.google.ar.core.Frame): DoubleArray? {
        val img = try { frame.acquireDepthImage16Bits() } catch (_: Exception) { return null }
        try {
            val w = img.width; val hgt = img.height
            val p = img.planes[0]
            val buf = p.buffer.order(ByteOrder.nativeOrder()).asShortBuffer()
            val stride = p.rowStride / 2
            // Sensor image of an upright phone is rotated: sensor bottom = screen left, sensor top = screen right.
            val xs = (w * 0.30).toInt()..(w * 0.80).toInt()
            val bands = arrayOf(hgt * 2 / 3 until hgt, hgt / 3 until hgt * 2 / 3, 0 until hgt / 3)
            return DoubleArray(3) { z ->
                val vals = ArrayList<Int>()
                for (y in bands[z]) for (x in xs) {
                    val mm = buf.get(y * stride + x).toInt() and 0xFFFF
                    if (mm in 500..5000) vals += mm
                }
                if (vals.size < 20) Double.NaN else vals.sorted()[vals.size / 10] / 1000.0
            }
        } finally {
            img.close()
        }
    }

    private fun warn(d: DoubleArray, now: Long) {
        if (now - lastWarn < 4_000) return
        val names = arrayOf("on your left", "ahead", "on your right")
        val i = d.indices.filter { !d[it].isNaN() && d[it] < 1.5 }.minByOrNull { d[it] } ?: return
        lastWarn = now
        val m = d[i]
        val dist = if (m < 0.8) "very close" else "about %.1f metres".format(m)
        tts.speak("${if (i == 1 && m < 0.8) "Stop. " else ""}Something $dist, ${names[i]}.", interrupt = i == 1 && m < 0.8)
    }

    private fun report(tracking: Boolean, d: DoubleArray?) {
        val zones = arrayOf("left", "forward", "right")
        val haz = JSONArray()
        d?.forEachIndexed { i, v -> if (!v.isNaN() && v < 2.0) haz.put(JSONObject().put("kind", "obstacle").put("zone", zones[i]).put("distance_m", v)) }
        val ahead = d?.get(1)
        val command = when {
            !tracking -> "caution"
            ahead != null && !ahead.isNaN() && ahead < 0.8 -> "stop"
            else -> "proceed"
        }
        val data = JSONObject().put("hazards", haz)
            .put("path_confidence", if (tracking) 0.4 else 0.0)
            .put("clearance_left_m", d?.get(0)?.takeIf { !it.isNaN() } ?: JSONObject.NULL)
            .put("clearance_right_m", d?.get(2)?.takeIf { !it.isNaN() } ?: JSONObject.NULL)
        NavReport.publish("arcore", command, data)
    }

    private fun close() {
        runCatching { session?.close() }
        session = null
        val disp = egl[0] as? EGLDisplay
        if (disp != null) {
            EGL14.eglMakeCurrent(disp, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            (egl[2] as? EGLSurface)?.let { EGL14.eglDestroySurface(disp, it) }
            (egl[1] as? EGLContext)?.let { EGL14.eglDestroyContext(disp, it) }
        }
        egl = arrayOfNulls(3)
        LiveState.arState = "off"
        LiveState.changed()
    }

    fun shutdown() {
        running = false
        h.post { close(); thread.quitSafely() }
    }
}
