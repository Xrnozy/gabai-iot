package com.example.gabai.safety

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import com.example.gabai.services.SettingsStore
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

/** One HTTP exchange with the relay. [body] is null on any failure and [error] says why, in words a status line can show. */
class Reply(val code: Int, val body: JSONObject?, val rttMs: Long, val error: String?) {
    val ok get() = code in 200..299 && body != null
}

/** Blocking client for gabai/relay. Call off the main thread. Auth role is the key into [SettingsStore] tokens. */
class RelayApi(context: Context, private val role: String) {

    private val app = context.applicationContext

    private fun base() = SettingsStore.getRelayUrl(app)
    private fun token() = SettingsStore.getRelayToken(app, role)

    fun ingest(batch: JSONObject) = call("POST", "/api/phone/ingest", batch)
    fun state() = call("GET", "/api/state")
    fun history(sinceEpoch: Double) = call("GET", "/api/history/locations?since=$sinceEpoch&limit=2000")
    fun events() = call("GET", "/api/history/events?limit=100")
    fun ack(id: String, who: String) =
        call("POST", "/api/events/${java.net.URLEncoder.encode(id, "UTF-8")}/ack?who=${java.net.URLEncoder.encode(who, "UTF-8")}")
    fun setRoute(name: String, points: List<Pair<Double, Double>>) = call("PUT", "/api/route", JSONObject()
        .put("name", name).put("points", JSONArray().also { a -> points.forEach { a.put(JSONArray().put(it.first).put(it.second)) } }))
    fun clearRoute() = call("DELETE", "/api/route")

    private fun call(method: String, path: String, body: JSONObject? = null): Reply {
        val t0 = SystemClock.elapsedRealtime()
        var conn: HttpURLConnection? = null
        return try {
            if (token().isBlank()) return Reply(0, null, 0, "No relay token set. Add it in Settings.")
            conn = (URL(base() + path).openConnection() as HttpURLConnection).apply {
                requestMethod = method
                connectTimeout = 4_000
                readTimeout = 6_000
                setRequestProperty("Authorization", "Bearer ${token()}")
                if (body != null) {
                    doOutput = true
                    setRequestProperty("Content-Type", "application/json")
                    outputStream.use { it.write(body.toString().toByteArray()) }
                }
            }
            val code = conn.responseCode
            val text = (if (code in 200..299) conn.inputStream else conn.errorStream)?.use { String(it.readBytes()) } ?: ""
            val json = runCatching { JSONObject(text) }.getOrNull()
            Reply(code, json, SystemClock.elapsedRealtime() - t0,
                if (code in 200..299) null else if (code == 401 || code == 403) "Relay rejected the token." else "Relay error $code.")
        } catch (e: Exception) {
            Reply(0, null, SystemClock.elapsedRealtime() - t0, "Cannot reach the relay: ${e.javaClass.simpleName}")
        } finally {
            conn?.disconnect()
        }
    }

    /** Opens the live MJPEG stream; the caller reads frames from it and must disconnect. Null when unavailable. */
    fun openStream(side: String): HttpURLConnection? = runCatching {
        (URL("${base()}/stream/$side.mjpg?fps=6").openConnection() as HttpURLConnection).apply {
            connectTimeout = 4_000
            readTimeout = 8_000
            setRequestProperty("Authorization", "Bearer ${token()}")
        }.takeIf { it.responseCode == 200 }
    }.getOrNull()

    /** Last frame on the relay and how old it is; works while the Pi is offline. */
    fun snapshot(side: String): Pair<Bitmap, Double>? = runCatching {
        val conn = (URL("${base()}/snapshot/$side.jpg").openConnection() as HttpURLConnection).apply {
            connectTimeout = 4_000
            readTimeout = 6_000
            setRequestProperty("Authorization", "Bearer ${token()}")
        }
        try {
            if (conn.responseCode != 200) return@runCatching null
            val age = conn.getHeaderField("X-Frame-Age-S")?.toDoubleOrNull() ?: 0.0
            val bmp = BitmapFactory.decodeStream(conn.inputStream) ?: return@runCatching null
            bmp to age
        } finally {
            conn.disconnect()
        }
    }.getOrNull()
}

/** Splits a multipart/x-mixed-replace JPEG stream into frames without extra libraries. */
class MjpegReader(private val input: InputStream) {
    private val buf = ByteArrayOutputStream(64 * 1024)

    /** Next JPEG, or null when the stream ended. Frames are found by their JPEG start/end markers. */
    fun next(): ByteArray? {
        buf.reset()
        var prev = -1
        var inJpeg = false
        while (true) {
            val b = input.read()
            if (b < 0) return null
            if (!inJpeg) {
                if (prev == 0xFF && b == 0xD8) {
                    inJpeg = true
                    buf.write(0xFF)
                    buf.write(0xD8)
                }
            } else {
                buf.write(b)
                if (prev == 0xFF && b == 0xD9) return buf.toByteArray()
                if (buf.size() > MAX_FRAME) return null
            }
            prev = b
        }
    }

    private companion object {
        const val MAX_FRAME = 4 * 1024 * 1024
    }
}
