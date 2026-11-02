package com.example.gabai.services

import android.util.Log
import okhttp3.MediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import org.json.JSONObject

/** ponytail: dev-only NDJSON ingest; fails silently off-device. */
internal object AgentDebugLog {
    private const val TAG = "AgentDebug"
    private const val ENDPOINT =
        "http://127.0.0.1:7526/ingest/f3f7c36d-cd2d-4cb6-a79e-dba10cd3e73a"
    private val client = OkHttpClient()

    fun log(
        location: String,
        message: String,
        hypothesisId: String,
        data: Map<String, Any?> = emptyMap(),
    ) {
        val payload = JSONObject().apply {
            put("sessionId", "be9aa7")
            put("timestamp", System.currentTimeMillis())
            put("location", location)
            put("message", message)
            put("hypothesisId", hypothesisId)
            put("data", JSONObject(data))
        }
        Log.d(TAG, payload.toString())
        // #region agent log
        Thread {
            try {
                val body = RequestBody.create(
                    MediaType.parse("application/json"),
                    payload.toString(),
                )
                val req = Request.Builder()
                    .url(ENDPOINT)
                    .header("X-Debug-Session-Id", "be9aa7")
                    .post(body)
                    .build()
                val response = client.newCall(req).execute()
                response.body()?.close()
            } catch (_: Exception) {
            }
        }.start()
        // #endregion
    }
}
