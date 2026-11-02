package com.example.gabai.safety

import org.json.JSONArray
import org.json.JSONObject

/** Turns a navigation payload (from the Pi pipeline over WebRTC, or from the ARCore fallback) into the compact report caregivers see. */
object NavReport {

    private fun num(d: JSONObject?, vararg keys: String): Double? =
        keys.firstNotNullOfOrNull { k -> d?.takeIf { it.has(k) && !it.isNull(k) }?.optDouble(k)?.takeIf { v -> !v.isNaN() } }

    fun publish(mode: String, command: String, data: JSONObject?) {
        LiveState.navMode = mode
        val obstacles = JSONArray()
        val hazards = data?.optJSONArray("hazards")
        if (hazards != null) for (i in 0 until minOf(hazards.length(), 8)) {
            val h = hazards.optJSONObject(i) ?: continue
            obstacles.put(JSONObject()
                .put("kind", h.optString("kind"))
                .put("zone", h.optString("zone"))
                .put("distance_m", num(h, "distance_m", "distance", "dist_m") ?: JSONObject.NULL)
                .put("approaching", h.optBoolean("approaching", false)))
        }
        val n = JSONObject()
            .put("t", System.currentTimeMillis() / 1000.0)
            .put("mode", mode)
            .put("direction", command)
            .put("instruction", data?.optJSONObject("speech")?.optString("text")?.takeIf { it.isNotBlank() } ?: JSONObject.NULL)
            .put("distance_m", num(data, "next_distance_m", "distance_m") ?: JSONObject.NULL)
            .put("obstacles", obstacles)
            // The payload names a collision stop or degraded perception explicitly; anything else is reported as safe to continue.
            .put("forward_safe", command != "stop" && command != "caution")
            .put("path_confidence", num(data, "path_confidence", "walkable_confidence", "confidence") ?: JSONObject.NULL)
            .put("clearance_left_m", num(data, "clearance_left_m", "left_clearance_m") ?: JSONObject.NULL)
            .put("clearance_right_m", num(data, "clearance_right_m", "right_clearance_m") ?: JSONObject.NULL)
        Uplink.setNav(n)
    }
}
