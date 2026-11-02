package com.example.gabai.caregiver

import java.text.DateFormat
import java.util.Date
import kotlin.math.roundToInt

/** Plain-language formatting shared by the caregiver tabs. */
object Fmt {

    /** "4 s", "3 min", "2 h 5 min". */
    fun age(seconds: Double?): String {
        if (seconds == null) return "never"
        val s = seconds.roundToInt().coerceAtLeast(0)
        return when {
            s < 60 -> "$s s"
            s < 3600 -> "${s / 60} min"
            s < 86400 -> "${s / 3600} h ${(s % 3600) / 60} min"
            else -> "${s / 86400} d"
        }
    }

    fun clock(epochSeconds: Double?): String =
        if (epochSeconds == null) "unknown" else DateFormat.getTimeInstance(DateFormat.MEDIUM).format(Date((epochSeconds * 1000).toLong()))

    fun dateTime(epochSeconds: Double?): String =
        if (epochSeconds == null) "unknown" else DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.MEDIUM).format(Date((epochSeconds * 1000).toLong()))

    fun metres(m: Double?): String = when {
        m == null -> "unknown"
        m < 1000 -> "${m.roundToInt()} m"
        else -> "%.2f km".format(m / 1000)
    }

    fun kmh(mps: Double?): String = if (mps == null) "unknown" else "%.1f km/h".format(mps * 3.6)

    fun compass(deg: Double?): String {
        if (deg == null) return "unknown"
        val names = arrayOf("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
        return "${names[(((deg % 360) + 360) % 360 / 45).roundToInt() % 8]} (${deg.roundToInt()}°)"
    }

    fun pct(v: Double?): String = if (v == null) "unknown" else "${v.roundToInt()}%"

    fun zoneWords(zone: String?): String = when (zone) {
        "left", "slight_left" -> "on your left"
        "right", "slight_right" -> "on your right"
        "ahead", "center", "centre", "forward" -> "ahead"
        else -> zone ?: "nearby"
    }

    /** Raspberry Pi `vcgencmd get_throttled` bits: 0-3 are happening now, 16-19 happened since boot. */
    fun powerFlags(hex: String?): Pair<List<String>, List<String>> {
        val v = hex?.removePrefix("0x")?.toLongOrNull(16) ?: return emptyList<String>() to emptyList()
        val names = listOf("under-voltage", "frequency capped", "throttled", "soft temperature limit")
        val now = names.filterIndexed { i, _ -> v and (1L shl i) != 0L }
        val past = names.filterIndexed { i, _ -> v and (1L shl (16 + i)) != 0L }
        return now to past
    }
}
