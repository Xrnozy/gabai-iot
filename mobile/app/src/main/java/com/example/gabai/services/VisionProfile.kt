package com.example.gabai.services

import android.content.Context

/**
 * Per-user visual field profile — which directions are blind spots.
 * Advisory warnings are only spoken for hazards in these zones; the
 * profile lives entirely on-device (health data never leaves the phone).
 */
enum class Zone(val wire: String) {
    LEFT("left"),
    SLIGHT_LEFT("slight_left"),
    FORWARD("forward"),
    SLIGHT_RIGHT("slight_right"),
    RIGHT("right");

    companion object {
        fun fromWire(s: String): Zone? = entries.find { it.wire == s }
    }
}

data class VisionProfile(
    val id: String,
    val label: String,
    val blindZones: Set<Zone>,
    val steeringEnabled: Boolean,
)

object VisionProfiles {
    val presets: List<VisionProfile> = listOf(
        VisionProfile(
            "left_loss", "Left field loss",
            setOf(Zone.LEFT, Zone.SLIGHT_LEFT),
            steeringEnabled = false,
        ),
        VisionProfile(
            "right_loss", "Right field loss",
            setOf(Zone.RIGHT, Zone.SLIGHT_RIGHT),
            steeringEnabled = false,
        ),
        VisionProfile(
            "tunnel", "Tunnel vision",
            setOf(Zone.LEFT, Zone.RIGHT),
            steeringEnabled = false,
        ),
        VisionProfile(
            "central", "Central vision loss",
            setOf(Zone.FORWARD, Zone.SLIGHT_LEFT, Zone.SLIGHT_RIGHT),
            steeringEnabled = true,
        ),
        VisionProfile(
            "all", "Monitor all zones",
            Zone.entries.toSet(),
            steeringEnabled = true,
        ),
        VisionProfile(
            "custom", "Custom zones",
            Zone.entries.toSet(),
            steeringEnabled = true,
        ),
    )

    fun byId(id: String?): VisionProfile =
        presets.find { it.id == id } ?: presets.first { it.id == "all" }
}

/** SharedPreferences-backed profile store. */
object VisionProfileStore {
    private const val PREFS = "gabai_vision_profile"
    private const val KEY_ID = "profile_id"
    private const val KEY_STEERING = "steering"
    private const val KEY_ZONES = "custom_zones"

    fun load(context: Context): VisionProfile {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val base = VisionProfiles.byId(p.getString(KEY_ID, null))
        val zones = if (base.id == "custom") {
            val raw = p.getString(KEY_ZONES, "") ?: ""
            val parsed = raw.split(",").mapNotNull { Zone.fromWire(it) }.toSet()
            if (parsed.isEmpty()) base.blindZones else parsed
        } else {
            base.blindZones
        }
        return base.copy(
            blindZones = zones,
            steeringEnabled = p.getBoolean(KEY_STEERING, base.steeringEnabled),
        )
    }

    fun saveProfile(context: Context, profile: VisionProfile) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putString(KEY_ID, profile.id)
            .putBoolean(KEY_STEERING, profile.steeringEnabled)
            .putString(
                KEY_ZONES,
                profile.blindZones.joinToString(",") { it.wire },
            )
            .apply()
    }

    fun setSteering(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putBoolean(KEY_STEERING, enabled).apply()
    }

    /** Clears the saved profile (used by "Reset everything" in Settings). */
    fun reset(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .clear().apply()
    }
}
