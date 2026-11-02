package com.example.gabai.services

import android.content.Context
import androidx.appcompat.app.AlertDialog
import com.example.gabai.R

/**
 * Shared vision-profile dialogs, usable from Navigation and Settings.
 * Presets cover the blindness types onboarding asks about (one-sided loss,
 * tunnel, central); "custom" lets the user pick zones directly.
 */
object VisionProfilePicker {

    fun interface OnProfileChanged {
        fun onChanged(profile: VisionProfile)
    }

    /**
     * Same big colored buttons as onboarding's vision step — uniform
     * shape groups the choice, distinct hue differentiates options.
     * Extra rows (custom zones, steering) use a neutral fill.
     */
    fun show(context: Context, current: VisionProfile, onChanged: OnProfileChanged) {
        val presets = VisionProfiles.presets.filter { it.id != "custom" }
        val steeringLabel = if (current.steeringEnabled)
            context.getString(R.string.nav_profile_steering_on)
        else
            context.getString(R.string.nav_profile_steering_off)

        val container = android.widget.LinearLayout(context).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            val pad = context.resources.getDimensionPixelSize(R.dimen.space_lg)
            setPadding(pad, 0, pad, pad)
        }

        val dialog = AlertDialog.Builder(context)
            .setTitle(R.string.nav_profile_picker_title)
            .setView(container)
            .setNegativeButton(android.R.string.cancel, null)
            .create()

        presets.forEach { preset ->
            container.addView(optionButton(context, preset.label, VisionOptionColors.resFor(preset.id)) {
                VisionProfileStore.saveProfile(context, preset)
                onChanged.onChanged(preset)
                dialog.dismiss()
            })
        }
        container.addView(optionButton(
            context, context.getString(R.string.nav_profile_zones_title) + "…",
            R.color.ob_neutral,
        ) {
            dialog.dismiss()
            showZonePicker(context, current, onChanged)
        })
        container.addView(optionButton(context, steeringLabel, R.color.ob_neutral) {
            dialog.dismiss()
            toggleSteering(context, current, onChanged)
        })

        dialog.show()
    }

    /** Large solid-color choice button (shared style with onboarding). */
    private fun optionButton(
        context: Context,
        label: String,
        colorRes: Int,
        onClick: () -> Unit,
    ): android.widget.TextView {
        val d = context.resources.displayMetrics.density
        val ripple = android.util.TypedValue().also {
            context.theme.resolveAttribute(android.R.attr.selectableItemBackground, it, true)
        }
        return android.widget.TextView(context).apply {
            text = label
            setTextColor(androidx.core.content.ContextCompat.getColor(context, R.color.on_tape))
            textSize = 18f
            typeface = com.example.gabai.ui.Ui.font(context, true)
            gravity = android.view.Gravity.CENTER
            isClickable = true
            isFocusable = true
            contentDescription = label
            minimumHeight = (88 * d).toInt()
            background = android.graphics.drawable.GradientDrawable().apply {
                setColor(androidx.core.content.ContextCompat.getColor(context, colorRes))
                cornerRadius =
                    context.resources.getDimensionPixelSize(R.dimen.radius_md).toFloat()
            }
            foreground =
                androidx.core.content.ContextCompat.getDrawable(context, ripple.resourceId)
            layoutParams = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply {
                bottomMargin = context.resources.getDimensionPixelSize(R.dimen.space_md)
            }
            setOnClickListener { onClick() }
        }
    }

    /** Multi-choice zone picker backing the "custom" profile. */
    fun showZonePicker(context: Context, current: VisionProfile, onChanged: OnProfileChanged) {
        val zoneLabels = arrayOf("Far left", "Slight left", "Ahead", "Slight right", "Far right")
        val allZones = Zone.entries.toList()
        val checked = allZones.map { it in current.blindZones }.toBooleanArray()

        AlertDialog.Builder(context)
            .setTitle(R.string.nav_profile_zones_title)
            .setMultiChoiceItems(zoneLabels, checked) { _, which, isChecked ->
                checked[which] = isChecked
            }
            .setPositiveButton(android.R.string.ok) { _, _ ->
                val zones = allZones.filterIndexed { i, _ -> checked[i] }.toSet()
                if (zones.isEmpty()) return@setPositiveButton
                val custom = VisionProfile("custom", "Custom", zones, current.steeringEnabled)
                VisionProfileStore.saveProfile(context, custom)
                onChanged.onChanged(custom)
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun toggleSteering(context: Context, current: VisionProfile, onChanged: OnProfileChanged) {
        val enabled = !current.steeringEnabled
        VisionProfileStore.setSteering(context, enabled)
        onChanged.onChanged(current.copy(steeringEnabled = enabled))
    }
}

/** Profile id → button color; shared by onboarding and the pickers. */
object VisionOptionColors {
    @androidx.annotation.ColorRes
    fun resFor(profileId: String): Int = when (profileId) {
        "left_loss" -> R.color.ob_opt_left
        "right_loss" -> R.color.ob_opt_right
        "tunnel" -> R.color.ob_opt_tunnel
        "central" -> R.color.ob_opt_central
        else -> R.color.ob_opt_all
    }
}
