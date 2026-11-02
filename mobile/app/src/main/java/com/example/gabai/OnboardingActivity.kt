package com.example.gabai

import android.Manifest
import android.content.Intent
import android.graphics.Typeface
import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.example.gabai.services.Haptics
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.TtsService
import com.example.gabai.services.VisionProfile
import com.example.gabai.services.VisionProfileStore
import com.example.gabai.services.VisionProfiles
import java.util.Locale

/**
 * First-run setup: welcome → blindness type → permissions → voice practice.
 * Every step is spoken; the flow is designed to be completable without
 * looking at the screen (large targets, spoken confirmations, auto-advance).
 */
class OnboardingActivity : BaseActivity() {

    private enum class Step { WELCOME, VISION, PERMISSIONS, EXPLORE }

    private data class VisionOption(
        val id: String,
        val titleRes: Int,
        val descRes: Int,
    )

    private lateinit var tts: TtsService
    private lateinit var haptics: Haptics

    private lateinit var tvTitle: TextView
    private lateinit var tvDesc: TextView
    private lateinit var content: LinearLayout
    private lateinit var btnPrimary: android.widget.Button
    private lateinit var btnSkip: TextView

    private var step = Step.WELCOME
    private lateinit var heroCard: View

    /** (profile id, title res, desc res) — shown as big tappable rows. */
    private val visionOptions = listOf(
        VisionOption("left_loss", R.string.onboard_vision_left,
            R.string.onboard_vision_left_desc),
        VisionOption("right_loss", R.string.onboard_vision_right,
            R.string.onboard_vision_right_desc),
        VisionOption("tunnel", R.string.onboard_vision_tunnel,
            R.string.onboard_vision_tunnel_desc),
        VisionOption("central", R.string.onboard_vision_central,
            R.string.onboard_vision_central_desc),
        VisionOption("all", R.string.onboard_vision_all,
            R.string.onboard_vision_all_desc),
    )

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        // Whatever the outcome, move on — each mode re-requests what it needs.
        goTo(Step.EXPLORE)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_onboarding)

        tts = TtsService(this, Locale.US)
        haptics = Haptics(this)

        tvTitle = findViewById(R.id.tv_onboard_title)
        androidx.core.view.ViewCompat.setAccessibilityHeading(tvTitle, true)
        tvDesc = findViewById(R.id.tv_onboard_desc)
        content = findViewById(R.id.onboard_content)
        heroCard = findViewById(R.id.onboard_hero_card)
        btnPrimary = findViewById(R.id.btn_onboard_primary)
        btnSkip = findViewById(R.id.btn_onboard_skip)

        renderStep()
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.shutdown()
    }

    // ---------- steps ----------

    private fun goTo(next: Step) {
        step = next
        renderStep()
    }

    private fun renderStep() {
        content.removeAllViews()
        tvDesc.visibility = View.VISIBLE
        heroCard.visibility = if (step == Step.WELCOME) View.VISIBLE else View.GONE
        when (step) {
            Step.WELCOME -> {
                tvTitle.setText(R.string.onboard_welcome_title)
                tvDesc.setText(R.string.onboard_welcome_desc)
                btnPrimary.setText(R.string.onboard_get_started)
                btnPrimary.setOnClickListener {
                    haptics.click()
                    goTo(Step.VISION)
                }
                btnSkip.visibility = View.GONE
                speakStep("Welcome to Gab Ai. " + getString(R.string.onboard_welcome_desc) +
                    " Tap Get started at the bottom of the screen.")
            }

            Step.VISION -> {
                tvTitle.setText(R.string.onboard_vision_title)
                // Gestalt figure/ground: the choices themselves carry the
                // information; a subtitle only adds noise for partially
                // blind users. (It is still spoken via speakStep.)
                tvDesc.visibility = View.GONE
                visionOptions.forEach { opt ->
                    content.addView(visionRow(opt.id, opt.titleRes, opt.descRes))
                }
                btnPrimary.setText(R.string.onboard_continue)
                btnPrimary.setOnClickListener {
                    haptics.click()
                    // Default for "not sure": monitor everything.
                    chooseProfile("all")
                }
                btnPrimary.contentDescription =
                    "Continue. Not sure, tells me about everything around me."
                btnSkip.visibility = View.GONE
                speakStep(getString(R.string.onboard_vision_title) + ". " +
                    getString(R.string.onboard_vision_desc) +
                    " The choices are: Blurry on my left. Blurry on my right. " +
                    "Tunnel vision. Blurry center. Little or no vision. " +
                    "Or tap Continue to hear about everything.")
            }

            Step.PERMISSIONS -> {
                tvTitle.setText(R.string.onboard_permissions_title)
                tvDesc.setText(R.string.onboard_permissions_desc)
                btnPrimary.setText(R.string.onboard_allow)
                btnPrimary.setOnClickListener {
                    haptics.click()
                    permissionLauncher.launch(
                        buildList {
                            add(Manifest.permission.CAMERA)
                            add(Manifest.permission.RECORD_AUDIO)
                            // Needed to show the "Gabai is listening" notification.
                            if (android.os.Build.VERSION.SDK_INT >=
                                android.os.Build.VERSION_CODES.TIRAMISU
                            ) {
                                add(Manifest.permission.POST_NOTIFICATIONS)
                            }
                        }.toTypedArray()
                    )
                }
                btnSkip.setText(R.string.onboard_continue)
                btnSkip.visibility = View.VISIBLE
                btnSkip.setOnClickListener { goTo(Step.EXPLORE) }
                speakStep(getString(R.string.onboard_permissions_title) + ". " +
                    getString(R.string.onboard_permissions_desc))
            }

            Step.EXPLORE -> {
                tvTitle.setText(R.string.onboard_explore_title)
                tvDesc.setText(R.string.onboard_explore_desc)
                btnPrimary.setText(R.string.onboard_finish)
                btnPrimary.setOnClickListener { finishOnboarding() }
                btnSkip.visibility = View.GONE
                speakStep(getString(R.string.onboard_explore_title) + ". " +
                    getString(R.string.onboard_explore_desc) +
                    " Tap Finish to start using GabAi.")
            }
        }
        animateStepIn()
    }

    /** Gentle cascade — title, body and options rise in sequence so each
     *  step change reads as a transition, not a hard swap. */
    private fun animateStepIn() {
        val d = resources.displayMetrics.density
        tvTitle.alpha = 0f
        tvTitle.animate().alpha(1f).setDuration(250).start()
        tvDesc.alpha = 0f
        tvDesc.animate().alpha(1f).setDuration(250).setStartDelay(60).start()
        for (i in 0 until content.childCount) {
            val child = content.getChildAt(i)
            child.alpha = 0f
            child.translationY = 24f * d
            child.animate().alpha(1f).translationY(0f)
                .setDuration(280).setStartDelay(80L + i * 50L).start()
        }
    }

    /**
     * One vision-profile choice as a large, solid-color button.
     * Same shape/size across options (similarity) groups them as one
     * decision; the distinct hue per option differentiates choices at a
     * glance; the label alone carries the information.
     */
    private fun visionRow(profileId: String, titleRes: Int, descRes: Int): View {
        val colorRes = com.example.gabai.services.VisionOptionColors.resFor(profileId)
        val title = getString(titleRes)
        val desc = getString(descRes)
        val d = resources.displayMetrics.density

        val ripple = android.util.TypedValue().also {
            theme.resolveAttribute(android.R.attr.selectableItemBackground, it, true)
        }

        val btn = TextView(this).apply {
            text = title
            setTextColor(android.graphics.Color.WHITE)
            textSize = 20f
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
            gravity = android.view.Gravity.CENTER
            isClickable = true
            isFocusable = true
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
            contentDescription = "$title. $desc. Double-tap to choose."
            minimumHeight = (104 * d).toInt()
            background = android.graphics.drawable.GradientDrawable().apply {
                setColor(ContextCompat.getColor(this@OnboardingActivity, colorRes))
                cornerRadius = resources.getDimensionPixelSize(R.dimen.radius_pill).toFloat()
            }
            foreground = ContextCompat.getDrawable(this@OnboardingActivity, ripple.resourceId)
            setPadding(
                resources.getDimensionPixelSize(R.dimen.space_xl), 0,
                resources.getDimensionPixelSize(R.dimen.space_xl), 0,
            )
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply {
                bottomMargin = resources.getDimensionPixelSize(R.dimen.space_lg)
            }
        }

        btn.setOnClickListener {
            haptics.click()
            chooseProfile(profileId)
        }
        btn.setOnFocusChangeListener { _, hasFocus ->
            if (hasFocus) tts.speak("$title. $desc.")
        }
        return btn
    }

    private fun chooseProfile(profileId: String) {
        val profile: VisionProfile = VisionProfiles.byId(profileId)
        VisionProfileStore.saveProfile(this, profile)
        tts.speak("${profile.label}. Saved.")
        // Small delay so the confirmation lands before the next step speaks.
        content.postDelayed({ goTo(Step.PERMISSIONS) }, 600)
    }

    private fun speakStep(text: String) {
        tvTitle.postDelayed({ tts.speak(text) }, 250)
    }

    private fun finishOnboarding() {
        haptics.click()
        SettingsStore.setOnboardingDone(this, true)
        tts.speak("All set. Welcome to Gab Ai.")
        tvTitle.postDelayed({
            startActivity(Intent(this, MainActivity::class.java))
            finish()
        }, 900)
    }
}
