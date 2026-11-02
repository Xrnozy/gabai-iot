package com.example.gabai

import android.content.Context
import android.content.res.Configuration
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.ViewGroup
import android.view.accessibility.AccessibilityManager
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.gabai.accessibility.AccessFocusController
import com.example.gabai.services.Contrast
import com.example.gabai.services.Haptics
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.TouchExploreOverlay
import com.example.gabai.services.TtsService

/**
 * Shared base for every screen: applies the display-related Settings
 * (large text via configuration fontScale, high contrast via a post-layout
 * view pass) and recreates the screen if those settings changed while the
 * user was away (e.g., toggled in SettingsActivity then navigated back).
 */
abstract class BaseActivity : AppCompatActivity() {

    private var appliedFontScale = 1f
    private var appliedHighContrast = false

    private var captionPanel: View? = null
    private var captionText: TextView? = null

    // ── Touch explore: warp overlay + dwell-to-announce ──
    private var exploreOverlay: TouchExploreOverlay? = null
    private var exploreTts: TtsService? = null
    private var exploreHaptics: Haptics? = null
    private val exploreHandler = Handler(Looper.getMainLooper())
    private val overlayLoc = IntArray(2)
    private var touchSlop = 0
    private var downX = 0f
    private var downY = 0f
    private var curX = 0f
    private var curY = 0f
    private var announcedTarget: View? = null
    private var hoverTarget: View? = null

    private val dwellRunnable = Runnable { onTouchDwell() }

    // ── Accessible navigation (tap to focus, double-tap to activate) ──
    private var accessFocus: AccessFocusController? = null

    override fun attachBaseContext(newBase: Context) {
        val scale = SettingsStore.getFontScale(newBase)
        val context = if (scale != 1.0f) {
            val cfg = Configuration(newBase.resources.configuration)
            cfg.fontScale = scale
            newBase.createConfigurationContext(cfg)
        } else {
            newBase
        }
        super.attachBaseContext(context)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        appliedFontScale = SettingsStore.getFontScale(this)
        appliedHighContrast = SettingsStore.isHighContrast(this)
        super.onCreate(savedInstanceState)
    }

    override fun onResume() {
        super.onResume()
        val scale = SettingsStore.getFontScale(this)
        val contrast = SettingsStore.isHighContrast(this)
        if (scale != appliedFontScale || contrast != appliedHighContrast) {
            recreate()
            return
        }
        if (contrast) {
            window.decorView.post { Contrast.apply(window.decorView) }
        }
        syncCaptionVisibility()
        syncAccessNav(announce = true)
    }

    private fun talkBackOn(): Boolean =
        (getSystemService(ACCESSIBILITY_SERVICE) as? AccessibilityManager)?.isTouchExplorationEnabled == true

    /** Installs or removes the focus-ring controller to match the setting. Never runs while TalkBack explores by touch. */
    private fun syncAccessNav(announce: Boolean) {
        val want = SettingsStore.isAccessNavEnabled(this) && !talkBackOn()
        if (want && accessFocus == null) {
            accessFocus = AccessFocusController(this, exploreTts(), exploreHaptics()) { accessFocus = null }.also { it.install(announce) }
        } else if (!want && accessFocus != null) {
            accessFocus?.uninstall()
            accessFocus = null
        }
    }

    /** Called by the labelled on/off buttons. [tabsOnly] limits swipe navigation to the tab bar. */
    fun setAccessNav(on: Boolean, tabsOnly: Boolean = false) {
        if (on && talkBackOn()) {
            exploreTts().speak("TalkBack is on, so use its gestures. Accessible navigation stays off while TalkBack is running.")
            return
        }
        SettingsStore.setAccessNavEnabled(this, on)
        SettingsStore.setAccessTabsOnly(this, on && tabsOnly)
        accessFocus?.uninstall()
        accessFocus = null
        syncAccessNav(announce = false)
        exploreTts().speak(
            if (!on) "Accessible navigation is off."
            else (if (tabsOnly) "Accessible tab navigation is on. Swipe right or left to move between tabs. " else "Accessible navigation is on. Tap an item to select it and hear what it is. Swipe right or left to move to the next or previous item. ") +
                "Double tap the item with the ring around it to open it. To exit, select the Exit button at the top right and double tap it.",
        )
    }

    fun isAccessNavOn() = accessFocus != null

    /**
     * Hooks the shared "Gabai said" caption panel (layouts that include
     * view_caption_panel.xml) up to this screen's TTS engine. Call after
     * setContentView; no-op on screens without the panel.
     */
    protected fun attachCaptions(tts: TtsService) {
        captionPanel = findViewById(R.id.caption_panel)
        captionText = findViewById(R.id.tv_caption)
        if (captionPanel != null) {
            tts.onSpoken = { line -> runOnUiThread { captionText?.text = line } }
        }
        syncCaptionVisibility()
    }

    protected fun syncCaptionVisibility() {
        captionPanel?.visibility =
            if (SettingsStore.isCaptionsEnabled(this)) View.VISIBLE else View.GONE
    }

    override fun onContentChanged() {
        super.onContentChanged()
        applySystemBarPadding()
        installTouchExplore()
        if (SettingsStore.isHighContrast(this)) {
            window.decorView.post { Contrast.apply(window.decorView) }
        }
    }

    /**
     * Touch explore — every screen gets the finger-tracking warp overlay
     * plus dwell announcements (hold ~1 s over a control → tick + spoken
     * label). Observation-only: touch events still flow to views normally.
     * Disabled when TalkBack's own explore-by-touch is active (it already
     * announces) or when turned off in Settings.
     */
    private fun installTouchExplore() {
        if (!SettingsStore.isTouchExploreEnabled(this) ||
            exploreOverlay != null) return
        touchSlop = ViewConfiguration.get(this).scaledTouchSlop
        val overlay = TouchExploreOverlay(this).apply {
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        // Assign BEFORE addContentView — adding a view re-enters
        // onContentChanged, so the null check must already see it.
        exploreOverlay = overlay
        addContentView(
            overlay,
            ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )
    }

    private fun touchExploreActive(): Boolean {
        if (exploreOverlay == null || accessFocus != null ||
            !SettingsStore.isTouchExploreEnabled(this)) return false
        val am = getSystemService(ACCESSIBILITY_SERVICE) as? AccessibilityManager
            ?: return false
        // TalkBack's explore-by-touch already announces — don't double-speak.
        return !am.isTouchExplorationEnabled
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (accessFocus?.onTouch(ev) == true) return true
        if (touchExploreActive()) trackTouch(ev)
        return super.dispatchTouchEvent(ev)
    }

    private fun trackTouch(ev: MotionEvent) {
        val overlay = exploreOverlay ?: return
        overlay.getLocationOnScreen(overlayLoc)
        val x = ev.rawX - overlayLoc[0]
        val y = ev.rawY - overlayLoc[1]
        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                downX = ev.rawX; downY = ev.rawY
                curX = ev.rawX; curY = ev.rawY
                announcedTarget = null
                hoverTarget = null
                overlay.onFingerDown(x, y)
                armDwell()
            }
            MotionEvent.ACTION_MOVE -> {
                curX = ev.rawX; curY = ev.rawY
                overlay.onFingerMove(x, y)
                // Hover feedback — a light tick each time the finger
                // crosses onto a new readable element.
                val hit = exploreTargetAt(curX, curY)
                if (hit !== hoverTarget) {
                    hoverTarget = hit
                    if (hit != null) exploreHaptics().hoverTick()
                }
                val dx = ev.rawX - downX
                val dy = ev.rawY - downY
                if (dx * dx + dy * dy > touchSlop * touchSlop) {
                    downX = ev.rawX; downY = ev.rawY
                    armDwell()
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                exploreHandler.removeCallbacks(dwellRunnable)
                hoverTarget = null
                overlay.onFingerUp()
            }
        }
    }

    private fun armDwell() {
        exploreHandler.removeCallbacks(dwellRunnable)
        exploreHandler.postDelayed(dwellRunnable, DWELL_MS)
    }

    private fun onTouchDwell() {
        val target = exploreTargetAt(curX, curY) ?: return
        if (target === announcedTarget) return
        announcedTarget = target
        val label = labelFor(target) ?: return
        exploreHaptics().heavy()
        exploreTts().speak(label.toString())
    }

    private fun exploreTts(): TtsService =
        exploreTts ?: TtsService(this).also { exploreTts = it }

    private fun exploreHaptics(): Haptics =
        exploreHaptics ?: Haptics(this).also { exploreHaptics = it }

    /** Deepest visible view under the point, skipping the overlay itself. */
    private fun deepestAt(v: View, rawX: Float, rawY: Float): View? {
        if (v === exploreOverlay || !v.isShown) return null
        v.getLocationOnScreen(overlayLoc)
        val inX = rawX >= overlayLoc[0] && rawX < overlayLoc[0] + v.width
        val inY = rawY >= overlayLoc[1] && rawY < overlayLoc[1] + v.height
        if (!inX || !inY) return null
        if (v is ViewGroup) {
            for (i in v.childCount - 1 downTo 0) {
                deepestAt(v.getChildAt(i), rawX, rawY)?.let { return it }
            }
        }
        return v
    }

    /**
     * What should be announced at this point: the nearest clickable
     * ancestor of whatever the finger rests on (its label wins), else the
     * deepest a11y-important view with text — e.g. a plain label counts.
     */
    private fun exploreTargetAt(rawX: Float, rawY: Float): View? {
        val root = findViewById<View>(android.R.id.content) ?: return null
        var v: View? = deepestAt(root, rawX, rawY) ?: return null
        var labeled: View? = null
        while (v != null && v !== root) {
            val label = labelFor(v)
            if (v.isClickable && label != null) return v
            if (labeled == null && label != null &&
                v.isImportantForAccessibility) labeled = v
            v = v.parent as? View
        }
        return labeled
    }

    private fun labelFor(v: View): CharSequence? =
        v.contentDescription?.takeIf { it.isNotBlank() }
            ?: (v as? TextView)?.text?.takeIf { it.isNotBlank() }

    /**
     * Edge-to-edge targets draw content under the status/nav bars; pad the
     * content frame by the system-bar insets so headers never sit under
     * the clock/battery area on any screen.
     */
    private fun applySystemBarPadding() {
        val content = findViewById<View>(android.R.id.content) ?: return
        val baseLeft = content.paddingLeft
        val baseTop = content.paddingTop
        val baseRight = content.paddingRight
        val baseBottom = content.paddingBottom
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(content) { v, insets ->
            val bars = insets.getInsets(
                androidx.core.view.WindowInsetsCompat.Type.systemBars()
            )
            v.setPadding(
                baseLeft + bars.left, baseTop + bars.top,
                baseRight + bars.right, baseBottom + bars.bottom,
            )
            insets
        }
        androidx.core.view.ViewCompat.requestApplyInsets(content)
    }

    /**
     * Re-snapshots the display prefs this instance was created with. Call
     * after changing them without a recreate (e.g. live-applying contrast)
     * so onResume doesn't fire a spurious recreate.
     */
    protected fun syncDisplaySettings() {
        appliedFontScale = SettingsStore.getFontScale(this)
        appliedHighContrast = SettingsStore.isHighContrast(this)
    }

    override fun onDestroy() {
        exploreHandler.removeCallbacks(dwellRunnable)
        exploreTts?.shutdown()
        super.onDestroy()
    }

    private companion object {
        const val DWELL_MS = 1_000L
    }
}
