package com.example.gabai.accessibility

import android.app.Activity
import android.graphics.RectF
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.ViewGroup
import android.view.inputmethod.InputMethodManager
import android.widget.Checkable
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.TextView
import androidx.core.view.ViewCompat
import androidx.core.view.accessibility.AccessibilityNodeInfoCompat
import com.example.gabai.R
import com.example.gabai.services.Haptics
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.TtsService
import com.example.gabai.ui.Ui
import kotlin.math.abs

/**
 * App-level accessible navigation for people who cannot see the screen. Tap an element: a focus ring appears and its name,
 * role and state are spoken. Double-tap the focused element: it activates, and the result is spoken. Swipe right or left
 * anywhere: focus moves to the next or previous element in reading order. Vertical drags scroll as usual.
 *
 * It is switched on by the user with a labelled button and is never installed while TalkBack's explore-by-touch is running:
 * TalkBack already does all of this system-wide, and two screen readers would fight over every touch.
 */
class AccessFocusController(
    private val activity: Activity,
    private val tts: TtsService,
    private val haptics: Haptics,
    private val onExit: () -> Unit,
) {

    private val handler = Handler(Looper.getMainLooper())
    private val density = activity.resources.displayMetrics.density
    private val slop = ViewConfiguration.get(activity).scaledTouchSlop
    private val ring = FocusRingView(activity)
    private val exitPill: TextView
    private val host = FrameLayout(activity)
    private val loc = IntArray(2)
    private val hostLoc = IntArray(2)

    private var focused: View? = null
    private var lastTapAt = 0L
    private var downX = 0f
    private var downY = 0f
    private var lastY = 0f
    private var dragging = false
    private var scrollTarget: View? = null
    private val relayout = Relayout()

    init {
        exitPill = Ui.button(activity, "Exit accessible navigation", Ui.Kind.SECONDARY, "Exit accessible navigation. Button.") { disable() }.apply {
            minHeight = Ui.dp(activity, 48)
            textSize = 14f
        }
        host.importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS
        host.addView(ring, FrameLayout.LayoutParams(-1, -1))
        host.addView(exitPill, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.END).apply {
            topMargin = Ui.dp(activity, 8); marginEnd = Ui.dp(activity, 8)
        })
    }

    fun install(announce: Boolean = true) {
        if (host.parent != null) return
        activity.addContentView(host, ViewGroup.LayoutParams(-1, -1))
        content().viewTreeObserver.addOnGlobalLayoutListener(relayout)
        content().viewTreeObserver.addOnScrollChangedListener(relayout)
        if (announce) handler.postDelayed({ tts.speak("${activity.title}. Accessible navigation is on.", interrupt = true) }, 300)
        if (SettingsStore.isAccessTabsOnly(activity)) handler.postDelayed({ focusSelectedTab() }, 900)
    }

    fun uninstall() {
        handler.removeCallbacksAndMessages(null)
        runCatching {
            content().viewTreeObserver.removeOnGlobalLayoutListener(relayout)
            content().viewTreeObserver.removeOnScrollChangedListener(relayout)
        }
        (host.parent as? ViewGroup)?.removeView(host)
        focused = null
        ring.clear()
    }

    private fun content(): ViewGroup = activity.findViewById(android.R.id.content)

    /** Switches the mode off for the whole app and says so. */
    fun disable() {
        SettingsStore.setAccessNavEnabled(activity, false)
        SettingsStore.setAccessTabsOnly(activity, false)
        tts.speak("Accessible navigation is off.", interrupt = true)
        haptics.click()
        uninstall()
        onExit()
    }

    // ---------- touch ----------
    /** Returns true when the event was consumed. Called first from the activity's dispatchTouchEvent. */
    fun onTouch(ev: MotionEvent): Boolean {
        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                downX = ev.rawX; downY = ev.rawY; lastY = ev.rawY
                dragging = false
                scrollTarget = scrollableAt(ev.rawX, ev.rawY)
            }
            MotionEvent.ACTION_MOVE -> {
                if (!dragging && (abs(ev.rawX - downX) > slop || abs(ev.rawY - downY) > slop)) dragging = true
                if (dragging && abs(ev.rawY - downY) > abs(ev.rawX - downX)) {
                    scrollTarget?.scrollBy(0, (lastY - ev.rawY).toInt())
                    lastY = ev.rawY
                }
            }
            MotionEvent.ACTION_UP -> {
                val dx = ev.rawX - downX
                val dy = ev.rawY - downY
                when {
                    !dragging -> tap(ev.rawX, ev.rawY)
                    abs(dx) > 60 * density && abs(dx) > 2 * abs(dy) -> move(if (dx > 0) 1 else -1)
                }
                dragging = false
            }
            MotionEvent.ACTION_CANCEL -> dragging = false
        }
        return true
    }

    private fun tap(x: Float, y: Float) {
        val target = elementAt(x, y) ?: return
        val now = SystemClock.uptimeMillis()
        if (target === focused && now - lastTapAt <= DOUBLE_TAP_MS) {
            lastTapAt = 0
            activate(target)
        } else {
            focus(target, repeat = target === focused)
            lastTapAt = now
        }
    }

    // ---------- focus ----------
    private fun focus(v: View, repeat: Boolean = false) {
        focused = v
        v.parent?.requestChildRectangleOnScreen(v, android.graphics.Rect(0, 0, v.width, v.height), false)
        refreshRing(animate = true)
        haptics.hoverTick()
        tts.speak(describe(v, withHint = !repeat), interrupt = true)
        v.sendAccessibilityEvent(android.view.accessibility.AccessibilityEvent.TYPE_VIEW_ACCESSIBILITY_FOCUSED)
    }

    private fun activate(v: View) {
        haptics.click()
        if (v is EditText) {
            v.requestFocus()
            (activity.getSystemService(Activity.INPUT_METHOD_SERVICE) as InputMethodManager).showSoftInput(v, 0)
            tts.speak("${nameOf(v)}. Text field. Keyboard open.", interrupt = true)
            return
        }
        v.performClick()
        // Read the result after the click handlers have run: a toggled switch, a new tab, a changed label.
        handler.postDelayed({
            if (!v.isAttachedToWindow) return@postDelayed
            tts.speak(confirmation(v), interrupt = true)
            refreshRing(animate = false)
        }, 180)
    }

    private fun confirmation(v: View): String {
        val name = nameOf(v).substringBefore(',')
        return when {
            isTab(v) -> "$name tab opened."
            v is Checkable -> "$name, now ${if (v.isChecked) "on" else "off"}."
            else -> "$name activated."
        }
    }

    private fun move(dir: Int) {
        val list = targets()
        if (list.isEmpty()) return
        val i = list.indexOfFirst { it === focused }
        val next = if (i < 0) (if (dir > 0) 0 else list.lastIndex) else (i + dir).coerceIn(0, list.lastIndex)
        if (i >= 0 && next == i) {
            haptics.heavy()
            tts.speak(if (dir > 0) "Last item." else "First item.", interrupt = true)
            return
        }
        focus(list[next])
        lastTapAt = 0
    }

    private fun focusSelectedTab() {
        val tabs = targets().filter(::isTab)
        (tabs.firstOrNull { it.isSelected } ?: tabs.firstOrNull())?.let { focus(it) }
    }

    private fun refreshRing(animate: Boolean = false) {
        val v = focused
        if (v == null || !v.isShown || v.width == 0) { ring.clear(); return }
        v.getLocationOnScreen(loc)
        host.getLocationOnScreen(hostLoc)
        val l = (loc[0] - hostLoc[0]).toFloat()
        val t = (loc[1] - hostLoc[1]).toFloat()
        ring.moveTo(RectF(l, t, l + v.width, t + v.height), animate)
    }

    private inner class Relayout : android.view.ViewTreeObserver.OnGlobalLayoutListener, android.view.ViewTreeObserver.OnScrollChangedListener {
        override fun onGlobalLayout() = refreshRing()
        override fun onScrollChanged() = refreshRing()
    }

    // ---------- finding elements ----------
    private fun isTab(v: View) = v.getTag(R.id.a11y_tab) == true

    private fun targets(): List<View> {
        val out = ArrayList<View>()
        walk(content(), out)
        val tabsOnly = SettingsStore.isAccessTabsOnly(activity)
        val list = if (tabsOnly) out.filter(::isTab) else out
        val sorted = list.sortedWith(compareBy<View>({ top(it) / (24 * density).toInt() }, { left(it) }))
        return sorted + exitPill
    }

    private fun top(v: View): Int { v.getLocationOnScreen(loc); return loc[1] }
    private fun left(v: View): Int { v.getLocationOnScreen(loc); return loc[0] }

    private fun walk(v: View, out: MutableList<View>) {
        if (v === host || !v.isShown || v.width == 0 || v.height == 0) return
        if (v.importantForAccessibility == View.IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS) return
        val important = v.importantForAccessibility != View.IMPORTANT_FOR_ACCESSIBILITY_NO
        if (important && (v.isClickable || v is EditText)) { out += v; return }
        if (important && v !is ViewGroup && labelOf(v) != null) { out += v; return }
        if (important && v is ViewGroup && !v.contentDescription.isNullOrBlank()) { out += v; return }
        if (v is ViewGroup) for (i in 0 until v.childCount) walk(v.getChildAt(i), out)
    }

    /** Deepest element under the point: the same set, order and hit-areas that swiping visits. */
    private fun elementAt(rawX: Float, rawY: Float): View? {
        host.getLocationOnScreen(hostLoc)
        exitPill.getLocationOnScreen(loc)
        if (rawX >= loc[0] && rawX <= loc[0] + exitPill.width && rawY >= loc[1] && rawY <= loc[1] + exitPill.height) return exitPill
        val hits = targets().filter {
            it.getLocationOnScreen(loc)
            rawX >= loc[0] && rawX <= loc[0] + it.width && rawY >= loc[1] && rawY <= loc[1] + it.height
        }
        return hits.minByOrNull { it.width * it.height }
    }

    private fun scrollableAt(rawX: Float, rawY: Float): View? {
        fun find(v: View): View? {
            if (!v.isShown || v === host) return null
            v.getLocationOnScreen(loc)
            if (rawX < loc[0] || rawX > loc[0] + v.width || rawY < loc[1] || rawY > loc[1] + v.height) return null
            if (v is ViewGroup) for (i in v.childCount - 1 downTo 0) find(v.getChildAt(i))?.let { return it }
            return v.takeIf { it.canScrollVertically(1) || it.canScrollVertically(-1) }
        }
        return find(content())
    }

    // ---------- speech ----------
    private fun labelOf(v: View): CharSequence? =
        v.contentDescription?.takeIf { it.isNotBlank() } ?: (v as? TextView)?.text?.takeIf { it.isNotBlank() }

    private fun nameOf(v: View): String {
        labelOf(v)?.let { return it.toString().trim().trimEnd('.') }
        if (v is ViewGroup) {
            val parts = ArrayList<String>()
            fun collect(g: View) {
                if (g is TextView && g.isShown && g.text.isNotBlank() && g.importantForAccessibility != View.IMPORTANT_FOR_ACCESSIBILITY_NO) parts += g.text.toString()
                if (g is ViewGroup) for (i in 0 until g.childCount) collect(g.getChildAt(i))
            }
            collect(v)
            if (parts.isNotEmpty()) return parts.joinToString(". ")
        }
        return "Unlabelled"
    }

    private fun describe(v: View, withHint: Boolean): String {
        val info = AccessibilityNodeInfoCompat.obtain()
        ViewCompat.onInitializeAccessibilityNodeInfo(v, info)
        val name = nameOf(v)
        val role = info.roleDescription?.toString() ?: when {
            isTab(v) -> "tab"
            v is EditText -> "text field"
            v is android.widget.CompoundButton -> "switch"
            v.isClickable -> "button"
            ViewCompat.isAccessibilityHeading(v) -> "heading"
            else -> null
        }
        val state = info.stateDescription?.toString() ?: when {
            v is Checkable -> if (v.isChecked) "on" else "off"
            isTab(v) -> if (v.isSelected) "selected" else "not selected"
            else -> null
        }
        info.recycle()
        val parts = ArrayList<String>()
        parts += name
        // Labels in this app often already end in "Button."; do not say the role twice.
        if (role != null && !wordIn(role, name)) parts += role
        if (state != null && !wordIn(state, name)) parts += state
        val text = parts.joinToString(", ") + "."
        val hint = (v.getTag(R.id.a11y_hint) as? String) ?: if (v.isClickable) "Double tap to activate." else null
        return if (withHint && hint != null) "$text $hint" else text
    }

    private fun wordIn(word: String, text: String) =
        Regex("\\b" + Regex.escape(word) + "\\b", RegexOption.IGNORE_CASE).containsMatchIn(text)

    private companion object {
        const val DOUBLE_TAP_MS = 650L
    }
}
