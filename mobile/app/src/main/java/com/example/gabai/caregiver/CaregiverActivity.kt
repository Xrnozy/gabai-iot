package com.example.gabai.caregiver

import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.accessibility.AccessibilityNodeInfo
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.core.view.AccessibilityDelegateCompat
import androidx.core.view.ViewCompat
import androidx.core.view.accessibility.AccessibilityNodeInfoCompat
import com.example.gabai.BaseActivity
import com.example.gabai.FamilyActivity
import com.example.gabai.R
import com.example.gabai.safety.RelayApi
import com.example.gabai.services.SettingsStore
import com.example.gabai.ui.Level
import com.example.gabai.ui.Ui

/**
 * Caregiver dashboard: four tabs (now, map, alerts, details) over one shared 2-second poll. The tab bar is
 * ordinary views with tab semantics, so TalkBack and the app's own accessible navigation both treat it as tabs.
 */
class CaregiverActivity : BaseActivity() {

    private lateinit var tabs: List<CaregiverTab>
    private lateinit var content: FrameLayout
    private lateinit var tabViews: List<LinearLayout>
    private lateinit var linkChip: TextView
    private lateinit var alertsBtn: TextView
    private lateinit var accessBtn: TextView
    private lateinit var tabsBtn: TextView
    private lateinit var optionsColumn: LinearLayout
    private var current = 0
    private val onHub: () -> Unit = { render() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val c = this
        val ack: (String) -> Unit = { id -> acknowledge(id) }
        tabs = listOf(OverviewTab(c, ack), MapTab(c), SafetyTab(c, ack), DeviceTab(c))

        // Options (background alerts, people, accessibility) live in a sheet, so live data starts at the top of the screen.
        alertsBtn = Ui.button(c, "", Ui.Kind.SECONDARY) { toggleAlerts() }
        accessBtn = Ui.button(c, "", Ui.Kind.SECONDARY) { setAccessNav(!isAccessNavOn()); window.decorView.postDelayed({ refreshAccessButtons() }, 250) }
        tabsBtn = Ui.button(c, "", Ui.Kind.SECONDARY) {
            setAccessNav(true, tabsOnly = !SettingsStore.isAccessTabsOnly(c))
            window.decorView.postDelayed({ refreshAccessButtons() }, 250)
        }
        optionsColumn = Ui.column(c, 16).apply {
            addView(Ui.button(c, "Connect to someone", Ui.Kind.PRIMARY, "Connect to someone. Scan or type their code, with step by step help. Button.") {
                startActivity(android.content.Intent(c, com.example.gabai.ConnectActivity::class.java))
            }, Ui.lp(ctx = c))
            addView(alertsBtn, Ui.lp(top = 12, ctx = c))
            addView(Ui.button(c, "People I look after", Ui.Kind.SECONDARY, "People I look after. Invite someone or see who has allowed you. Button.") {
                startActivity(android.content.Intent(c, FamilyActivity::class.java))
            }, Ui.lp(top = 12, ctx = c))
            addView(accessBtn, Ui.lp(top = 12, ctx = c))
            addView(tabsBtn, Ui.lp(top = 12, ctx = c))
            addView(Ui.button(c, "App settings", Ui.Kind.SECONDARY, "App settings. Text size, contrast, speech and haptics. Button.") {
                startActivity(android.content.Intent(c, com.example.gabai.SettingsActivity::class.java))
            }, Ui.lp(top = 12, ctx = c))
            if (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0) {       // a test control: never shown in a release build
                addView(Ui.button(c, "Reset app (for testing)", Ui.Kind.DANGER, "Reset app for testing. Clears every setting, token and role on this phone. Button.") { confirmReset() },
                    Ui.lp(top = 12, ctx = c))
            }
        }
        val header = Ui.column(c, 16).apply {
            setBackgroundColor(Ui.color(c, R.color.canvas))
            addView(Ui.row(c).apply {
                addView(Ui.title(c, "Caregiver"), LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
                linkChip = Ui.chip(c, Level.OFF, "Connecting")
                addView(linkChip)
                addView(Ui.button(c, "Options", Ui.Kind.SECONDARY, "Options. Background alerts, people I look after, accessible navigation. Button.") { showOptions() },
                    LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).also { it.marginStart = Ui.dp(c, 12) })
            })
        }
        content = FrameLayout(c)
        tabs.forEach { content.addView(it.view, FrameLayout.LayoutParams(-1, -1)) }

        val bar = LinearLayout(c).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Ui.color(c, R.color.surface_card))
            setPadding(Ui.dp(c, 4), Ui.dp(c, 4), Ui.dp(c, 4), Ui.dp(c, 4))
        }
        tabViews = tabs.mapIndexed { i, t -> tabView(t, i).also { bar.addView(it, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)) } }

        setContentView(LinearLayout(c).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Ui.color(c, R.color.canvas))
            addView(header, Ui.lp(ctx = c))
            addView(content, LinearLayout.LayoutParams(-1, 0, 1f))
            addView(bar, Ui.lp(ctx = c))
        })
        select(0)
        refreshAlertsButton()
    }

    private fun showOptions() {
        refreshAlertsButton(); refreshAccessButtons()
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Options")
            .setView(optionsColumn)
            .setNegativeButton("Close", null)
            .create()
            .also { d -> d.setOnDismissListener { (optionsColumn.parent as? android.view.ViewGroup)?.removeView(optionsColumn) } }
            .show()
    }

    /** Public so accessible navigation can open a tab from its focus ring. */
    fun select(index: Int) {
        tabs.getOrNull(current)?.onHide()
        current = index
        tabs.forEachIndexed { i, t -> t.view.visibility = if (i == index) View.VISIBLE else View.GONE }
        tabViews.forEachIndexed { i, v -> styleTab(v, i == index) }
        tabs[index].onShow()
        tabs[index].update(StateHub.snapshot, StateHub.link)
        title = tabs[index].label
    }

    private fun tabView(t: CaregiverTab, index: Int) = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        gravity = Gravity.CENTER
        minimumHeight = Ui.dp(context, 64)
        setPadding(Ui.dp(context, 4), Ui.dp(context, 8), Ui.dp(context, 4), Ui.dp(context, 8))
        isClickable = true
        isFocusable = true
        setTag(R.id.a11y_tab, true)
        addView(ImageView(context).apply { setImageResource(t.icon); importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO }, LinearLayout.LayoutParams(Ui.dp(context, 24), Ui.dp(context, 24)))
        addView(Ui.text(context, t.label, 13f, R.color.muted, bold = true).apply { tag = "label"; gravity = Gravity.CENTER; importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO })
        setOnClickListener { select(index) }
        contentDescription = "${t.label}, tab ${index + 1} of ${tabs.size}. ${t.purpose}"
        ViewCompat.setAccessibilityDelegate(this, object : AccessibilityDelegateCompat() {
            override fun onInitializeAccessibilityNodeInfo(host: View, info: AccessibilityNodeInfoCompat) {
                super.onInitializeAccessibilityNodeInfo(host, info)
                info.roleDescription = "tab"
                info.isSelected = index == current
                info.stateDescription = if (index == current) "selected" else "not selected"
                info.addAction(AccessibilityNodeInfoCompat.AccessibilityActionCompat(AccessibilityNodeInfo.ACTION_CLICK, "open ${t.label}"))
            }
        })
    }

    private fun styleTab(v: LinearLayout, selected: Boolean) {
        v.setBackgroundResource(if (selected) R.drawable.bg_tab_selected else 0)
        (v.findViewWithTag<TextView>("label")).setTextColor(Ui.color(this, if (selected) R.color.primary else R.color.muted))
        v.getChildAt(0).let { (it as ImageView).setColorFilter(Ui.color(this, if (selected) R.color.primary else R.color.muted)) }
        v.sendAccessibilityEvent(android.view.accessibility.AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED)
    }

    override fun onResume() {
        super.onResume()
        refreshAccessButtons()
    }

    private fun refreshAccessButtons() {
        val nav = isAccessNavOn()
        val tabs = nav && SettingsStore.isAccessTabsOnly(this)
        accessBtn.text = "Accessible navigation: ${if (nav) "On" else "Off"}"
        accessBtn.contentDescription = "Accessible navigation is ${if (nav) "on" else "off"}. Tap to hear an item, double tap to open it. Button."
        tabsBtn.text = "Accessible tabs: ${if (tabs) "On" else "Off"}"
        tabsBtn.contentDescription = "Accessible tab navigation is ${if (tabs) "on" else "off"}. Swipe to move between the tabs, double tap to open one. Button."
    }

    override fun onStart() {
        super.onStart()
        StateHub.listen(onHub)
        StateHub.acquire(this, "ui")
        tabs[current].onShow()
        render()
    }

    override fun onStop() {
        tabs[current].onHide()
        StateHub.unlisten(onHub)
        StateHub.release("ui")
        super.onStop()
    }

    private var promptedForLink = false

    companion object {
        /** Debug builds: open the dashboard without a token or an active link, only to look at the layout. */
        const val EXTRA_DEBUG_PREVIEW = "debug_preview"
    }

    private fun render() {
        val link = StateHub.link
        // Nobody has allowed this caregiver yet (or the token is not set up): go straight to the invite screen, once.
        if (!promptedForLink && !intent.getBooleanExtra(EXTRA_DEBUG_PREVIEW, false) && link.error?.let { it.contains("no active caregiver link", true) || it.startsWith("No relay token") } == true) {
            promptedForLink = true
            startActivity(android.content.Intent(this, FamilyActivity::class.java))
        }
        val l = if (link.ok) Level.OK else if (StateHub.snapshot == null && link.error == null) Level.OFF else Level.BAD
        linkChip.text = "${l.symbol} " + if (link.ok) "Live · ${link.rttMs} ms" else if (StateHub.snapshot != null) "Offline · showing last known" else "Connecting"
        linkChip.setTextColor(Ui.color(this, l.color))
        tabs[current].update(StateHub.snapshot, link)
    }

    private fun acknowledge(id: String) {
        val who = SettingsStore.getCaregiverName(this)
        Thread {
            val ok = RelayApi(this, SettingsStore.ROLE_CAREGIVER).ack(id, who).ok
            runOnUiThread { Toast.makeText(this, if (ok) "Acknowledged. The user has been told you are responding." else "Could not acknowledge. Try again.", Toast.LENGTH_LONG).show() }
        }.start()
    }

    private fun toggleAlerts() {
        val on = !SettingsStore.isCaregiverAlertsEnabled(this)
        SettingsStore.setCaregiverAlertsEnabled(this, on)
        if (on) CaregiverAlertService.start(this) else CaregiverAlertService.stop(this)
        refreshAlertsButton()
    }

    /** Same wipe as Settings > Reset, plus stopping the background alert service; the app then restarts at the role choice. */
    private fun confirmReset() {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Reset app?")
            .setMessage("This clears every setting, the relay address, your tokens and your role on this phone. You will start again at the role choice.")
            .setPositiveButton("Reset") { _, _ ->
                CaregiverAlertService.stop(this)
                SettingsStore.resetAll(this)
                com.example.gabai.services.VisionProfileStore.reset(this)
                startActivity(android.content.Intent(this, com.example.gabai.MainActivity::class.java))
                finishAffinity()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun refreshAlertsButton() {
        val on = SettingsStore.isCaregiverAlertsEnabled(this)
        alertsBtn.text = if (on) "Alerts when the app is closed: On" else "Alerts when the app is closed: Off"
        alertsBtn.contentDescription = (if (on) "Background alerts are on. " else "Background alerts are off. ") + "Double tap to turn them ${if (on) "off" else "on"}. Button."
    }
}
