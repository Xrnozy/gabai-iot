package com.example.gabai

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.text.format.DateUtils
import android.view.View
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import com.example.gabai.caregiver.CaregiverActivity
import com.example.gabai.safety.Emergency
import com.example.gabai.safety.LiveState
import com.example.gabai.safety.TrackingService
import com.example.gabai.safety.Uplink
import com.example.gabai.services.SettingsStore
import com.example.gabai.ui.Level
import com.example.gabai.ui.Ui
import androidx.activity.result.contract.ActivityResultContracts

/**
 * The blind user's side of "Family & safety": switch location sharing and fall detection on, press SOS, and connect this phone
 * to the relay. Caregivers open the dashboard from the same screen on their own phone.
 */
class FamilyActivity : BaseActivity() {

    private lateinit var status: LinearLayout
    private lateinit var toggle: android.widget.TextView
    private lateinit var url: EditText
    private lateinit var phoneToken: EditText
    private lateinit var caregiverToken: EditText
    private lateinit var name: EditText
    private lateinit var userLinks: FamilyLinks
    private lateinit var caregiverLinks: FamilyLinks
    private val onChange: () -> Unit = { runOnUiThread { renderStatus() } }

    private val permissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { r ->
        if (r[Manifest.permission.ACCESS_FINE_LOCATION] == true || r[Manifest.permission.ACCESS_COARSE_LOCATION] == true) {
            TrackingService.start(this)
            renderStatus()
        } else {
            LiveState.gpsState = "denied"
            renderStatus()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val c = this
        val col = Ui.column(c, 24)

        col.addView(Ui.button(c, "← Back", Ui.Kind.SECONDARY, "Back to home. Button.") { finish() }, Ui.lp(ctx = c))
        col.addView(Ui.title(c, "Family & safety"), Ui.lp(top = 16, ctx = c))
        col.addView(Ui.caption(c, "Share your location with people you trust. They can see where you are, how the device is doing, and get alerted if you need help."), Ui.lp(top = 4, ctx = c))

        val helper = SettingsStore.getAppRole(c) != SettingsStore.APP_CAREGIVER
        val connect = Ui.card(c, if (helper) "Connect a caregiver" else "Connect to someone").also { col.addView(it, Ui.lp(top = 16, ctx = c)) }
        connect.addView(Ui.body(c, if (helper) "Show a code to the person who looks after you. Step by step help is on the next screen."
            else "Scan the code on the other phone, or type it. Step by step help is on the next screen."), Ui.lp(top = 4, ctx = c))
        connect.addView(Ui.button(c, if (helper) "Show my code" else "Scan or type a code", Ui.Kind.PRIMARY,
            (if (helper) "Show my code to a caregiver" else "Scan or type a code to connect") + ". Button.") {
            startActivity(Intent(c, ConnectActivity::class.java))
        }, Ui.lp(top = 12, ctx = c))

        val share = Ui.card(c, "Share my location").also { col.addView(it, Ui.lp(top = 16, ctx = c)) }
        status = Ui.column(c)
        share.addView(status, Ui.lp(top = 8, ctx = c))
        toggle = Ui.button(c, "", Ui.Kind.PRIMARY) { toggleSharing() }
        share.addView(toggle, Ui.lp(top = 16, ctx = c))
        share.addView(Ui.button(c, "Emergency SOS", Ui.Kind.DANGER, "Emergency SOS. Alerts your caregivers after a 5 second countdown you can cancel. Button.") {
            Emergency.startSos(c); Emergency.openScreen()
        }, Ui.lp(top = 12, ctx = c))
        share.addView(Ui.caption(c, "Sharing and fall detection turn on by themselves once you are connected to a caregiver. You always get a 20 second countdown to cancel before anyone is alerted."), Ui.lp(top = 12, ctx = c))

        val conn = Ui.card(c, "Connection").also { col.addView(it, Ui.lp(top = 16, ctx = c)) }
        url = Ui.field(c, "Relay address, for example http://192.168.137.1:8090", SettingsStore.getRelayUrl(c))
        phoneToken = Ui.field(c, "This phone's token", SettingsStore.getRelayToken(c, SettingsStore.ROLE_PHONE), secret = true)
        caregiverToken = Ui.field(c, "Caregiver token (only on a caregiver's phone)", SettingsStore.getRelayToken(c, SettingsStore.ROLE_CAREGIVER), secret = true)
        name = Ui.field(c, "Your name, shown when you acknowledge an alert", SettingsStore.getCaregiverName(c))
        conn.addView(Ui.caption(c, "Nothing to type here. This phone signs up with the relay by itself and keeps its own private key (a token). Use Connect above to link people."), Ui.lp(top = 4, ctx = c))
        listOf(url, name).forEach { conn.addView(it, Ui.lp(top = 12, ctx = c)) }

        // Tokens fill themselves in; the fields stay only for a relay set up with shared tokens, so they start hidden.
        val advanced = Ui.column(c).apply { visibility = View.GONE }
        advanced.addView(Ui.caption(c, "A token is the private key this phone uses to talk to the relay. It is created automatically and filled in below. Only type one if your relay was set up with shared tokens (relay_tokens.json)."), Ui.lp(ctx = c))
        listOf(phoneToken, caregiverToken).forEach { advanced.addView(it, Ui.lp(top = 12, ctx = c)) }
        conn.addView(Ui.button(c, "Advanced: tokens", Ui.Kind.SECONDARY, "Show or hide the token fields. Button.") {
            advanced.visibility = if (advanced.visibility == View.GONE) View.VISIBLE else View.GONE
        }, Ui.lp(top = 12, ctx = c))
        conn.addView(advanced, Ui.lp(top = 4, ctx = c))
        conn.addView(Ui.button(c, "Save connection", Ui.Kind.SECONDARY) { saveConnection() }, Ui.lp(top = 16, ctx = c))

        val mine = Ui.card(c, "My caregivers").also { col.addView(it, Ui.lp(top = 16, ctx = c)) }
        userLinks = FamilyLinks(c, "user") { phoneToken.setText(it) }
        mine.addView(userLinks.view, Ui.lp(ctx = c))

        val cared = Ui.card(c, "People I look after").also { col.addView(it, Ui.lp(top = 16, ctx = c)) }
        caregiverLinks = FamilyLinks(c, "caregiver") { caregiverToken.setText(it) }
        cared.addView(caregiverLinks.view, Ui.lp(ctx = c))

        val cg = Ui.card(c, "Caregiver dashboard").also { col.addView(it, Ui.lp(top = 16, ctx = c)) }
        cg.addView(Ui.caption(c, "For family and caregivers: live location, camera, alerts and device health."), Ui.lp(top = 4, ctx = c))
        val openDashboard = Ui.button(c, "Open caregiver dashboard", Ui.Kind.PRIMARY) {
            saveConnection()
            if (SettingsStore.getRelayToken(c, SettingsStore.ROLE_CAREGIVER).isBlank()) {
                android.widget.Toast.makeText(c, "Connect to the relay first. Opening Connect.", android.widget.Toast.LENGTH_LONG).show()
                startActivity(Intent(c, ConnectActivity::class.java))
            } else startActivity(Intent(c, CaregiverActivity::class.java))
        }
        // Debug builds only: hold the button to preview the dashboard layout with no token and no relay.
        if (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0) {
            openDashboard.setOnLongClickListener {
                startActivity(Intent(c, CaregiverActivity::class.java).putExtra(CaregiverActivity.EXTRA_DEBUG_PREVIEW, true))
                true
            }
        }
        cg.addView(openDashboard, Ui.lp(top = 12, ctx = c))

        // Relay address, name and tokens are set up by Connect; nobody needs to see them to be safe.
        conn.visibility = View.GONE
        col.addView(Ui.button(c, "Advanced: connection settings", Ui.Kind.SECONDARY, "Show or hide the relay address and name fields. Button.") {
            conn.visibility = if (conn.visibility == View.GONE) View.VISIBLE else View.GONE
        }, Ui.lp(top = 16, ctx = c))

        // One role per phone: hide what belongs to the other one. Views stay built so renderStatus() never meets an uninitialised field.
        val caregiverPhone = SettingsStore.getAppRole(c) == SettingsStore.APP_CAREGIVER
        (if (caregiverPhone) listOf(share, mine) else listOf(cared, cg)).forEach { it.visibility = View.GONE }
        col.addView(Ui.button(c, "Switch role", Ui.Kind.SECONDARY, "Switch role between navigation help and caregiver. Button.") {
            SettingsStore.setAppRole(c, "")
            startActivity(Intent(c, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK))
        }, Ui.lp(top = 16, ctx = c))

        col.addView(Ui.space(c, 48))
        setContentView(ScrollView(c).apply { setBackgroundColor(Ui.color(c, R.color.canvas)); addView(col); isFillViewport = true })
    }

    override fun onStart() {
        super.onStart(); LiveState.listen(onChange); renderStatus()
        // The Connect screen can change the relay and tokens; reload so "Save connection" never writes stale values back.
        url.setText(SettingsStore.getRelayUrl(this))
        phoneToken.setText(SettingsStore.getRelayToken(this, SettingsStore.ROLE_PHONE))
        caregiverToken.setText(SettingsStore.getRelayToken(this, SettingsStore.ROLE_CAREGIVER))
        if (SettingsStore.getAppRole(this) == SettingsStore.APP_CAREGIVER) caregiverLinks.refresh() else userLinks.refresh()
    }
    override fun onStop() { LiveState.unlisten(onChange); super.onStop() }

    private fun saveConnection() {
        SettingsStore.setRelayUrl(this, url.text.toString())
        SettingsStore.setRelayToken(this, SettingsStore.ROLE_PHONE, phoneToken.text.toString())
        SettingsStore.setRelayToken(this, SettingsStore.ROLE_CAREGIVER, caregiverToken.text.toString())
        SettingsStore.setCaregiverName(this, name.text.toString())
    }

    private fun toggleSharing() {
        saveConnection()
        if (LiveState.trackingRunning) {
            TrackingService.stop(this)
            toggle.postDelayed({ renderStatus() }, 300)
            return
        }
        if (SettingsStore.getRelayToken(this, SettingsStore.ROLE_PHONE).isBlank()) {
            android.widget.Toast.makeText(this, "This phone is not connected to the relay yet. Opening Connect.", android.widget.Toast.LENGTH_LONG).show()
            startActivity(Intent(this, ConnectActivity::class.java)); return
        }
        val wanted = mutableListOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
        if (Build.VERSION.SDK_INT >= 33) wanted += Manifest.permission.POST_NOTIFICATIONS
        permissions.launch(wanted.toTypedArray())
    }

    private fun renderStatus() {
        val c = this
        status.removeAllViews()
        val running = LiveState.trackingRunning
        toggle.text = if (running) "Stop sharing" else "Start sharing"
        toggle.contentDescription = if (running) "Stop sharing my location. Button." else "Start sharing my location and fall detection. Button."
        status.addView(Ui.kv(c, "Sharing", if (running) "On" else "Off", if (running) Level.OK else Level.OFF))
        val gps = LiveState.gpsState
        status.addView(Ui.kv(c, "GPS", when (gps) {
            "ok" -> "Good signal"; "weak" -> "Weak signal"; "searching" -> "Searching for signal"
            "denied" -> "Permission needed"; "disabled" -> "Location is switched off"; else -> "Starts with Navigation" },
            when (gps) { "ok" -> Level.OK; "weak", "searching" -> Level.WARN; "denied", "disabled" -> Level.BAD; else -> Level.OFF }))
        val relay = if (!running) null else LiveState.relayOk
        status.addView(Ui.kv(c, "Relay server", when (relay) { true -> "Connected"; false -> LiveState.relayError ?: "Not connected"; null -> "Not used" },
            when (relay) { true -> Level.OK; false -> Level.BAD; null -> Level.OFF }))
        if (LiveState.lastSuccessMs > 0) {
            status.addView(Ui.kv(c, "Last upload", DateUtils.getRelativeTimeSpanString(LiveState.lastSuccessMs, System.currentTimeMillis(), DateUtils.SECOND_IN_MILLIS).toString()))
        }
        val queued = Uplink.pending()
        if (queued > 0) status.addView(Ui.kv(c, "Waiting to send", "$queued", Level.WARN))
        status.visibility = View.VISIBLE
    }
}
