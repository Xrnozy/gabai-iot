package com.example.gabai

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.ScrollView
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.TtsService
import com.example.gabai.ui.Ui
import java.util.Locale

/**
 * First launch only: who is using this phone? The top button is for the person who needs navigation help (one tap, nothing to
 * type, and the prompt is read aloud); the caregiver asks for one thing, their name, because the person they look after sees it.
 */
class RoleActivity : BaseActivity() {

    private var tts: TtsService? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val c = this
        val col = Ui.column(c, 24)
        col.addView(Ui.title(c, "Welcome to GabAi"), Ui.lp(top = 24, ctx = c))
        col.addView(Ui.body(c, "Who will use this phone?"), Ui.lp(top = 8, ctx = c))

        val name = Ui.field(c, "Your name, shown to the person you look after", "").apply { visibility = View.GONE }
        val go = Ui.button(c, "Continue", Ui.Kind.PRIMARY) { continueAsCaregiver(name) }.apply { visibility = View.GONE }

        col.addView(Ui.button(c, "I need navigation help", Ui.Kind.PRIMARY, "I need navigation help. Button.") {
            SettingsStore.setAppRole(c, SettingsStore.APP_USER)
            startActivity(Intent(c, MainActivity::class.java))
            finish()
        }, Ui.lp(h = Ui.dp(c, 120), top = 24, ctx = c))
        col.addView(Ui.button(c, "I look after someone", Ui.Kind.SECONDARY, "I look after someone. Button.") {
            name.visibility = View.VISIBLE
            go.visibility = View.VISIBLE
            name.requestFocus()
        }, Ui.lp(h = Ui.dp(c, 120), top = 16, ctx = c))
        col.addView(name, Ui.lp(top = 16, ctx = c))
        col.addView(go, Ui.lp(top = 12, ctx = c))

        setContentView(ScrollView(c).apply { setBackgroundColor(Ui.color(c, R.color.canvas)); addView(col); isFillViewport = true })

        tts = TtsService(this, Locale.US).also {
            it.speak("Welcome to GabAi. If you need navigation help, tap the top button. If you look after someone, tap the bottom button.")
        }
    }

    private fun continueAsCaregiver(name: android.widget.EditText) {
        val n = name.text.toString().trim()
        if (n.isEmpty()) { name.error = "Enter your name"; name.requestFocus(); return }
        SettingsStore.setCaregiverName(this, n)
        SettingsStore.setAppRole(this, SettingsStore.APP_CAREGIVER)
        startActivity(Intent(this, FamilyActivity::class.java))
        finish()
    }
}
