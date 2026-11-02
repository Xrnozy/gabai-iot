package com.example.gabai

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.gabai.modules.AslModule
import com.example.gabai.services.Haptics
import com.example.gabai.services.TtsService
import java.util.Locale

class AslActivity : AppCompatActivity() {

    private lateinit var tts: TtsService
    private lateinit var haptics: Haptics
    private lateinit var module: AslModule

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraGranted = permissions[Manifest.permission.CAMERA] == true
        if (cameraGranted) {
            startAsl()
        } else {
            tts.speak("Camera permission denied. Cannot start A S L.")
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_asl)

        tts = TtsService(this, Locale.US)
        haptics = Haptics(this)

        val tvStatus = findViewById<TextView>(R.id.tv_asl_status)

        module = AslModule(this, tts).apply {
            onStatusText = { text ->
                runOnUiThread { tvStatus.text = text }
            }
        }

        // Back button is now an ImageView
        findViewById<ImageView>(R.id.btn_asl_back).setOnClickListener {
            haptics.click()
            tts.speak("Back")
            finish()
        }

        // Help — speaks how this screen works (accessibility-first:
        // a blind user can always learn the screen hands-free)
        findViewById<ImageView>(R.id.btn_asl_help).setOnClickListener {
            haptics.click()
            tts.speak(getString(R.string.asl_help))
        }

        // Speak button
        findViewById<android.widget.Button>(R.id.btn_asl_speak).setOnClickListener {
            haptics.click()
            val phrase = tvStatus.text?.toString().orEmpty()
            if (phrase.isNotBlank()) {
                tts.speak(phrase)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        tts.speak("A S L mode")
        ensurePermissionsAndStart()
    }

    override fun onPause() {
        super.onPause()
        module.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        module.stop()
        tts.shutdown()
    }

    private fun ensurePermissionsAndStart() {
        val cameraGranted =
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

        if (cameraGranted) {
            startAsl()
        } else {
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
        }
    }

    private fun startAsl() {
        module.start()
    }
}
