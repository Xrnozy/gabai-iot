package com.example.gabai

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale

class TtsManager(context: Context) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = null
    private var isInitialized = false
    private var lastSpokenText: String = ""

    init {
        tts = TextToSpeech(context, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                println("TTS: Language not supported")
            } else {
                isInitialized = true
            }
        } else {
            println("TTS: Initialization failed")
        }
    }

    fun speak(text: String) {
        if (!isInitialized || text.isBlank()) return
        
        val newText = text.trim().lowercase(Locale.ROOT)
        
        // If it's already speaking this exact text, don't interrupt it.
        // This prevents the "st- st- stop" stuttering if we receive multiple identical commands in a row.
        if (newText == lastSpokenText && tts?.isSpeaking == true) {
            return
        }
        
        lastSpokenText = newText
        
        // QUEUE_FLUSH interrupts whatever is currently playing
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
    }
}
