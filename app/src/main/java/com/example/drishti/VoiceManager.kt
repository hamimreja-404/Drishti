package com.example.drishti

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale
import java.util.UUID

// ============================================================
//  VOICE OUTPUT MANAGER
// ============================================================

class VoiceManager(context: Context) : TextToSpeech.OnInitListener {

    private val tts = TextToSpeech(context, this)
    private val cooldownMap = mutableMapOf<String, Long>()
    private var isReady = false

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
            tts.setSpeechRate(1.05f)
            tts.setPitch(0.95f)
            isReady = true
        }
    }

    fun speak(
        text: String,
        flush: Boolean = false,
        cooldownKey: String? = null,
        cooldownMs: Long = 0L
    ) {
        if (!isReady) return
        if (cooldownKey != null) {
            val now = System.currentTimeMillis()
            if (now - (cooldownMap[cooldownKey] ?: 0L) < cooldownMs) return
            cooldownMap[cooldownKey] = now
        }
        val mode = if (flush) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        tts.speak(text, mode, null, UUID.randomUUID().toString())
    }

    fun isSpeaking(): Boolean = tts.isSpeaking

    fun shutdown() {
        if (isReady) { tts.stop(); tts.shutdown() }
    }
}