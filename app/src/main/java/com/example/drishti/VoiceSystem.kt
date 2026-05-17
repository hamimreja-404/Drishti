package com.example.drishti

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale
import java.util.UUID

class VoiceSystem(context: Context) : TextToSpeech.OnInitListener {

    private val tts = TextToSpeech(context, this)
    private val cooldownMap = mutableMapOf<String, Long>()
    private var isReady = false

    // Priority tracking
    // 1: Critical (System, Navigation)
    // 2: LLM Q&A
    // 3: Routine (Object/Face)
    @Volatile
    private var currentSpeakingPriority = 0

    private var lastSpokenText = ""
    private var lastSpokenTime = 0L

    var onSpeechDone: (() -> Unit)? = null

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val preferredLocale = Locale("en", "IN")
            val languageResult = tts.setLanguage(preferredLocale)
            if (languageResult == TextToSpeech.LANG_MISSING_DATA || languageResult == TextToSpeech.LANG_NOT_SUPPORTED) {
                tts.language = Locale.UK
            }
            val clearVoice = tts.voices
                ?.filter { !it.isNetworkConnectionRequired && it.locale.language == "en" }
                ?.minByOrNull { voice ->
                    val localeScore = if (voice.locale.country == "IN" || voice.locale.country == "GB") 0 else 1
                    val qualityScore = -voice.quality
                    localeScore * 10 + qualityScore
                }
            if (clearVoice != null) tts.voice = clearVoice
            tts.setSpeechRate(1.03f)
            tts.setPitch(1.05f)
            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    // Extract priority from utteranceId (e.g., "1_uuid")
                    utteranceId?.split("_")?.firstOrNull()?.toIntOrNull()?.let {
                        currentSpeakingPriority = it
                    }
                }
                override fun onDone(utteranceId: String?) {
                    currentSpeakingPriority = 0
                    onSpeechDone?.invoke()
                }
                override fun onError(utteranceId: String?) {
                    currentSpeakingPriority = 0
                    onSpeechDone?.invoke()
                }
            })
            isReady = true
        }
    }

    fun speak(
        text: String,
        priority: Int,
        cooldownKey: String? = null,
        cooldownMs: Long = 0L
    ) {
        val cleanText = text
            .replace(Regex("\\s+"), " ")
            .replace("CRITICAL_CLOSE", "very close")
            .replace("NEARBY", "nearby")
            .trim()
        if (!isReady || cleanText.isBlank()) return

        val now = System.currentTimeMillis()

        // Reject duplicates within 3 seconds for the same text
        if (cleanText == lastSpokenText && (now - lastSpokenTime < 3000)) {
            return
        }

        if (cooldownKey != null) {
            if (now - (cooldownMap[cooldownKey] ?: 0L) < cooldownMs) return
        }

        // Priority Logic: Higher priority = lower number (1 is highest)
        if (currentSpeakingPriority in 1..2) {
            if (priority > currentSpeakingPriority) {
                // Ignore lower priority messages if a higher one is currently speaking
                return
            }
        }

        // Always interrupt (flush) if the new message is higher or equal priority
        val mode = if (currentSpeakingPriority == 0 || priority <= currentSpeakingPriority) {
            TextToSpeech.QUEUE_FLUSH
        } else {
            TextToSpeech.QUEUE_ADD
        }

        if (cooldownKey != null) {
            cooldownMap[cooldownKey] = now
        }

        lastSpokenText = cleanText
        lastSpokenTime = now

        val utteranceId = "${priority}_${UUID.randomUUID()}"
        val params = Bundle().apply {
            putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, 1.0f)
            putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId)
        }
        tts.speak(cleanText, mode, params, utteranceId)
    }

    fun isSpeaking(): Boolean = tts.isSpeaking

    fun shutdown() {
        if (isReady) { tts.stop(); tts.shutdown() }
    }
}
