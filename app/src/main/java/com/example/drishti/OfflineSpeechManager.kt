package com.example.drishti

import android.content.Context
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import org.vosk.android.StorageService
import org.json.JSONObject
import java.io.IOException

class OfflineSpeechManager(
    private val context: Context,
    private val onResult: (String) -> Unit
) : RecognitionListener {

    private var speechService: SpeechService? = null
    private var model: Model? = null
    private var startPending = false
    private var keepAlive = false

    /**
     * Call once (e.g. in onCreate / LaunchedEffect) to unpack and load the model.
     * The "model-en-us" folder must be placed in assets/.
     */
    fun initModel() {
        StorageService.unpack(
            context,
            "model-en-in",          // folder name inside assets/
            "model",                // destination name in filesDir
            { unpackedModel: Model ->
                model = unpackedModel
                if (startPending) {
                    startPending = false
                    startListening()
                }
            },
            { exception: IOException ->
                exception.printStackTrace()
            }
        )
    }

    /** Start continuous offline recognition. */
    fun startListening() {
        keepAlive = true
        val m = model
        if (m == null) {
            startPending = true
            return
        }
        try {
            if (speechService != null) return // Already running
            val recognizer = Recognizer(m, 16000.0f)
            speechService = SpeechService(recognizer, 16000.0f)
            speechService!!.startListening(this)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    /** Stop recognition and release the service (model is kept for reuse). */
    fun stopListening() {
        keepAlive = false
        speechService?.stop()
        speechService?.shutdown()
        speechService = null
    }

    /** Release everything including the model. Call from onDestroy. */
    fun destroy() {
        stopListening()
        model?.close()
        model = null
    }

    // ── RecognitionListener callbacks ────────────────────────────

    override fun onResult(hypothesis: String?) {
        if (hypothesis.isNullOrBlank()) return
        runCatching {
            val text = JSONObject(hypothesis).optString("text", "").trim()
            if (text.isNotEmpty()) onResult(text)
        }
    }

    override fun onPartialResult(hypothesis: String?) {
        // Partial results can be used for real-time UI feedback if needed
    }

    override fun onFinalResult(hypothesis: String?) {
        // Called after stopListening(); treat same as onResult
        onResult(hypothesis)
    }

    override fun onError(exception: Exception?) {
        exception?.printStackTrace()
        speechService = null
        if (keepAlive) startListening()
    }

    override fun onTimeout() {
        speechService = null
        if (keepAlive) startListening()
        // Recognition timed out — restart if needed
    }
}
