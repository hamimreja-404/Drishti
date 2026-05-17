package com.example.drishti

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognizerIntent
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.Image
import androidx.compose.ui.graphics.asImageBitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import kotlinx.coroutines.channels.Channel

enum class AppScreen { CONNECT, DASHBOARD, ADD_FACE, NAVIGATION }
enum class CameraMode { NONE, LOCAL, EXTERNAL } // <-- ADD THIS
// ============================================================
//  FACE RECOGNITION MANAGER
// ============================================================
class FaceRecognitionManager(
    private val context: Context,
    private val voiceSystem: VoiceSystem,
    private val onStartListening: () -> Unit
) {
    private val interpreter: Interpreter

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setMinFaceSize(0.15f)
            .build()
    )

    private val embeddingSize = 192
    private val matchThreshold = 0.65f
    private val minFaceAreaForEnrollment = 0.12f
    private val recognitionCooldownMs = 7000L
    private val analysisInFlight = AtomicBoolean(false)

    val isEnrolling = AtomicBoolean(false)
    private var pendingEnrollEmbedding: FloatArray? = null
    private val knownEmbeddings = mutableMapOf<String, FloatArray>()
    private val recCooldowns = mutableMapOf<String, Long>()
    private val recentlySeen = java.util.Collections.synchronizedSet(mutableSetOf<String>())

    fun getRecentlySeenFaces(): List<String> {
        return synchronized(recentlySeen) {
            val faces = recentlySeen.toList()
            recentlySeen.clear()
            faces
        }
    }

    init {
        val fd = context.assets.openFd("mobilefacenet.tflite")
        val buf = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(buf, Interpreter.Options().apply {
            setUseXNNPACK(true)
            numThreads = 4
        })
        loadPersistedFaces()
    }

    fun startEnrollmentFlow() {
        isEnrolling.set(true)
        voiceSystem.speak(
            "Enrollment mode. Please face the camera and move close. I will capture your face.",
            priority = 1
        )
    }

    fun analyze(bitmap: Bitmap) {
        if (!analysisInFlight.compareAndSet(false, true)) return
        val image = InputImage.fromBitmap(bitmap, 0)
        detector.process(image).addOnSuccessListener { faces ->
            if (faces.isEmpty()) return@addOnSuccessListener
            val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                ?: return@addOnSuccessListener
            val box = face.boundingBox
            val faceArea = (box.width().toFloat() / bitmap.width) *
                    (box.height().toFloat() / bitmap.height)
            val faceBmp = safeCropFace(bitmap, box) ?: return@addOnSuccessListener
            val embedding = getEmbedding(faceBmp)

            if (isEnrolling.get()) {
                if (faceArea >= minFaceAreaForEnrollment) {
                    isEnrolling.set(false)
                    pendingEnrollEmbedding = embedding
                    voiceSystem.speak("Face captured. Now say the person's name.", priority = 1)
                    Handler(Looper.getMainLooper()).postDelayed({ onStartListening() }, 2200)
                } else {
                    voiceSystem.speak(
                        "Please come closer.",
                        priority = 3,
                        cooldownKey = "closer_hint",
                        cooldownMs = 4000
                    )
                }
            } else {
                if (faceArea >= 0.08f) {
                    val name = findBestMatch(embedding)
                    if (name != null) {
                        recentlySeen.add(name)
                        val now = System.currentTimeMillis()
                        if (now - (recCooldowns[name] ?: 0L) > recognitionCooldownMs) {
                            recCooldowns[name] = now
                            voiceSystem.speak("$name is in front of you.", priority = 3)
                        }
                    }
                }
            }
        }.addOnCompleteListener {
            analysisInFlight.set(false)
        }
    }

    fun onNameHeard(name: String?) {
        val emb = pendingEnrollEmbedding
        if (emb == null) {
            voiceSystem.speak("No face was captured. Please try again.", priority = 1)
            return
        }
        if (name.isNullOrBlank()) {
            voiceSystem.speak("Name not understood. Face not saved.", priority = 1)
            isEnrolling.set(false)
            pendingEnrollEmbedding = null
            return
        }
        val cleanName = name.trim().replaceFirstChar { it.uppercaseChar() }
        knownEmbeddings[cleanName] = emb
        persistFace(cleanName, emb)
        pendingEnrollEmbedding = null
        voiceSystem.speak("$cleanName has been registered successfully.", priority = 1)
    }

    private fun safeCropFace(bitmap: Bitmap, box: Rect): Bitmap? {
        val pad = (box.width() * 0.15f).toInt()
        val l = (box.left   - pad).coerceAtLeast(0)
        val t = (box.top    - pad).coerceAtLeast(0)
        val r = (box.right  + pad).coerceAtMost(bitmap.width)
        val b = (box.bottom + pad).coerceAtMost(bitmap.height)
        if (r <= l || b <= t) return null
        val crop = Bitmap.createBitmap(bitmap, l, t, r - l, b - t)
        return Bitmap.createScaledBitmap(crop, 112, 112, true)
    }

    private fun getEmbedding(face112: Bitmap): FloatArray {
        val buf = ByteBuffer.allocateDirect(1 * 112 * 112 * 3 * 4)
            .apply { order(ByteOrder.nativeOrder()) }
        val pixels = IntArray(112 * 112)
        face112.getPixels(pixels, 0, 112, 0, 0, 112, 112)
        for (px in pixels) {
            buf.putFloat(((px shr 16 and 0xFF) - 127.5f) / 128f)
            buf.putFloat(((px shr 8  and 0xFF) - 127.5f) / 128f)
            buf.putFloat(((px        and 0xFF) - 127.5f) / 128f)
        }
        val out = Array(1) { FloatArray(embeddingSize) }
        interpreter.run(buf, out)
        return out[0]
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return if (na == 0f || nb == 0f) 0f else dot / (sqrt(na) * sqrt(nb))
    }

    private fun findBestMatch(emb: FloatArray): String? {
        var best: String? = null
        var bestScore = matchThreshold
        for ((name, stored) in knownEmbeddings) {
            val score = cosineSimilarity(emb, stored)
            if (score > bestScore) { bestScore = score; best = name }
        }
        return best
    }

    private fun persistFace(name: String, emb: FloatArray) {
        context.getSharedPreferences("drishti_faces", Context.MODE_PRIVATE)
            .edit().putString("face_$name", emb.joinToString(",")).apply()
    }

    private fun loadPersistedFaces() {
        val prefs = context.getSharedPreferences("drishti_faces", Context.MODE_PRIVATE)
        prefs.all.forEach { (key, value) ->
            if (key.startsWith("face_")) {
                runCatching {
                    val emb = value.toString().split(",").map { it.toFloat() }.toFloatArray()
                    if (emb.size == embeddingSize) knownEmbeddings[key.removePrefix("face_")] = emb
                }
            }
        }
    }
}

// ============================================================
//  NAVIGATION MANAGER
// ============================================================
// ============================================================
//  NAVIGATION MANAGER
// ============================================================
class NavigationManager(private val voiceSystem: VoiceSystem) {

    var isNavigating = false
        private set
    var destination = ""
        private set

    private var lastNavTime = 0L
    private val navCooldownMs = 1800L

    fun start(dest: String) {
        destination = dest.trim().replaceFirstChar { it.uppercaseChar() }
        isNavigating = true
        voiceSystem.speak(
            "Navigation started. Heading to $destination. I will alert you to obstacles.",
            priority = 1
        )
    }

    fun stop() {
        isNavigating = false
        voiceSystem.speak("Navigation stopped.", priority = 1)
    }

    fun processDetections(detections: List<DetectionResult>) {
        if (!isNavigating) return
        val now = System.currentTimeMillis()
        if (now - lastNavTime < navCooldownMs) return
        lastNavTime = now

        val center = detections.filter { it.cx in 0.30f..0.70f && it.isRelevant }
        val left   = detections.filter { it.cx < 0.35f && it.isRelevant }
        val right  = detections.filter { it.cx > 0.65f && it.isRelevant }

        val instruction = when {
            center.any { it.distanceCategory == "CRITICAL_CLOSE" } -> {
                val obs = center.filter { it.distanceCategory == "CRITICAL_CLOSE" }.minByOrNull { it.estimatedDistanceMeters }!!
                "Stop! ${obs.label} directly ahead, very close."
            }
            center.any { it.distanceCategory == "NEARBY" } -> {
                val obs = center.minByOrNull { it.estimatedDistanceMeters }!!
                "${obs.label} ahead. Move slowly."
            }
            left.isNotEmpty() && right.isEmpty() -> {
                val obs = left.minByOrNull { it.estimatedDistanceMeters }!!
                "Bear right. ${obs.label} on your left."
            }
            right.isNotEmpty() && left.isEmpty() -> {
                val obs = right.minByOrNull { it.estimatedDistanceMeters }!!
                "Bear left. ${obs.label} on your right."
            }
            left.isNotEmpty() && right.isNotEmpty() ->
                "Obstacles on both sides. Proceed carefully."
            else -> "Path clear."
        }

        voiceSystem.speak(instruction, priority = 2, cooldownKey = "nav_$instruction", cooldownMs = 2500)
    }
}
// ============================================================
//  MAIN ACTIVITY
// ============================================================
class MainActivity : ComponentActivity() {

    private lateinit var voiceSystem: VoiceSystem

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        voiceSystem = VoiceSystem(this)
        setContent {
            MaterialTheme {
                Surface(color = Color(0xFF08080F), modifier = Modifier.fillMaxSize()) {
                    DrishtiApp(voiceSystem)
                }
            }
        }
    }

    override fun onDestroy() {
        if (::voiceSystem.isInitialized) voiceSystem.shutdown()
        super.onDestroy()
    }
}

// ============================================================
//  ROOT COMPOSABLE
// ============================================================
@Composable
fun DrishtiApp(voiceSystem: VoiceSystem) {
    val context = LocalContext.current
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()

    var screen       by remember { mutableStateOf(AppScreen.CONNECT) }
    var isConnecting by remember { mutableStateOf(false) }
    var isConnected  by remember { mutableStateOf(false) }
    var cameraMode   by remember { mutableStateOf(CameraMode.NONE) }
    var detections   by remember { mutableStateOf(listOf<DetectionResult>()) }
    var statusMsg    by remember { mutableStateOf("") }
    var showStream   by remember { mutableStateOf(false) }
    var audioPermissionGranted by remember { mutableStateOf(androidx.core.content.ContextCompat.checkSelfPermission(context, android.Manifest.permission.RECORD_AUDIO) == android.content.pm.PackageManager.PERMISSION_GRANTED) }

    val latestBitmapRef = remember { java.util.concurrent.atomic.AtomicReference<android.graphics.Bitmap?>(null) }
    var latestBitmapState by remember { mutableStateOf<android.graphics.Bitmap?>(null) }
    
    val objectDetector = remember { ObjectDetector(context) }
    val llmAssistant   = remember { LLMClient() }
    
    val cameraManager  = remember { CameraManager(context, lifecycleOwner) }
    val connectivityManager = remember { ConnectivityManager() }
    val navMgr         = remember { NavigationManager(voiceSystem) }
    val faceRef        = remember { mutableStateOf<FaceRecognitionManager?>(null) }
    
    var hardwareDistState by remember { mutableStateOf(0) }
    var isAssistantActive by remember { mutableStateOf(false) }
    var isListeningForQuery by remember { mutableStateOf(false) }
    var lastDetectionTime by remember { mutableStateOf(System.currentTimeMillis()) }
    var lastFallbackTime by remember { mutableStateOf(0L) }

    // Tie VoiceSystem callback to state
    DisposableEffect(voiceSystem) {
        voiceSystem.onSpeechDone = { isAssistantActive = false }
        onDispose { voiceSystem.onSpeechDone = null }
    }

    // WAKE WORD LISTENER
    val offlineSpeechMgr = remember {
        OfflineSpeechManager(context) { recognizedText ->
            val textLower = recognizedText.lowercase()
            
            // Continuous always-on listening for the wake word "drishti"
            if (textLower.contains("drishti")) {
                val query = textLower.substringAfter("drishti").trim()
                
                if (query.contains("navigate to")) {
                    val dest = query.substringAfter("navigate to").trim()
                    if (dest.isNotEmpty()) {
                        screen = AppScreen.NAVIGATION
                        navMgr.start(dest)
                    }
                } else if (query.contains("who is this") || query.contains("who is there")) {
                    // Trigger face recognition specifically
                    val faces = faceRef.value?.getRecentlySeenFaces()
                    if (faces.isNullOrEmpty()) {
                        voiceSystem.speak("I don't see any recognized faces right now.", priority = 2)
                    } else {
                        voiceSystem.speak("I see " + faces.joinToString(" and "), priority = 2)
                    }
                } else if (query.isNotEmpty() && !isAssistantActive) {
                    // General query through Gemini.
                    isAssistantActive = true
                    voiceSystem.speak("Thinking.", priority = 2)
                    coroutineScope.launch(Dispatchers.IO) {
                        val bmp = latestBitmapRef.get()
                        val answer = llmAssistant.askQuestionAboutScene(query, bmp)
                        withContext(Dispatchers.Main) {
                            voiceSystem.speak(answer, priority = 2)
                        }
                    }
                } else if (!isAssistantActive) {
                    // Just "Drishti" was said
                    isListeningForQuery = true
                    voiceSystem.speak("Yes, how can I help?", priority = 2)
                }
            } else if (isListeningForQuery && textLower.isNotBlank() && !isAssistantActive) {
                // Pick up the command after the wake word was acknowledged
                isListeningForQuery = false
                isAssistantActive = true
                voiceSystem.speak("Thinking.", priority = 2)
                coroutineScope.launch(Dispatchers.IO) {
                    val bmp = latestBitmapRef.get()
                    val answer = llmAssistant.askQuestionAboutScene(textLower, bmp)
                    withContext(Dispatchers.Main) {
                        voiceSystem.speak(answer, priority = 2)
                    }
                }
            }
        }.apply { initModel() }
    }

    val speechLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val text = if (result.resultCode == Activity.RESULT_OK)
            result.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)?.firstOrNull()
        else null
        when (screen) {
            AppScreen.ADD_FACE   -> faceRef.value?.onNameHeard(text)
            AppScreen.NAVIGATION -> {
                if (!text.isNullOrBlank()) navMgr.start(text)
                else voiceSystem.speak("Destination not understood. Tap mic to try again.", priority = 1)
            }
            AppScreen.DASHBOARD -> {
                if (!text.isNullOrBlank()) {
                    isAssistantActive = true
                    voiceSystem.speak("Thinking.", priority = 2)
                    coroutineScope.launch(Dispatchers.IO) {
                        val bmp = latestBitmapRef.get()
                        val answer = llmAssistant.askQuestionAboutScene(text, bmp)
                        withContext(Dispatchers.Main) {
                            voiceSystem.speak(answer, priority = 2)
                        }
                    }
                }
            }
            else -> Unit
        }
    }

    val faceMgr = remember {
        FaceRecognitionManager(context, voiceSystem) {
            speechLauncher.launch(
                Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_PROMPT, "Say the person's name clearly")
                }
            )
        }.also { faceRef.value = it }
    }

    val permLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        val cameraOk = grants[Manifest.permission.CAMERA] ?: false
        val audioOk = grants[Manifest.permission.RECORD_AUDIO] ?: false
        audioPermissionGranted = audioOk

        when (screen) {
            AppScreen.CONNECT -> {
                if (cameraOk) {
                    isConnecting = true
                    voiceSystem.speak("Activating mobile camera.", priority = 1)
                    Handler(Looper.getMainLooper()).postDelayed({
                        isConnected  = true
                        isConnecting = false
                        screen       = AppScreen.DASHBOARD
                        voiceSystem.speak("Connected. Vision system active.", priority = 1)
                    }, 1400)
                } else {
                    voiceSystem.speak("Camera permission is required to see.", priority = 1)
                    isConnecting = false
                    cameraMode = CameraMode.NONE
                }
            }
            AppScreen.ADD_FACE -> {
                if (audioOk) {
                    faceMgr.startEnrollmentFlow()
                } else {
                    voiceSystem.speak("Microphone permission is required to save names.", priority = 1)
                    screen = AppScreen.DASHBOARD
                }
            }
            AppScreen.NAVIGATION -> {
                if (audioOk) {
                    speechLauncher.launch(
                        Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                            putExtra(RecognizerIntent.EXTRA_PROMPT, "Where do you want to go?")
                        }
                    )
                } else {
                    voiceSystem.speak("Microphone permission is required to set a destination.", priority = 1)
                    screen = AppScreen.DASHBOARD
                }
            }
            AppScreen.DASHBOARD -> {
                if (audioOk) {
                    speechLauncher.launch(
                        Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                            putExtra(RecognizerIntent.EXTRA_PROMPT, "Ask the AI assistant...")
                        }
                    )
                } else {
                    voiceSystem.speak("Microphone permission is required.", priority = 1)
                }
            }
            else -> Unit
        }
    }

    // Keep the wake listener alive whenever microphone permission exists.
    LaunchedEffect(audioPermissionGranted) {
        if (audioPermissionGranted) {
            withContext(Dispatchers.IO) { offlineSpeechMgr.startListening() }
        } else {
            offlineSpeechMgr.stopListening()
        }
    }

    val detCooldowns = remember { mutableMapOf<String, Long>() }
    val frameChannel = remember { Channel<Bitmap>(Channel.CONFLATED) }
    // AUTO-SCAN FALLBACK FOR EXTERNAL CAMERA
    LaunchedEffect(cameraMode, isConnected, isAssistantActive) {
        if (cameraMode == CameraMode.EXTERNAL && isConnected && !isAssistantActive) {
            while(true) {
                kotlinx.coroutines.delay(1000)
                val now = System.currentTimeMillis()
                if (now - lastDetectionTime > 7000 && now - lastFallbackTime > 20000) {
                    lastFallbackTime = now
                    isAssistantActive = true
                    voiceSystem.speak("Scanning.", priority = 2)
                    coroutineScope.launch(Dispatchers.IO) {
                        val bmp = latestBitmapRef.get()
                        val answer = llmAssistant.askQuestionAboutScene("Describe the nearby indoor obstacles and people.", bmp)
                        withContext(Dispatchers.Main) {
                            voiceSystem.speak(answer, priority = 2)
                        }
                    }
                }
            }
        }
    }

    // ROUTER: Decide which camera provides the frames
    val yoloCooldowns = remember { mutableMapOf<String, Long>() }
    val yoloCooldownMs = 5000L

    LaunchedEffect(isConnected, cameraMode) {
        if (isConnected) {

            // 1. THE CONSUMER (The AI Pipeline)
            val indoorPriority = setOf(
                "person", "laptop", "bed", "chair", "dining table", "couch",
                "bottle", "cup", "door", "tv", "cell phone", "book",
                "keyboard", "mouse", "remote", "clock", "vase", "backpack", "suitcase"
            )
            launch {
                for (bitmap in frameChannel) {
                    latestBitmapRef.getAndSet(bitmap)
                    if (showStream) latestBitmapState = bitmap
                    
                    when (screen) {
                        AppScreen.ADD_FACE -> {
                            withContext(Dispatchers.Default) { faceMgr.analyze(bitmap) }
                        }
                        AppScreen.NAVIGATION -> {
                            coroutineScope {
                                val fastDetections = async(Dispatchers.Default) { 
                                    objectDetector.analyze(bitmap).filter { 
                                        indoorPriority.contains(it.label) || it.distanceCategory == "CRITICAL_CLOSE" 
                                    }
                                }
                                val rawDetections = fastDetections.await()
                                
                                withContext(Dispatchers.Main) {
                                    detections = rawDetections
                                    navMgr.processDetections(rawDetections)
                                }
                            }
                        }
                        AppScreen.DASHBOARD -> {
                            if (!isAssistantActive) {
                                coroutineScope {
                                    val faceJob = launch(Dispatchers.Default) { faceMgr.analyze(bitmap) }
                                    
                                    val fastDetections = async(Dispatchers.Default) {
                                        objectDetector.analyze(bitmap).filter { 
                                            indoorPriority.contains(it.label) 
                                        }
                                    }

                                    val rawDetections = fastDetections.await()
                                    faceJob.join()

                                    // ── YOLO Voice Feedback with per-object cooldown ──
                                    val now = System.currentTimeMillis()
                                    val toAnnounce = rawDetections.filter { det ->
                                        val lastTime = yoloCooldowns[det.label] ?: 0L
                                        now - lastTime > yoloCooldownMs
                                    }.take(2) // max 2 objects per frame to avoid overlap

                                    if (toAnnounce.isNotEmpty() && !voiceSystem.isSpeaking()) {
                                        toAnnounce.forEach { det ->
                                            yoloCooldowns[det.label] = now
                                        }
                                        val announcement = toAnnounce.joinToString(", then ") { det ->
                                            "${det.label} ${det.position}"
                                        }
                                        withContext(Dispatchers.Main) {
                                            voiceSystem.speak(announcement, priority = 3)
                                        }
                                    }

                                    // Gemini fallback only when YOLO has been quiet for a while.
                                    if (rawDetections.isEmpty() && now - lastFallbackTime > 18000) {
                                        lastFallbackTime = now
                                        launch(Dispatchers.Default) {
                                            val description = llmAssistant.askQuestionAboutScene(
                                                "Briefly describe nearby indoor objects or say path clear.",
                                                bitmap
                                            )
                                            withContext(Dispatchers.Main) {
                                                if (!isAssistantActive) {
                                                    voiceSystem.speak(description, priority = 3, cooldownKey = "gemini_fallback", cooldownMs = 18000)
                                                }
                                            }
                                        }
                                    }

                                    withContext(Dispatchers.Main) {
                                        detections = rawDetections
                                    }
                                }
                            }
                        }
                        AppScreen.CONNECT -> Unit
                    }
                }
            }

            // 2. THE PRODUCER (The Camera)
            if (cameraMode == CameraMode.LOCAL) {
                cameraManager.startReceivingFrames { bitmap, _ -> 
                    frameChannel.trySend(bitmap)
                }
            } else if (cameraMode == CameraMode.EXTERNAL) {
                connectivityManager.startReceivingFrames { bitmap, hardwareDistance ->
                    if (hardwareDistance != null) {
                        hardwareDistState = hardwareDistance
                        if (hardwareDistance <= 30) {
                            vibratePhoneDynamic(context, hardwareDistance)
                        }
                    }
                    frameChannel.trySend(bitmap)
                }
            }

        } else {
            cameraManager.stopReceiving()
            connectivityManager.stopReceiving()
        }
    }
    when (screen) {
        AppScreen.CONNECT -> ConnectScreen(
            isConnecting = isConnecting,
            onSelectLocal = {
                cameraMode = CameraMode.LOCAL
                permLauncher.launch(arrayOf(Manifest.permission.CAMERA))
            },
            onSelectExternal = {
                cameraMode = CameraMode.EXTERNAL
                isConnecting = true
                voiceSystem.speak("Connecting to wearable camera.", priority = 1)
                Handler(Looper.getMainLooper()).postDelayed({
                    isConnected = true
                    isConnecting = false
                    screen = AppScreen.DASHBOARD
                    voiceSystem.speak("Connected to external vision system.", priority = 1)
                }, 1500)
            }
        )
        AppScreen.DASHBOARD -> DashboardScreen(
            detections = detections,
            statusMsg  = if (cameraMode == CameraMode.LOCAL) "Mobile Camera Active" else "Wearable Camera Active",
            currentBitmap = latestBitmapState,
            showStream = showStream,
            onToggleStream = { showStream = !showStream },
            onAddFace  = {
                screen = AppScreen.ADD_FACE
                detections = emptyList()
                permLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
            },
            onNavigation = {
                screen = AppScreen.NAVIGATION
                detections = emptyList()
                permLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
            },
            onAskAi = {
                permLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
            },
            onDisconnect = {
                isConnected = false
                cameraMode  = CameraMode.NONE
                screen      = AppScreen.CONNECT
                detections  = emptyList()
                voiceSystem.speak("Camera disconnected.", priority = 1)
            }
        )
        AppScreen.ADD_FACE -> AddFaceScreen(
            onBack = {
                faceMgr.isEnrolling.set(false)
                screen = AppScreen.DASHBOARD
                voiceSystem.speak("Back to main mode.", priority = 1)
            }
        )
        AppScreen.NAVIGATION -> NavigationScreen(
            destination      = navMgr.destination,
            isNavigating     = navMgr.isNavigating,
            detections       = detections,
            onBack           = {
                navMgr.stop()
                screen = AppScreen.DASHBOARD
            },
            onSetDestination = {
                permLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
            }
        )
    }
}
// ============================================================
//  HELPER: GET LOCAL IP ADDRESS
// ============================================================
fun getLocalIpAddress(): String {
    try {
        val interfaces = java.net.NetworkInterface.getNetworkInterfaces()
        while (interfaces.hasMoreElements()) {
            val iface = interfaces.nextElement()
            val addresses = iface.inetAddresses
            while (addresses.hasMoreElements()) {
                val addr = addresses.nextElement()
                if (!addr.isLoopbackAddress && addr is java.net.Inet4Address) {
                    return addr.hostAddress ?: "Unknown"
                }
            }
        }
    } catch (e: Exception) {
        e.printStackTrace()
    }
    return "Unknown"
}

// ============================================================
//  SCREEN: CONNECT
// ============================================================
@Composable
fun ConnectScreen(
    isConnecting: Boolean,
    onSelectLocal: () -> Unit,
    onSelectExternal: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.radialGradient(listOf(Color(0xFF0D1A2E), Color(0xFF08080F)), radius = 1200f)),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("DRISHTI", fontSize = 60.sp, fontWeight = FontWeight.ExtraBold, color = Color.White, letterSpacing = 12.sp)
            Text("AI VISION ASSISTANT", fontSize = 12.sp, color = Color(0xFF4D90FE), letterSpacing = 5.sp)
            Spacer(Modifier.height(12.dp))
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .clip(RoundedCornerShape(10.dp))
                    .background(Color(0xFF0D1A2E))
                    .border(1.dp, Color(0xFF1E3A5F), RoundedCornerShape(10.dp))
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                Icon(Icons.Default.Wifi, contentDescription = null, tint = Color(0xFF4D90FE), modifier = Modifier.size(14.dp))
                Spacer(Modifier.width(8.dp))
                Text("PHONE IP: ", color = Color(0xFF546E7A), fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
                Text(getLocalIpAddress(), color = Color(0xFF90CAF9), fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.height(48.dp))

            if (isConnecting) {
                Text("CONNECTING…", color = Color(0xFF64B5F6), fontSize = 14.sp, fontWeight = FontWeight.Bold, letterSpacing = 2.sp)
                Spacer(Modifier.height(20.dp))
                LinearProgressIndicator(color = Color(0xFF42A5F5), trackColor = Color(0xFF0D1A2E), modifier = Modifier.width(180.dp).clip(RoundedCornerShape(50)))
            } else {
                Text("SELECT CAMERA SOURCE", color = Color(0xFF546E7A), fontSize = 13.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 2.sp)
                Spacer(Modifier.height(30.dp))

                Row(horizontalArrangement = Arrangement.spacedBy(20.dp)) {
                    // Local Camera Button
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Box(
                            modifier = Modifier
                                .size(120.dp).clip(CircleShape)
                                .background(Brush.radialGradient(listOf(Color(0xFF1E88E5), Color(0xFF0A3880))))
                                .border(2.dp, Color(0xFF42A5F5), CircleShape)
                                .clickable { onSelectLocal() },
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Default.PhoneAndroid, contentDescription = "Phone", tint = Color.White, modifier = Modifier.size(40.dp))
                        }
                        Spacer(Modifier.height(12.dp))
                        Text("MOBILE CAMERA", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                    }

                    // External Camera Button
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Box(
                            modifier = Modifier
                                .size(120.dp).clip(CircleShape)
                                .background(Brush.radialGradient(listOf(Color(0xFF00897B), Color(0xFF004D40))))
                                .border(2.dp, Color(0xFF26A69A), CircleShape)
                                .clickable { onSelectExternal() },
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Default.Wifi, contentDescription = "WiFi", tint = Color.White, modifier = Modifier.size(40.dp))
                        }
                        Spacer(Modifier.height(12.dp))
                        Text("WEARABLE (UDP)", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}
// ============================================================
//  SCREEN: DASHBOARD
// ============================================================
@Composable
fun DashboardScreen(
    detections    : List<DetectionResult>,
    statusMsg     : String,
    currentBitmap : Bitmap?,
    showStream    : Boolean,
    onToggleStream: () -> Unit,
    onAddFace     : () -> Unit,
    onNavigation  : () -> Unit,
    onAskAi       : () -> Unit,
    onDisconnect  : () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(Color(0xFF08080F), Color(0xFF0A1220))))
            .padding(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment     = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    "DRISHTI",
                    fontSize      = 30.sp,
                    fontWeight    = FontWeight.ExtraBold,
                    color         = Color.White,
                    letterSpacing = 5.sp
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(7.dp).clip(CircleShape).background(Color(0xFF4CAF50)))
                    Spacer(Modifier.width(6.dp))
                    Text("Local Camera Active", color = Color(0xFF81C784), fontSize = 12.sp)
                }
            }
            Box(
                modifier = Modifier
                    .size(44.dp).clip(CircleShape)
                    .background(Color(0xFF1A0000))
                    .border(1.dp, Color(0xFF7F0000), CircleShape)
                    .clickable { onDisconnect() },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.PowerSettingsNew,
                    contentDescription = "Disconnect",
                    tint     = Color(0xFFEF5350),
                    modifier = Modifier.size(22.dp)
                )
            }
        }

        if (statusMsg.isNotEmpty()) {
            Spacer(Modifier.height(8.dp))
            Text(statusMsg, color = Color(0xFFFFB74D), fontSize = 12.sp, textAlign = TextAlign.Center)
        }

        // ── Camera Stream Toggle ───────────────────────────────
        Spacer(Modifier.height(10.dp))
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(if (showStream) Color(0xFF0D2B1E) else Color(0xFF12131A))
                .border(1.dp, if (showStream) Color(0xFF43A047) else Color(0xFF1A2235), RoundedCornerShape(14.dp))
                .clickable { onToggleStream() }
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.Videocam,
                    contentDescription = null,
                    tint = if (showStream) Color(0xFF66BB6A) else Color(0xFF546E7A),
                    modifier = Modifier.size(20.dp)
                )
                Spacer(Modifier.width(10.dp))
                Column {
                    Text(
                        "CAMERA STREAM",
                        color = if (showStream) Color.White else Color(0xFF546E7A),
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.sp,
                        letterSpacing = 1.sp
                    )
                    Text(
                        if (showStream) "Live feed visible" else "Feed hidden (audio only)",
                        color = if (showStream) Color(0xFF81C784) else Color(0xFF37474F),
                        fontSize = 10.sp
                    )
                }
            }
            Switch(
                checked = showStream,
                onCheckedChange = { onToggleStream() },
                colors = SwitchDefaults.colors(
                    checkedThumbColor = Color(0xFF4CAF50),
                    checkedTrackColor = Color(0xFF1B5E20),
                    uncheckedThumbColor = Color(0xFF546E7A),
                    uncheckedTrackColor = Color(0xFF1A2235)
                )
            )
        }

        // ── Live Camera Preview ───────────────────────────────
        if (showStream && currentBitmap != null) {
            Spacer(Modifier.height(10.dp))
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(220.dp)
                    .clip(RoundedCornerShape(18.dp))
                    .border(1.5.dp, Color(0xFF43A047), RoundedCornerShape(18.dp))
            ) {
                Image(
                    bitmap = currentBitmap.asImageBitmap(),
                    contentDescription = "Camera Stream",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = androidx.compose.ui.layout.ContentScale.Crop
                )
                // Overlay: LIVE badge
                Box(
                    modifier = Modifier
                        .padding(8.dp)
                        .clip(RoundedCornerShape(6.dp))
                        .background(Color(0xFFEF5350).copy(alpha = 0.85f))
                        .padding(horizontal = 8.dp, vertical = 3.dp)
                        .align(Alignment.TopStart)
                ) {
                    Text("● LIVE", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                }
            }
        }

        Spacer(Modifier.height(28.dp))

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            ActionCard(
                title       = "ADD FACE",
                subtitle    = "Register person",
                icon        = Icons.Default.Face,
                bgColor     = Color(0xFF0D2B45),
                accentColor = Color(0xFF42A5F5),
                modifier    = Modifier.weight(1f),
                onClick     = onAddFace
            )
            ActionCard(
                title       = "NAVIGATE",
                subtitle    = "Voice guidance",
                icon        = Icons.Default.Navigation,
                bgColor     = Color(0xFF0A2010),
                accentColor = Color(0xFF66BB6A),
                modifier    = Modifier.weight(1f),
                onClick     = onNavigation
            )
        }

        Spacer(Modifier.height(14.dp))

        ActionCard(
            title       = "ASK AI ASSISTANT",
            subtitle    = "Ask what's around you",
            icon        = Icons.Default.ChatBubble,
            bgColor     = Color(0xFF2E0D45),
            accentColor = Color(0xFFAB47BC),
            modifier    = Modifier.fillMaxWidth(),
            onClick     = onAskAi
        )

        Spacer(Modifier.height(20.dp))

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .background(Color(0xFF0C0F18), RoundedCornerShape(20.dp))
                .border(1.dp, Color(0xFF1A2235), RoundedCornerShape(20.dp))
                .padding(18.dp)
                .verticalScroll(rememberScrollState())
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier.size(8.dp).clip(CircleShape)
                        .background(if (detections.isNotEmpty()) Color(0xFF4CAF50) else Color(0xFF37474F))
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    "LIVE DETECTION",
                    color         = Color(0xFF546E7A),
                    fontSize      = 11.sp,
                    fontWeight    = FontWeight.ExtraBold,
                    letterSpacing = 2.sp
                )
            }
            Spacer(Modifier.height(14.dp))
            if (detections.isEmpty()) {
                Box(
                    Modifier.fillMaxWidth().padding(vertical = 32.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text("✓  Path is clear", color = Color(0xFF37474F), fontSize = 18.sp)
                }
            } else {
                detections.forEach { det ->
                    DetectionRow(det)
                    Spacer(Modifier.height(8.dp))
                }
            }
        }

        Spacer(Modifier.height(16.dp))

        OutlinedButton(
            onClick  = onDisconnect,
            colors   = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFFEF5350)),
            border   = BorderStroke(1.dp, Brush.horizontalGradient(listOf(Color(0xFF7F0000), Color(0xFFEF5350)))),
            shape    = RoundedCornerShape(14.dp),
            modifier = Modifier.fillMaxWidth().height(52.dp)
        ) {
            Icon(Icons.Default.PowerSettingsNew, null, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(8.dp))
            Text("STOP CAMERA", fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
        }
    }
}

// ── Detection Row ─────────────────────────────────────────────
// ── Detection Row ─────────────────────────────────────────────
@Composable
fun DetectionRow(det: DetectionResult) {
    val arrowIcon = when {
        det.cx < 0.33f -> "◀"
        det.cx > 0.67f -> "▶"
        else           -> "▲"
    }
    val distColor = when (det.distanceCategory.replace("_", " ")) {
        "CRITICAL_CLOSE" -> Color(0xFFEF5350)
        "NEARBY"         -> Color(0xFFFFB74D)
        else             -> Color(0xFF546E7A)
    }

    val friendlyDistance = det.distanceCategory.replace("_", " ").replace("_", " ")

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xFF111827), RoundedCornerShape(12.dp))
            .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalAlignment     = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(arrowIcon, color = Color(0xFF42A5F5), fontSize = 20.sp)
            Spacer(Modifier.width(12.dp))
            Column {
                Text(det.label.uppercase(), color = Color.White, fontWeight = FontWeight.ExtraBold, fontSize = 16.sp)
                Text(det.position, color = Color(0xFF78909C), fontSize = 11.sp)
            }
        }
        Column(horizontalAlignment = Alignment.End) {
            Text(friendlyDistance, color = distColor, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            Text("${(det.confidence * 100).toInt()}%", color = Color(0xFF546E7A), fontSize = 10.sp)
        }
    }
}
// ── Action Card ───────────────────────────────────────────────
@Composable
fun ActionCard(
    title      : String,
    subtitle   : String,
    icon       : ImageVector,
    bgColor    : Color,
    accentColor: Color,
    modifier   : Modifier,
    onClick    : () -> Unit
) {
    Box(
        modifier = modifier
            .height(156.dp)
            .clip(RoundedCornerShape(20.dp))
            .background(bgColor)
            .border(1.5.dp, accentColor.copy(alpha = 0.55f), RoundedCornerShape(20.dp))
            .clickable { onClick() }
            .padding(20.dp),
        contentAlignment = Alignment.BottomStart
    ) {
        Column {
            Icon(icon, null, tint = accentColor, modifier = Modifier.size(38.dp))
            Spacer(Modifier.height(14.dp))
            Text(title, color = Color.White, fontWeight = FontWeight.ExtraBold, fontSize = 15.sp)
            Text(subtitle, color = accentColor.copy(alpha = 0.75f), fontSize = 11.sp)
        }
    }
}

// ============================================================
//  SCREEN: ADD FACE
// ============================================================
@Composable
fun AddFaceScreen(onBack: () -> Unit) {

    val anim = rememberInfiniteTransition(label = "face_scan")
    val alpha by anim.animateFloat(
        0.3f, 1f,
        infiniteRepeatable(tween(750), RepeatMode.Reverse),
        label = "a"
    )
    val outerScale by anim.animateFloat(
        1f, 1.06f,
        infiniteRepeatable(tween(1100, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "s"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(Color(0xFF08080F), Color(0xFF0D1A2E))))
    ) {
        Column(
            modifier = Modifier.fillMaxSize().padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, null, tint = Color(0xFF64B5F6))
                }
                Text(
                    "ADD FACE",
                    color         = Color.White,
                    fontWeight    = FontWeight.ExtraBold,
                    fontSize      = 20.sp,
                    letterSpacing = 3.sp
                )
            }

            Spacer(Modifier.weight(1f))

            Box(contentAlignment = Alignment.Center) {
                Box(
                    Modifier.size(240.dp).scale(outerScale).clip(CircleShape)
                        .background(Color(0xFF0D47A1).copy(alpha = alpha * 0.25f))
                        .border(2.dp, Color(0xFF42A5F5).copy(alpha = alpha), CircleShape)
                )
                Box(
                    Modifier.size(190.dp).clip(CircleShape)
                        .background(Color(0xFF1565C0).copy(alpha = 0.20f))
                        .border(1.dp, Color(0xFF64B5F6).copy(alpha = alpha * 0.85f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        Icons.Default.Face,
                        contentDescription = null,
                        tint     = Color(0xFF42A5F5).copy(alpha = alpha),
                        modifier = Modifier.size(90.dp)
                    )
                }
            }

            Spacer(Modifier.height(40.dp))

            Text(
                "SCANNING FOR FACE",
                color         = Color(0xFF42A5F5),
                fontSize      = 14.sp,
                fontWeight    = FontWeight.ExtraBold,
                letterSpacing = 3.sp
            )
            Spacer(Modifier.height(10.dp))
            Text(
                "Face the camera directly\nMove close until the system captures you",
                color     = Color(0xFF78909C),
                fontSize  = 13.sp,
                textAlign = TextAlign.Center
            )

            Spacer(Modifier.height(36.dp))

            listOf(
                Icons.Default.LightMode         to "Ensure good, even lighting",
                Icons.Default.CenterFocusStrong to "Face the camera straight on",
                Icons.Default.Landscape         to "Clear, non-busy background"
            ).forEach { (icon, tip) ->
                Row(
                    Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 5.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(icon, null, tint = Color(0xFF1E4976), modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(10.dp))
                    Text(tip, color = Color(0xFF37474F), fontSize = 12.sp)
                }
            }

            Spacer(Modifier.weight(1f))

            OutlinedButton(
                onClick  = onBack,
                colors   = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFF546E7A)),
                shape    = RoundedCornerShape(14.dp),
                modifier = Modifier.fillMaxWidth().height(50.dp)
            ) {
                Text("CANCEL", fontWeight = FontWeight.Bold, letterSpacing = 2.sp)
            }
            Spacer(Modifier.height(8.dp))
        }
    }
}

// ============================================================
//  SCREEN: NAVIGATION
// ============================================================
@Composable
fun NavigationScreen(
    destination     : String,
    isNavigating    : Boolean,
    detections      : List<DetectionResult>,
    onBack          : () -> Unit,
    onSetDestination: () -> Unit
) {
    val centerObstacle = detections.firstOrNull { it.cx in 0.28f..0.72f && it.isRelevant }

    val pathColor = when {
        centerObstacle != null && centerObstacle.distanceCategory == "CRITICAL_CLOSE" -> Color(0xFFEF5350)
        centerObstacle != null                                                        -> Color(0xFFFFB74D)
        else                                                                          -> Color(0xFF4CAF50)
    }
    val pathIcon = when {
        centerObstacle != null && centerObstacle.distanceCategory == "CRITICAL_CLOSE" -> Icons.Default.Warning
        centerObstacle != null                                                        -> Icons.Default.SlowMotionVideo
        else                                                                          -> Icons.Default.CheckCircle
    }
    val pathStatus = when {
        centerObstacle != null && centerObstacle.distanceCategory == "CRITICAL_CLOSE" -> "STOP — OBSTACLE CLOSE"
        centerObstacle != null                                                        -> "OBSTACLE AHEAD"
        else                                                                          -> "PATH CLEAR"
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(Color(0xFF080F09), Color(0xFF0A1510))))
            .padding(20.dp)
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, null, tint = Color(0xFF66BB6A))
            }
            Text(
                "NAVIGATION",
                color         = Color.White,
                fontWeight    = FontWeight.ExtraBold,
                fontSize      = 20.sp,
                letterSpacing = 3.sp
            )
            Spacer(Modifier.weight(1f))
            if (isNavigating) {
                Row(
                    Modifier.clip(RoundedCornerShape(20.dp))
                        .background(Color(0xFF1B5E20).copy(alpha = 0.7f))
                        .padding(horizontal = 10.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(Modifier.size(6.dp).clip(CircleShape).background(Color(0xFF4CAF50)))
                    Spacer(Modifier.width(6.dp))
                    Text("LIVE", color = Color(0xFF81C784), fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
            }
        }

        Spacer(Modifier.height(20.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF0A1A0C), RoundedCornerShape(16.dp))
                .border(1.dp, Color(0xFF2E7D32), RoundedCornerShape(16.dp))
                .padding(18.dp)
        ) {
            Column {
                Text(
                    "HEADING TO",
                    color         = Color(0xFF66BB6A),
                    fontSize      = 10.sp,
                    fontWeight    = FontWeight.ExtraBold,
                    letterSpacing = 2.sp
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    if (destination.isNotEmpty()) destination.uppercase() else "— NOT SET —",
                    color      = if (destination.isNotEmpty()) Color.White else Color(0xFF37474F),
                    fontSize   = 26.sp,
                    fontWeight = FontWeight.ExtraBold
                )
            }
        }

        Spacer(Modifier.height(16.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth().height(100.dp)
                .background(pathColor.copy(alpha = 0.12f), RoundedCornerShape(16.dp))
                .border(1.5.dp, pathColor.copy(alpha = 0.5f), RoundedCornerShape(16.dp)),
            contentAlignment = Alignment.Center
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(pathIcon, null, tint = pathColor, modifier = Modifier.size(32.dp))
                Spacer(Modifier.width(14.dp))
                Text(
                    pathStatus,
                    color         = pathColor,
                    fontSize      = 18.sp,
                    fontWeight    = FontWeight.ExtraBold,
                    letterSpacing = 1.sp
                )
            }
        }

        Spacer(Modifier.height(16.dp))

        Column(
            modifier = Modifier
                .fillMaxWidth().weight(1f)
                .background(Color(0xFF0A0F0A), RoundedCornerShape(16.dp))
                .border(1.dp, Color(0xFF1B5E20), RoundedCornerShape(16.dp))
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            Text(
                "OBSTACLES DETECTED",
                color         = Color(0xFF546E7A),
                fontSize      = 10.sp,
                fontWeight    = FontWeight.ExtraBold,
                letterSpacing = 2.sp
            )
            Spacer(Modifier.height(12.dp))
            if (detections.isEmpty()) {
                Box(
                    Modifier.fillMaxWidth().padding(vertical = 20.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text("No obstacles", color = Color(0xFF2E4A30), fontSize = 16.sp)
                }
            } else {
                detections.forEach { det ->
                    DetectionRow(det)
                    Spacer(Modifier.height(8.dp))
                }
            }
        }

        Spacer(Modifier.height(16.dp))

        Button(
            onClick  = onSetDestination,
            colors   = ButtonDefaults.buttonColors(containerColor = Color(0xFF1B5E20)),
            shape    = RoundedCornerShape(14.dp),
            modifier = Modifier.fillMaxWidth().height(58.dp)
        ) {
            Icon(Icons.Default.Mic, null, modifier = Modifier.size(22.dp))
            Spacer(Modifier.width(10.dp))
            Text(
                if (isNavigating) "CHANGE DESTINATION" else "SET DESTINATION",
                fontWeight    = FontWeight.ExtraBold,
                fontSize      = 15.sp,
                letterSpacing = 1.sp
            )
        }

        Spacer(Modifier.height(8.dp))

        OutlinedButton(
            onClick  = onBack,
            colors   = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFF546E7A)),
            shape    = RoundedCornerShape(14.dp),
            modifier = Modifier.fillMaxWidth().height(50.dp)
        ) {
            Text("STOP & GO BACK", fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
        }
    }
}

// Helper function to trigger dynamic vibration based on distance
fun vibratePhoneDynamic(context: Context, distanceCm: Int) {
    if (distanceCm > 30 || distanceCm < 0) return
    val amplitude = 255 - ((distanceCm / 30f) * 200).toInt()
    val safeAmplitude = amplitude.coerceIn(1, 255)
    
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        val vibrator = vibratorManager.defaultVibrator
        vibrator.vibrate(VibrationEffect.createOneShot(100, safeAmplitude))
    } else {
        @Suppress("DEPRECATION")
        val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(100, safeAmplitude))
        } else {
            vibrator.vibrate(100)
        }
    }
}
