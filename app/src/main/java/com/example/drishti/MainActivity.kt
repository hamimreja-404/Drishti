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

enum class AppScreen { CONNECT, DASHBOARD, ADD_FACE, NAVIGATION }

// ============================================================
//  FACE RECOGNITION MANAGER
// ============================================================
class FaceRecognitionManager(
    private val context: Context,
    private val voiceManager: VoiceManager,
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
    private val recognitionCooldownMs = 8000L

    val isEnrolling = AtomicBoolean(false)
    private var pendingEnrollEmbedding: FloatArray? = null
    private val knownEmbeddings = mutableMapOf<String, FloatArray>()
    private val recCooldowns = mutableMapOf<String, Long>()

    init {
        val fd = context.assets.openFd("mobilefacenet.tflite")
        val buf = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(buf, Interpreter.Options().apply { numThreads = 2 })
        loadPersistedFaces()
    }

    fun startEnrollmentFlow() {
        isEnrolling.set(true)
        voiceManager.speak(
            "Enrollment mode. Please face the camera and move close. I will capture your face.",
            flush = true
        )
    }

    fun analyze(bitmap: Bitmap) {
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
                    voiceManager.speak("Face captured. Now say the person's name.", flush = true)
                    Handler(Looper.getMainLooper()).postDelayed({ onStartListening() }, 2200)
                } else {
                    voiceManager.speak(
                        "Please come closer.",
                        flush = false,
                        cooldownKey = "closer_hint",
                        cooldownMs = 4000
                    )
                }
            } else {
                if (faceArea >= 0.08f) {
                    val name = findBestMatch(embedding)
                    if (name != null) {
                        val now = System.currentTimeMillis()
                        if (now - (recCooldowns[name] ?: 0L) > recognitionCooldownMs) {
                            recCooldowns[name] = now
                            voiceManager.speak("$name is in front of you.", flush = false)
                        }
                    }
                }
            }
        }
    }

    fun onNameHeard(name: String?) {
        val emb = pendingEnrollEmbedding
        if (emb == null) {
            voiceManager.speak("No face was captured. Please try again.", flush = true)
            return
        }
        if (name.isNullOrBlank()) {
            voiceManager.speak("Name not understood. Face not saved.", flush = true)
            isEnrolling.set(false)
            pendingEnrollEmbedding = null
            return
        }
        val cleanName = name.trim().replaceFirstChar { it.uppercaseChar() }
        knownEmbeddings[cleanName] = emb
        persistFace(cleanName, emb)
        pendingEnrollEmbedding = null
        voiceManager.speak("$cleanName has been registered successfully.", flush = true)
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
class NavigationManager(private val voiceManager: VoiceManager) {

    var isNavigating = false
        private set
    var destination = ""
        private set

    private var lastNavTime = 0L
    private val navCooldownMs = 2500L

    fun start(dest: String) {
        destination = dest.trim().replaceFirstChar { it.uppercaseChar() }
        isNavigating = true
        voiceManager.speak(
            "Navigation started. Heading to $destination. I will alert you to obstacles.",
            flush = true
        )
    }

    fun stop() {
        isNavigating = false
        voiceManager.speak("Navigation stopped.", flush = true)
    }

    fun processDetections(detections: List<DetectionResult>) {
        if (!isNavigating) return
        val now = System.currentTimeMillis()
        if (now - lastNavTime < navCooldownMs) return
        lastNavTime = now

        val center = detections.filter { it.cx in 0.28f..0.72f && it.isRelevant }
        val left   = detections.filter { it.cx < 0.35f && it.isRelevant }
        val right  = detections.filter { it.cx > 0.65f && it.isRelevant }

        val instruction = when {
            center.any { it.area > 0.20f } -> {
                val obs = center.maxByOrNull { it.area }!!
                "Stop! ${obs.label} directly ahead, very close."
            }
            center.isNotEmpty() -> {
                val obs = center.maxByOrNull { it.priorityScore }!!
                if (obs.area > 0.08f) "${obs.label} ahead. Move slowly."
                else "Caution. ${obs.label} in path."
            }
            left.isNotEmpty() && right.isEmpty() -> {
                val obs = left.maxByOrNull { it.priorityScore }!!
                "Bear right. ${obs.label} on your left."
            }
            right.isNotEmpty() && left.isEmpty() -> {
                val obs = right.maxByOrNull { it.priorityScore }!!
                "Bear left. ${obs.label} on your right."
            }
            left.isNotEmpty() && right.isNotEmpty() ->
                "Obstacles on both sides. Proceed carefully."
            else -> "Path clear. Continue forward."
        }

        voiceManager.speak(instruction, flush = false)
    }
}

// ============================================================
//  MAIN ACTIVITY
// ============================================================
class MainActivity : ComponentActivity() {

    private lateinit var voiceManager: VoiceManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        voiceManager = VoiceManager(this)
        setContent {
            MaterialTheme {
                Surface(color = Color(0xFF08080F), modifier = Modifier.fillMaxSize()) {
                    DrishtiApp(voiceManager)
                }
            }
        }
    }

    override fun onDestroy() {
        if (::voiceManager.isInitialized) voiceManager.shutdown()
        super.onDestroy()
    }
}

// ============================================================
//  ROOT COMPOSABLE
// ============================================================
@Composable
fun DrishtiApp(voiceManager: VoiceManager) {
    val context = LocalContext.current
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()

    var screen       by remember { mutableStateOf(AppScreen.CONNECT) }
    var isConnecting by remember { mutableStateOf(false) }
    var isConnected  by remember { mutableStateOf(false) }
    var detections   by remember { mutableStateOf(listOf<DetectionResult>()) }
    var statusMsg    by remember { mutableStateOf("") }

    val objectDetector = remember { ObjectDetector(context) }
    val cameraManager  = remember { CameraManager(context, lifecycleOwner) }
    val navMgr         = remember { NavigationManager(voiceManager) }
    val faceRef        = remember { mutableStateOf<FaceRecognitionManager?>(null) }

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
                else voiceManager.speak("Destination not understood. Tap mic to try again.", true)
            }
            else -> Unit
        }
    }

    val faceMgr = remember {
        FaceRecognitionManager(context, voiceManager) {
            speechLauncher.launch(
                Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_PROMPT, "Say the person's name clearly")
                }
            )
        }.also { faceRef.value = it }
    }

    // UPDATED: Now requests both CAMERA and RECORD_AUDIO
    val permLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        // Safely check if the specific permission was granted, defaulting to false if it wasn't even requested
        val cameraOk = grants[Manifest.permission.CAMERA] ?: false
        val audioOk = grants[Manifest.permission.RECORD_AUDIO] ?: false

        when (screen) {
            AppScreen.CONNECT -> {
                if (cameraOk) {
                    isConnecting = true
                    voiceManager.speak("Activating camera.", true)
                    Handler(Looper.getMainLooper()).postDelayed({
                        isConnected  = true
                        isConnecting = false
                        screen       = AppScreen.DASHBOARD
                        voiceManager.speak("Connected. Vision system active.", true)
                    }, 1400)
                } else {
                    voiceManager.speak("Camera permission is required to see.", true)
                    isConnecting = false
                }
            }
            AppScreen.ADD_FACE -> {
                if (audioOk) {
                    faceMgr.startEnrollmentFlow()
                } else {
                    voiceManager.speak("Microphone permission is required to save names.", true)
                    screen = AppScreen.DASHBOARD // Kick them back if denied
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
                    voiceManager.speak("Microphone permission is required to set a destination.", true)
                    screen = AppScreen.DASHBOARD
                }
            }
            else -> Unit
        }
    }

    val detCooldowns = remember { mutableMapOf<String, Long>() }

    // UPDATED: Replaced UDP with CameraManager
    LaunchedEffect(isConnected) {
        if (isConnected) {
            var lastAnalysisTime = 0L
            cameraManager.startReceivingFrames { bitmap ->
                val now = System.currentTimeMillis()
                // Throttle frames to process roughly 1 per second to save battery
                if (now - lastAnalysisTime < 1000) return@startReceivingFrames
                lastAnalysisTime = now

                coroutineScope.launch {
                    when (screen) {
                        AppScreen.ADD_FACE -> {
                            faceMgr.analyze(bitmap)
                        }
                        AppScreen.NAVIGATION -> {
                            val dets = withContext(Dispatchers.Default) { objectDetector.analyze(bitmap) }
                            withContext(Dispatchers.Main) { detections = dets }
                            navMgr.processDetections(dets)
                        }
                        AppScreen.DASHBOARD -> {
                            val dets = withContext(Dispatchers.Default) { objectDetector.analyze(bitmap) }
                            faceMgr.analyze(bitmap)
                            withContext(Dispatchers.Main) {
                                detections = dets
                                val t = System.currentTimeMillis()
                                for (det in dets) {
                                    val last = detCooldowns[det.label] ?: 0L
                                    if (t - last > 5000L) {
                                        detCooldowns[det.label] = t
                                        voiceManager.speak(
                                            "${det.label}, ${det.distanceLabel}, ${det.position}",
                                            flush = false
                                        )
                                    }
                                }
                            }
                        }
                        AppScreen.CONNECT -> Unit
                    }
                }
            }
        } else {
            cameraManager.stopReceiving()
        }
    }

    when (screen) {
        AppScreen.CONNECT -> ConnectScreen(
            isConnecting = isConnecting,
            onConnect = {
                // FIX: ONLY ask for Camera when starting the app!
                permLauncher.launch(arrayOf(Manifest.permission.CAMERA))
            }
        )
        AppScreen.DASHBOARD -> DashboardScreen(
            detections   = detections,
            statusMsg    = statusMsg,
            onAddFace    = {
                screen = AppScreen.ADD_FACE
                detections = emptyList()
                // FIX: ONLY ask for Microphone when saving a face
                permLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
            },
            onNavigation = {
                screen = AppScreen.NAVIGATION
                detections = emptyList()
                // FIX: ONLY ask for Microphone when starting navigation
                permLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
            },
            onDisconnect = {
                isConnected = false
                screen      = AppScreen.CONNECT
                detections  = emptyList()
                voiceManager.speak("Camera disconnected.", true)
            }
        )
        AppScreen.ADD_FACE -> AddFaceScreen(
            onBack = {
                faceMgr.isEnrolling.set(false)
                screen = AppScreen.DASHBOARD
                voiceManager.speak("Back to main mode.", true)
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
                // FIX: ONLY ask for Microphone when setting a new destination
                permLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
            }
        )
    }
}

// ============================================================
//  SCREEN: CONNECT
// ============================================================
@Composable
fun ConnectScreen(isConnecting: Boolean, onConnect: () -> Unit) {

    val transition = rememberInfiniteTransition(label = "connect_anim")

    val outerRingScale by transition.animateFloat(
        1f, if (isConnecting) 1.22f else 1.08f,
        infiniteRepeatable(tween(1100, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "outer"
    )
    val middleRingScale by transition.animateFloat(
        1f, if (isConnecting) 1.14f else 1.04f,
        infiniteRepeatable(tween(900, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "mid"
    )
    val glowAlpha by transition.animateFloat(
        0.25f, if (isConnecting) 0.9f else 0.5f,
        infiniteRepeatable(tween(1300), RepeatMode.Reverse),
        label = "glow"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.radialGradient(
                    listOf(Color(0xFF0D1A2E), Color(0xFF08080F)),
                    radius = 1200f
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {

            Text(
                "DRISHTI",
                fontSize      = 60.sp,
                fontWeight    = FontWeight.ExtraBold,
                color         = Color.White,
                letterSpacing = 12.sp
            )
            Text(
                "AI VISION ASSISTANT",
                fontSize      = 12.sp,
                color         = Color(0xFF4D90FE),
                letterSpacing = 5.sp,
                modifier      = Modifier.padding(bottom = 80.dp)
            )

            Box(contentAlignment = Alignment.Center) {
                Box(
                    modifier = Modifier
                        .size(260.dp).scale(outerRingScale).clip(CircleShape)
                        .background(Color(0xFF1565C0).copy(alpha = glowAlpha * 0.18f))
                        .border(1.dp, Color(0xFF42A5F5).copy(alpha = glowAlpha * 0.6f), CircleShape)
                )
                Box(
                    modifier = Modifier
                        .size(210.dp).scale(middleRingScale).clip(CircleShape)
                        .background(Color(0xFF1976D2).copy(alpha = 0.22f))
                        .border(1.5.dp, Color(0xFF64B5F6).copy(alpha = glowAlpha * 0.8f), CircleShape)
                )
                Box(
                    modifier = Modifier
                        .size(168.dp).clip(CircleShape)
                        .background(Brush.radialGradient(listOf(Color(0xFF1E88E5), Color(0xFF0A3880))))
                        .border(3.dp, Color(0xFF42A5F5), CircleShape)
                        .clickable(enabled = !isConnecting) { onConnect() },
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(
                            Icons.Default.PowerSettingsNew,
                            contentDescription = "Connect",
                            tint     = Color.White,
                            modifier = Modifier.size(54.dp)
                        )
                        Spacer(Modifier.height(6.dp))
                        Text(
                            "START",
                            color         = Color.White,
                            fontSize      = 11.sp,
                            fontWeight    = FontWeight.ExtraBold,
                            letterSpacing = 2.5.sp
                        )
                    }
                }
            }

            Spacer(Modifier.height(50.dp))

            Text(
                if (isConnecting) "ACTIVATING CAMERA…" else "TAP TO START VISION SYSTEM",
                color         = if (isConnecting) Color(0xFF64B5F6) else Color(0xFF546E7A),
                fontSize      = 13.sp,
                fontWeight    = FontWeight.SemiBold,
                letterSpacing = 2.sp
            )

            if (isConnecting) {
                Spacer(Modifier.height(20.dp))
                LinearProgressIndicator(
                    color      = Color(0xFF42A5F5),
                    trackColor = Color(0xFF0D1A2E),
                    modifier   = Modifier.width(180.dp).clip(RoundedCornerShape(50))
                )
            }

            Spacer(Modifier.height(80.dp))
            Text("Using Local Device Camera", color = Color(0xFF263238), fontSize = 11.sp, letterSpacing = 1.sp)
        }
    }
}

// ============================================================
//  SCREEN: DASHBOARD
// ============================================================
@Composable
fun DashboardScreen(
    detections  : List<DetectionResult>,
    statusMsg   : String,
    onAddFace   : () -> Unit,
    onNavigation: () -> Unit,
    onDisconnect: () -> Unit
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
@Composable
fun DetectionRow(det: DetectionResult) {
    val arrowIcon = when {
        det.cx < 0.33f -> "◀"
        det.cx > 0.67f -> "▶"
        else           -> "▲"
    }
    val distColor = when (det.distanceLabel) {
        "very close" -> Color(0xFFEF5350)
        "nearby"     -> Color(0xFFFFB74D)
        "ahead"      -> Color(0xFFFFF176)
        else         -> Color(0xFF546E7A)
    }
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
            Text(det.distanceLabel.uppercase(), color = distColor, fontSize = 11.sp, fontWeight = FontWeight.Bold)
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
    val centerObstacle = detections.firstOrNull { it.cx in 0.28f..0.72f && it.area > 0.05f }
    val pathColor = when {
        centerObstacle != null && centerObstacle.area > 0.20f -> Color(0xFFEF5350)
        centerObstacle != null                                  -> Color(0xFFFFB74D)
        else                                                    -> Color(0xFF4CAF50)
    }
    val pathIcon = when {
        centerObstacle != null && centerObstacle.area > 0.20f -> Icons.Default.Warning
        centerObstacle != null                                  -> Icons.Default.SlowMotionVideo
        else                                                    -> Icons.Default.CheckCircle
    }
    val pathStatus = when {
        centerObstacle != null && centerObstacle.area > 0.20f -> "STOP — OBSTACLE CLOSE"
        centerObstacle != null                                  -> "OBSTACLE AHEAD"
        else                                                    -> "PATH CLEAR"
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