package com.example.drishti

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

// ============================================================
//  DATA MODEL
// ============================================================

data class DetectionResult(
    val label: String,
    val confidence: Float,
    val cx: Float,
    val cy: Float,
    val bw: Float,
    val bh: Float,
) {
    // Extensive real-world height mapping (in meters)
    private val realWorldHeights = mapOf(
        "person" to 1.7f, "car" to 1.5f, "bus" to 3.0f, "truck" to 3.0f,
        "motorcycle" to 1.1f, "bicycle" to 1.0f, "chair" to 0.9f,
        "dog" to 0.6f, "cat" to 0.3f, "bottle" to 0.25f, "cup" to 0.15f,
        "door" to 2.1f, "fire hydrant" to 0.8f, "stop sign" to 2.0f,
        "potted plant" to 0.5f, "tv" to 0.6f, "laptop" to 0.25f,
        "bed" to 0.6f, "dining table" to 0.8f, "couch" to 0.8f,
        "backpack" to 0.45f, "umbrella" to 1.0f, "traffic light" to 0.8f
    )

    // FIX 1: Math Bug Resolved. bh is now a percentage (0.0 to 1.0)
    // K = 1.0f works well for standard mobile phone camera lenses
    val estimatedDistanceMeters: Float get() {
        val realHeight = realWorldHeights[label] ?: 1.0f
        val k = 1.0f
        return (realHeight / bh) * k
    }

    // Adjusted distance thresholds
    val distanceCategory: String get() = when {
        estimatedDistanceMeters < 1.2f -> "CRITICAL_CLOSE"
        estimatedDistanceMeters < 3.5f -> "NEARBY"
        else -> "FAR"
    }

    val position: String get() = when {
        cx < 0.35f -> "on left"
        cx > 0.65f -> "on right"
        else       -> "ahead"
    }

    val isRelevant: Boolean get() = distanceCategory != "FAR"
}

// ============================================================
//  YOLO OBJECT DETECTOR
// ============================================================

class ObjectDetector(context: Context) {

    private val interpreter: Interpreter
    private val inputSize = 640

    private val confThreshold = 0.38f

    private val labels = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    )

    private val indoorLabels = setOf(
        "person", "chair", "couch", "dining table", "table", "bed", "door", "tv", "laptop",
        "keyboard", "mouse", "remote", "cell phone", "book", "clock", "vase", "backpack",
        "suitcase", "bottle", "cup", "sink", "refrigerator", "microwave", "oven", "toaster",
        "potted plant"
    )
    private val inputBuf = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        .apply { order(ByteOrder.nativeOrder()) }
    private val pixels = IntArray(inputSize * inputSize)
    private val outBoxes = Array(1) { Array(8400) { FloatArray(4) } }
    private val outScores = Array(1) { FloatArray(8400) }
    private val outClasses = Array(1) { FloatArray(8400) }

    init {
        val fd = context.assets.openFd("YOLOv11-Detection.tflite")
        val buffer = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)

        val compatList = CompatibilityList()
        val options = Interpreter.Options().apply {
            setUseXNNPACK(true)
            numThreads = 4
            if (compatList.isDelegateSupportedOnThisDevice) {
                addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
            }
        }
        interpreter = Interpreter(buffer, options)
    }

    private fun letterboxBitmap(bitmap: Bitmap, targetSize: Int): Bitmap {
        val scale = minOf(targetSize.toFloat() / bitmap.width, targetSize.toFloat() / bitmap.height)
        val scaledWidth = (bitmap.width * scale).toInt()
        val scaledHeight = (bitmap.height * scale).toInt()

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)

        val outputBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(outputBitmap)
        canvas.drawColor(android.graphics.Color.BLACK)

        val left = (targetSize - scaledWidth) / 2f
        val top = (targetSize - scaledHeight) / 2f

        canvas.drawBitmap(scaledBitmap, left, top, null)
        return outputBitmap
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        if (degrees == 0f) return bitmap
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun runInference(bitmap: Bitmap): List<DetectionResult> {
        val resized = letterboxBitmap(bitmap, inputSize)
        inputBuf.rewind()
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (px in pixels) {
            inputBuf.putFloat(((px shr 16) and 0xFF) / 255f)
            inputBuf.putFloat(((px shr 8) and 0xFF) / 255f)
            inputBuf.putFloat((px and 0xFF) / 255f)
        }

        interpreter.runForMultipleInputsOutputs(
            arrayOf(inputBuf),
            mapOf(0 to outBoxes, 1 to outScores, 2 to outClasses)
        )

        val raw = mutableListOf<DetectionResult>()
        for (i in 0 until 8400) {
            val score = outScores[0][i]
            if (score < confThreshold) continue
            val classId = outClasses[0][i].toInt().coerceIn(labels.indices)
            val lbl = labels[classId]
            if (lbl !in indoorLabels) continue
            val cx = outBoxes[0][i][0] / inputSize.toFloat()
            val cy = outBoxes[0][i][1] / inputSize.toFloat()
            val bw = outBoxes[0][i][2] / inputSize.toFloat()
            val bh = outBoxes[0][i][3] / inputSize.toFloat()
            if (lbl == "person" && bw > (bh * 1.1f)) continue
            raw.add(DetectionResult(lbl, score, cx, cy, bw, bh))
        }
        return raw
    }

    // AUTO-ROTATION: Try 4 orientations and pick the one with the best detections.
    // Solves Pi Zero / ESP32 cameras that may be mounted sideways.
    // Uses a "fast-path": if 0° rotation works, skip the other 3 rotations.
    private var bestRotation = 0f

    fun analyze(bitmap: Bitmap): List<DetectionResult> {
        // Fast path: use cached best rotation first
        val fastResult = runInference(rotateBitmap(bitmap, bestRotation))
        if (fastResult.isNotEmpty()) {
            return fastResult
                .filter { it.isRelevant }
                .sortedBy { it.estimatedDistanceMeters }
                .distinctBy { it.label }
                .take(6)
        }

        // If fast path returns nothing, try all 4 orientations to find the best
        val rotations = listOf(0f, 90f, 180f, 270f).filter { it != bestRotation }
        var bestResults = emptyList<DetectionResult>()
        var bestScore = 0f

        for (deg in rotations) {
            val rotated = rotateBitmap(bitmap, deg)
            val results = runInference(rotated)
            val totalScore = results.sumOf { it.confidence.toDouble() }.toFloat()
            if (totalScore > bestScore) {
                bestScore = totalScore
                bestResults = results
                bestRotation = deg // cache this rotation for future frames
            }
        }

        return bestResults
            .filter { it.isRelevant }
            .sortedBy { it.estimatedDistanceMeters }
            .distinctBy { it.label }
            .take(6)
    }
}
