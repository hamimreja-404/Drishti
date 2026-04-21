package com.example.drishti

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
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
    val area: Float get() = bw * bh

    val position: String get() = when {
        cx < 0.33f -> "on your left"
        cx > 0.67f -> "on your right"
        else       -> "ahead"
    }

    val distanceLabel: String get() = when {
        area > 0.25f -> "very close"
        area > 0.10f -> "nearby"
        area > 0.04f -> "ahead"
        else         -> "far away"
    }

    val priorityScore: Float get() = confidence * (area * area)
    val isRelevant: Boolean get() = area > 0.015f
}

// ============================================================
//  YOLO OBJECT DETECTOR  (YOLOv11-Detection.tflite)
// ============================================================

class ObjectDetector(context: Context, private val usePixelCoords: Boolean = false) {

    private val interpreter: Interpreter
    private val inputSize = 640
    private val confThreshold = 0.45f
    private val minAreaThreshold = 0.015f

    private val highDangerLabels = setOf(
        "person", "car", "motorcycle", "bus", "truck", "bicycle",
        "dog", "stop sign", "traffic light"
    )

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

    init {
        val fd = context.assets.openFd("YOLOv11-Detection.tflite")
        val buffer = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(buffer, Interpreter.Options().apply { numThreads = 4 })
    }

    fun analyze(bitmap: Bitmap): List<DetectionResult> {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuf = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
            .apply { order(ByteOrder.nativeOrder()) }
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (px in pixels) {
            inputBuf.putFloat(((px shr 16) and 0xFF) / 255f)
            inputBuf.putFloat(((px shr 8)  and 0xFF) / 255f)
            inputBuf.putFloat(( px         and 0xFF) / 255f)
        }

        val outBoxes   = Array(1) { Array(8400) { FloatArray(4) } }
        val outScores  = Array(1) { FloatArray(8400) }
        val outClasses = Array(1) { FloatArray(8400) }
        interpreter.runForMultipleInputsOutputs(
            arrayOf(inputBuf),
            mapOf(0 to outBoxes, 1 to outScores, 2 to outClasses)
        )

        val raw = mutableListOf<DetectionResult>()
        for (i in 0 until 8400) {
            val score = outScores[0][i]
            if (score < confThreshold) continue
            val classId = outClasses[0][i].toInt().coerceIn(labels.indices)
            val box = outBoxes[0][i]
            val sc = if (usePixelCoords) inputSize.toFloat() else 1f
            val cx = box[0] / sc
            val cy = box[1] / sc
            val bw = box[2] / sc
            val bh = box[3] / sc
            val area = bw * bh
            val lbl = labels[classId]
            if (area < minAreaThreshold && lbl !in highDangerLabels) continue
            raw.add(DetectionResult(lbl, score, cx, cy, bw, bh))
        }

        return raw
            .sortedByDescending { it.priorityScore }
            .distinctBy { it.label }
            .take(5)
    }
}