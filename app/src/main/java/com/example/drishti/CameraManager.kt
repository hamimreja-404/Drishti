package com.example.drishti

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

// ============================================================
//  CAMERA MANAGER (Replaces ConnectivityManager/UDP)
// ============================================================

class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var imageAnalyzer: ImageAnalysis? = null
    private var lastAnalysisTime = 0L

    fun startReceivingFrames(onFrameReceived: (Bitmap, Int?) -> Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Setup ImageAnalysis to get frames as fast as possible
            imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        val currentTime = System.currentTimeMillis()
                        if (currentTime - lastAnalysisTime >= 1500) {
                            lastAnalysisTime = currentTime
                            val bitmap = imageProxy.toBitmap()
                            val rotatedBitmap = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)
                            // LOCAL camera has no hardware sensor, so distance is always null
                            onFrameReceived(rotatedBitmap, null)
                        }

                        // Critical: close the proxy to receive the next frame
                        imageProxy.close()
                    }
                }

            // Default to the rear camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind any previous use cases before binding new ones
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    imageAnalyzer
                )
                Log.d("CameraManager", "Local Camera bound successfully")
            } catch (e: Exception) {
                Log.e("CameraManager", "Camera binding failed", e)
            }

        }, ContextCompat.getMainExecutor(context))
    }

    fun stopReceiving() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.get().unbindAll()
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}