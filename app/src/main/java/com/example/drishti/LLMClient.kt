package com.example.drishti

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL

class LLMClient {

    private val apiKey = BuildConfig.GEMINI_API_KEY
    private val models = listOf("gemini-2.5-flash-lite", "gemini-2.5-flash")
    private var lastFailureReason = ""

    suspend fun askQuestionAboutScene(userQuestion: String, bitmap: Bitmap?): String {
        return withContext(Dispatchers.IO) {
            try {
                if (apiKey.isBlank()) {
                    Log.e("LLMClient", "Missing GEMINI_API_KEY in BuildConfig")
                    return@withContext "Gemini API key is missing."
                }

                val imageRequest = buildRequest(userQuestion, bitmap)
                askWithModels(imageRequest)?.let { return@withContext it }

                if (bitmap != null) {
                    Log.w("LLMClient", "Image request failed; retrying Gemini without image. Last failure: $lastFailureReason")
                    askWithModels(buildRequest(userQuestion, null))?.let { answer ->
                        return@withContext "$answer I could not upload the camera image."
                    }
                }

                if (lastFailureReason.isNotBlank()) {
                    "Gemini connection failed: $lastFailureReason."
                } else {
                    "I encountered an error connecting to the AI system."
                }
            } catch (e: Exception) {
                Log.e("LLMClient", "Exception", e)
                "AI request failed: ${e.javaClass.simpleName}."
            }
        }
    }

    private fun buildRequest(userQuestion: String, bitmap: Bitmap?): JSONObject {
        val requestBody = JSONObject()
        val contents = JSONArray()
        val parts = JSONArray()

        val textPart = JSONObject()
        val prompt = "You are Drishti, a fast indoor navigation assistant for a visually impaired user. The user asks: '$userQuestion'. Answer in one short, clear sentence. Prioritize people, chairs, tables, doors, laptops, stairs, and immediate obstacles. Do not mention that you are looking at an image."
        textPart.put("text", prompt)
        parts.put(textPart)

        if (bitmap != null) {
            val base64Image = encodeImageToBase64(bitmap)
            val inlineData = JSONObject()
            inlineData.put("mime_type", "image/jpeg")
            inlineData.put("data", base64Image)

            val imagePart = JSONObject()
            imagePart.put("inline_data", inlineData)
            parts.put(imagePart)
        }

        val content = JSONObject()
        content.put("parts", parts)
        contents.put(content)
        requestBody.put("contents", contents)
        requestBody.put(
            "generationConfig",
            JSONObject()
                .put("maxOutputTokens", 80)
                .put("temperature", 0.2)
        )
        return requestBody
    }

    private fun askWithModels(requestBody: JSONObject): String? {
        for (model in models) {
            val result = sendRequest(model, requestBody)
            if (result != null) return result
        }
        Log.e("LLMClient", "All Gemini models failed. Last failure: $lastFailureReason")
        return null
    }

    private fun sendRequest(model: String, requestBody: JSONObject): String? {
        val apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=$apiKey"
        return try {
                val url = URL(apiUrl)
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                connection.connectTimeout = 8000
                connection.readTimeout = 15000
                connection.doOutput = true

                connection.outputStream.use { os ->
                    val input = requestBody.toString().toByteArray(Charsets.UTF_8)
                    os.write(input, 0, input.size)
                }

                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val response = connection.inputStream.bufferedReader().use { it.readText() }
                    val jsonResponse = JSONObject(response)
                    val candidates = jsonResponse.getJSONArray("candidates")
                    if (candidates.length() > 0) {
                        val contentObj = candidates.getJSONObject(0).getJSONObject("content")
                        val partsArr = contentObj.getJSONArray("parts")
                        if (partsArr.length() > 0) {
                            return partsArr.getJSONObject(0).getString("text").trim()
                        }
                    }
                    "I couldn't generate a description for that."
                } else {
                    val errorResponse = connection.errorStream?.bufferedReader()?.use { it.readText() }
                    Log.e("LLMClient", "$model API Error: $responseCode - $errorResponse")
                    lastFailureReason = when (responseCode) {
                        400 -> "bad image request"
                        401, 403 -> "API key permission denied"
                        429 -> "Gemini quota exceeded"
                        else -> "HTTP $responseCode"
                    }
                    null
                }
            } catch (e: SocketTimeoutException) {
                lastFailureReason = "timeout"
                Log.e("LLMClient", "$model timeout", e)
                null
            } catch (e: IOException) {
                lastFailureReason = e.javaClass.simpleName
                Log.e("LLMClient", "$model network exception", e)
                null
            } catch (e: Exception) {
                lastFailureReason = e.javaClass.simpleName
                Log.e("LLMClient", "$model Exception", e)
                null
            }
    }

    private fun encodeImageToBase64(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()
        // Resize bitmap to reduce payload size and speed up the request
        val scaledBitmap = scaleBitmapDown(bitmap, 512)
        scaledBitmap.compress(Bitmap.CompressFormat.JPEG, 60, outputStream)
        val byteArray = outputStream.toByteArray()
        Log.d("LLMClient", "Sending Gemini image: ${scaledBitmap.width}x${scaledBitmap.height}, ${byteArray.size} bytes")
        return Base64.encodeToString(byteArray, Base64.NO_WRAP)
    }

    private fun scaleBitmapDown(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val originalWidth = bitmap.width
        val originalHeight = bitmap.height
        var resizedWidth = maxDimension
        var resizedHeight = maxDimension

        if (originalHeight > originalWidth) {
            resizedHeight = maxDimension
            resizedWidth = (resizedHeight * originalWidth.toFloat() / originalHeight.toFloat()).toInt()
        } else if (originalWidth > originalHeight) {
            resizedWidth = maxDimension
            resizedHeight = (resizedWidth * originalHeight.toFloat() / originalWidth.toFloat()).toInt()
        }

        return if (resizedWidth == originalWidth && resizedHeight == originalHeight) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, false)
        }
    }
}
