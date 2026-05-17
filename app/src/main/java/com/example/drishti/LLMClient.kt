package com.example.drishti

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL

class LLMClient {

    private val apiKey = BuildConfig.GEMINI_API_KEY
    private val models = listOf("gemini-2.5-flash", "gemini-2.0-flash")

    suspend fun askQuestionAboutScene(userQuestion: String, bitmap: Bitmap?): String {
        return withContext(Dispatchers.IO) {
            try {
                val requestBody = JSONObject()
                val contents = JSONArray()
                val parts = JSONArray()

                // Add text part
                val textPart = JSONObject()
                val prompt = "You are Drishti, a fast indoor navigation assistant for a visually impaired user. The user asks: '$userQuestion'. Answer in one short, clear sentence. Prioritize people, chairs, tables, doors, laptops, stairs, and immediate obstacles. Do not mention that you are looking at an image."
                textPart.put("text", prompt)
                parts.put(textPart)

                // Add image part if available
                if (bitmap != null) {
                    val base64Image = encodeImageToBase64(bitmap)
                    val inlineData = JSONObject()
                    inlineData.put("mimeType", "image/jpeg")
                    inlineData.put("data", base64Image)

                    val imagePart = JSONObject()
                    imagePart.put("inlineData", inlineData)
                    parts.put(imagePart)
                }

                val content = JSONObject()
                content.put("parts", parts)
                contents.put(content)
                requestBody.put("contents", contents)

                var lastError = ""
                for (model in models) {
                    val result = sendRequest(model, requestBody)
                    if (result != null) return@withContext result
                    lastError = model
                }

                Log.e("LLMClient", "All Gemini models failed. Last tried: $lastError")
                "I encountered an error connecting to the AI system."
            } catch (e: Exception) {
                Log.e("LLMClient", "Exception", e)
                "I encountered an error processing your request."
            }
        }
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
                    null
                }
            } catch (e: Exception) {
                Log.e("LLMClient", "$model Exception", e)
                null
            }
    }

    private fun encodeImageToBase64(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()
        // Resize bitmap to reduce payload size and speed up the request
        val scaledBitmap = scaleBitmapDown(bitmap, 800)
        scaledBitmap.compress(Bitmap.CompressFormat.JPEG, 70, outputStream)
        val byteArray = outputStream.toByteArray()
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
