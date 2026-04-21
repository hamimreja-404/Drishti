package com.example.drishti

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.SocketTimeoutException

class ConnectivityManager {

    private var udpSocket: DatagramSocket? = null
    private var isReceiving = false

    // Standard port for ESP32/UDP streaming, change if your hardware uses a different one
    private val PORT = 6868
    private val MAX_PACKET_SIZE = 65507 // Max UDP packet size

    suspend fun startReceivingFrames(onFrameReceived: (Bitmap) -> Unit) {
        withContext(Dispatchers.IO) {
            try {
                udpSocket = DatagramSocket(PORT)
                udpSocket?.soTimeout = 2000 // 2 seconds timeout
                isReceiving = true

                val buffer = ByteArray(MAX_PACKET_SIZE)
                val packet = DatagramPacket(buffer, buffer.size)

                Log.d("ConnectivityManager", "Started listening for UDP frames on port $PORT")

                while (isReceiving) {
                    try {
                        udpSocket?.receive(packet)
                        // Decode the received JPEG byte array into a Bitmap
                        val bitmap = BitmapFactory.decodeByteArray(packet.data, 0, packet.length)
                        if (bitmap != null) {
                            withContext(Dispatchers.Main) {
                                onFrameReceived(bitmap)
                            }
                        }
                    } catch (e: SocketTimeoutException) {
                        // Timeout is fine, just loop again if still receiving
                        continue
                    } catch (e: Exception) {
                        Log.e("ConnectivityManager", "Error receiving frame: ${e.message}")
                    }
                }
            } catch (e: Exception) {
                Log.e("ConnectivityManager", "Failed to setup UDP Socket: ${e.message}")
            } finally {
                stopReceiving()
            }
        }
    }

    fun stopReceiving() {
        isReceiving = false
        udpSocket?.close()
        udpSocket = null
    }
}