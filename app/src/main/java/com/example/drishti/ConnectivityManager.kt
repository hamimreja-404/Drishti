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

    private val PORT = 6868
    private val MAX_PACKET_SIZE = 65507

    // UPDATED: Callback now expects a Bitmap AND an Int? (nullable integer) for distance
    suspend fun startReceivingFrames(onFrameReceived: (Bitmap, Int?) -> Unit) {
        withContext(Dispatchers.IO) {
            try {
                udpSocket = DatagramSocket(PORT)
                udpSocket?.soTimeout = 2000
                isReceiving = true

                val buffer = ByteArray(MAX_PACKET_SIZE)
                val packet = DatagramPacket(buffer, buffer.size)

                Log.d("ConnectivityManager", "Listening for UDP frames on port $PORT")

                while (isReceiving) {
                    try {
                        udpSocket?.receive(packet)

                        // 1. EXTRACT THE DISTANCE (The very first byte)
                        // Use bitwise AND 0xFF to ensure it reads as a positive unsigned integer (0-255)
                        val distanceCm = packet.data[0].toInt() and 0xFF
                        Log.d("DRISHTI_NETWORK", "SUCCESS! Packet received. Size: ${packet.length} bytes")
                        // 2. EXTRACT THE IMAGE (Skip the first byte)
                        val bitmap = BitmapFactory.decodeByteArray(
                            packet.data,
                            1,                    // Start at index 1
                            packet.length - 1     // Total length minus the 1 byte we removed
                        )

                        if (bitmap != null) {
                            withContext(Dispatchers.Main) {
                                // Send BOTH pieces of data back to the UI
                                onFrameReceived(bitmap, distanceCm)
                            }
                        }else {
                            // 🚨 ADD THIS LINE: It proves the packet arrived, but wasn't a valid JPEG image
                            Log.e("DRISHTI_NETWORK", "ERROR: Packet received, but failed to decode into an image!")
                        }
                    } catch (e: SocketTimeoutException) {
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