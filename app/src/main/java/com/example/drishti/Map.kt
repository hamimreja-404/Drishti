package com.example.drishti

import android.content.Context
import android.content.Intent
import android.location.Geocoder
import android.location.Location
import android.location.LocationManager
import android.net.Uri
import java.util.Locale

data class NavigationStep(
    val instruction: String,
    val maneuverLocation: Location,
    val distance: Double
)

object MapSystem {
    fun isLocationEnabled(context: Context): Boolean {
        val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        return locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER) ||
                locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)
    }

    fun buildGoogleMapsNavigationIntent(destination: String): Intent {
        val encodedDestination = Uri.encode(destination.trim())
        val uri = Uri.parse("google.navigation:q=$encodedDestination&mode=w")
        return Intent(Intent.ACTION_VIEW, uri).apply {
            setPackage("com.google.android.apps.maps")
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
    }

    fun parseCoordinateDestination(destination: String): Location? {
        val parts = destination.split(",").map { it.trim() }
        if (parts.size != 2) return null
        val latitude = parts[0].toDoubleOrNull() ?: return null
        val longitude = parts[1].toDoubleOrNull() ?: return null
        if (latitude !in -90.0..90.0 || longitude !in -180.0..180.0) return null
        return Location("manual").apply {
            this.latitude = latitude
            this.longitude = longitude
        }
    }

    @Suppress("DEPRECATION")
    fun resolveDestination(context: Context, destination: String): Location? {
        parseCoordinateDestination(destination)?.let { return it }
        val addresses = runCatching {
            Geocoder(context, Locale.getDefault()).getFromLocationName(destination, 1)
        }.getOrNull()
        val address = addresses?.firstOrNull() ?: return null
        return Location("geocoder").apply {
            latitude = address.latitude
            longitude = address.longitude
        }
    }

    fun startGoogleMapsNavigation(context: Context, destination: String): Boolean {
        val cleanDestination = destination.trim()
        if (cleanDestination.isEmpty() || !isLocationEnabled(context)) return false
        val intent = buildGoogleMapsNavigationIntent(cleanDestination)
        return runCatching {
            context.startActivity(intent)
            true
        }.getOrElse {
            val fallbackUri = Uri.parse("https://www.google.com/maps/dir/?api=1&destination=${Uri.encode(cleanDestination)}&travelmode=walking")
            context.startActivity(Intent(Intent.ACTION_VIEW, fallbackUri).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            true
        }
    }
}
