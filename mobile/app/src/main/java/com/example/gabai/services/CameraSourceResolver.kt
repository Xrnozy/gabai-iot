package com.example.gabai.services

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import org.webrtc.Camera2Enumerator

/**
 * Picks which camera feeds WebRTC: built-in (rear preferred) or USB / external.
 */
object CameraSourceResolver {

    fun phoneCameraId(context: Context): String? {
        val enumerator = Camera2Enumerator(context)
        val names = enumerator.deviceNames
        if (names.isEmpty()) return null
        return names.firstOrNull { enumerator.isBackFacing(it) }
            ?: names.firstOrNull { enumerator.isFrontFacing(it) }
            ?: names.first()
    }

    /** Camera2 external (OTG UVC) when the OS exposes it — API 28+. */
    fun externalCamera2Id(context: Context): String? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) return null
        val mgr = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        for (id in mgr.cameraIdList) {
            val facing = mgr.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_EXTERNAL) return id
        }
        val enumerator = Camera2Enumerator(context)
        return enumerator.deviceNames.firstOrNull { name ->
            !enumerator.isFrontFacing(name) && !enumerator.isBackFacing(name)
        }
    }

    fun label(context: Context, source: String): String = when (source) {
        SettingsStore.CAMERA_UVC ->
            context.getString(com.example.gabai.R.string.camera_source_uvc)
        else ->
            context.getString(com.example.gabai.R.string.camera_source_phone)
    }
}
