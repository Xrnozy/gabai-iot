package com.example.lumenguide

import org.webrtc.*

object FramePreprocessor {

    /**
     * Apply edge enhancement directly on I420 frame (Y plane only)
     * FAST + real-time safe
     */
    fun processFrame(frame: VideoFrame): VideoFrame {
        val buffer = frame.buffer

        if (buffer !is VideoFrame.I420Buffer) {
            return frame
        }

        val width = buffer.width
        val height = buffer.height

        val yPlane = buffer.dataY
        val stride = buffer.strideY

        // Copy original (avoid modifying in-place badly)
        val temp = ByteArray(width * height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                temp[y * width + x] = yPlane.get(y * stride + x)
            }
        }

        // 🔥 SHARPEN (fast Laplacian)
        for (y in 1 until height - 1 step 2) {
            for (x in 1 until width - 1 step 2) {

                val idx = y * width + x

                val center = temp[idx].toInt() and 0xFF
                val top = temp[(y - 1) * width + x].toInt() and 0xFF
                val bottom = temp[(y + 1) * width + x].toInt() and 0xFF
                val left = temp[y * width + x - 1].toInt() and 0xFF
                val right = temp[y * width + x + 1].toInt() and 0xFF

                var newValue = (5 * center - top - bottom - left - right)

                if (newValue < 0) newValue = 0
                if (newValue > 255) newValue = 255

                yPlane.put(y * stride + x, newValue.toByte())
            }
        }

        return frame
    }
}