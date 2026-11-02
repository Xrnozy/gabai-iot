package com.example.gabai.services

import android.content.Context
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.herohan.uvcapp.CameraException
import com.herohan.uvcapp.ICameraHelper
import com.serenegiant.usb.IFrameCallback
import com.serenegiant.usb.UVCCamera
import org.webrtc.CapturerObserver
import org.webrtc.CameraVideoCapturer
import org.webrtc.NV21Buffer
import org.webrtc.SurfaceTextureHelper
import org.webrtc.VideoFrame
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

/**
 * WebRTC capturer for USB UVC webcams (OTG) via [com.herohan.uvcapp.CameraHelper].
 * Used when Camera2 does not expose the device as LENS_FACING_EXTERNAL.
 */
class UvcUsbVideoCapturer(
    private val appContext: Context,
) : CameraVideoCapturer {

    companion object {
        private const val TAG = "UvcUsbCapturer"
        /** If the camera opens but delivers zero valid frames within
         *  this window, the negotiated format is likely not converting
         *  to NV21 — report it instead of streaming silence. */
        private const val FRAME_WATCHDOG_MS = 6_000L
        /** How long startCapture waits for a camera to enumerate before
         *  declaring "no USB camera" — onAttach is asynchronous. */
        private const val DEVICE_WAIT_MS = 8_000L

        fun hasUsbCamera(context: Context): Boolean {
            return try {
                val helper = com.herohan.uvcapp.CameraHelper()
                val list = helper.deviceList
                helper.release()
                !list.isNullOrEmpty()
            } catch (e: Exception) {
                false
            }
        }
    }

    /** Called with a plain-language reason when capture cannot produce
     *  video (no device, permission denied, no frames). A blind user
     *  cannot see a black preview — the app must SAY the camera died. */
    var onFailure: ((String) -> Unit)? = null

    private val helper = com.herohan.uvcapp.CameraHelper()
    private var observer: CapturerObserver? = null
    private var captureFps = 10
    @Volatile private var frameWidth = 640
    @Volatile private var frameHeight = 480
    private val capturing = AtomicBoolean(false)
    @Volatile private var startRequested = false
    @Volatile private var firstFrameSeen = false
    @Volatile private var deviceSelected = false
    private var selectedDevice: UsbDevice? = null
    private var opened = false
    private var loggedFrameMismatch = false
    private val mainHandler = Handler(Looper.getMainLooper())
    private var frameWatchdog: Runnable? = null
    private var deviceWatchdog: Runnable? = null

    /** Only real video devices — an OTG hub, serial adapter or charger
     *  IC in deviceList must never be picked and opened as a camera. */
    private fun isVideoDevice(d: UsbDevice): Boolean {
        if (d.deviceClass == UsbConstants.USB_CLASS_VIDEO) return true
        for (i in 0 until d.interfaceCount) {
            if (d.getInterface(i).interfaceClass ==
                    UsbConstants.USB_CLASS_VIDEO) return true
        }
        return false
    }

    private fun pickVideoDevice(): UsbDevice? {
        val devices = helper.deviceList ?: return null
        return devices.firstOrNull { isVideoDevice(it) }
            ?: devices.firstOrNull()
    }

    private fun reportFailure(msg: String) {
        Log.e(TAG, "capture failed: $msg")
        mainHandler.post { onFailure?.invoke(msg) }
    }

    private fun armFrameWatchdog() {
        cancelFrameWatchdog()
        val r = Runnable {
            if (opened && !firstFrameSeen) {
                reportFailure(
                    "The USB camera opened but is not delivering video.")
            }
        }
        frameWatchdog = r
        mainHandler.postDelayed(r, FRAME_WATCHDOG_MS)
    }

    private fun cancelFrameWatchdog() {
        frameWatchdog?.let { mainHandler.removeCallbacks(it) }
        frameWatchdog = null
    }

    private fun selectDevice(d: UsbDevice) {
        deviceSelected = true
        selectedDevice = d
        helper.selectDevice(d)
    }

    private fun armDeviceWatchdog() {
        cancelDeviceWatchdog()
        val r = Runnable {
            if (startRequested && !deviceSelected) {
                reportFailure(
                    "No USB camera detected — check the OTG cable.")
                observer?.onCapturerStarted(false)
            }
        }
        deviceWatchdog = r
        mainHandler.postDelayed(r, DEVICE_WAIT_MS)
    }

    private fun cancelDeviceWatchdog() {
        deviceWatchdog?.let { mainHandler.removeCallbacks(it) }
        deviceWatchdog = null
    }

    private val stateCallback = object : ICameraHelper.StateCallback {
        override fun onAttach(device: UsbDevice) {
            // Device hot-plugged (or enumerated after register). Only
            // grab it once capture was actually requested, and only if
            // it is a video-class device.
            if (startRequested && isVideoDevice(device)) {
                selectDevice(device)
            }
        }

        override fun onDeviceOpen(device: UsbDevice, isFirstOpen: Boolean) {
            // Native/default preview size — forcing 320×240 breaks NV21 stride (green bars).
            helper.openCamera()
        }

        override fun onCameraOpen(device: UsbDevice) {
            opened = true
            helper.previewSize?.let { size ->
                frameWidth = size.width
                frameHeight = size.height
                captureFps = size.fps
                // #region agent log
                AgentDebugLog.log(
                    "UvcUsbVideoCapturer.kt:onCameraOpen",
                    "uvc preview size",
                    "A",
                    mapOf(
                        "w" to size.width,
                        "h" to size.height,
                        "fps" to size.fps,
                        "type" to size.type,
                    ),
                )
                // #endregion
            }
            helper.setFrameCallback(frameCallback, UVCCamera.PIXEL_FORMAT_NV21)
            helper.startPreview()
            observer?.onCapturerStarted(true)
            armFrameWatchdog()
        }

        override fun onCameraClose(device: UsbDevice) {
            opened = false
        }

        override fun onDeviceClose(device: UsbDevice) {}
        override fun onDetach(device: UsbDevice) {
            // Cable yanked mid-navigation — silence here means the user
            // walks blind without knowing the feed died.
            if (device.deviceName == selectedDevice?.deviceName) {
                deviceSelected = false
                selectedDevice = null
                opened = false
                reportFailure("The USB camera was disconnected.")
                observer?.onCapturerStopped()
            }
        }
        override fun onCancel(device: UsbDevice) {
            Log.w(TAG, "USB permission denied for ${device.deviceName}")
            reportFailure("USB camera permission was denied.")
            observer?.onCapturerStarted(false)
        }

        override fun onError(device: UsbDevice, e: CameraException) {
            Log.e(TAG, "UVC error: ${e.message}")
            reportFailure("USB camera error: ${e.message ?: "unknown"}")
            observer?.onCapturerStarted(false)
        }
    }

    private val frameCallback = IFrameCallback { frame: ByteBuffer ->
        if (!capturing.get()) return@IFrameCallback
        val obs = observer ?: return@IFrameCallback
        val w = frameWidth
        val h = frameHeight
        val copy = ByteArray(frame.remaining())
        frame.get(copy)
        val expected = w * h * 3 / 2
        if (copy.size != expected) {
            if (!loggedFrameMismatch) {
                loggedFrameMismatch = true
                // #region agent log
                AgentDebugLog.log(
                    "UvcUsbVideoCapturer.kt:frameCallback",
                    "nv21 size mismatch",
                    "A",
                    mapOf(
                        "bytes" to copy.size,
                        "expected" to expected,
                        "w" to w,
                        "h" to h,
                    ),
                )
                // #endregion
            }
            return@IFrameCallback
        }
        val nv21 = NV21Buffer(copy, w, h, null)
        val i420 = nv21.toI420()
        val videoFrame = VideoFrame(i420, 0, System.nanoTime())
        firstFrameSeen = true
        obs.onFrameCaptured(videoFrame)
        videoFrame.release()
    }

    override fun initialize(
        surfaceTextureHelper: SurfaceTextureHelper?,
        context: Context?,
        capturerObserver: CapturerObserver?,
    ) {
        observer = capturerObserver
        helper.setStateCallback(stateCallback)
    }

    override fun startCapture(width: Int, height: Int, framerate: Int) {
        captureFps = framerate
        loggedFrameMismatch = false
        firstFrameSeen = false
        deviceSelected = false
        selectedDevice = null
        capturing.set(true)
        startRequested = true
        val device = pickVideoDevice()
        if (device == null) {
            // The helper's USB monitor may still be enumerating — an
            // attached camera arrives via onAttach shortly. Do NOT
            // declare failure yet; that would permanently kill the
            // capture session the moment the app starts.
            Log.w(TAG, "No USB video device yet — waiting for attach")
            armDeviceWatchdog()
            return
        }
        selectDevice(device)
    }

    override fun stopCapture() {
        startRequested = false
        capturing.set(false)
        cancelFrameWatchdog()
        cancelDeviceWatchdog()
        if (opened) {
            helper.stopPreview()
            helper.closeCamera()
        }
        observer?.onCapturerStopped()
    }

    override fun changeCaptureFormat(width: Int, height: Int, framerate: Int) {
        captureFps = framerate
        // WebRTC may request 320×240; UVC keeps negotiated native size.
    }

    override fun dispose() {
        startRequested = false
        capturing.set(false)
        cancelFrameWatchdog()
        cancelDeviceWatchdog()
        try {
            helper.stopPreview()
            helper.closeCamera()
            helper.release()
        } catch (e: Exception) {
            Log.w(TAG, "dispose: ${e.message}")
        }
        observer = null
    }

    override fun isScreencast(): Boolean = false

    override fun switchCamera(handler: CameraVideoCapturer.CameraSwitchHandler?) {
        handler?.onCameraSwitchError("USB camera only")
    }

    override fun switchCamera(
        handler: CameraVideoCapturer.CameraSwitchHandler?,
        cameraName: String?,
    ) {
        switchCamera(handler)
    }
}
