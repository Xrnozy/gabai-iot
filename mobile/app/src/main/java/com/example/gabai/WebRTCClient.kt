package com.example.gabai

import com.example.lumenguide.FramePreprocessor


import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.example.gabai.services.CameraSourceResolver
import com.example.gabai.services.SettingsStore
import com.example.gabai.services.UvcUsbVideoCapturer
import org.webrtc.*
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.charset.Charset
import java.util.concurrent.atomic.AtomicBoolean

class WebRTCClient(
    private val context: Context,
    private val mode: String = "navigation",
) {

    interface NavigationListener {
        fun onNavigationCommand(command: String, fullData: JSONObject)

        /** Fired when the peer connection reaches CONNECTED — callers hide
         *  the "connecting" indicator and can confirm readiness. */
        fun onConnected() {}

        /** Fired on the main thread when the peer connection fails —
         *  callers should announce honest degraded state. */
        fun onConnectionFailed() {}

        /** Fired on the main thread when the camera cannot produce
         *  video — a blind user cannot see a black preview, so callers
         *  should SAY the reason. */
        fun onCameraError(message: String) {}

        /** Fired on the main thread when the peer enters DISCONNECTED —
         *  a transient ICE gap that usually self-heals. Callers arm a
         *  grace timer and rebuild the session if it doesn't recover. */
        fun onDisconnected() {}
    }

    var navigationListener: NavigationListener? = null

    // Server address comes from Settings (Helper section); the default is
    // the laptop's LAN IP. Any host used must be whitelisted in
    // res/xml/network_security_config.xml.
    private val serverUrl = SettingsStore.getServerUrl(context)
    // Optional bearer token — must match the server's GABAI_TOKEN env var.
    // Leave empty when the server runs without a token (dev default).
    private val serverToken = ""

    private val eglBase = EglBase.create()

    private var peerConnectionFactory: PeerConnectionFactory
    private var peerConnection: PeerConnection? = null

    private var videoCapturer: CameraVideoCapturer? = null
    private var videoSource: VideoSource? = null
    private var localVideoTrack: VideoTrack? = null
    private var navigationChannel: DataChannel? = null

    @Volatile
    private var isReleased = false

    // The offer must only be sent once ICE gathering has finished — otherwise
    // the SDP carries no candidates and the server can never reach the phone
    // (root cause of the "connected but blank feed" bug over Tailscale).
    private val offerSent = AtomicBoolean(false)
    private var pendingOffer: SessionDescription? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    companion object {
        private const val TAG = "WebRTCClient"
        private const val ICE_GATHER_TIMEOUT_MS = 5000L
    }

    init {
        PeerConnectionFactory.initialize(
            PeerConnectionFactory.InitializationOptions.builder(context)
                .createInitializationOptions()
        )

        peerConnectionFactory = PeerConnectionFactory.builder()
            .setVideoEncoderFactory(
                DefaultVideoEncoderFactory(
                    eglBase.eglBaseContext,
                    true,
                    true
                )
            )
            .setVideoDecoderFactory(
                DefaultVideoDecoderFactory(eglBase.eglBaseContext)
            )
            .createPeerConnectionFactory()

        initPeerConnection()
    }

    private fun initPeerConnection() {

        val iceServers = listOf(
            PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer()
        )

        val config = PeerConnection.RTCConfiguration(iceServers).apply {
            sdpSemantics = PeerConnection.SdpSemantics.UNIFIED_PLAN
            bundlePolicy = PeerConnection.BundlePolicy.MAXBUNDLE
            rtcpMuxPolicy = PeerConnection.RtcpMuxPolicy.REQUIRE
            continualGatheringPolicy =
                PeerConnection.ContinualGatheringPolicy.GATHER_CONTINUALLY
        }

        peerConnection = peerConnectionFactory.createPeerConnection(
            config,
            object : PeerConnection.Observer {

                override fun onIceCandidate(candidate: IceCandidate) {
                    // Candidates are gathered into localDescription; we send
                    // the offer once gathering is COMPLETE (see sendOfferWhenReady).
                    Log.d(TAG, "[ICE] candidate gathered")
                }

                override fun onConnectionChange(newState: PeerConnection.PeerConnectionState) {
                    Log.d(TAG, "[PC] connection state: $newState")
                    if (newState == PeerConnection.PeerConnectionState.CONNECTED) {
                        Log.d(TAG, "[RTC] Peer connection established — streaming")
                        mainHandler.post { navigationListener?.onConnected() }
                    } else if (newState == PeerConnection.PeerConnectionState.FAILED) {
                        Log.e(TAG, "[PC] Connection failed — is the server reachable over Tailscale?")
                        mainHandler.post { navigationListener?.onConnectionFailed() }
                    } else if (newState == PeerConnection.PeerConnectionState.DISCONNECTED) {
                        Log.w(TAG, "[PC] Connection disconnected — transient ICE gap")
                        mainHandler.post { navigationListener?.onDisconnected() }
                    }
                }

                override fun onTrack(transceiver: RtpTransceiver) {
                    Log.d(TAG, "[TRACK] Remote: ${transceiver.receiver.track()?.kind()}")
                }

                override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) {
                    Log.d(TAG, "[ICE] connection state: $state")
                }

                override fun onIceGatheringChange(state: PeerConnection.IceGatheringState) {
                    Log.d(TAG, "[ICE] gathering state: $state")
                    if (state == PeerConnection.IceGatheringState.COMPLETE) {
                        sendOfferWhenReady()
                    }
                }
                override fun onSignalingChange(p0: PeerConnection.SignalingState) {}
                override fun onAddStream(p0: MediaStream) {}
                override fun onRemoveStream(p0: MediaStream) {}
                override fun onIceConnectionReceivingChange(p0: Boolean) {}
                override fun onIceCandidatesRemoved(p0: Array<out IceCandidate>) {}
                override fun onDataChannel(dc: DataChannel) {
                    Log.d(TAG, "[DC] onDataChannel: ${dc.label()} id=${dc.id()}")
                }
                override fun onRenegotiationNeeded() {}
            }
        )

        // Create negotiated navigation channel (must match server id=2)
        val init = DataChannel.Init().apply {
            negotiated = true
            id = 2
        }
        navigationChannel = peerConnection?.createDataChannel("navigation", init)
        Log.d(TAG, "[DC] Created negotiated navigation channel (id=2)")

        // Scene channel (id=3): created symmetrically per spec — the phone
        // never reads it; the server only sends scene packets to viewers.
        val sceneInit = DataChannel.Init().apply {
            negotiated = true
            id = 3
        }
        val sceneChannel = peerConnection?.createDataChannel("scene", sceneInit)
        sceneChannel?.registerObserver(object : DataChannel.Observer {
            override fun onBufferedAmountChange(previous: Long) {}
            override fun onStateChange() {
                Log.d(TAG, "[DC] Scene channel state: ${sceneChannel.state()}")
            }
            override fun onMessage(buffer: DataChannel.Buffer) {
                // Server only sends scene packets to viewer connections.
            }
        })

        navigationChannel?.registerObserver(object : DataChannel.Observer {
            override fun onBufferedAmountChange(previous: Long) {}

            override fun onStateChange() {
                Log.d(TAG, "[DC] Navigation channel state: ${navigationChannel?.state()}")
            }

            override fun onMessage(buffer: DataChannel.Buffer) {
                val data = buffer.data
                val bytes = ByteArray(data.remaining())
                data.get(bytes)
                val json = String(bytes, Charset.forName("UTF-8"))
                try {
                    val obj = JSONObject(json)
                    val command = obj.optString("nav_command", "Forward")
                    Log.d(TAG, "[NAV] Received: $command")
                    navigationListener?.onNavigationCommand(command, obj)
                } catch (e: Exception) {
                    Log.e(TAG, "[NAV] Parse error: ${e.message}")
                }
            }
        })
    }

    // -------------------------
    // CAMERA
    // -------------------------
    fun startCamera() {
        if (videoCapturer != null) return

        val source = SettingsStore.getCameraSource(context)
        val cameraEnumerator = Camera2Enumerator(context)
        videoCapturer = when (source) {
            SettingsStore.CAMERA_UVC -> {
                val externalId = CameraSourceResolver.externalCamera2Id(context)
                if (externalId != null) {
                    Log.d(TAG, "[CAM] Using Camera2 external id=$externalId")
                    cameraEnumerator.createCapturer(externalId, null)
                } else {
                    Log.d(TAG, "[CAM] Using USB UVC capturer")
                    UvcUsbVideoCapturer(context.applicationContext).also { uvc ->
                        uvc.onFailure = { msg ->
                            mainHandler.post {
                                navigationListener?.onCameraError(msg)
                            }
                        }
                    }
                }
            }
            else -> {
                val phoneId = CameraSourceResolver.phoneCameraId(context)
                if (phoneId == null) {
                    Log.e(TAG, "[CAM] No built-in camera found")
                    null
                } else {
                    Log.d(TAG, "[CAM] Using phone camera id=$phoneId")
                    cameraEnumerator.createCapturer(phoneId, null)
                }
            }
        }
        if (videoCapturer == null) {
            Log.e(TAG, "[CAM] Could not open camera (source=$source)")
            mainHandler.post {
                navigationListener?.onCameraError("No camera is available.")
            }
            return
        }

        videoSource = peerConnectionFactory.createVideoSource(false)
        val sharpen = videoCapturer !is UvcUsbVideoCapturer
        videoSource?.setVideoProcessor(CustomVideoProcessor(sharpen))

        val surfaceTextureHelper =
            SurfaceTextureHelper.create("CaptureThread", eglBase.eglBaseContext)

        videoCapturer?.initialize(
            surfaceTextureHelper,
            context,
            videoSource?.capturerObserver
        )

        // 🔥 OPTIMIZED FOR LOW INTERNET / MOBILE DATA
        videoCapturer?.startCapture(320, 240, 10)

        localVideoTrack =
            peerConnectionFactory.createVideoTrack("video", videoSource)

        val sender = peerConnection?.addTrack(localVideoTrack)

        // 🔥 FORCE STABLE QUALITY & LOW DELAY (MAIN FIX)
        val parameters = sender?.parameters
        parameters?.encodings?.forEach {
            it.maxBitrateBps = 250_000   // 250 kbps (mobile data friendly)
            it.minBitrateBps = 100_000   // prevent blur
            it.maxFramerate = 10         // smooth enough for nav, low bandwidth
            it.scaleResolutionDownBy = 1.0
        }
        sender?.parameters = parameters
    }
    // -------------------------
    // OFFER
    // -------------------------
    fun createOffer() {
        if (isReleased) return

        val constraints = MediaConstraints().apply {
            mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo", "false"))

            // Prefer H264
            optional.add(MediaConstraints.KeyValuePair("preferredVideoCodec", "H264"))
        }

        peerConnection?.createOffer(object : SdpObserver {

            override fun onCreateSuccess(desc: SessionDescription) {
                if (isReleased) return
                peerConnection?.setLocalDescription(object : SdpObserver {

                    override fun onSetSuccess() {
                        if (isReleased) return
                        pendingOffer = desc
                        sendOfferWhenReady()
                        // Fallback: if gathering stalls, send partial
                        // candidates rather than never sending at all.
                        mainHandler.postDelayed(
                            { sendOfferWhenReady(force = true) },
                            ICE_GATHER_TIMEOUT_MS
                        )
                    }

                    override fun onSetFailure(p0: String?) {
                        Log.e(TAG, "[SIG] setLocalDescription failed: $p0")
                    }
                    override fun onCreateSuccess(p0: SessionDescription?) {}
                    override fun onCreateFailure(p0: String?) {}

                }, desc)
            }

            override fun onCreateFailure(p0: String?) {
                Log.e(TAG, "[SIG] createOffer failed: $p0")
            }
            override fun onSetSuccess() {}
            override fun onSetFailure(p0: String?) {}

        }, constraints)
    }

    /**
     * Sends the offer once ICE gathering has finished so the SDP actually
     * contains the phone's candidates (including its Tailscale 100.x address).
     * [force] is used by a timeout fallback so a stalled gather still emits
     * an offer with whatever candidates were found.
     */
    private fun sendOfferWhenReady(force: Boolean = false) {
        if (isReleased || offerSent.get()) return
        val pc = peerConnection ?: return
        val complete = pc.iceGatheringState() == PeerConnection.IceGatheringState.COMPLETE
        if (!complete && !force) return
        // localDescription is mutated as candidates gather — it now includes them.
        val local = pc.localDescription ?: pendingOffer ?: return
        if (!offerSent.compareAndSet(false, true)) return
        if (!complete) {
            Log.w(TAG, "[ICE] Gathering incomplete after ${ICE_GATHER_TIMEOUT_MS}ms — sending offer with partial candidates")
        }
        sendOfferToServer(local)
    }

    private fun sendOfferToServer(offer: SessionDescription) {
        Thread {
            try {
                if (isReleased) return@Thread

                val json = JSONObject().apply {
                    put("type", offer.type.canonicalForm())
                    put("sdp", offer.description)
                    put("mode", mode)
                }

                val nCandidates = offer.description
                    .lineSequence()
                    .count { it.startsWith("a=candidate") }
                Log.d(TAG, "[SIG] POST /offer to $serverUrl " +
                        "(${offer.description.length} bytes, $nCandidates candidates)")

                val url = java.net.URL("$serverUrl/offer")
                val conn = url.openConnection() as java.net.HttpURLConnection

                conn.requestMethod = "POST"
                conn.setRequestProperty("Content-Type", "application/json")
                if (serverToken.isNotEmpty()) {
                    conn.setRequestProperty("Authorization", "Bearer $serverToken")
                }
                conn.doOutput = true
                conn.connectTimeout = 10_000
                conn.readTimeout = 10_000

                conn.outputStream.bufferedWriter().use {
                    it.write(json.toString())
                }

                val code = conn.responseCode
                val response = runCatching {
                    (if (code in 200..299) conn.inputStream else conn.errorStream)
                        ?.bufferedReader()?.use { it.readText() } ?: ""
                }.getOrDefault("")

                if (code !in 200..299) {
                    Log.e(TAG, "[SIG] /offer failed: HTTP $code — $response")
                    return@Thread
                }

                if (isReleased) return@Thread

                val answerJson = JSONObject(response)

                val answer = SessionDescription(
                    SessionDescription.Type.fromCanonicalForm(
                        answerJson.getString("type")
                    ),
                    answerJson.getString("sdp")
                )

                if (!isReleased && peerConnection != null) {
                    peerConnection?.setRemoteDescription(
                        object : SdpObserver {
                            override fun onSetSuccess() {
                                Log.d(TAG, "[RTC] Answer applied — ICE checks starting")
                            }

                            override fun onSetFailure(p0: String?) {
                                Log.e(TAG, "[RTC] setRemoteDescription failed: $p0")
                            }
                            override fun onCreateSuccess(p0: SessionDescription?) {}
                            override fun onCreateFailure(p0: String?) {}
                        },
                        answer
                    )
                }

            } catch (e: Exception) {
                if (!isReleased) {
                    Log.e(TAG, "[SIG] /offer request failed: ${e.message}", e)
                }
            }
        }.start()
    }

    fun release() {
        isReleased = true
        // Stale events (a late DISCONNECTED/FAILED) from a released peer
        // must not reach the module — they would trigger a phantom
        // reconnect on top of the fresh session.
        navigationListener = null
        mainHandler.removeCallbacksAndMessages(null)
        try {
            videoCapturer?.stopCapture()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        videoCapturer?.dispose()
        videoCapturer = null
        videoSource?.dispose()
        videoSource = null
        peerConnection?.close()
        peerConnection = null
        peerConnectionFactory.dispose()
        eglBase.release()
    }
    private class CustomVideoProcessor(
        private val sharpenFrames: Boolean,
    ) : VideoProcessor {

        private var sink: VideoSink? = null

        override fun onCapturerStarted(success: Boolean) {}
        override fun onCapturerStopped() {}

        override fun setSink(videoSink: VideoSink?) {
            sink = videoSink
        }

        override fun onFrameCaptured(frame: VideoFrame) {
            val out = if (sharpenFrames) {
                FramePreprocessor.processFrame(frame)
            } else {
                frame
            }
            sink?.onFrame(out)
        }
    }
}