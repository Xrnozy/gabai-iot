package com.example.gabai.services

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** A finished utterance: its transcript, the raw PCM and the language used. */
data class Utterance(val text: String, val pcm: FloatArray, val language: String?)

/**
 * Continuous voice-command pipeline:
 *
 *   mic -> [MicEffects] (NS/AEC/AGC) -> [MicNormalizer] -> [WakeWordDetector]
 *       -> WHISPER_NATIVE mode: rolling window -> whisper.cpp internal VAD
 *          (only speech spans are decoded) -> [TranscriptFilter]
 *       -> ONNX_SILERO mode: [SileroVad] (or [EnergyVad] when no ONNX is
 *          installed) -> [VadSegmenter] -> speech segments -> [Recognizer]
 *       -> onCommand -> [CommandSession]
 *
 * In window mode whisper_full runs on every hop, but its internal VAD means
 * silent windows cost only the tiny VAD pass — the encoder never runs — so
 * the mic listens continuously without Whisper churning on noise. The ONNX
 * mode keeps Whisper idle until a segment forms.
 *
 * [commandWindowOpen] and [requireWakeName] gate which transcripts reach
 * the session; wire them to the session/state.
 */
class VoiceCommandService(
    private val context: Context,
    private val locale: Locale = Locale.US,
    private val onCommand: (Utterance) -> Boolean,
    private val onError: (String?) -> Unit = {},
    private val speechGate: () -> Boolean = { true },
    private val config: VoiceConfig = VoiceConfig.DEFAULT,
    private val requireWakeName: () -> Boolean = { true },
    private val commandWindowOpen: () -> Boolean = { false },
    /** Main-thread callback: models/mic are warming up (true) or the
     *  pipeline is ready/failed/stopped (false). The Whisper model copy +
     *  native init can take seconds — listeners should show a loading
     *  state until this reports false. */
    private val onLoadingChanged: (Boolean) -> Unit = {},
    /** Main-thread callback: the mic is genuinely capturing (true) or the
     *  capture loop has ended (false). Fires only after every model loaded
     *  and AudioRecord started — the honest "listening" signal. */
    private val onListeningChanged: (Boolean) -> Unit = {},
    /** Main-thread callback: the latest text Whisper produced — both from
     *  command segments and from live-caption windows — so the UI can
     *  always show what the mic heard. */
    private val onHeard: (String) -> Unit = {},
    /** "Show what I say" mode: rolling windows of raw mic audio go to
     *  Whisper on a fixed hop, independent of VAD/wake gating — captions
     *  and commands keep working even where Silero never fires. */
    private val liveTranscriptEnabled: () -> Boolean = { false },
    /** The app's own most-recent spoken line (TtsService.lastSpoken) — the
     *  mic keeps capturing through TTS, so transcripts can carry our own
     *  prompts glued onto the user's command ("back, please try again").
     *  Stripped in [stripSelfSpeech] before parsing. */
    private val selfSpeech: () -> String? = { null },
    private val recognizerFactory: (Context, VoiceConfig) -> Recognizer =
        { c, cfg -> WhisperRecognizer(c, cfg) },
    private val vadFactory: (Context, VoiceConfig) -> FrameVad =
        { c, cfg ->
            try {
                SileroVad.create(c, cfg)
            } catch (e: Exception) {
                Log.w(TAG, "silero VAD unavailable (${e.message}) — " +
                    "using energy-VAD fallback (no ONNX needed)")
                EnergyVad()
            }
        },
    private val wakeFactory: (Context, VoiceConfig) -> WakeWordDetector? =
        { c, cfg -> OnnxWakeWordDetector.create(c, cfg) },
) {

    private val mainHandler = Handler(Looper.getMainLooper())
    private val enabled = AtomicBoolean(false)
    @Volatile private var worker: Thread? = null

    private val whisperExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "voice-whisper").apply { isDaemon = true }
    }
    /** Bound the whisper queue — stale audio is worse than a dropped segment. */
    private val whisperPending = AtomicInteger(0)
    /** At most one live-caption job at a time; command segments win. */
    private val captionInFlight = AtomicBoolean(false)
    /** whisper-native mode: at most one window transcription in flight. */
    private val windowInFlight = AtomicBoolean(false)
    /** Bring-up diagnostics — windows dumped / window jobs run / the
     *  one-shot no-VAD probe (whisperExecutor thread only). */
    private var windowsDumped = 0
    private var windowJobs = 0
    private val probeDone = AtomicBoolean(false)

    /** Last transcript shown — consecutive duplicates are suppressed. */
    @Volatile private var lastHeard: String? = null
    /** Dedup for caption->command forwarding: overlapping windows repeat. */
    @Volatile private var lastCaptionCommand: String? = null
    @Volatile private var lastCaptionCommandAtMs = 0L

    private var vad: FrameVad? = null
    private var segmenter: VadSegmenter? = null
    private var wake: WakeWordDetector? = null
    private var recognizer: Recognizer? = null
    private var clipStore: ClipStore? = null

    /** Fallback mode only (no wake model): consecutive wake transcripts. */
    private var consecutiveWakeHits = 0
    /** Stream sample index of the last wake trigger — lets the segment that
     *  contains "hello gab walk" reach Whisper. */
    private var lastWakeAtSample = -1L
    private var streamSamples = 0L

    fun start() {
        if (!enabled.compareAndSet(false, true)) return
        lastHeard = null
        lastCaptionCommand = null
        worker = Thread({ captureLoop() }, "voice-commands").also { it.start() }
    }

    fun stop() {
        enabled.set(false)
        worker?.join(1000)
        worker = null
    }

    fun destroy() {
        stop()
        whisperExecutor.shutdown()   // queued release still runs
    }

    @SuppressLint("MissingPermission") // RECORD_AUDIO is granted before start().
    private fun captureLoop() {
        SoundCues.configure(context)
        mainHandler.post { onLoadingChanged(true) }

        val rec = try {
            recognizerFactory(context, config).apply { load() }
        } catch (e: Throwable) {
            // #region agent log
            AgentDebugLog.log(
                "VoiceCommandService.kt:captureLoop",
                "recognizer load failed",
                "B",
                mapOf("type" to e.javaClass.simpleName, "msg" to (e.message ?: "")),
            )
            // #endregion
            reportError("Voice commands are unavailable: ${e.message}")
            return
        }
        recognizer = rec

        // whisper.cpp's built-in Silero VAD gates speech inside each rolling
        // window; the per-frame ONNX Silero + VadSegmenter chain stays as
        // the fallback mode. Window mode still runs without the ggml VAD
        // model — whisper just decodes the whole window and the
        // transcript filters do the work.
        val windowed = config.vadMode == VadMode.WHISPER_NATIVE
        val nativeVad = windowed &&
            (rec as? WhisperRecognizer)?.nativeVadEnabled == true
        if (windowed && !nativeVad) {
            Log.w(TAG, "${config.whisperVadModelFile} unavailable — " +
                "windows transcribe without internal VAD")
        }

        // Optional: the intent-reasoning model. Without it, keyword +
        // fuzzy matching still cover the standard commands.
        try {
            Needle.load(context)
        } catch (e: Exception) {
            Log.w(TAG, "intent model unavailable, matching only: ${e.message}")
        }

        clipStore = ClipStore(context, config)

        var v: FrameVad? = null
        var seg: VadSegmenter? = null
        if (!windowed) {
            v = try {
                vadFactory(context, config)
            } catch (e: Exception) {
                reportError("Voice commands are unavailable: ${e.message}")
                rec.release()
                return
            }
            vad = v
            seg = VadSegmenter(config).also { s ->
                segmenter = s
                s.onSegment = { dispatch(it) }
                s.onTooShortSegment = { pcm, _ ->
                    clipStore?.save(pcm, "too_short")
                    Log.d(TAG, "segment %.1fs dropped — shorter than %dms"
                        .format(pcm.size.toFloat() / config.sampleRate,
                            config.vadMinSegmentMs))
                }
            }
        }

        wake = wakeFactory(context, config)
        if (wake == null) {
            Log.w(TAG, "wake model ${config.wakeModelFile} missing — " +
                "falling back to transcript wake (VAD + " +
                "${config.fallbackWakeConsecutive}x consecutive)")
        }

        val record = openAudioRecord() ?: run {
            v?.close()
            whisperExecutor.execute { rec.release() }
            return
        }

        MicEffects.attach(record, config)
        if (config.enableDenoiser) {
            Log.w(TAG, "enableDenoiser is set but no denoiser is bundled — " +
                "running without RNNoise")
        }
        val normalizer = if (config.micNormalize) MicNormalizer(config) else null

        try {
            record.startRecording()
            Log.i(TAG, "mic capture started (mode=%s, wake=%s, stt=whisper.cpp)"
                .format(if (windowed) "whisper-vad" else v!!.javaClass.simpleName,
                    if (wake != null) "onnx" else "fallback"))
            Log.i(TAG, "audiorecord rate=%d ch=%d src=%d"
                .format(record.sampleRate, record.channelCount,
                    record.audioSource))
            if (windowed) {
                Log.i(TAG, "window mode: %dms window, %dms hop, vad=%s"
                    .format(config.listenWindowMs, config.listenHopMs,
                        if (nativeVad) "whisper" else "off"))
            } else {
                Log.i(TAG, "live transcript %s (window=%dms hop=%dms)"
                    .format(if (liveTranscriptEnabled()) "on" else "off",
                        config.liveCaptionWindowMs, config.liveCaptionHopMs))
            }
            mainHandler.post {
                onLoadingChanged(false)
                if (enabled.get()) onListeningChanged(true)
            }
            if (windowed) windowLoop(record, normalizer)
            else segmentLoop(record, v!!, seg!!, normalizer)
        } finally {
            try { record.stop() } catch (_: Exception) {}
            record.release()
            MicEffects.release()
            seg?.flush()
            v?.close()
            wake?.close()
            // Release on the whisper thread — serialised after any in-flight
            // transcription so nativeRelease never races whisper_full.
            recognizer = null
            whisperExecutor.execute { rec.release() }
            mainHandler.post { onListeningChanged(false) }
        }
    }

    /** Probes [VoiceConfig.audioSources] in order; the first source that
     *  initializes wins. UNPROCESSED bypasses the vendor audio chain on
     *  devices where MIC/VOICE_RECOGNITION is pre-processed. */
    @SuppressLint("MissingPermission")
    private fun openAudioRecord(): AudioRecord? {
        for (src in config.audioSources) {
            val record = try {
                AudioRecord(
                    src,
                    config.sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    maxOf(
                        AudioRecord.getMinBufferSize(
                            config.sampleRate,
                            AudioFormat.CHANNEL_IN_MONO,
                            AudioFormat.ENCODING_PCM_16BIT,
                        ),
                        config.frameSamples * 2 * 10,
                    ),
                )
            } catch (e: SecurityException) {
                reportError("Microphone permission is required for voice commands.")
                return null
            } catch (e: Exception) {
                Log.w(TAG, "audio source $src failed to open: ${e.message}")
                continue
            }
            if (record.state == AudioRecord.STATE_INITIALIZED) {
                applyMicPreference(record)
                Log.i(TAG, "audio source %d selected".format(src))
                return record
            }
            record.release()
            Log.w(TAG, "audio source $src did not initialize — trying next")
        }
        reportError("Voice commands are unavailable on this device.")
        return null
    }

    /** Applies the Settings "Microphone" choice to a live record session.
     *  "auto" leaves routing to the OS (headset mic wins when plugged).
     *  Lets users split the pair — e.g. prompts in headphones while the
     *  phone's own mic keeps listening for "Hello". */
    private fun applyMicPreference(record: AudioRecord) {
        val src = SettingsStore.getMicSource(context)
        if (src == SettingsStore.MIC_AUTO) return
        try {
            val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            val devices = am.getDevices(AudioManager.GET_DEVICES_INPUTS)
            // Legacy picks map to type groups; anything else is a specific
            // device id stored from the live picker in Settings.
            val device = when (src) {
                SettingsStore.MIC_PHONE -> devices.firstOrNull {
                    it.type == AudioDeviceInfo.TYPE_BUILTIN_MIC
                }
                SettingsStore.MIC_HEADSET -> devices.firstOrNull {
                    it.type == AudioDeviceInfo.TYPE_WIRED_HEADSET ||
                        it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO ||
                        it.type == AudioDeviceInfo.TYPE_USB_HEADSET
                }
                else -> devices.firstOrNull { it.id.toString() == src }
            }
            if (device != null) {
                val ok = record.setPreferredDevice(device)
                Log.i(TAG, "mic preferred device ${device.type} applied=$ok")
            } else {
                // Device unplugged since it was chosen — fall back to the
                // system default rather than a dead mic.
                Log.i(TAG, "mic preference set but no matching device present")
            }
        } catch (e: Exception) {
            Log.w(TAG, "setPreferredDevice failed: ${e.message}")
        }
    }

    /**
     * whisper-native mode: the last [VoiceConfig.listenWindowMs] of mic
     * audio goes to whisper_full (with the internal Silero VAD) every
     * [VoiceConfig.listenHopMs]. A silent window only costs the tiny VAD
     * pass — the encoder/decoder never run — so the mic listens
     * continuously without Whisper churning on noise.
     */
    private fun windowLoop(record: AudioRecord, normalizer: MicNormalizer?) {
        val frame = ShortArray(config.frameSamples * 2)
        val pending = FloatArray(config.frameSamples)
        var pendingLen = 0
        var gateClosedSinceMs = 0L
        var peakSample = 0f
        var logFrames = 0

        val ring = FloatArray(config.listenWindowMs * config.sampleRate / 1000)
        val hopSamples = config.listenHopMs * config.sampleRate / 1000
        var ringPos = 0
        var ringLen = 0
        var sinceHop = 0

        // Endpointing state: the early decode fires once per speech burst.
        val endpointSilenceSamples =
            config.endpointSilenceMs.toLong() * config.sampleRate / 1000
        val endpointMinSamples =
            config.endpointMinSpeechMs * config.sampleRate / 1000
        var lastVoicedAtSample = 0L
        var endpointFired = false
        // Absolute sample index where the current voiced burst began —
        // endpoint decodes send just that span, not the whole window.
        var burstStartAbs = -1L

        while (enabled.get()) {
            val read = record.read(frame, 0, frame.size)
            if (read < 0) break // AudioRecord errors are sticky; give up.
            if (read == 0) continue
            // The gate only mutes DECODING, never capture — the ring keeps
            // filling through the wake ding and "I'm listening" prompt, so
            // a command spoken over the prompt isn't thrown away; it lands
            // in the first post-gate window and fires normally.
            val gated = !speechGate()
            if (gated) {
                // Speech during TTS/cues isn't burst-tracked — the first
                // post-gate voiced frame starts a fresh one.
                burstStartAbs = -1L
                if (gateClosedSinceMs == 0L) {
                    gateClosedSinceMs = SystemClock.elapsedRealtime()
                } else if (SystemClock.elapsedRealtime() - gateClosedSinceMs >
                    GATE_RESET_MS
                ) {
                    wake?.reset()
                }
            } else {
                gateClosedSinceMs = 0L
            }

            var i = 0
            while (i < read) {
                val n = minOf(config.frameSamples - pendingLen, read - i)
                for (j in 0 until n) {
                    pending[pendingLen + j] = frame[i + j] / 32768f
                }
                pendingLen += n
                i += n
                if (pendingLen == config.frameSamples) {
                    normalizer?.process(pending)
                    streamSamples += config.frameSamples
                    if (!gated && wake?.accept(pending) == true) onWakeDetected()
                    var sumSq = 0f
                    for (s in pending) {
                        ring[ringPos] = s
                        ringPos = (ringPos + 1) % ring.size
                        val a = kotlin.math.abs(s)
                        if (a > peakSample) peakSample = a
                        sumSq += s * s
                    }
                    ringLen = minOf(ringLen + config.frameSamples, ring.size)
                    if (++logFrames >= VAD_LOG_FRAMES) {
                        Log.d(TAG, "mic amp=%.4f window=%.1fs"
                            .format(peakSample,
                                ringLen.toFloat() / config.sampleRate))
                        peakSample = 0f
                        logFrames = 0
                    }
                    if (!gated) {
                        sinceHop += config.frameSamples
                        // Endpointing: speech ended -> decode NOW rather
                        // than waiting for the hop boundary.
                        val rms = kotlin.math.sqrt(sumSq / pending.size)
                        if (rms >= config.endpointVoicedRms) {
                            if (burstStartAbs < 0) {
                                burstStartAbs = streamSamples -
                                    config.frameSamples
                            }
                            lastVoicedAtSample = streamSamples
                            endpointFired = false
                        } else {
                            if (!endpointFired && lastVoicedAtSample > 0 &&
                                streamSamples - lastVoicedAtSample >=
                                    endpointSilenceSamples &&
                                ringLen >= endpointMinSamples
                            ) {
                                endpointFired = true
                                sinceHop = 0
                                queueBurst(
                                    ring, ringLen, ringPos,
                                    burstStartAbs, lastVoicedAtSample)
                            }
                            // Sustained silence closes the burst even when
                            // no decode fired — the next word is a new one.
                            if (streamSamples - lastVoicedAtSample >=
                                endpointSilenceSamples
                            ) burstStartAbs = -1L
                        }
                        if (sinceHop >= hopSamples) {
                            sinceHop = 0
                            queueWindow(ring, ringLen, ringPos)
                        }
                    }
                    pendingLen = 0
                }
            }
        }
    }

    /**
     * ONNX Silero mode (fallback): per-frame VAD scores drive
     * [VadSegmenter]; finished segments go to Whisper via [dispatch], and
     *  the optional live-caption ring keeps "Show what I say" alive.
     */
    private fun segmentLoop(
        record: AudioRecord,
        v: FrameVad,
        seg: VadSegmenter,
        normalizer: MicNormalizer?,
    ) {
        val frame = ShortArray(config.frameSamples * 2)
        // Accumulate reads into exact VAD-size frames — AudioRecord may
        // return partial reads and Silero wants exactly frameSamples.
        val pending = FloatArray(config.frameSamples)
        var pendingLen = 0
        var gateClosedSinceMs = 0L
        // Field debugging: the loudest VAD score + loudest sample over each
        // ~2 s window. "vad peak" staying under the start threshold while
        // someone talks points at mic/gain problems, not Whisper; "amp"
        // pinned at zero means the mic itself is delivering silence.
        var peakVadProb = 0f
        var peakSample = 0f
        var vadLogFrames = 0

        // Live-transcript ring: the last ~[liveCaptionWindowMs] of raw mic
        // audio, re-transcribed every [liveCaptionHopMs] — the caption
        // path stays alive even when VAD/wake gating yields no segments.
        val captionRing = FloatArray(
            config.liveCaptionWindowMs * config.sampleRate / 1000)
        val captionHopSamples = config.liveCaptionHopMs * config.sampleRate / 1000
        var captionPos = 0
        var captionLen = 0
        var captionHop = 0

        while (enabled.get()) {
            val read = record.read(frame, 0, frame.size)
            if (read < 0) break // AudioRecord errors are sticky; give up.
            if (read == 0) continue
            if (!speechGate()) {
                pendingLen = 0
                if (gateClosedSinceMs == 0L) {
                    gateClosedSinceMs = SystemClock.elapsedRealtime()
                } else if (SystemClock.elapsedRealtime() - gateClosedSinceMs >
                    GATE_RESET_MS
                ) {
                    seg.reset()
                    v.reset()
                    wake?.reset()
                }
                continue
            }
            gateClosedSinceMs = 0L

            var i = 0
            while (i < read) {
                val n = minOf(config.frameSamples - pendingLen, read - i)
                for (j in 0 until n) {
                    pending[pendingLen + j] = frame[i + j] / 32768f
                }
                pendingLen += n
                i += n
                if (pendingLen == config.frameSamples) {
                    normalizer?.process(pending)
                    streamSamples += config.frameSamples
                    if (wake?.accept(pending) == true) onWakeDetected()
                    val prob = v.speechProb(pending)
                    if (prob > peakVadProb) peakVadProb = prob
                    for (s in pending) {
                        val a = kotlin.math.abs(s)
                        if (a > peakSample) peakSample = a
                    }
                    if (++vadLogFrames >= VAD_LOG_FRAMES) {
                        Log.d(TAG, "vad peak=%.2f amp=%.4f inSpeech=%s"
                            .format(peakVadProb, peakSample, seg.inSpeech))
                        peakVadProb = 0f
                        peakSample = 0f
                        vadLogFrames = 0
                    }
                    // copyOf: the segmenter keeps the array; pending is
                    // reused for the next frame.
                    seg.accept(pending.copyOf(), prob)
                    for (s in pending) {
                        captionRing[captionPos] = s
                        captionPos = (captionPos + 1) % captionRing.size
                    }
                    captionLen = minOf(
                        captionLen + config.frameSamples, captionRing.size)
                    captionHop += config.frameSamples
                    if (captionHop >= captionHopSamples) {
                        captionHop = 0
                        if (liveTranscriptEnabled()) {
                            queueCaption(captionRing, captionLen, captionPos)
                        }
                    }
                    pendingLen = 0
                }
            }
        }
    }

    /** Wake word fired — synthetic wake utterance opens the session window;
     *  the segment containing it is flagged so one-breath commands survive. */
    private fun onWakeDetected() {
        lastWakeAtSample = streamSamples
        consecutiveWakeHits = 0
        mainHandler.post {
            if (enabled.get()) {
                onCommand(Utterance(CommandParser.WAKE_NAME, FloatArray(0), null))
            }
        }
    }

    /**
     * Decides whether a completed segment goes to Whisper. Whisper stays
     * idle while nothing is expected of it: segments reach it only when the
     * command window is open, wake-free mode is on, the wake word fired
     * inside this segment, or the transcript fallback is in charge of waking.
     */
    private fun dispatch(segment: VadSegment) {
        if (!enabled.get()) return
        val wakeInside = lastWakeAtSample >= 0 &&
            lastWakeAtSample >= segment.startMs * config.sampleRate / 1000 &&
            lastWakeAtSample <= segment.endMs * config.sampleRate / 1000 +
                WAKE_PAD_MS * config.sampleRate / 1000
        val send = wake == null || !requireWakeName() || commandWindowOpen() || wakeInside
        if (!send) {
            clipStore?.save(segment.pcm, "idle_drop")
            Log.d(TAG, "segment %.1fs dropped — command window closed"
                .format(segment.pcm.size.toFloat() / config.sampleRate))
            return
        }
        if (whisperPending.incrementAndGet() > MAX_PENDING_TRANSCRIPTS) {
            whisperPending.decrementAndGet()
            Log.w(TAG, "whisper busy — dropping segment")
            return
        }
        Log.d(TAG, "segment %.1fs -> whisper"
            .format(segment.pcm.size.toFloat() / config.sampleRate))
        whisperExecutor.execute { transcribe(segment) }
    }

    /**
     * Live-transcript caption job: snapshots the ring's last [len] samples
     * (oldest -> newest) and runs Whisper on them off the capture thread.
     * Captions always yield to command segments — one job in flight max.
     */
    private fun queueCaption(ring: FloatArray, len: Int, writePos: Int) {
        if (!enabled.get() || len <= 0) return
        if (whisperPending.get() > 0) return
        if (!captionInFlight.compareAndSet(false, true)) return
        val pcm = FloatArray(len)
        val start = (writePos - len + ring.size) % ring.size
        if (start + len <= ring.size) {
            ring.copyInto(pcm, 0, start, start + len)
        } else {
            val first = ring.size - start
            ring.copyInto(pcm, 0, start, ring.size)
            ring.copyInto(pcm, first, 0, len - first)
        }
        Log.d(TAG, "caption %.1fs -> whisper"
            .format(len.toFloat() / config.sampleRate))
        whisperExecutor.execute {
            try {
                val t = recognizer?.transcribe(pcm) ?: return@execute
                // Lenient display gate: keep the real confidence checks but
                // let blocklisted words through — "hello" IS worth showing.
                val d = TranscriptFilter.evaluate(
                    t, vadConfirmed = true, cfg = config, skipBlocklist = true)
                TranscriptFilter.log(TAG, t, vadConfirmed = true, d = d)
                // The visible-in-logcat line: every caption window reports
                // what the model actually heard, rejected or not.
                Log.i(TAG, "caption heard: \"%s\" -> %s".format(t.text, d.reason))
                if (!d.accepted || !enabled.get()) return@execute
                reportHeard(t.text)
                forwardCaptionCommand(t, pcm)
            } catch (e: Exception) {
                Log.w(TAG, "caption transcribe failed: ${e.message}")
            } finally {
                captionInFlight.set(false)
            }
        }
    }

    /**
     * Endpoint decode: Whisper sees only the detected speech burst — padded
     * by [VoiceConfig.burstLeadPadMs]/[burstTailPadMs] — instead of the
     * whole rolling window. A one-word command decodes ~1.5 s of audio
     * rather than the full window, so the encoder finishes several times
     * faster. Falls back to the full window when no burst was tracked.
     */
    private fun queueBurst(
        ring: FloatArray,
        ringLen: Int,
        ringPos: Int,
        burstStartAbs: Long,
        burstEndAbs: Long,
    ) {
        if (burstStartAbs < 0 || burstEndAbs <= burstStartAbs) {
            queueWindow(ring, ringLen, ringPos)
            return
        }
        val streamEnd = streamSamples
        val lead = config.burstLeadPadMs.toLong() * config.sampleRate / 1000
        val tail = config.burstTailPadMs.toLong() * config.sampleRate / 1000
        // Clamp to samples still live in the ring (oldest = end - ringLen).
        val startAbs = maxOf(burstStartAbs - lead, streamEnd - ringLen, 0L)
        val endAbs = minOf(burstEndAbs + tail, streamEnd)
        if (endAbs <= startAbs) {
            queueWindow(ring, ringLen, ringPos)
            return
        }
        val len = (endAbs - startAbs).toInt()
        // Ring slot of sample i is i % ring.size; writePos is the slot
        // after the burst's last sample.
        queueWindow(ring, len, (endAbs % ring.size).toInt(), dropIfBusy = false)
        Log.d(TAG, "burst %.1fs -> whisper (trimmed)"
            .format(len.toFloat() / config.sampleRate))
    }

    /**
     * whisper-native mode: snapshots the ring's last [len] samples and runs
     * whisper_full (internal VAD) off the capture thread. Hop decodes are
     * dropped while one is in flight — stale audio is worse than a dropped
     * hop. Endpoint bursts ([dropIfBusy] = false) always queue instead: a
     * burst is a finished utterance, and waiting one decode behind a hop
     * beats losing the fast path entirely.
     */
    private fun queueWindow(
        ring: FloatArray,
        len: Int,
        writePos: Int,
        dropIfBusy: Boolean = true,
    ) {
        if (!enabled.get() || len <= 0) return
        if (dropIfBusy && !windowInFlight.compareAndSet(false, true)) return
        val pcm = FloatArray(len)
        val start = (writePos - len + ring.size) % ring.size
        if (start + len <= ring.size) {
            ring.copyInto(pcm, 0, start, start + len)
        } else {
            val first = ring.size - start
            ring.copyInto(pcm, 0, start, ring.size)
            ring.copyInto(pcm, first, 0, len - first)
        }
        whisperExecutor.execute {
            try {
                val wr = recognizer as? WhisperRecognizer
                val vadProb = wr?.vadMaxProb(pcm) ?: -1f
                // Bring-up aid: raw window WAVs to external files dir —
                // `adb pull` them to hear what the mic actually captured
                // when VAD reports nothing despite healthy amplitude.
                val fullWindow = len >= ring.size
                if (config.dumpDebugWindows && fullWindow &&
                    windowsDumped < config.dumpWindowCount
                ) {
                    windowsDumped++
                    clipStore?.saveProbe(
                        pcm, "window_%03d.wav".format(windowsDumped))
                }
                val t = recognizer?.transcribe(pcm) ?: return@execute
                // Whisper's internal VAD verdict gates the transcript
                // filter; without a VAD model every window is "confirmed"
                // and the text-quality filters do the work alone.
                val hasNativeVad = wr?.nativeVadEnabled == true
                val vadConfirmed = !hasNativeVad || t.vadSegments.isNotEmpty()
                val d = TranscriptFilter.evaluate(
                    t, vadConfirmed = vadConfirmed, cfg = config,
                    skipBlocklist = true)
                TranscriptFilter.log(TAG, t, vadConfirmed = vadConfirmed, d = d)
                // The visible-in-logcat line: every window reports what the
                // model actually heard, rejected or not.
                Log.i(TAG, "window heard: \"%s\" vad=%d prob=%.2f %dms -> %s"
                    .format(t.text, t.vadSegments.size, vadProb,
                        t.elapsedMs, d.reason))
                // One-shot no-VAD probe per session: a plain decode of a
                // full window separates "VAD rejects the audio" (probe
                // returns text -> tune the threshold) from "the PCM has no
                // speech" (probe returns nothing -> check the dumped WAV).
                if (!probeDone.get() && fullWindow && vadProb >= 0f &&
                    (vadProb > 0.05f || windowJobs >= 4)
                ) {
                    probeDone.set(true)
                    val plain = wr?.transcribeNoVad(pcm)
                    Log.w(TAG, "probe(no-vad) text=\"${plain?.text}\" " +
                        "no_speech=%s elapsed=%dms".format(
                            plain?.noSpeechProb?.let { "%.2f".format(it) }
                                ?: "n/a", plain?.elapsedMs ?: -1L))
                }
                windowJobs++
                if (!d.accepted || !enabled.get()) return@execute
                reportHeard(t.text)
                forwardCaptionCommand(t, pcm)
            } catch (e: Exception) {
                Log.w(TAG, "window transcribe failed: ${e.message}")
            } finally {
                windowInFlight.set(false)
            }
        }
    }

    /** Posts a transcript to the "You said" strip — blanks and consecutive
     *  repeats are suppressed so the display isn't spammed by one phrase. */
    private fun reportHeard(text: String) {
        if (!enabled.get() || text.isBlank() || text == lastHeard) return
        lastHeard = text
        Log.i(TAG, "heard: \"$text\"")
        HeardLog.append(context, text)
        mainHandler.post { if (enabled.get()) onHeard(text) }
    }

    /**
     * Caption transcripts can drive commands too — the segment path is the
     * preferred route, but on devices where VAD/wake gating starves it,
     * this is the only way "hello, walk" reaches the session. Gated hard:
     * the transcript must carry the wake name, land inside an open command
     * window, or run in wake-free mode — ambient speech stays display-only.
     */
    private fun forwardCaptionCommand(t: Transcription, pcm: FloatArray) {
        // Self-speech echo: the ring keeps buffering through TTS, so the
        // wake/timeout prompts and the last spoken line can land in a
        // window transcript — glued onto the user's command ("back,
        // please try again"). Strip them; if nothing remains it was pure
        // echo and never becomes a command.
        val text = stripSelfSpeech(t.text) ?: return
        val go = !requireWakeName() || commandWindowOpen() ||
            CommandParser.hasWakeName(text)
        if (!go) return
        // Overlapping windows re-transcribe the same speech — don't fire
        // the identical utterance twice within the dedup window. Wake-only
        // utterances are exempt: CommandSession has its own re-wake
        // suppressor, and a 6 s dedup would eat a legitimate second "hello"
        // (no re-ding, dead-feeling device).
        val isWake = CommandParser.parse(
            text, requireWakeName = false) is ParseResult.WakeOnly
        if (!isWake) {
            val now = SystemClock.elapsedRealtime()
            if (text == lastCaptionCommand &&
                now - lastCaptionCommandAtMs < CAPTION_COMMAND_DEDUP_MS
            ) return
            lastCaptionCommand = text
            lastCaptionCommandAtMs = now
        }
        Log.d(TAG, "caption -> command: \"$text\" (raw: \"${t.text}\")")
        mainHandler.post {
            if (enabled.get()) onCommand(Utterance(text, pcm, t.language))
        }
    }

    /**
     * Removes the app's own spoken phrases — wake/timeout prompts, the
     * wake hint, and the last TTS line — from a transcript, returning the
     * remainder or null when nothing user-spoken is left. Echo lands in
     * the rolling window mixed with the command, so exact-match rejection
     * isn't enough: the phrases are removed as token subsequences.
     */
    private fun stripSelfSpeech(raw: String): String? {
        val phrases = buildList {
            add(config.wakePrompt)
            add(config.windowTimeoutPrompt)
            add(CommandParser.WAKE_HINT)
            selfSpeech()?.let { add(it) }
        }.map { TranscriptFilter.normalize(it) }
            .filter { it.isNotBlank() }
        if (phrases.isEmpty()) {
            return TranscriptFilter.normalize(raw)
                .takeIf { it.isNotBlank() }
        }
        val tokens = TranscriptFilter.normalize(raw).split(' ')
            .filter { it.isNotBlank() }.toMutableList()
        for (p in phrases) {
            val pt = p.split(' ').filter { it.isNotBlank() }
            if (pt.isEmpty() || pt.size > tokens.size) continue
            var i = 0
            while (i <= tokens.size - pt.size) {
                if (tokens.subList(i, i + pt.size) == pt) {
                    repeat(pt.size) { tokens.removeAt(i) }
                } else {
                    i++
                }
            }
        }
        return tokens.joinToString(" ").takeIf { it.isNotBlank() }
    }

    private fun transcribe(segment: VadSegment) {
        try {
            val t = recognizer?.transcribe(segment.pcm)
                ?: return
            // Surface every display-worthy segment transcript on the "You
            // said" strip — commands still go through the stricter filter.
            if (TranscriptFilter.evaluate(
                    t, vadConfirmed = true, cfg = config,
                    skipBlocklist = true).accepted
            ) {
                reportHeard(t.text)
            }
            // Fallback mode only: a wake-only transcript ("hello") gets a
            // blocklist pass — it's on the blocklist itself, so the filter
            // would eat every fallback wake attempt. It still needs the
            // consecutive-hit rule before anything fires.
            val isWakeTranscript = wake == null &&
                (CommandParser.parse(t.text, requireWakeName = true)
                    is ParseResult.WakeOnly)
            val decision = TranscriptFilter.evaluate(
                t, vadConfirmed = true, cfg = config,
                skipBlocklist = isWakeTranscript,
            )
            TranscriptFilter.log(TAG, t, vadConfirmed = true, d = decision)
            if (!decision.accepted) {
                clipStore?.save(segment.pcm, decision.reason)
                return
            }
            if (!enabled.get()) return
            val text = stripSelfSpeech(t.text) ?: return
            if (wake == null && !fallbackWakeAllows(text)) return
            // Stamp the caption dedup keys — an identical phrase coming
            // back through a live-caption window shouldn't fire twice.
            lastCaptionCommand = text
            lastCaptionCommandAtMs = SystemClock.elapsedRealtime()
            mainHandler.post {
                if (enabled.get()) onCommand(Utterance(text, segment.pcm, t.language))
            }
        } catch (e: Exception) {
            reportError("Voice transcription failed: ${e.message}")
        } finally {
            whisperPending.decrementAndGet()
        }
    }

    /**
     * Transcript-fallback wake gating: a bare "hello" only fires after
     * [VoiceConfig.fallbackWakeConsecutive] consecutive wake transcripts
     * (VAD already confirmed speech for every segment that gets here).
     * Wake+command transcripts pass immediately.
     */
    private fun fallbackWakeAllows(text: String): Boolean {
        val result = CommandParser.parse(text, requireWakeName = true)
        return when (result) {
            is ParseResult.WakeOnly -> {
                consecutiveWakeHits++
                val ok = consecutiveWakeHits >= config.fallbackWakeConsecutive
                Log.d(TAG, "fallback wake hit $consecutiveWakeHits -> ${if (ok) "wake" else "waiting"}")
                ok
            }
            is ParseResult.Parsed -> { consecutiveWakeHits = 0; true }
            else -> {
                consecutiveWakeHits = 0
                // Non-wake transcripts still flow to the session — inside an
                // open window they may be commands; otherwise they're dropped.
                true
            }
        }
    }

    private fun reportError(message: String) {
        Log.e(TAG, message)
        mainHandler.post {
            onLoadingChanged(false)
            onError(message)
        }
    }

    private companion object {
        const val TAG = "GabVoice"
        const val GATE_RESET_MS = 1_000L
        /** How far past a segment's end a wake trigger still counts as
         *  belonging to it ("hello gab <pause> walk" hits the next segment). */
        const val WAKE_PAD_MS = 400
        const val MAX_PENDING_TRANSCRIPTS = 2
        /** Identical caption transcripts inside this window fire once —
         *  overlapping windows re-transcribe the same speech. */
        const val CAPTION_COMMAND_DEDUP_MS = 6_000L
        /** ~2 s at 32 ms/frame — how often the "vad peak" debug line prints. */
        const val VAD_LOG_FRAMES = 62
    }
}
