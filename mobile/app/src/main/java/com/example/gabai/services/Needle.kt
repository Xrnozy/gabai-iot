package com.example.gabai.services

import android.content.Context
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject

/**
 * On-device intent reasoning backed by Needle 3 — a tiny tool-calling model
 * that shares the [Whistle] speech engine: `needle_load` keeps whichever
 * model a .cact carries, so one runtime does STT and reasoning side by side.
 *
 * The model gets a fixed system prompt and the app's tool catalogue at load
 * time ([SYSTEM_PROMPT]/[TOOLS_JSON]); each utterance then returns JSON with
 * `function_calls`, `reasoning` and a calibrated `confidence`. An off-topic
 * phrase returns an empty call list rather than a guess.
 *
 * Threading: the engine is process-global and not thread-safe, so every
 * call is serialised on [Whistle]'s monitor — the same lock its own
 * `@Synchronized` methods take.
 */
object Needle {

    private const val TAG = "GabVoice"
    private const val MODEL_ASSET = "needle3.cact"
    private const val MAX_NEW_TOKENS = 96

    /**
     * The users are blind or low-vision and many speak simple, broken or
     * accented English — the prompt tells the model to reason about intent,
     * not exact words.
     */
    private const val SYSTEM_PROMPT =
        "You are Gab, the voice assistant inside GabAi, an app that helps " +
        "blind and low-vision users. Input is a short transcribed voice " +
        "command. Users may speak simple or broken English, single words, " +
        "or describe a function instead of naming it. Call the one tool " +
        "that matches the user's intent. If nothing matches an app " +
        "function, call no tools."

    /** The app's "buttons", declared so the model can pick among them. */
    private const val TOOLS_JSON = """[
        {"name":"open_navigation","description":"Start walking navigation with obstacle guidance","parameters":{"type":"object","properties":{}}},
        {"name":"read_text","description":"Read printed text out loud — signs, labels, papers","parameters":{"type":"object","properties":{}}},
        {"name":"detect_color","description":"Say the color of what the camera points at","parameters":{"type":"object","properties":{}}},
        {"name":"open_settings","description":"Open app settings","parameters":{"type":"object","properties":{}}},
        {"name":"go_back","description":"Go back to the previous screen or home","parameters":{"type":"object","properties":{}}},
        {"name":"stop","description":"Stop talking or stop the current mode","parameters":{"type":"object","properties":{}}},
        {"name":"repeat","description":"Repeat the last spoken message","parameters":{"type":"object","properties":{}}},
        {"name":"help","description":"Explain the voice commands the user can say","parameters":{"type":"object","properties":{}}}
    ]"""

    private val TOOL_TO_COMMAND = mapOf(
        "open_navigation" to Command.NAVIGATION,
        "read_text" to Command.READ_TEXT,
        "detect_color" to Command.COLOR,
        "open_settings" to Command.SETTINGS,
        "go_back" to Command.BACK,
        "stop" to Command.STOP,
        "repeat" to Command.REPEAT,
        "help" to Command.HELP,
    )

    private val loaded = AtomicBoolean(false)

    /** Set only after the model AND its tool prompt are live — readable
     *  without the engine lock so callers on the main thread never stall
     *  behind an in-flight transcription. */
    @Volatile
    private var ready = false

    private external fun nativeLoad(cact: ByteArray): Int
    private external fun nativeModels(): Int
    private external fun nativeInit(systemPrompt: String, toolsJson: String): Int
    private external fun nativeComplete(input: String, maxNewTokens: Int): String?
    private external fun nativeCompleteAudio(pcm: FloatArray, maxNewTokens: Int): String?
    private external fun nativeSetAudio(
        language: String?,
        keywords: String?,
        wordTimestamps: Int,
    )
    private external fun nativeReset()
    private external fun nativeLastError(): String?

    /** True when the text model loaded and initialised — the keyword parser
     *  alone covers everything when this is false. */
    val isAvailable: Boolean
        get() = ready

    /** Loads needle3.cact and installs the system prompt + tools. Safe to
     *  call repeatedly; loads once. Throws when the model is missing or
     *  init fails — callers should treat that as "feature off". */
    fun load(context: Context) {
        synchronized(Whistle) {
            if (loaded.get()) return
            val bytes = context.applicationContext.assets.open(MODEL_ASSET)
                .use { it.readBytes() }
            if (nativeLoad(bytes) < 0) {
                throw IllegalStateException(nativeLastError() ?: "needle failed to load")
            }
            if (nativeInit(SYSTEM_PROMPT, TOOLS_JSON) < 0) {
                throw IllegalStateException(nativeLastError() ?: "needle_init failed")
            }
            loaded.set(true)
            ready = true
            Log.d(TAG, "needle3 loaded — intent reasoning on")
        }
    }

    /**
     * Reasons about [phrase] against the tool catalogue. Returns matching
     * [Command]s (usually zero or one); empty on off-topic phrases, engine
     * errors or when the model isn't loaded. Runs synchronously on the
     * calling thread — call off the main thread.
     */
    fun commandsFor(phrase: String): List<Command> {
        if (!ready) return emptyList()
        return synchronized(Whistle) {
            nativeReset()
            val raw = nativeComplete(phrase, MAX_NEW_TOKENS)
            if (raw == null) {
                Log.w(TAG, "needle_complete failed: ${nativeLastError()}")
                emptyList()
            } else {
                parseCommands(raw, phrase)
            }
        }
    }

    /**
     * Speech-in tool calling: hands the raw 16 kHz clip to
     * `needle_complete`, which transcribes it in-engine and answers the
     * transcript — no text round-trip, so an STT mishear can't garble the
     * intent before the model sees it. Preferred over [commandsFor]
     * whenever the utterance PCM is at hand. Same threading rules.
     */
    fun commandsForAudio(pcm: FloatArray, language: String? = null): List<Command> {
        if (!ready || pcm.isEmpty()) return emptyList()
        return synchronized(Whistle) {
            nativeReset()
            nativeSetAudio(language, CommandParser.KEYWORDS.joinToString("\n"), 0)
            val raw = nativeCompleteAudio(pcm, MAX_NEW_TOKENS)
            if (raw == null) {
                Log.w(TAG, "needle_complete(audio) failed: ${nativeLastError()}")
                emptyList()
            } else {
                parseCommands(raw, "<audio ${pcm.size / 16000}s>")
            }
        }
    }

    private fun parseCommands(raw: String, phrase: String): List<Command> {
        return try {
            val calls = JSONObject(raw).optJSONArray("function_calls")
                ?: return emptyList()
            val commands = buildList {
                for (i in 0 until calls.length()) {
                    val name = calls.getJSONObject(i).optString("name")
                    TOOL_TO_COMMAND[name]?.let { add(it) }
                }
            }.distinct()
            Log.d(TAG, "needle \"$phrase\" -> $commands (${raw.take(200)})")
            commands
        } catch (e: Exception) {
            Log.e(TAG, "bad needle response for \"$phrase\": $raw", e)
            emptyList()
        }
    }
}
