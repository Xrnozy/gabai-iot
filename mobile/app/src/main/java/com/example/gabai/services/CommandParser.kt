package com.example.gabai.services

import java.util.Locale

/** High-level voice commands the app understands. */
enum class Command {
    NAVIGATION,   // open Navigation
    READ_TEXT,    // open OCR / trigger read-aloud inside OCR
    COLOR,        // open Color / announce inside Color
    SETTINGS,     // open Settings
    BACK,         // leave the current screen
    STOP,         // stop current mode / go home
    REPEAT,       // repeat the last status line
    HELP,         // speak the command list
    SOS,          // "help me", "I need help": start the emergency countdown from any screen

    // Settings toggles — "voice commands off", "captions on", ...
    SET_VOICE_ON, SET_VOICE_OFF,           // voice commands enabled
    SET_WAKE_ON, SET_WAKE_OFF,             // wake-name required
    SET_TRANSCRIPT_ON, SET_TRANSCRIPT_OFF, // live transcript panel
    SET_LISTENING_ON, SET_LISTENING_OFF,   // screen-off listening
    SET_SPEECH_ON, SET_SPEECH_OFF,         // spoken feedback (TTS)
    SET_HAPTICS_ON, SET_HAPTICS_OFF,       // vibration
    SET_CONTRAST_ON, SET_CONTRAST_OFF,     // high contrast
    SET_BIG_TEXT_ON, SET_BIG_TEXT_OFF,     // large text
    SET_CAPTIONS_ON, SET_CAPTIONS_OFF,     // spoken-text panel
    RATE_SLOW, RATE_NORMAL, RATE_FAST,     // speech speed
}

sealed interface ParseResult {
    data class Parsed(val command: Command) : ParseResult

    /**
     * The phrase was just the wake name ("Gab", "hey gabai") with no command —
     * the assistant should acknowledge and wait for the next utterance.
     */
    data object WakeOnly : ParseResult

    /** A bare command word was heard without the wake name. */
    data object BareCommand : ParseResult

    data object NoMatch : ParseResult
}

/**
 * Maps raw STT phrases to [Command]s using a "wake name + command" grammar:
 *   "Gabai, walk" → NAVIGATION,  "Gabai, read" → READ_TEXT, ...
 *
 * The synonym table deliberately favors short, common words so speakers with
 * limited English can drive the whole app ("walk", "read", "color", "back").
 * Multi-word synonyms are matched before single words so "go back" resolves
 * to BACK instead of NAVIGATION via "go".
 */
object CommandParser {

    const val WAKE_NAME = "gab"

    /**
     * Wake-word variants. "Gab" is what's taught — "gab", "hello gab" and
     * "hey gab" all work. "Hello" alone stays accepted for users who picked
     * it up earlier. First tokens starting with "gab"/"hell"/"hall" also
     * count (see [stripWake]). Multi-word mishears are listed verbatim.
     */
    private val WAKE_WORD_VARIANTS = listOf(
        "hello", "hullo", "hallo", "halo", "hellow", "hell o", "hell oh", "ello",
        "gab", "gabai", "gabay", "gabaey", "gabey", "gabei", "gabee", "gabi",
        "gabby", "gab eh", "gab ay", "gab ee", "go buy", "guy bye",
    )

    /**
     * Two-word "hello gab" mishears — the second token drifts phonetically
     * far enough ("hello god", "yellow cab") that the single-token fuzzy
     * matcher can't rescue it ('c'/'d' tokens aren't even anchored on a
     * wake first letter). Listed verbatim; [stripWake] tries the longest
     * phrases first, so these also strip as a prefix before a command.
     */
    private val WAKE_PHRASE_VARIANTS = listOf(
        "hello gab", "hello gabai", "hello god", "hello gabe", "hello gav",
        "hello gob", "hello gap", "hello grab", "hello gad", "hello gov",
        "hello gub", "hello gaap", "hello gabriel",
        "hello cab", "hello cap", "hello cat", "hello kev", "hello jab",
        "hello dab", "hello dad",
        "hello dog", "hello doug", "hello dock", "hello dawg",
        "yellow gab", "yellow cab", "yellow god", "yellow cap",
        "yellow gap", "yellow gav", "yellow cat",
        "hellow gab", "hallo gab", "hullo gab", "ello gab", "a low gab",
    )

    private val WAKE_VARIANTS = WAKE_WORD_VARIANTS + WAKE_PHRASE_VARIANTS

    /** Optional fillers allowed before the wake name: "hey gabai …". */
    private val WAKE_PREFIXES = listOf("hey", "okay", "ok", "um", "uh")

    /**
     * Single-token wake variants eligible for fuzzy matching. A lone wake
     * word gets the worst STT spellings ("gav", "gap", "grab") because the
     * model has no phrase context — a stray ding is cheap, a missed wake
     * word is confusing. Only applied to one-word utterances.
     */
    private val SINGLE_WAKE_VARIANTS = WAKE_VARIANTS.filter { ' ' !in it }

    /** Command → spoken synonyms. Internal visibility so [CommandMatcher]
     *  scores the same phrases without duplicating the table. */
    internal val synonymTable: Map<Command, List<String>> = mapOf(
        Command.NAVIGATION to listOf(
            "open navigation", "open nav",
            "navigation", "navigate", "nav", "directions",
            "walk", "walking", "start walking", "go", "move",
        ),
        Command.READ_TEXT to listOf(
            "what does it say", "what is written", "read text", "scan text",
            "open read", "open ocr",
            "read", "text", "ocr", "o c r", "scan", "scanning", "picture",
            "letter", "letters", "words",
        ),
        Command.COLOR to listOf(
            "what color", "what colour",
            "identify color", "identify colour", "open color",
            "color", "colour", "colors", "colours",
        ),
        Command.SETTINGS to listOf(
            "open settings", "change settings", "settings menu",
            "settings", "setting", "options", "option", "set up", "setup",
            "preferences", "preference", "configure",
        ),
        Command.BACK to listOf(
            "go back", "back", "home", "return",
        ),
        Command.STOP to listOf(
            "shut up", "stop", "quiet", "cancel", "off",
        ),
        Command.REPEAT to listOf(
            "say again", "repeat", "again", "pardon",
        ),
        Command.SOS to listOf(
            "i need help", "need help", "help me", "send help", "get help", "call for help", "call help", "please help",
            "i need emergency", "need emergency", "emergency", "sos", "s o s",
        ),
        Command.HELP to listOf(
            "what can i say", "command list", "commands", "help",
        ),

        // Settings toggles — explicit "X on"/"X off" pairs so a spoken
        // intent is idempotent ("captions on" never turns captions off).
        Command.SET_VOICE_ON to listOf(
            "voice commands on", "voice control on",
            "turn voice on", "turn on voice", "enable voice",
        ),
        Command.SET_VOICE_OFF to listOf(
            "voice commands off", "voice control off", "voice off",
            "turn voice off", "turn off voice", "disable voice",
        ),
        Command.SET_WAKE_ON to listOf(
            "wake word on", "wake name on", "say hello first",
            "require wake word",
        ),
        Command.SET_WAKE_OFF to listOf(
            "wake word off", "wake name off", "no wake word",
            "no wake name",
        ),
        Command.SET_TRANSCRIPT_ON to listOf(
            "live transcript on", "transcript on",
            "show what i say", "show transcript",
        ),
        Command.SET_TRANSCRIPT_OFF to listOf(
            "live transcript off", "transcript off",
            "hide what i say", "hide transcript",
        ),
        Command.SET_LISTENING_ON to listOf(
            "screen off listening on", "background listening on",
            "listen in background", "listening on",
        ),
        Command.SET_LISTENING_OFF to listOf(
            "screen off listening off", "background listening off",
            "listening off",
        ),
        Command.SET_SPEECH_ON to listOf(
            "speech on", "voice output on", "spoken feedback on",
            "talk on", "unmute",
        ),
        Command.SET_SPEECH_OFF to listOf(
            "speech off", "voice output off", "spoken feedback off",
            "talk off", "mute speech", "mute",
        ),
        Command.SET_HAPTICS_ON to listOf(
            "haptics on", "vibration on", "vibrate on",
        ),
        Command.SET_HAPTICS_OFF to listOf(
            "haptics off", "vibration off", "vibrate off",
        ),
        Command.SET_CONTRAST_ON to listOf(
            "high contrast on", "contrast on",
        ),
        Command.SET_CONTRAST_OFF to listOf(
            "high contrast off", "contrast off", "normal contrast",
        ),
        Command.SET_BIG_TEXT_ON to listOf(
            "large text on", "big text on", "large font on", "bigger text",
        ),
        Command.SET_BIG_TEXT_OFF to listOf(
            "large text off", "big text off", "large font off",
            "normal text", "smaller text",
        ),
        Command.SET_CAPTIONS_ON to listOf(
            "text panel on", "captions on", "caption on",
            "show captions", "spoken text on",
        ),
        Command.SET_CAPTIONS_OFF to listOf(
            "text panel off", "captions off", "caption off",
            "hide captions", "spoken text off",
        ),
        Command.RATE_SLOW to listOf(
            "speech slow", "speak slower", "talk slower",
            "speak slowly", "talk slowly",
            "slow speech", "slow down", "slower", "slowly",
        ),
        Command.RATE_NORMAL to listOf(
            "speech normal", "normal speed", "normal speech",
            "speak normally", "normal rate",
        ),
        Command.RATE_FAST to listOf(
            "speech fast", "speak faster", "talk faster",
            "fast speech", "speed up", "faster",
        ),
    )

    /** Keyword-bias list handed to the Whistle decoder. */
    val KEYWORDS: List<String> = buildList {
        addAll(WAKE_VARIANTS)
        addAll(WAKE_PREFIXES)
        synonymTable.values.forEach { addAll(it) }
    }.distinct()

    /**
     * Bias only toward the wake name — the word STT is worst at ("gabai" is
     * not a real word). Command words ("walk", "read", "settings") are
     * common English and transcribe fine unaided; biasing toward them made
     * the decoder hallucinate them out of background noise.
     */
    val WAKE_KEYWORDS: List<String> = (WAKE_VARIANTS + WAKE_PREFIXES).distinct()

    /** Short line spoken when the user asks for help — also the home
     *  screen's one-time intro (MainActivity speaks it verbatim). */
    const val HELP_HINT =
        "Say: walk. Read. Color. Or settings. " +
        "where commands like captions off and speak faster work too. " +
        "Say go back to leave a screen."

    /** Hint spoken when a bare command was heard but the wake name is required. */
    const val WAKE_HINT =
        "Say Gab first, like: Gab, walk."

    /**
     * Parses [raw] into a [ParseResult].
     * When [requireWakeName] is true, phrases must start with the wake name
     * ("gabai walk"); a bare command word returns [ParseResult.BareCommand] so
     * callers can teach the user instead of silently ignoring them.
     * Saying just "gabai" returns [ParseResult.WakeOnly] — callers ack with a
     * chime and hand the next utterance to [parseAfterWake].
     */
    fun parse(
        raw: String,
        requireWakeName: Boolean = true,
        allowed: Set<Command>? = null,
    ): ParseResult {
        val tokens = normalize(raw)
        if (tokens.isEmpty()) return ParseResult.NoMatch
        if (isEmergency(tokens)) return ParseResult.Parsed(Command.SOS)

        val commandText = stripWake(tokens)
        if (commandText != null) {
            if (commandText.isEmpty()) return ParseResult.WakeOnly
            // "gab gab gab" said in one breath merges into a single
            // utterance — repeated wake tokens still mean "listen to me".
            if (commandText.all { isWakeToken(it) }) return ParseResult.WakeOnly
            return matchCommand(commandText, allowed)
                ?.let { ParseResult.Parsed(it) }
                ?: ParseResult.NoMatch
        }

        // A lone near-miss of the wake name ("gav", "grab") still wakes —
        // fuzzy match only ever applies to one-word utterances.
        if (tokens.size == 1 && isFuzzyWake(tokens[0])) return ParseResult.WakeOnly

        if (!requireWakeName) {
            // Wake-free mode still demands the command span the whole
            // utterance — loose substring matching would fire on ambient
            // speech like "we got the settings".
            val command = matchCommand(tokens, allowed) ?: return ParseResult.NoMatch
            return if (consumedAll(tokens, command)) {
                ParseResult.Parsed(command)
            } else {
                ParseResult.NoMatch
            }
        }

        // The whole phrase is a plausible command attempt minus the wake name —
        // only flag it when the match covers the ENTIRE phrase ("walk", "go
        // back"), so background speech containing a stray keyword is ignored.
        val whole = matchCommand(tokens, allowed)
        return if (whole != null && consumedAll(tokens, whole)) {
            ParseResult.BareCommand
        } else {
            ParseResult.NoMatch
        }
    }

    /**
     * Parses an utterance spoken inside the command window that the wake ding
     * opened — the wake name is implicit ("Gab" … "walk").
     *
     * The window is generous: a literal command phrase anywhere in the
     * utterance fires, so noise or repeated words around it are ignored
     * ("uh walk maybe" → NAVIGATION). The user was just asked for a command,
     * so a stray keyword is far more likely intent than accident — except
     * commands in [strict] (session-ending/destructive), which still must
     * span the WHOLE utterance. Wake-prefixed phrases still work
     * ("gab walk"), and repeating just the wake name returns
     * [ParseResult.WakeOnly] so the window can be re-armed.
     */
    fun parseAfterWake(
        raw: String,
        allowed: Set<Command>? = null,
        strict: Set<Command> = emptySet(),
    ): ParseResult {
        val tokens = normalize(raw)
        if (tokens.isEmpty()) return ParseResult.NoMatch
        if (isEmergency(tokens)) return ParseResult.Parsed(Command.SOS)

        val commandText = stripWake(tokens)
        if (commandText != null) {
            if (commandText.isEmpty()) return ParseResult.WakeOnly
            if (commandText.all { isWakeToken(it) }) return ParseResult.WakeOnly
            return matchCommand(commandText, allowed)
                ?.let { ParseResult.Parsed(it) }
                ?: ParseResult.NoMatch
        }

        // A lone near-miss of the wake name re-arms the window.
        if (tokens.size == 1 && isFuzzyWake(tokens[0])) return ParseResult.WakeOnly

        val command = matchCommand(tokens, allowed) ?: return ParseResult.NoMatch
        return if (consumedAll(tokens, command) || command !in strict) {
            ParseResult.Parsed(command)
        } else {
            ParseResult.NoMatch
        }
    }

    /**
     * True when the phrase opens with the wake name (exact, "gab*" prefix or
     * a fuzzy lone-word mishear). Used to gate the LLM fallback — ambient
     * speech must never reach the intent model.
     */
    fun hasWakeName(raw: String): Boolean {
        val tokens = normalize(raw)
        if (tokens.isEmpty()) return false
        if (stripWake(tokens) != null) return true
        return tokens.size == 1 && isFuzzyWake(tokens[0])
    }

    /** Any token that on its own would count as the wake name —
     *  exact variant, "gab*" prefix, or fuzzy single-word mishear. */
    private fun isWakeToken(token: String): Boolean =
        token in SINGLE_WAKE_VARIANTS || isWakePrefixToken(token) || isFuzzyWake(token)

    /** First letters every wake variant starts with — the fuzzy matcher only
     *  applies when the token opens with one, so "cab"/"tab"/"cello"/"jello"
     *  can't trigger it. */
    private val WAKE_FIRST_LETTERS = SINGLE_WAKE_VARIANTS.map { it.first() }.toSet()

    /** Prefix rule shared by [stripWake]/[isWakeToken] — covers wake-word
     *  pronunciations STT spells unpredictably (gabaey, hellow…). */
    private fun isWakePrefixToken(token: String): Boolean =
        token.startsWith("gab") || token.startsWith("hell") || token.startsWith("hall")

    /** True for a lone token within one edit of a wake variant (see
     *  [SINGLE_WAKE_VARIANTS]) — anchored on a wake-word first letter. */
    private fun isFuzzyWake(token: String): Boolean {
        if (token.length !in 2..7 || token.first() !in WAKE_FIRST_LETTERS) return false
        return SINGLE_WAKE_VARIANTS.any { editDistance(token, it) <= 1 }
    }

    /** Levenshtein distance — only ever run on single short tokens. */
    private fun editDistance(a: String, b: String): Int {
        val prev = IntArray(b.length + 1) { it }
        for (i in 1..a.length) {
            var diag = prev[0]
            prev[0] = i
            for (j in 1..b.length) {
                val above = prev[j]
                prev[j] = minOf(
                    prev[j] + 1,
                    prev[j - 1] + 1,
                    diag + if (a[i - 1] == b[j - 1]) 0 else 1,
                )
                diag = above
            }
        }
        return prev[b.length]
    }

    /** Tokens after the wake name in [raw], or null when absent — lets
     *  [CommandMatcher] fuzzy-match just the command part of wake-gated
     *  utterances. */
    internal fun textAfterWake(raw: String): List<String>? {
        val tokens = normalize(raw)
        if (tokens.isEmpty()) return null
        return stripWake(tokens)
    }

    /** Returns tokens after the wake name, or null if the wake name is absent. */
    private fun stripWake(tokens: List<String>): List<String>? {
        var t = tokens
        // Drop a leading filler like "hey" in "hey gabai".
        if (t.isNotEmpty() && t.first() in WAKE_PREFIXES) t = t.drop(1)
        for (variant in WAKE_VARIANTS.sortedByDescending { it.split(' ').size }) {
            val v = variant.split(' ')
            if (t.size >= v.size && t.subList(0, v.size) == v) {
                return t.drop(v.size)
            }
        }
        // Prefix rule: any first token starting like a wake word counts —
        // covers pronunciations the STT spells unpredictably (gabaey,
        // hellow…). (t may be empty after dropping a lone "hey"/"ok" filler.)
        if (t.isNotEmpty() && isWakePrefixToken(t.first())) return t.drop(1)
        return null
    }

    /**
     * First matching command; synonyms with more words are tried first so
     * "go back" → BACK (not "go" → NAVIGATION) and "what color" → COLOR
     * (not "what" → REPEAT, "color" also wins by phrase specificity).
     */
    private fun matchCommand(tokens: List<String>, allowed: Set<Command>? = null): Command? {
        val phrases = synonymTable.entries
            .filter { (cmd, _) -> allowed == null || cmd in allowed }
            .flatMap { (cmd, syns) -> syns.map { cmd to it } }
            .sortedByDescending { (_, syn) -> syn.split(' ').size }
        for ((cmd, syn) in phrases) {
            val s = syn.split(' ')
            if (containsPhrase(tokens, s)) return cmd
        }
        return null
    }

    /**
     * An emergency works from any screen, with or without the wake name, and is never limited by a screen's allowed set.
     * A multi-word phrase counts anywhere in the utterance ("please, help me now"); a one-word phrase ("emergency", "sos")
     * only when it is the whole utterance, so ambient talk does not set it off. It only starts the cancelable countdown.
     */
    internal fun isEmergency(tokens: List<String>): Boolean {
        val body = stripWake(tokens) ?: tokens
        return synonymTable.getValue(Command.SOS).map { it.split(' ') }.any { p ->
            if (p.size > 1) containsPhrase(body, p) else body == p
        }
    }

    /** True when the matched command's synonym spans every token — used for
     *  the strict BareCommand check so "uh walk maybe" doesn't false-positive. */
    private fun consumedAll(tokens: List<String>, command: Command): Boolean {
        val syns = synonymTable.getValue(command)
            .map { it.split(' ') }
            .sortedByDescending { it.size }
        for (s in syns) {
            if (s.size == tokens.size && containsPhrase(tokens, s)) return true
        }
        return false
    }

    private fun containsPhrase(tokens: List<String>, phrase: List<String>): Boolean {
        if (phrase.size > tokens.size) return false
        for (i in 0..tokens.size - phrase.size) {
            if (tokens.subList(i, i + phrase.size) == phrase) return true
        }
        return false
    }

    private fun normalize(raw: String): List<String> {
        val tokens = raw.lowercase(Locale.ROOT)
            .replace(Regex("[^a-z0-9 ]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
            .takeIf { it.isNotBlank() }
            ?.split(' ')
            ?: return emptyList()
        // STT stutters on noise ("walk, walk, walk") — collapse consecutive
        // repeats so a spammed command still parses.
        return tokens.fold(mutableListOf<String>()) { acc, t ->
            if (acc.isEmpty() || acc.last() != t) acc += t
            acc
        }
    }
}
