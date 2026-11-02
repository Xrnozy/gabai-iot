package com.example.gabai

import com.example.gabai.services.Command
import com.example.gabai.services.CommandParser
import com.example.gabai.services.ParseResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CommandParserTest {

    private fun parsed(raw: String, wake: Boolean = true): Command? =
        (CommandParser.parse(raw, requireWakeName = wake) as? ParseResult.Parsed)?.command

    @Test
    fun wakeNamePlusCommand() {
        assertEquals(Command.NAVIGATION, parsed("Gab, walk"))
        assertEquals(Command.NAVIGATION, parsed("gab navigation"))
        assertEquals(Command.NAVIGATION, parsed("Gabai, walk"))
        assertEquals(Command.READ_TEXT, parsed("Gab read"))
        assertEquals(Command.READ_TEXT, parsed("Gab, text"))
        assertEquals(Command.COLOR, parsed("Gab color"))
        assertEquals(Command.SETTINGS, parsed("Gab settings"))
        assertEquals(Command.BACK, parsed("Gab back"))
        assertEquals(Command.STOP, parsed("Gab stop"))
        assertEquals(Command.HELP, parsed("Gab help"))
    }

    @Test
    fun wakeNameVariants() {
        // TTS says "Gabei"; Filipino-accented speech comes out "Gabaey"/"gabey"
        assertEquals(Command.NAVIGATION, parsed("gabei walk"))
        assertEquals(Command.NAVIGATION, parsed("gabaey walk"))
        assertEquals(Command.NAVIGATION, parsed("gabey walk"))
        assertEquals(Command.NAVIGATION, parsed("hey gab walk"))
        assertEquals(Command.NAVIGATION, parsed("gabay walk"))
        assertEquals(Command.NAVIGATION, parsed("gabi go"))
        assertEquals(Command.NAVIGATION, parsed("okay gabai walk"))
        // Prefix rule: any "gab*"-leading token
        assertEquals(Command.NAVIGATION, parsed("gabbb walk"))
    }

    @Test
    fun lowEnglishSynonyms() {
        assertEquals(Command.NAVIGATION, parsed("gabai go"))
        assertEquals(Command.NAVIGATION, parsed("gabai start walking"))
        assertEquals(Command.READ_TEXT, parsed("gabai words"))
        assertEquals(Command.READ_TEXT, parsed("gabai letter"))
        assertEquals(Command.COLOR, parsed("gabai what color"))
        assertEquals(Command.SETTINGS, parsed("gabai options"))
        assertEquals(Command.STOP, parsed("gabai quiet"))
        assertEquals(Command.REPEAT, parsed("gabai say again"))
    }

    @Test
    fun multiWordBeatsSingleWord() {
        // "go back" must be BACK, not "go" → NAVIGATION
        assertEquals(Command.BACK, parsed("gabai go back"))
        // "what color" must be COLOR, not "what" → REPEAT
        assertEquals(Command.COLOR, parsed("gabai what color"))
        // "read text" → READ_TEXT regardless
        assertEquals(Command.READ_TEXT, parsed("gabai read text"))
        // "shut up" → STOP
        assertEquals(Command.STOP, parsed("gabai shut up"))
    }

    @Test
    fun bareWakeNameOpensCommandWindow() {
        // "Gab" alone asks for the command window, not a command — the
        // session plays the wake ding and waits for the next utterance.
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("gabai"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("gab"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hey gabai"))
        // ...but asking for help explicitly still works
        assertEquals(Command.HELP, parsed("gabai help"))
    }

    @Test
    fun helloWakeWord() {
        // "Hello" is the taught wake word — everyone can say it and STT
        // transcribes it reliably.
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("Hello!"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hey hello"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hullo"))
        // Common mishears/spellings still wake.
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hellow"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hell oh"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hella"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("helo"))
        // Wake + command in one breath.
        assertEquals(Command.NAVIGATION, parsed("hello walk"))
        assertEquals(Command.READ_TEXT, parsed("hello read"))
        assertEquals(Command.SETTINGS, parsed("hello settings"))
        assertEquals(Command.STOP, parsed("hello off"))
        // ...but words merely near "hello" on other letters don't trigger.
        assertEquals(ParseResult.NoMatch, CommandParser.parse("cello"))
        assertEquals(ParseResult.NoMatch, CommandParser.parse("jello"))
        assertEquals(ParseResult.NoMatch, CommandParser.parse("yellow"))
        // "help" stays a command, not a wake mishear.
        assertEquals(Command.HELP, parsed("gabai help"))
        assertEquals(ParseResult.BareCommand, CommandParser.parse("help"))
    }

    @Test
    fun fuzzyWakeSpellings() {
        // Lone-word STT mishears of "gab" still wake — a stray ding is
        // cheaper than a missed wake word.
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("gav"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("gap"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("grab"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("gob"))
        // Repeated wake words merged into one utterance.
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("gab gab"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("gab gabai gab"))
        // ...but near-misses anchored on other letters don't trigger,
        // and commands inside longer phrases still can't fire.
        assertEquals(ParseResult.NoMatch, CommandParser.parse("cab"))
        assertEquals(ParseResult.NoMatch, CommandParser.parse("gap walk"))
        assertEquals(Command.NAVIGATION, parsed("gab gab walk"))
    }

    @Test
    fun repeatedWordsStillParse() {
        // STT stutter-repetition: a spammed command collapses to itself.
        assertEquals(Command.NAVIGATION, parsed("hello walk walk"))
        assertEquals(Command.NAVIGATION, parsed("hello walk walk walk walk"))
        assertEquals(Command.BACK, parsed("hello go go back"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello hello"))
        // Inside the command window the same applies.
        assertEquals(
            Command.NAVIGATION,
            (CommandParser.parseAfterWake("walk walk") as? ParseResult.Parsed)?.command)
    }

    @Test
    fun wakeNameGate() {
        // The intent model only sees wake-gated speech.
        assertTrue(CommandParser.hasWakeName("gab please help me walk outside"))
        assertTrue(CommandParser.hasWakeName("hey gabai do something"))
        assertTrue(CommandParser.hasWakeName("gav"))   // fuzzy lone mishear
        assertTrue(!CommandParser.hasWakeName("please help me walk outside"))
        assertTrue(!CommandParser.hasWakeName("the weather is nice"))
        assertTrue(!CommandParser.hasWakeName(""))
    }

    @Test
    fun afterWakeCommands() {
        // Inside the window the wake name is implicit.
        assertEquals(
            Command.NAVIGATION,
            (CommandParser.parseAfterWake("walk") as? ParseResult.Parsed)?.command)
        assertEquals(
            Command.READ_TEXT,
            (CommandParser.parseAfterWake("read") as? ParseResult.Parsed)?.command)
        assertEquals(
            Command.BACK,
            (CommandParser.parseAfterWake("go back") as? ParseResult.Parsed)?.command)
        assertEquals(
            Command.HELP,
            (CommandParser.parseAfterWake("help") as? ParseResult.Parsed)?.command)
        // Wake-prefixed phrases still work inside the window.
        assertEquals(
            Command.NAVIGATION,
            (CommandParser.parseAfterWake("gab walk") as? ParseResult.Parsed)?.command)
        // Re-saying the wake name re-arms the window.
        assertEquals(ParseResult.WakeOnly, CommandParser.parseAfterWake("gabai"))
    }

    @Test
    fun afterWakeIsGenerous() {
        // Inside the window a literal command phrase anywhere in the
        // utterance fires — noise and repeated words around it are ignored.
        assertEquals(
            Command.NAVIGATION,
            (CommandParser.parseAfterWake("uh walk maybe") as? ParseResult.Parsed)?.command)
        assertEquals(
            Command.COLOR,
            (CommandParser.parseAfterWake("what is the color") as? ParseResult.Parsed)?.command)
        assertEquals(
            Command.READ_TEXT,
            (CommandParser.parseAfterWake("please read this for me") as? ParseResult.Parsed)?.command)
        assertEquals(
            Command.SET_CAPTIONS_OFF,
            (CommandParser.parseAfterWake("turn the captions off please") as? ParseResult.Parsed)?.command)
        // Session-ending commands (the strict set) still need the whole
        // utterance — window chatter can't stop a session.
        assertEquals(ParseResult.NoMatch,
            CommandParser.parseAfterWake("don't turn it off", strict = setOf(Command.STOP)))
        assertEquals(
            Command.STOP,
            (CommandParser.parseAfterWake("off", strict = setOf(Command.STOP))
                as? ParseResult.Parsed)?.command)
        // ...but utterances with no command at all still stay silent.
        assertEquals(ParseResult.NoMatch, CommandParser.parseAfterWake(""))
        assertEquals(ParseResult.NoMatch, CommandParser.parseAfterWake("banana"))
        // Lone filler after the wake doesn't crash.
        assertEquals(ParseResult.NoMatch, CommandParser.parse("hey"))
        assertEquals(ParseResult.NoMatch, CommandParser.parseAfterWake("hey"))
    }

    @Test
    fun helloGabMishears() {
        // Two-word "hello gab" mishears — phonetic drift far from the
        // single-token variants ("hello god", "yellow cab", ...).
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello gab"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello god"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello cab"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello gabe"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello gabriel"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("yellow cab"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("yellow god"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("a low gab"))
        // ...and they still strip as a prefix before a command.
        assertEquals(Command.NAVIGATION, parsed("hello god walk"))
        assertEquals(Command.READ_TEXT, parsed("yellow cab read"))
        assertEquals(Command.COLOR, parsed("hello gab color"))
    }

    @Test
    fun heardOnDevice() {
        // Every line below is a real Whisper transcript from the field
        // (heard_transcripts.txt) — pin the behavior they produced.
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello gabber"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello Gaba"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello God"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello gab,"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse(
            "hello gababababababababababababababababab"))
        // Second field batch: "hello dog" and friends.
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello dog"))
        assertEquals(ParseResult.WakeOnly, CommandParser.parse("hello doug"))
        // Stutter-collapsed wake+command still fires.
        assertEquals(Command.NAVIGATION, parsed("hello, go, go, go"))
        // Natural phrasings resolve inside the window.
        assertEquals(
            Command.COLOR,
            (CommandParser.parseAfterWake("what color is this")
                as? ParseResult.Parsed)?.command)
        assertEquals(
            Command.SET_LISTENING_ON,
            (CommandParser.parseAfterWake("listening on")
                as? ParseResult.Parsed)?.command)
        // User's actual phrasings for slower speech and opening a mode.
        assertEquals(Command.RATE_SLOW, parsed("hello speak slowly"))
        assertEquals(Command.RATE_SLOW, parsed("hello can you speak slower"))
        assertEquals(Command.NAVIGATION, parsed("hello open navigation"))
        assertEquals(Command.BACK, parsed("hello go back"))
        // Inside the window the same phrasings resolve too.
        assertEquals(
            Command.RATE_SLOW,
            (CommandParser.parseAfterWake("can you speak slowly")
                as? ParseResult.Parsed)?.command)
        assertEquals(
            Command.NAVIGATION,
            (CommandParser.parseAfterWake("open navigation")
                as? ParseResult.Parsed)?.command)
        // ...while noise transcripts stay dead.
        assertEquals(ParseResult.NoMatch, CommandParser.parse("notification."))
    }

    @Test
    fun wakeRequired() {
        // Exact bare command attempt → hint, not silent ignore
        assertEquals(ParseResult.BareCommand, CommandParser.parse("walk"))
        assertEquals(ParseResult.BareCommand, CommandParser.parse("read"))
        assertEquals(ParseResult.BareCommand, CommandParser.parse("settings"))
        // Stray keyword inside longer speech → ignore (no false positives)
        assertEquals(ParseResult.NoMatch, CommandParser.parse("uh walk maybe"))
        assertEquals(ParseResult.NoMatch, CommandParser.parse("what is the color"))
        // Empty / noise
        assertEquals(ParseResult.NoMatch, CommandParser.parse(""))
        assertEquals(ParseResult.NoMatch, CommandParser.parse("!!!"))
    }

    @Test
    fun wakeNotRequired() {
        assertEquals(Command.NAVIGATION, parsed("walk", wake = false))
        assertEquals(Command.READ_TEXT, parsed("read", wake = false))
        assertEquals(Command.BACK, parsed("go back", wake = false))
        // Wake-name phrases still work when the requirement is off
        assertEquals(Command.NAVIGATION, parsed("gabai walk", wake = false))
        // ...but bare mode is strict too — the command must span the whole
        // utterance, so ambient speech can't fire it.
        assertEquals(ParseResult.NoMatch, CommandParser.parse("we got the settings", requireWakeName = false))
        assertEquals(ParseResult.NoMatch, CommandParser.parse("please walk", requireWakeName = false))
    }

    @Test
    fun requestedCommandAliases() {
        // The aliases the user asked for — inside and outside the window.
        assertEquals(Command.READ_TEXT, parsed("hello scan"))
        assertEquals(Command.READ_TEXT, parsed("hello scan text"))
        assertEquals(Command.READ_TEXT, parsed("hello ocr"))
        assertEquals(Command.READ_TEXT, parsed("hello what does it say"))
        assertEquals(Command.NAVIGATION, parsed("hello nav"))
        assertEquals(Command.NAVIGATION, parsed("hello navigate"))
        assertEquals(Command.NAVIGATION, parsed("hello directions"))
        assertEquals(Command.COLOR, parsed("hello what color"))
        assertEquals(Command.COLOR, parsed("hello color"))
        assertEquals(Command.SETTINGS, parsed("hello open settings"))
        assertEquals(Command.SETTINGS, parsed("hello preferences"))
        assertEquals(
            Command.READ_TEXT,
            (CommandParser.parseAfterWake("scan") as? ParseResult.Parsed)?.command)
        assertEquals(
            Command.NAVIGATION,
            (CommandParser.parseAfterWake("nav") as? ParseResult.Parsed)?.command)
    }

    @Test
    fun settingsCommands() {
        // Explicit on/off phrases — idempotent intent, so "captions on" can
        // never toggle captions off.
        assertEquals(Command.SET_CAPTIONS_ON, parsed("hello captions on"))
        assertEquals(Command.SET_CAPTIONS_OFF, parsed("hello captions off"))
        assertEquals(Command.SET_CAPTIONS_ON, parsed("hello show captions"))
        assertEquals(Command.SET_VOICE_ON, parsed("hello voice commands on"))
        // "voice commands off" (3 words) must beat STOP's bare "off".
        assertEquals(Command.SET_VOICE_OFF, parsed("hello voice commands off"))
        assertEquals(Command.SET_WAKE_ON, parsed("hello wake word on"))
        assertEquals(Command.SET_WAKE_OFF, parsed("hello wake word off"))
        assertEquals(Command.SET_TRANSCRIPT_ON, parsed("hello live transcript on"))
        assertEquals(Command.SET_TRANSCRIPT_OFF, parsed("hello live transcript off"))
        assertEquals(Command.SET_LISTENING_ON, parsed("hello screen off listening on"))
        assertEquals(Command.SET_LISTENING_OFF, parsed("hello screen off listening off"))
        assertEquals(Command.SET_SPEECH_ON, parsed("hello speech on"))
        assertEquals(Command.SET_SPEECH_OFF, parsed("hello speech off"))
        assertEquals(Command.SET_HAPTICS_ON, parsed("hello vibration on"))
        assertEquals(Command.SET_HAPTICS_OFF, parsed("hello vibration off"))
        assertEquals(Command.SET_CONTRAST_ON, parsed("hello high contrast on"))
        assertEquals(Command.SET_CONTRAST_OFF, parsed("hello high contrast off"))
        assertEquals(Command.SET_BIG_TEXT_ON, parsed("hello large text on"))
        assertEquals(Command.SET_BIG_TEXT_OFF, parsed("hello large text off"))
        assertEquals(Command.RATE_SLOW, parsed("hello speak slower"))
        assertEquals(Command.RATE_SLOW, parsed("hello speech slow"))
        assertEquals(Command.RATE_NORMAL, parsed("hello normal speed"))
        assertEquals(Command.RATE_FAST, parsed("hello speak faster"))
        // ...and inside the command window too.
        assertEquals(
            Command.SET_CAPTIONS_OFF,
            (CommandParser.parseAfterWake("captions off") as? ParseResult.Parsed)?.command)
        assertEquals(
            Command.RATE_FAST,
            (CommandParser.parseAfterWake("speak faster") as? ParseResult.Parsed)?.command)
        // Bare "off" inside settings speech is still STOP — the settings
        // phrases all carry an explicit object word.
        assertEquals(Command.STOP, parsed("hello off"))
    }

    @Test
    fun punctuationAndCase() {
        assertEquals(Command.NAVIGATION, parsed("GABAI, WALK!"))
        assertEquals(Command.COLOR, parsed("Gabai: color?"))
    }

    @Test
    fun keywordsBiasListComplete() {
        // Every synonym should be in the bias list handed to Whistle
        assertTrue(CommandParser.KEYWORDS.contains("gabai"))
        assertTrue(CommandParser.KEYWORDS.contains("walk"))
        assertTrue(CommandParser.KEYWORDS.contains("read"))
        assertTrue(CommandParser.KEYWORDS.contains("settings"))
        assertTrue(CommandParser.KEYWORDS.contains("go back"))
    }
}
