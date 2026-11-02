package com.example.gabai

import com.example.gabai.services.Command
import com.example.gabai.services.CommandClassifier
import com.example.gabai.services.CommandMatcher
import com.example.gabai.services.TranscriptFilter
import com.example.gabai.services.Transcription
import com.example.gabai.services.VoiceConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CommandMatcherTest {

    private val cfg = VoiceConfig.DEFAULT

    private fun matched(raw: String): Command? = CommandMatcher.match(raw, cfg)?.command

    @Test
    fun nearMissesResolve() {
        assertEquals(Command.NAVIGATION, matched("wulk"))       // walk
        assertEquals(Command.READ_TEXT, matched("reed"))        // read
        assertEquals(Command.BACK, matched("block"))            // back
        assertEquals(Command.BACK, matched("go bock"))          // go back
        assertEquals(Command.COLOR, matched("collar"))          // color
        assertEquals(Command.SETTINGS, matched("setings"))      // settings
        assertEquals(Command.STOP, matched("stop"))             // exact
    }

    @Test
    fun unrelatedWordsAreNoAction() {
        assertNull(matched("banana"))
        assertNull(matched("the weather is nice"))
        assertNull(matched("what time is it"))
        // Stray keyword inside longer speech can't fire.
        assertNull(matched("we got the settings"))
        assertNull(matched("please walk with me to the store"))
    }

    @Test
    fun silenceHallucinationsAreNoAction() {
        // The classic Whisper-on-noise outputs must never become commands.
        assertNull(matched("thank you"))
        assertNull(matched("thanks for watching"))
        assertNull(matched("okay"))
        assertNull(matched("okay."))
        assertNull(matched("you"))
        assertNull(matched("bye"))
        assertNull(matched("hello hulk"))
    }

    @Test
    fun destructiveCommandsNeedNearExact() {
        // Near-misses of "stop" never reach the destructive bar (0.88, no
        // phonetic rescue) — no accidental session end. ("stap" still lands
        // on the harmless fuzzy "setup" match — just never STOP.)
        assertTrue(matched("stap") != Command.STOP)
        assertNull(matched("sop"))
        assertNull(matched("stopper"))
        assertNull(matched("stoppit"))
        assertEquals(Command.STOP, matched("stop"))
        assertEquals(Command.STOP, matched("quiet"))     // exact synonym
        assertEquals(Command.STOP, matched("shut up"))   // exact phrase
    }

    @Test
    fun settingsCommandsFuzzy() {
        // Mishears of the new settings phrases still resolve.
        assertEquals(Command.READ_TEXT, matched("scam"))          // scan
        assertEquals(Command.SET_CAPTIONS_ON, matched("captons on"))
        assertEquals(Command.SET_CAPTIONS_OFF, matched("captens off"))
        assertEquals(Command.RATE_FAST, matched("speak fastre"))
        // ...but ambiguous "what X is it" phrasing stays no-action.
        assertNull(matched("what time is it"))
    }

    @Test
    fun sessionLearning() {
        CommandClassifier.reset()
        // A mishear resolves once through the fuzzy scorer...
        assertEquals(
            Command.NAVIGATION,
            CommandClassifier.match("wulk", cfg)?.command)
        // ...its token equivalence is remembered for the session...
        assertEquals("walk", CommandClassifier.learnedSnapshot()["wulk"])
        // ...and the same utterance hits the phrase cache identically.
        assertEquals(
            Command.NAVIGATION,
            CommandClassifier.match("wulk", cfg)?.command)
        // Below-threshold utterances are cached as misses, not re-scored.
        assertNull(CommandClassifier.match("the weather is nice", cfg))
        assertNull(CommandClassifier.match("the weather is nice", cfg))
        // Learned equivalences can never empower a destructive command.
        assertTrue(CommandClassifier.match("stap", cfg)?.command != Command.STOP)
        CommandClassifier.reset()
    }

    @Test
    fun allowedSetRestrictsMatches() {
        // Only COLOR valid in this state — "walk" must not resolve.
        assertNull(CommandMatcher.match("walk", cfg, allowed = setOf(Command.COLOR)))
        assertEquals(
            Command.COLOR,
            CommandMatcher.match("collar", cfg, allowed = setOf(Command.COLOR))?.command,
        )
    }
}

class TranscriptFilterTest {

    private val cfg = VoiceConfig.DEFAULT

    private fun t(
        text: String,
        noSpeech: Float? = null,
        logprob: Float? = null,
        dur: Float = 2f,
    ) = Transcription(text, "en", noSpeech, logprob, dur)

    @Test
    fun rejectsSilenceDecodes() {
        assertFalse(TranscriptFilter.evaluate(t(""), true, cfg).accepted)
        assertFalse(TranscriptFilter.evaluate(t("   "), true, cfg).accepted)
        assertFalse(
            TranscriptFilter.evaluate(t("thank you", dur = 1.5f), true, cfg).accepted)
        assertFalse(
            TranscriptFilter.evaluate(t("okay.", dur = 1.0f), true, cfg).accepted)
        assertFalse(TranscriptFilter.evaluate(t("you", dur = 0.8f), true, cfg).accepted)
    }

    @Test
    fun rejectsLowConfidenceAndNonSpeech() {
        assertFalse(
            TranscriptFilter.evaluate(t("walk", noSpeech = 0.9f), true, cfg).accepted)
        assertFalse(
            TranscriptFilter.evaluate(t("walk", logprob = -1.4f), true, cfg).accepted)
        // VAD disagreement: clip that didn't come from a speech segment.
        assertFalse(TranscriptFilter.evaluate(t("walk"), false, cfg).accepted)
    }

    @Test
    fun rejectsGarbageAndForeignScript() {
        assertFalse(TranscriptFilter.evaluate(t("you you you you"), true, cfg).accepted)
        assertFalse(TranscriptFilter.evaluate(t("!!!!"), true, cfg).accepted)
        assertFalse(TranscriptFilter.evaluate(t("歩いてください"), true, cfg).accepted)
    }

    @Test
    fun acceptsRealCommands() {
        assertTrue(TranscriptFilter.evaluate(
            t("hello walk", noSpeech = 0.05f, logprob = -0.2f), true, cfg).accepted)
        assertTrue(TranscriptFilter.evaluate(t("settings"), true, cfg).accepted)
        // A genuinely long, confident sentence starting with "okay" is not a
        // silence hallucination — the blocklist only bites on short/weak clips.
        assertTrue(TranscriptFilter.evaluate(
            t("okay", dur = 6f, logprob = -0.1f, noSpeech = 0.02f), true, cfg).accepted)
    }

    private fun assertFalse(b: Boolean) = org.junit.Assert.assertFalse(b)
}
