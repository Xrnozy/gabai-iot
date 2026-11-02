package com.example.gabai.services

import android.util.Log
import java.util.concurrent.ConcurrentHashMap

/**
 * Lightweight session lexicon layered on [CommandMatcher] — the app's
 * "close enough" word classifier.
 *
 * Three kinds of memory, all in-memory only (lifetime = app process):
 *
 *  - **phraseCache**: normalized utterance → command. Once "wulk" scored
 *    as NAVIGATION, the same utterance is an O(1) hit forever after — and
 *    resolves identically, so a word can never flip commands mid-session.
 *  - **learnedTokens**: single-word equivalences learned from accepted
 *    matches ("wulk"→"walk"). Later utterances containing that token score
 *    it as an exact hit, so learned mishears generalize to new phrasings.
 *  - **missCache**: utterances that scored below threshold — repeated
 *    background noise stops paying the fuzzy-match cost every hop.
 *
 * Safety: session-ending commands ([VoiceConfig.destructiveCommands])
 * neither learn nor benefit from learned tokens — a learned mishear must
 * never gain the power to end a session.
 */
object CommandClassifier {

    private const val TAG = "GabVoice"
    /** A token pairing is worth learning once it's already this close. */
    private const val LEARN_THRESHOLD = 0.75f
    private const val MAX_CACHE_ENTRIES = 500

    private val phraseCache = ConcurrentHashMap<String, Command>()
    private val missCache = ConcurrentHashMap.newKeySet<String>()
    private val learnedTokens = ConcurrentHashMap<String, String>()

    /**
     * Classify [raw] to a [CommandMatcher.Match], consulting the session
     * memory first. Same contract as [CommandMatcher.match]: null = no
     * action.
     */
    fun match(
        raw: String,
        config: VoiceConfig = VoiceConfig.DEFAULT,
        allowed: Set<Command>? = null,
    ): CommandMatcher.Match? {
        val key = TranscriptFilter.normalize(raw)
        if (key.isEmpty()) return null

        phraseCache[key]?.let { learned ->
            if (allowed == null || learned in allowed) {
                Log.d(TAG, "learned \"$key\" -> $learned")
                return CommandMatcher.Match(learned, 1f, "learned")
            }
        }
        if (key in missCache) return null

        val match = CommandMatcher.match(raw, config, allowed, learnedTokens)
        if (match == null) {
            if (missCache.size < MAX_CACHE_ENTRIES) missCache += key
            return null
        }

        if (phraseCache.size < MAX_CACHE_ENTRIES) phraseCache[key] = match.command
        if (match.command !in config.destructiveCommands) {
            learn(raw, match.synonym)
        }
        return match
    }

    /** For each utterance token, remember its best-matching synonym token
     *  when they were already near-identical — generalizes a one-off
     *  mishear ("wulk") into a standing equivalence ("wulk" = "walk"). */
    private fun learn(raw: String, synonym: String) {
        val utterance = TranscriptFilter.normalize(raw).split(' ')
        val synTokens = synonym.split(' ')
        for (u in utterance) {
            for (s in synTokens) {
                if (u == s) break
                // Score the unlearned pair — bootstrap only from real
                // similarity so noise tokens can't hijack a command word.
                if (CommandMatcher.tokenSimilarity(u, s, phonetic = true) >=
                    LEARN_THRESHOLD
                ) {
                    learnedTokens.putIfAbsent(u, s)
                    Log.d(TAG, "learned token: \"$u\" = \"$s\"")
                    break
                }
            }
        }
    }

    /** Snapshot of learned equivalences (debug/inspection). */
    fun learnedSnapshot(): Map<String, String> = learnedTokens.toMap()

    /** Drops the session memory — call when starting a fresh listen run. */
    fun reset() {
        phraseCache.clear()
        missCache.clear()
        learnedTokens.clear()
    }
}
