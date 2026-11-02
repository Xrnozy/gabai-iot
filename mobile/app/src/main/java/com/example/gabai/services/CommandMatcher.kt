package com.example.gabai.services

import kotlin.math.max

/**
 * Fuzzy + phonetic command matcher for the gap between exact keyword
 * matching ([CommandParser]) and intent reasoning ([Needle]).
 *
 * For each command synonym, every synonym token is scored against the best-
 * matching utterance token — character edit similarity OR Soundex-key
 * similarity, whichever is higher — then scaled by phrase coverage so a
 * stray keyword inside longer speech cannot score well ("we got the
 * settings" stays dead). Below [VoiceConfig.commandMatchThreshold] the
 * result is NO ACTION; commands in [VoiceConfig.destructiveCommands]
 * (session-ending) additionally need [VoiceConfig.destructiveMatchThreshold].
 */
object CommandMatcher {

    data class Match(val command: Command, val score: Float, val synonym: String)

    /**
     * Best fuzzy match for the command text (already stripped of any wake
     * name — callers feed the post-wake remainder or a window utterance).
     * Null = no action.
     */
    fun match(
        raw: String,
        config: VoiceConfig = VoiceConfig.DEFAULT,
        allowed: Set<Command>? = null,
        learned: Map<String, String> = emptyMap(),
    ): Match? {
        val tokens = TranscriptFilter.normalize(raw).split(' ')
            .filter { it.isNotBlank() }
        if (tokens.isEmpty()) return null

        var best: Match? = null
        for ((command, synonyms) in CommandParser.synonymTable) {
            if (allowed != null && command !in allowed) continue
            // Session-ending commands get no phonetic rescue — they need a
            // literal near-exact phrase, so "stap" can never fire "stop".
            // The same rule blocks learned equivalences: a remembered
            // mishear must never gain the power to end a session.
            val phonetic = command !in config.destructiveCommands
            for (syn in synonyms) {
                val synTokens = syn.split(' ')
                val score = phraseScore(
                    tokens, synTokens, phonetic,
                    if (phonetic) learned else emptyMap(),
                )
                if (score > (best?.score ?: 0f)) {
                    best = Match(command, score, syn)
                }
            }
        }
        val match = best ?: return null
        val threshold = if (match.command in config.destructiveCommands)
            config.destructiveMatchThreshold else config.commandMatchThreshold
        return if (match.score >= threshold) match else null
    }

    /**
     * Whole-utterance similarity: mean of each synonym token's best token
     * score, times coverage = |syn| / |utterance|. Extra unmatched words drag
     * the score down — a lone keyword inside longer speech can't fire.
     */
    private fun phraseScore(
        utterance: List<String>,
        synonym: List<String>,
        phonetic: Boolean,
        learned: Map<String, String>,
    ): Float {
        if (utterance.isEmpty() || synonym.isEmpty()) return 0f
        var sum = 0f
        for (s in synonym) {
            var bestTok = 0f
            for (u in utterance) {
                bestTok = max(bestTok, tokenSimilarity(u, s, phonetic, learned))
            }
            sum += bestTok
        }
        val sim = sum / synonym.size
        val coverage = synonym.size.toFloat() / max(synonym.size, utterance.size)
        return sim * coverage
    }

    /**
     * Character-edit similarity plus a small consonant-skeleton bonus — the
     * phonetic key only nudges a word that already looks close ("block"→
     * "back"), it can never carry a distant one ("stap"→"setup" stays out).
     * [learned] session equivalences ("wulk"→"walk") count as exact hits —
     * only ever supplied for non-destructive commands.
     */
    internal fun tokenSimilarity(
        a: String,
        b: String,
        phonetic: Boolean,
        learned: Map<String, String> = emptyMap(),
    ): Float {
        if (a == b || learned[a] == b) return 1f
        val edit = 1f - editDistance(a, b).toFloat() / max(a.length, b.length)
        if (!phonetic) return edit
        val skel = 1f - editDistance(skeleton(a), skeleton(b)).toFloat() /
            max(skeleton(a).length, skeleton(b).length).coerceAtLeast(1)
        return minOf(1f, edit + 0.3f * skel)
    }

    /** First letter + consonants — a coarse pronunciation key. */
    private fun skeleton(s: String): String {
        if (s.isEmpty()) return s
        return s[0] + s.drop(1).filter { it !in "aeiou" }
    }

    /** Classic Soundex — enough to rescue mishears like "block"→"back"
     *  (B420 vs B200) that edit distance alone would fail. */
    fun soundex(s: String): String {
        if (s.isEmpty()) return ""
        val code = CharArray(4) { '0' }
        code[0] = s[0].uppercaseChar()
        var idx = 1
        var prev = soundexDigit(s[0])
        for (i in 1 until s.length) {
            val d = soundexDigit(s[i])
            if (d != '0' && d != prev && idx < 4) {
                code[idx++] = d
            }
            if (s[i] != 'h' && s[i] != 'w') prev = d
        }
        return String(code)
    }

    private fun soundexDigit(c: Char): Char = when (c.lowercaseChar()) {
        'b', 'f', 'p', 'v' -> '1'
        'c', 'g', 'j', 'k', 'q', 's', 'x', 'z' -> '2'
        'd', 't' -> '3'
        'l' -> '4'
        'm', 'n' -> '5'
        'r' -> '6'
        else -> '0'
    }

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
}
