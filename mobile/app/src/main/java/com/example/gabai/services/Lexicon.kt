package com.example.gabai.services

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/** Decodes the server's numeric messages ({"d":[[1,1,2]],"e":[],"c":[]}) with assets/lexicon.json; same rules as gabai/net/lexicon.py. */
class Lexicon(context: Context) {
    private val lex = JSONObject(context.assets.open("lexicon.json").bufferedReader().use { it.readText() })

    private fun word(table: String, id: Int) = if (id == 0) "" else lex.getJSONObject(table).optString(id.toString())
    private fun cap(s: String) = s.replaceFirstChar { it.uppercase() } + "."

    /** One detection packet [obj, dist, dir, (rel, obj2), (state)] -> "Human near you on your right." */
    fun say(p: List<Int>): String {
        var s = listOf(word("obj", p[0]), word("dist", p[1]), word("dir", p[2])).filter { it.isNotEmpty() }.joinToString(" ")
        if (p.size >= 5 && p[3] != 0) s += ", ${word("rel", p[3])} ${word("obj", p[4])}"
        if (p.size >= 6 && p[5] != 0) s += ", ${word("state", p[5])}"
        return cap(s)
    }

    fun decode(m: JSONObject): List<String> {
        fun ints(a: JSONArray?) = List(a?.length() ?: 0) { a!!.getInt(it) }
        val d = m.optJSONArray("d")
        return List(d?.length() ?: 0) { say(ints(d!!.getJSONArray(it))) } +
            ints(m.optJSONArray("e")).map { cap(word("env", it)) } + ints(m.optJSONArray("c")).map { cap(word("cue", it)) }
    }
}
