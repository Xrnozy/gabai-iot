package com.example.gabai.safety

import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import com.google.zxing.BarcodeFormat
import com.google.zxing.qrcode.QRCodeWriter

/**
 * The pairing code a blind user shows and a caregiver scans: the relay address and the user's public handle.
 * It never carries a token. Scanning only sends an invite; nothing is shared until the user taps Allow.
 */
object PairQr {

    class Payload(val relay: String, val handle: String)

    fun encode(relay: String, handle: String): String =
        Uri.Builder().scheme("gabai").authority("pair").appendQueryParameter("relay", relay).appendQueryParameter("handle", handle).build().toString()

    fun decode(text: String): Payload? {
        val u = runCatching { Uri.parse(text.trim()) }.getOrNull() ?: return null
        if (u.scheme != "gabai" || u.host != "pair") return null
        val relay = u.getQueryParameter("relay")?.trim()?.trimEnd('/').orEmpty()
        val handle = u.getQueryParameter("handle")?.trim().orEmpty()
        if (handle.isEmpty() || !(relay.startsWith("http://") || relay.startsWith("https://"))) return null
        return Payload(relay, handle)
    }

    /** How a handle is shown and read aloud: "gab-k3m7x2" -> "GAB-K3M7X2". */
    fun formatCode(handle: String): String = handle.trim().uppercase()

    /** What a person typed back into a handle, or null if it cannot be one. Accepts any case, spaces, and a missing dash. */
    fun parseCode(text: String): String? {
        var s = text.trim().lowercase().replace(Regex("\\s+"), "-")
        if (Regex("gab[a-z0-9]{6}").matches(s)) s = "gab-" + s.substring(3)
        return s.takeIf { Regex("[a-z0-9._@+-]{3,40}").matches(it) }
    }

    fun bitmap(text: String, size: Int = 720): Bitmap {
        val m = QRCodeWriter().encode(text, BarcodeFormat.QR_CODE, size, size)
        val px = IntArray(size * size) { if (m.get(it % size, it / size)) Color.BLACK else Color.WHITE }
        return Bitmap.createBitmap(px, size, size, Bitmap.Config.ARGB_8888)
    }
}
