package com.example.gabai.services

import android.view.View
import android.widget.TextView
import com.example.gabai.R

/**
 * Drives the shared "You said" live-transcript strip (view_heard_panel.xml).
 *
 *  - not listening (or the feature is off) → hidden — honest: the mic
 *    isn't producing anything to show
 *  - models loading                          → "Loading voice model…"
 *  - listening, nothing heard yet            → idle line inviting speech
 *  - a transcript arrives                    → the latest thing Whisper
 *    produced, verbatim
 *
 * Call [setListening] / [setLoading] / [show] on the main thread; call
 * [refresh] after the Settings toggle flips so visibility re-evaluates.
 */
class HeardPanel(private val panel: View?) {

    private val text: TextView? = panel?.findViewById(R.id.tv_heard)
    private var listening = false
    private var loading = false

    fun setListening(value: Boolean) {
        listening = value
        render()
    }

    fun setLoading(value: Boolean) {
        loading = value
        render()
    }

    /** Latest thing Whisper heard — replaces the idle/loading line. */
    fun show(heard: String) {
        val p = panel ?: return
        p.visibility = View.VISIBLE
        text?.text = heard
    }

    /** Re-evaluate visibility after the Settings toggle changes. */
    fun refresh() = render()

    private fun render() {
        val p = panel ?: return
        when {
            !SettingsStore.isLiveTranscriptEnabled(p.context) ||
                (!listening && !loading) -> p.visibility = View.GONE
            loading -> {
                p.visibility = View.VISIBLE
                text?.setText(R.string.heard_loading)
            }
            else -> {
                p.visibility = View.VISIBLE
                text?.setText(R.string.heard_idle)
            }
        }
    }
}
