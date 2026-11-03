package com.example.gabai

import com.example.gabai.safety.PairQr
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PairQrCodeTest {

    @Test fun formatsHandleInCapitals() = assertEquals("GAB-K3M7X2", PairQr.formatCode("gab-k3m7x2"))

    @Test fun parsesTypedVariants() {
        for (typed in listOf("gab-k3m7x2", "GAB-K3M7X2", " gab k3m7x2 ", "GABK3M7X2")) {
            assertEquals(typed, "gab-k3m7x2", PairQr.parseCode(typed))
        }
    }

    @Test fun keepsCustomHandles() = assertEquals("mum@home", PairQr.parseCode("Mum@Home"))

    @Test fun rejectsJunk() {
        assertNull(PairQr.parseCode(""))
        assertNull(PairQr.parseCode("a"))
        assertNull(PairQr.parseCode("not/a handle!"))
    }
}
