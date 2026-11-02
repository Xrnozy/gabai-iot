package com.example.gabai.modules

import java.util.concurrent.atomic.AtomicBoolean

class ModelManager {
    private val navigationInitialized = AtomicBoolean(false)
    private val ocrInitialized = AtomicBoolean(false)
    private val colorInitialized = AtomicBoolean(false)
    private val aslInitialized = AtomicBoolean(false)

    fun ensureNavigationModelsInitialized() {
        if (!navigationInitialized.compareAndSet(false, true)) return
        // Lazy-load navigation models/pipelines here (YOLO, depth, optical flow, etc.)
    }

    fun ensureOcrModelsInitialized() {
        if (!ocrInitialized.compareAndSet(false, true)) return
        // Lazy-load OCR/text recognition models here.
    }

    fun ensureColorModelsInitialized() {
        if (!colorInitialized.compareAndSet(false, true)) return
        // Lazy-load color understanding pipeline here (if needed).
    }

    fun ensureAslModelsInitialized() {
        if (!aslInitialized.compareAndSet(false, true)) return
        // Lazy-load ASL recognition models/pipeline here (if needed).
    }

    fun disposeNavigationModels() {
        // Optional: release navigation-only resources if you want aggressive teardown.
    }

    fun disposeOcrModels() {
        // Optional: release OCR-only resources if you want aggressive teardown.
    }

    fun disposeColorModels() {
        // Optional: release color-only resources if you want aggressive teardown.
    }

    fun disposeAslModels() {
        if (!aslInitialized.compareAndSet(true, false)) return
        // Optional: release ASL-only resources if you want aggressive teardown.
    }
}

