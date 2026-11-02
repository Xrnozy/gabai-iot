#include <jni.h>
#include <android/log.h>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include "whisper.h"

// Thin JNI bridge to whisper.cpp for GabVoice. One whisper_context per
// handle; the Kotlin side guarantees only one handle is alive at a time
// (low-RAM devices) and serialises all calls.

namespace {

constexpr const char* kTag = "GabVoice";

std::mutex g_errorMutex;
std::string g_lastError;

void setError(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_errorMutex);
    g_lastError = msg;
    __android_log_write(ANDROID_LOG_ERROR, kTag, msg.c_str());
}

std::string toUtf8(JNIEnv* env, jstring value) {
    if (value == nullptr) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

void jsonEscape(std::string& out, const std::string& in) {
    for (char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_example_gabai_services_WhisperRecognizer_nativeInit(JNIEnv* env, jobject /*thiz*/,
                                                              jstring modelPath, jint nThreads) {
    const std::string path = toUtf8(env, modelPath);

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;  // CPU + NEON only — deterministic on low-RAM phones.

    whisper_context* ctx = whisper_init_from_file_with_params(path.c_str(), cparams);
    if (ctx == nullptr) {
        setError("whisper_init failed for " + path);
        return 0;
    }
    __android_log_print(ANDROID_LOG_INFO, kTag,
        "whisper context ready: %s (threads=%d, multilingual=%d)",
        path.c_str(), nThreads, whisper_is_multilingual(ctx));
    return reinterpret_cast<jlong>(ctx);
}

// Transcribes 16 kHz mono float PCM. Returns JSON:
// {"text":"...","language":"en","n_segments":N,
//  "no_speech_prob":0.12,"avg_logprob":-0.31,
//  "n_vad_segments":K,"vad_segments_cs":[[t0,t1],...],"elapsed_ms":123}
// no_speech_prob is the max across segments; avg_logprob the mean token
// logprob. vad_segments_cs are whisper's internal VAD spans on the input
// timeline in centiseconds (only when params.vad was on). Both are real
// whisper.cpp decoder outputs.
JNIEXPORT jstring JNICALL
Java_com_example_gabai_services_WhisperRecognizer_nativeTranscribe(
        JNIEnv* env, jobject /*thiz*/, jlong handle, jfloatArray pcm,
        jstring language, jstring initialPrompt, jint nThreads,
        jstring vadModelPath,
        jfloat vadThreshold, jint vadMinSpeechMs, jint vadMinSilenceMs,
        jint vadSpeechPadMs, jfloat vadMaxSpeechS, jfloat vadOverlapS,
        jint audioCtx) {
    whisper_context* ctx = reinterpret_cast<whisper_context*>(handle);
    if (ctx == nullptr) { setError("null whisper context"); return nullptr; }

    const jsize count = env->GetArrayLength(pcm);
    jfloat* samples = env->GetFloatArrayElements(pcm, nullptr);
    if (samples == nullptr) { setError("no pcm"); return nullptr; }
    std::vector<float> audio(samples, samples + count);
    env->ReleaseFloatArrayElements(pcm, samples, JNI_ABORT);

    const std::string lang = toUtf8(env, language);
    const std::string prompt = toUtf8(env, initialPrompt);
    const std::string vadPath = toUtf8(env, vadModelPath);

    whisper_full_params params =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = nThreads > 0 ? nThreads : 4;
    params.no_context = true;        // each window stands alone
    params.single_segment = false;
    params.no_timestamps = true;
    params.translate = false;
    params.detect_language = false;  // explicit language, never auto
    params.language = lang.empty() ? "en" : lang.c_str();
    params.initial_prompt = prompt.empty() ? nullptr : prompt.c_str();
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.suppress_blank = true;
    params.suppress_nst = true;
    params.temperature = 0.0f;
    // temperature_inc 0 = no silent retry passes; GabVoice does its own
    // rejection on the returned probs instead of a slow second decode.
    params.temperature_inc = 0.0f;
    // Encoder-context clamp (EXPERIMENTAL in whisper.cpp) — 0 keeps the
    // full 1500-frame context.
    if (audioCtx > 0) params.audio_ctx = audioCtx;

    if (!vadPath.empty()) {
        params.vad = true;
        params.vad_model_path = vadPath.c_str();
        whisper_vad_params vp = whisper_vad_default_params();
        if (vadThreshold > 0.0f) vp.threshold = vadThreshold;
        if (vadMinSpeechMs > 0) vp.min_speech_duration_ms = vadMinSpeechMs;
        if (vadMinSilenceMs > 0) vp.min_silence_duration_ms = vadMinSilenceMs;
        if (vadSpeechPadMs >= 0) vp.speech_pad_ms = vadSpeechPadMs;
        if (vadMaxSpeechS > 0.0f) vp.max_speech_duration_s = vadMaxSpeechS;
        if (vadOverlapS >= 0.0f) vp.samples_overlap = vadOverlapS;
        params.vad_params = vp;
    }

    const auto start = std::chrono::steady_clock::now();
    if (whisper_full(ctx, params, audio.data(), static_cast<int>(audio.size())) != 0) {
        setError("whisper_full failed");
        return nullptr;
    }
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    std::string text;
    float maxNoSpeech = 0.0f;
    double logprobSum = 0.0;
    int logprobCount = 0;

    const int nSegments = whisper_full_n_segments(ctx);
    for (int i = 0; i < nSegments; ++i) {
        const char* segText = whisper_full_get_segment_text(ctx, i);
        if (segText != nullptr) text += segText;
        const float nsp = whisper_full_get_segment_no_speech_prob(ctx, i);
        if (nsp > maxNoSpeech) maxNoSpeech = nsp;
        const int nTokens = whisper_full_n_tokens(ctx, i);
        for (int t = 0; t < nTokens; ++t) {
            const whisper_token_data td = whisper_full_get_token_data(ctx, i, t);
            if (td.plog > -20.0f) {  // plog 0 marks unset/special slots
                logprobSum += td.plog;
                logprobCount++;
            }
        }
    }

    const int langId = whisper_full_lang_id(ctx);
    const char* langStr = langId >= 0 ? whisper_lang_str(langId) : "";

    std::string json = "{\"text\":\"";
    jsonEscape(json, text);
    json += "\",\"language\":\"";
    json += langStr != nullptr ? langStr : "";
    json += "\",\"n_segments\":" + std::to_string(nSegments);
    json += ",\"no_speech_prob\":" + std::to_string(maxNoSpeech);
    if (logprobCount > 0) {
        json += ",\"avg_logprob\":" + std::to_string(logprobSum / logprobCount);
    }
    const int nVad = whisper_full_n_vad_segments(ctx);
    json += ",\"n_vad_segments\":" + std::to_string(nVad);
    if (nVad > 0) {
        json += ",\"vad_segments_cs\":[";
        for (int i = 0; i < nVad; ++i) {
            if (i > 0) json += ",";
            json += "[" + std::to_string(whisper_full_get_vad_segment_t0(ctx, i)) +
                    "," + std::to_string(whisper_full_get_vad_segment_t1(ctx, i)) + "]";
        }
        json += "]";
    }
    json += ",\"elapsed_ms\":" + std::to_string(elapsedMs);
    json += "}";
    return env->NewStringUTF(json.c_str());
}

JNIEXPORT void JNICALL
Java_com_example_gabai_services_WhisperRecognizer_nativeRelease(JNIEnv* /*env*/,
                                                                 jobject /*thiz*/,
                                                                 jlong handle) {
    whisper_context* ctx = reinterpret_cast<whisper_context*>(handle);
    if (ctx != nullptr) whisper_free(ctx);
}

// Standalone VAD scoring — whisper_full's internal VAD reports only
// segment boundaries, not probabilities. This probe runs the same ggml
// Silero model on a window and returns the max per-frame probability, so
// field logs show exactly how far under the threshold the audio lands.
JNIEXPORT jlong JNICALL
Java_com_example_gabai_services_WhisperRecognizer_nativeVadInit(
        JNIEnv* env, jobject /*thiz*/, jstring modelPath, jint nThreads) {
    const std::string path = toUtf8(env, modelPath);
    whisper_vad_context_params vp = whisper_vad_default_context_params();
    vp.n_threads = nThreads > 0 ? nThreads : 2;
    vp.use_gpu = false;
    whisper_vad_context* vctx =
        whisper_vad_init_from_file_with_params(path.c_str(), vp);
    if (vctx == nullptr) {
        setError("whisper_vad_init failed for " + path);
        return 0;
    }
    return reinterpret_cast<jlong>(vctx);
}

JNIEXPORT jfloat JNICALL
Java_com_example_gabai_services_WhisperRecognizer_nativeVadMaxProb(
        JNIEnv* env, jobject /*thiz*/, jlong handle, jfloatArray pcm) {
    whisper_vad_context* vctx = reinterpret_cast<whisper_vad_context*>(handle);
    if (vctx == nullptr) { setError("null vad context"); return -1.0f; }

    const jsize count = env->GetArrayLength(pcm);
    jfloat* samples = env->GetFloatArrayElements(pcm, nullptr);
    if (samples == nullptr) { setError("no pcm"); return -1.0f; }
    std::vector<float> audio(samples, samples + count);
    env->ReleaseFloatArrayElements(pcm, samples, JNI_ABORT);

    if (!whisper_vad_detect_speech(vctx, audio.data(),
                                   static_cast<int>(audio.size()))) {
        return 0.0f;  // detect path declined entirely
    }
    const int n = whisper_vad_n_probs(vctx);
    const float* probs = whisper_vad_probs(vctx);
    float best = 0.0f;
    for (int i = 0; i < n; ++i) if (probs[i] > best) best = probs[i];
    return best;
}

JNIEXPORT void JNICALL
Java_com_example_gabai_services_WhisperRecognizer_nativeVadRelease(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    whisper_vad_context* vctx = reinterpret_cast<whisper_vad_context*>(handle);
    if (vctx != nullptr) whisper_vad_free(vctx);
}

JNIEXPORT jstring JNICALL
Java_com_example_gabai_services_WhisperRecognizer_nativeLastError(JNIEnv* env,
                                                                  jobject /*thiz*/) {
    std::lock_guard<std::mutex> lock(g_errorMutex);
    return env->NewStringUTF(g_lastError.c_str());
}

}  // extern "C"
