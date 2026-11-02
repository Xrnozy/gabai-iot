#include <jni.h>
#include <cstring>
#include <vector>
#include "needle.h"

// The armeabi-v7a build of libneedle.a was compiled against a libc++ that still
// exported std::__ndk1::__hash_memory (the murmur2 hash behind std::hash for
// strings). Newer NDKs made it an internal symbol, so we supply the identical
// implementation for the 32-bit link.
#if defined(__arm__) && !defined(__aarch64__)
namespace std {
inline namespace __ndk1 {
size_t __hash_memory(const void* ptr, size_t len) noexcept {
    const size_t m = 0x5bd1e995;
    const size_t r = 24;
    size_t h = len;
    const auto* data = static_cast<const unsigned char*>(ptr);
    while (len >= 4) {
        size_t k;
        memcpy(&k, data, sizeof(k));
        k *= m;
        k ^= k >> r;
        k *= m;
        h *= m;
        h ^= k;
        data += 4;
        len -= 4;
    }
    switch (len) {
        case 3:
            h ^= static_cast<size_t>(data[2]) << 16;
            [[fallthrough]];
        case 2:
            h ^= static_cast<size_t>(data[1]) << 8;
            [[fallthrough]];
        case 1:
            h ^= data[0];
            h *= m;
    }
    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;
    return h;
}
}  // namespace __ndk1
}  // namespace std
#endif

namespace {

constexpr int kOutCapacity = 1 << 18;

std::string toUtf8(JNIEnv* env, jstring value) {
    if (value == nullptr) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

jint loadCact(JNIEnv* env, jbyteArray cact) {
    const jsize n = env->GetArrayLength(cact);
    jbyte* bytes = env->GetByteArrayElements(cact, nullptr);
    if (bytes == nullptr) return -1;
    const int result =
        needle_load(reinterpret_cast<const unsigned char*>(bytes), static_cast<unsigned long long>(n));
    env->ReleaseByteArrayElements(cact, bytes, JNI_ABORT);
    return result;
}

}  // namespace

extern "C" {

JNIEXPORT jint JNICALL
Java_com_example_gabai_services_Whistle_nativeLoad(JNIEnv* env, jobject /*thiz*/, jbyteArray cact) {
    return loadCact(env, cact);
}

JNIEXPORT jint JNICALL
Java_com_example_gabai_services_Whistle_nativeModels(JNIEnv* /*env*/, jobject /*thiz*/) {
    return needle_models();
}

JNIEXPORT jstring JNICALL
Java_com_example_gabai_services_Whistle_nativeTranscribe(JNIEnv* env, jobject /*thiz*/,
                                                         jfloatArray pcm, jstring language,
                                                         jstring keywords) {
    const jsize count = env->GetArrayLength(pcm);
    jfloat* samples = env->GetFloatArrayElements(pcm, nullptr);
    if (samples == nullptr) return nullptr;
    const std::string lang = toUtf8(env, language);
    const std::string words = toUtf8(env, keywords);
    std::vector<char> buffer(kOutCapacity, '\0');
    const int result = needle_transcribe(
        samples, count,
        lang.empty() ? nullptr : lang.c_str(),
        words.empty() ? nullptr : words.c_str(),
        1, buffer.data(), static_cast<int>(buffer.size()));
    env->ReleaseFloatArrayElements(pcm, samples, JNI_ABORT);
    if (result < 0) return nullptr;
    return env->NewStringUTF(buffer.data());
}

JNIEXPORT jstring JNICALL
Java_com_example_gabai_services_Whistle_nativeLastError(JNIEnv* env, jobject /*thiz*/) {
    const char* error = needle_last_error();
    return error == nullptr ? nullptr : env->NewStringUTF(error);
}

// ---------- Needle (text / tool-calling model, same engine) ----------

JNIEXPORT jint JNICALL
Java_com_example_gabai_services_Needle_nativeLoad(JNIEnv* env, jobject /*thiz*/, jbyteArray cact) {
    return loadCact(env, cact);
}

JNIEXPORT jint JNICALL
Java_com_example_gabai_services_Needle_nativeModels(JNIEnv* /*env*/, jobject /*thiz*/) {
    return needle_models();
}

JNIEXPORT jint JNICALL
Java_com_example_gabai_services_Needle_nativeInit(JNIEnv* env, jobject /*thiz*/,
                                                  jstring systemPrompt, jstring toolsJson) {
    const std::string sys = toUtf8(env, systemPrompt);
    const std::string tools = toUtf8(env, toolsJson);
    return needle_init(
        sys.empty() ? nullptr : sys.c_str(),
        tools.empty() ? nullptr : tools.c_str(),
        nullptr);
}

JNIEXPORT jstring JNICALL
Java_com_example_gabai_services_Needle_nativeComplete(JNIEnv* env, jobject /*thiz*/,
                                                      jstring input, jint maxNewTokens) {
    const std::string text = toUtf8(env, input);
    std::vector<char> buffer(kOutCapacity, '\0');
    const int result = needle_complete(
        text.empty() ? nullptr : text.c_str(), nullptr, 0,
        maxNewTokens, buffer.data(), static_cast<int>(buffer.size()));
    if (result < 0) return nullptr;
    return env->NewStringUTF(buffer.data());
}

/* Speech-in tool calling: needle_complete transcribes the clip itself and
   answers the transcript — same engine pass for STT and reasoning. */
JNIEXPORT jstring JNICALL
Java_com_example_gabai_services_Needle_nativeCompleteAudio(JNIEnv* env, jobject /*thiz*/,
                                                           jfloatArray pcm, jint maxNewTokens) {
    const jsize count = env->GetArrayLength(pcm);
    jfloat* samples = env->GetFloatArrayElements(pcm, nullptr);
    if (samples == nullptr) return nullptr;
    std::vector<char> buffer(kOutCapacity, '\0');
    const int result = needle_complete(
        nullptr, samples, count,
        maxNewTokens, buffer.data(), static_cast<int>(buffer.size()));
    env->ReleaseFloatArrayElements(pcm, samples, JNI_ABORT);
    if (result < 0) return nullptr;
    return env->NewStringUTF(buffer.data());
}

JNIEXPORT void JNICALL
Java_com_example_gabai_services_Needle_nativeSetAudio(JNIEnv* env, jobject /*thiz*/,
                                                      jstring language, jstring keywords,
                                                      jint wordTimestamps) {
    const std::string lang = toUtf8(env, language);
    const std::string words = toUtf8(env, keywords);
    needle_set_audio(
        lang.empty() ? nullptr : lang.c_str(),
        words.empty() ? nullptr : words.c_str(),
        wordTimestamps);
}

JNIEXPORT void JNICALL
Java_com_example_gabai_services_Needle_nativeReset(JNIEnv* /*env*/, jobject /*thiz*/) {
    needle_reset();
}

JNIEXPORT jstring JNICALL
Java_com_example_gabai_services_Needle_nativeLastError(JNIEnv* env, jobject /*thiz*/) {
    const char* error = needle_last_error();
    return error == nullptr ? nullptr : env->NewStringUTF(error);
}

}  // extern "C"
