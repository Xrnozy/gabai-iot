cmake_minimum_required(VERSION 3.22.1)
project(gabai_jni CXX)

# ---------- Cactus "Needle" engine (Whistle STT + Needle tool-calling) ----------
add_library(needle STATIC IMPORTED GLOBAL)
set_target_properties(needle PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/needle/${ANDROID_ABI}/libneedle.a
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/needle/include
)

add_library(whistle_jni SHARED whistle_jni.cpp)

# libneedle.a is built against the NDK libc++; the shared STL runtime must be
# linked explicitly so libc++_shared.so is packaged into the APK.
target_link_libraries(whistle_jni needle c++_shared)

# ---------- whisper.cpp (vendored v1.9.5 under whisper/) ----------
# Mirrors examples/whisper.android: whisper.cpp + ggml (static CPU backend
# with NEON), one shared lib per ABI with the matching -march flags.
set(WHISPER_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/whisper)

set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
# ponytail: ggml defaults BUILD_SHARED_LIBS=ON and GGML_OPENMP=ON on Android.
# libggml-cpu.so then dlopens libomp.so, which the APK does not ship → crash at
# WhisperRecognizer.load(). Static ggml + no OpenMP matches whisper.android.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
add_subdirectory(whisper/ggml)

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    set(GAB_WHISPER_ARCH -march=armv8.2-a+fp16)
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    set(GAB_WHISPER_ARCH -mfpu=neon-vfpv4)
else ()
    set(GAB_WHISPER_ARCH "")
endif ()

add_library(whisper_gab SHARED
    ${WHISPER_LIB_DIR}/src/whisper.cpp
    ${WHISPER_LIB_DIR}/src/aneforge/whisper-aneforge.cpp
    whisper_gab.cpp
)

target_include_directories(whisper_gab PRIVATE
    ${WHISPER_LIB_DIR}/include
    ${WHISPER_LIB_DIR}/src
    ${WHISPER_LIB_DIR}/ggml/include
    ${WHISPER_LIB_DIR}/ggml/src
    ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu
)

target_compile_definitions(whisper_gab PUBLIC GGML_USE_CPU)
target_compile_definitions(whisper_gab PRIVATE WHISPER_VERSION="1.9.5")
target_compile_options(whisper_gab PRIVATE ${GAB_WHISPER_ARCH})
# 32-bit size_t vs %ld in vendored whisper.cpp trips -Werror=format-security.
target_compile_options(whisper_gab PRIVATE -Wno-error=format)
# NEON for the ggml CPU backend too (same trick as whisper.android).
target_compile_options(ggml PRIVATE ${GAB_WHISPER_ARCH})

# Always optimize whisper/ggml — a Gradle Debug build without -O3 runs
# whisper ~10x slower than realtime and the whole pipeline starves.
foreach(tgt whisper_gab ggml ggml-base ggml-cpu)
    if (TARGET ${tgt})
        target_compile_options(${tgt} PRIVATE -O3
            -fvisibility=hidden -ffunction-sections -fdata-sections)
    endif ()
endforeach ()
target_compile_options(whisper_gab PRIVATE -fvisibility-inlines-hidden)
target_link_options(whisper_gab PRIVATE -Wl,--gc-sections)
if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    target_link_options(whisper_gab PRIVATE -flto)
endif ()

find_library(LOG_LIB log)
target_link_libraries(whisper_gab ggml ${LOG_LIB} android c++_shared)
