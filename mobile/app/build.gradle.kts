plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.gabai"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.example.gabai"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                arguments += "-DANDROID_STL=c++_shared"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    aaptOptions {
        noCompress += "cact"
        // Model files: mmap-friendly (noCompress) so native code and ONNX
        // Runtime can read them without an inflated copy.
        noCompress += "bin"
        noCompress += "onnx"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    testOptions {
        // android.util.Log is a stub in unit tests — tolerate it.
        unitTests.isReturnDefaultValues = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    // Desktop ONNX Runtime for the clip-replay harness (test classpath only —
    // the APK uses onnxruntime-android above).
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.23.2")
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    // ONNX Runtime: Silero VAD segmentation and the "Hello Gab" wake model.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
    implementation("com.infobip:google-webrtc:1.0.0035529")
    implementation("com.herohan:UVCAndroid:1.0.13")
    implementation("com.squareup.okhttp3:okhttp:3.2.0")
    implementation("org.java-websocket:Java-WebSocket:1.5.3")
    // Caregiver map (OpenStreetMap tiles, no API key) and the ARCore offline-navigation fallback.
    implementation("org.osmdroid:osmdroid-android:6.1.20")
    // QR pairing: ZXing core draws the user's code; the embedded scanner reads it on the caregiver's phone.
    implementation("com.google.ar:core:1.45.0")
}