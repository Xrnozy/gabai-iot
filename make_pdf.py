from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=22, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], spaceBefore=14, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], spaceBefore=8, spaceAfter=3)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10.5, leading=15)

def bullets(items):
    return ListFlowable([ListItem(Paragraph(i, B), leftIndent=12) for i in items], bulletType="bullet", leftIndent=14)

abstract = [
    "Blind people often bump into things they can't see. A cane only reaches so far. Gabai helps with that.",
    "Gabai is a small bar you wear on your chest. It has two cameras and a motion sensor. The two cameras see depth, the way your two eyes do. Gabai builds a map of the space around you. It spots people, chairs, and poles. Then it talks to you: \"Chair, one metre, left.\"",
    "You can also ask, \"Where are my keys?\" Gabai remembers where you put things down. It tells you which way to go.",
    "You wear it, walk, and listen. You don't need to look at a screen. Your Android phone plays the voice and shows a simple status screen for helpers.",
]

story = [Paragraph("Gabai: Hear the Space Around You", H1), Paragraph("Project Abstract", H2)]
story += [Paragraph(p, B) for p in abstract]
story += [Spacer(1, 6), Paragraph("Technical Specifications", H2), Paragraph("Native Android Architecture &amp; Sensor Stack", H3), Paragraph("Android app (phone)", H3)]
story.append(bullets([
    "<b>SDK:</b> minSdk 26, targetSdk 36, compileSdk 36",
    "<b>Language:</b> Kotlin, with Coroutines and Flow",
    "<b>UI:</b> Jetpack Compose (Material 3), TalkBack-ready, audio-first",
    "<b>Jetpack:</b> Foreground Service, Hilt, Room (object memory), DataStore, ViewModel/Lifecycle, WorkManager",
    "<b>Phone sensors:</b> SensorManager (gyroscope, accelerometer, magnetometer) for heading and stop/fall detection",
    "<b>CameraX:</b> fallback object recognition when the chest bar is off",
    "<b>Audio/haptics:</b> TextToSpeech, accessibility audio usage, VibrationEffect",
    "<b>Voice input:</b> SpeechRecognizer (\"Where are my keys?\")",
    "<b>Location:</b> Fused Location (GPS) for outdoor mode and saved places",
]))
story.append(Paragraph("Chest bar (Raspberry Pi 3B, Python)", H3))
story.append(bullets([
    "2x EMEET C100 webcams, 640x480 MJPEG at 15 FPS, via v4l2 with manual exposure",
    "BMI160 IMU (200 Hz) and HMC5883L compass (50 Hz) via smbus2",
    "Forwards compressed frames and sensor data with Pi capture timestamps over UDP/TCP, with no decoding on the Pi",
]))
story.append(Paragraph("Compute server (PC, Python, ZeroMQ + msgpack between processes)", H3))
story.append(bullets([
    "<b>Tracking (CPU):</b> OpenCV decode and rectify, Basalt stereo-inertial tracking, Madgwick compass heading (ahrs)",
    "<b>Depth (GPU):</b> Lite Any Stereo V2 (TensorRT FP16), with CUDA SGM as the fallback when hot",
    "<b>Detection:</b> YOLO11n-seg / YOLOE (TensorRT FP16) for people, cars, obstacles and named items",
    "<b>3D objects and hands:</b> Boxer 3D boxes, MediaPipe Hands with grab and release rules",
    "<b>Mapping:</b> rolling obstacle grid (NumPy), Open3D floor and wall planes, nvblox voxel map, DINOv2 + ORB relocalisation for the saved home map",
    "<b>Memory:</b> SQLite object memory (\"where are my keys?\")",
    "<b>Safety:</b> single priority alert queue, health monitor (stale data, cable loss), thermal governor",
    "<b>Voice:</b> offline TTS (Piper); alert text sent to the phone",
]))
story.append(Paragraph("Rerun (rerun-sdk, Python)", H3))
story.append(bullets(["Live 3D view for developers and caregivers: camera feeds, depth, hand skeletons, 3D object boxes, obstacle grid and walking path, at 10-15 Hz"]))
story.append(Paragraph("Data flow", H3))
story.append(Paragraph("Chest bar &rarr; compute server (tracking, depth, detection, mapping) &rarr; alerts &rarr; Android app speaks them. Rerun shows the live 3D view alongside.", B))

SimpleDocTemplate("Gabai_Abstract_and_Tech_Specs.pdf", pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54, title="Gabai: Hear the Space Around You").build(story)
