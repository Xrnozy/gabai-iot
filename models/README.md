# Model requirements

Weights are **not committed** (see `.gitignore`). Download each one to the path shown.
Gabai never downloads models itself. Third-party code is cloned into `third_party/` (also not committed).

## Weights in `models/` and `checkpoints/`

| Model | File / path | Used for | Source |
|---|---|---|---|
| YOLO11n-seg (Ultralytics) | `models/yolo11n-seg.pt` (~6 MB; or set `GABAI_YOLO`) | People/object masks | https://github.com/ultralytics/assets/releases (`yolo11n-seg.pt`) |
| MediaPipe Hand Landmarker | `checkpoints/hand_landmarker.task` (~7.5 MB) | Hand tracking | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task |
| SceneScript (ASE) | `models/scenescript_model_ase.ckpt` | Unused, deferred | https://github.com/facebookresearch/scenescript |

## Stereo / depth models (`third_party/`)

| Model | Clone | Commit used | Weights → path | Weights source |
|---|---|---|---|---|
| LiteAnyStereo v2 (LAS2-S/M/H) | https://github.com/TomTomTommi/LiteAnyStereo | `8c97bd4` | `third_party/LiteAnyStereo/checkpoints/LAS2_{S,M,H}.pth` | Hugging Face `tomtomtommi/LiteAnyStereoV2` (repo's `download_checkpoint.py`) |
| DEFOM-Stereo (ViT-L, SceneFlow) | https://github.com/Insta360-Research-Team/DEFOM-Stereo | `5b27591` | `third_party/DEFOM-Stereo/checkpoints/defomstereo_vitl_sceneflow.pth` (~1.5 GB) | Link in the repo README |
| Fast-FoundationStereo | https://github.com/NVlabs/Fast-FoundationStereo | `476f424` | `third_party/Fast-FoundationStereo/weights/hf-c-fast/model_best_bp2_serialize.pth` | Link in the repo README |
| PromptStereo | https://github.com/Windsrain/PromptStereo | `77d8954` | `third_party/PromptStereo/checkpoint/unlimited_192.safetensors`, `depth_anything_v2_vitl.pth` | Links in the repo README |
| SimpleRecon | https://github.com/nianticlabs/simplerecon | `113b825` | `third_party/simplerecon/weights/hero_model.ckpt` | Link in the repo README (stripped with `strip_checkpoint.py`) |
| Boxer (BoxerNet, DINOv3 ViT-S/16+, OWLv2 base) | https://github.com/facebookresearch/boxer | `1f86542` | `third_party/boxer/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt`, `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`, `owlv2-base-patch16-ensemble.pt` (+ `_textemb_*.pt`) | Links in the repo README (OWLv2: Hugging Face `google/owlv2-base-patch16-ensemble`) |

Each upstream repo has its own `requirements.txt` (torch/xformers pins differ); install in a separate env if they conflict with the root `requirements.txt`.

## Optional

| Model | Source |
|---|---|
| MoGe-2 (`moge-s`/`b`/`l`) | `pip install git+https://github.com/microsoft/MoGe.git`; weights from Hugging Face `Ruicheng/moge-2-vits-normal`, `Ruicheng/moge-2-vitb-normal`, `Ruicheng/moge-2-vitl` |

Quick setup of the clones:

```bash
cd third_party
git clone https://github.com/TomTomTommi/LiteAnyStereo && git -C LiteAnyStereo checkout 8c97bd4
git clone https://github.com/Insta360-Research-Team/DEFOM-Stereo && git -C DEFOM-Stereo checkout 5b27591
git clone https://github.com/NVlabs/Fast-FoundationStereo && git -C Fast-FoundationStereo checkout 476f424
git clone https://github.com/Windsrain/PromptStereo && git -C PromptStereo checkout 77d8954
git clone https://github.com/nianticlabs/simplerecon && git -C simplerecon checkout 113b825
git clone https://github.com/facebookresearch/boxer && git -C boxer checkout 1f86542
```

## Android app on-device assets (`mobile/app/src/main/assets/`, not committed)

| File | Size | Notes |
|---|---|---|
| `models/ggml-small-q5_1.bin`, `ggml-base-q8_0.bin`, `ggml-base.en-q8_0.bin` | 182 / 78 / 78 MB | whisper.cpp GGML models: https://huggingface.co/ggerganov/whisper.cpp |
| `models/ggml-silero-v6.2.0.bin` | <1 MB | Silero VAD GGML: https://huggingface.co/ggml-org/whisper-vad |
| `needle3.cact`, `whistle.cact` | 34 / 17 MB | Cactus-format models; source not recorded in the repo, re-export from the original model |
| `assets.7z` | 191 MB | Asset bundle; source not recorded in the repo |
