# AI policy — Modal only

**Decision date:** 2026-07-24  
**Status:** Locked

## Rule

Do **not** download or run any AI model on the developer laptop or in the local `worker/`.

All model **weights** and **inference** are hosted and executed on **Modal**.

## Includes (non-exhaustive)

- Meta SAM 3.1 (player detect / segment / track)
- Ball detectors
- Pose estimators
- Action / event models
- Fine-tunes, embeddings, re-ID nets
- Any `torch` / CUDA neural workload

## Local / Cloud Run Job may run (non-AI)

- Next.js UI
- **Google Cloud Run Jobs** for analysis execution (CPU)
- ffmpeg / ffprobe inside the job container
- Job metadata + `.data/` or cloud object storage
- Homography / PnP / court geometry (`numpy` / OpenCV geometry OK)
- Mock JSON fixtures for UI without calling Modal (`USE_MOCK_TRACKS=1`)
- Modal **client** SDK (from laptop or job container) to invoke remote functions

Cloud Run Jobs orchestrate the pipeline; they do **not** host model weights.

## Local must NOT include

- `torch`, `torchvision`, SAM packages, Ultralytics, etc. in `worker/` requirements
- Checking model weights into the repo
- Documented “install SAM locally” paths

## Why

Laptop resources are insufficient / undesirable for large CV models. Modal scales GPUs, caches weights in the cloud image/volume, and keeps the local loop light.

## Enforcement

- Local `worker/` deps = ffmpeg wrappers, HTTP/Modal client, light geometry only
- Every ML pipeline stage is a function under `modal/`
- README and tickets state this policy; see Notion decision ticket

## Testing without GPU spend

Use mock track/event JSON locally to build UI. Call Modal only when validating real model output.
