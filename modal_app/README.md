# Modal AI stages

All AI model download and inference lives here (SAM 3.1, ball, …).

**Do not** install model weights on the laptop or in `worker/`.

> Folder is named `modal_app/` (not `modal/`) so it does not shadow the
> Modal Python SDK package.

## Deploy

```powershell
$env:PYTHONUTF8='1'
modal deploy modal_app/app.py
# or: npm run modal:deploy
```

App: https://modal.com/apps/jamesyang663/main/deployed/volleyball-ai

Requires secret `huggingface` with `HF_TOKEN` and HF access to
[facebook/sam3](https://huggingface.co/facebook/sam3) / sam3.1.

## Local worker

```powershell
# .env
USE_MOCK_TRACKS=0
SAM3_PROMPT=person

.\.venv\Scripts\pip install modal
npm run worker
```

Functions:
- `track_players` — SAM 3.1 text-prompted video tracking (`person` by default)
- `track_ball` — **best quality** VballNet (sliding-window center-frame)
- `track_ball_fast` — same weights, non-overlapping batches (cheaper / faster)
- `detect_court` — YOLOv11n-pose **14 court keypoints** ([Davidsv/volley-ref-ai](https://huggingface.co/Davidsv/volley-ref-ai))
- `compare_court_models` — side-by-side **volley-ref + Kaggle YOLOv8x + TennisCourtDetector** (normalized `volleyball_court_v1` schema)
- `fetch_court_models` — download Kaggle + tennis weights onto Modal Volume `court-extra-models`
- `fetch_kimi_k3` — download full [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) (~1.56 TB) onto Volume `kimi-k3-weights` (resume-safe)
- `KimiK3Server` — vLLM serve of that Volume checkpoint on **8×B300** (`tensor-parallel-size 8`)
- `analyze_court_with_kimi_k3` — multimodal court → `volleyball_court_v1` keypoints via **self-hosted** K3 (not API proxy, not Kimi-VL-A3B)
- `build_spatial_scene` — Nerfstudio **splatfacto-big** Gaussian env splat on A100-80GB
- `download_spatial_scene_ply` — pull published `.ply` from Volume `spatial-scenes`

### Spatial scene (best-quality Gaussian splat env)

Hybrid “live spatial video”: **static gym** from open-source Nerfstudio **splatfacto-big** on Modal, plus your existing **live player/ball tracks** in the web 3D view.

| Item | Value |
|---|---|
| Method | `splatfacto-big` (higher quality / ~12GB+ VRAM) |
| GPU | `A100-80GB` |
| Volume | `spatial-scenes` → `/spatial/{video_id}/publish/scene.ply` |
| Transient handling | Optional burn of SAM player bboxes before train |
| Local artifacts | `.data/videos/<id>/spatial/scene.ply` + `meta.json` |

**Policy:** Nerfstudio/COLMAP/train run on Modal only — not on the laptop.

```powershell
$env:PYTHONUTF8='1'
npm run modal:deploy

# Best-quality rebuild (often 30–90+ min). Prefer a 20–40s clip with camera motion.
.\.venv\Scripts\python.exe -m worker.test_spatial_scene .data/videos/<id>/work.mp4 --video-id <id>

# or:
modal run modal_app/app.py::build_spatial_scene_local `
  --video-path .data/videos/<id>/work.mp4 `
  --video-id <id> `
  --tracks-path .data/videos/<id>/players.tracks.json
```

Open the video page → Synced view: toggle **Splat on** / **Court on**. Orbit the reconstructed environment; players and ball still follow the playhead from tracks.

Capture tips: continuous shot, physical camera movement / parallax, minimal cuts. Fixed broadcast cams may fail COLMAP — try a slow sideline pan or empty-court walkthrough.

Cost ballpark: A100-80GB for ~1h train ≈ tens of dollars per clip (not Kimi-K3 B300 territory). Volume storage for `.ply` is usually small (tens–hundreds of MB).

### Full Kimi K3 (self-hosted on Modal)

**Policy:** weights stay on Modal only — never snapshot to the laptop.

Modal also offers a managed Shared API for K3; this project path **self-hosts the HF checkpoint on our Volume** so experiments use our exact copy.

| Resource | Value |
|---|---|
| HF repo | `moonshotai/Kimi-K3` (~1.56 TB, 118 files, not gated) |
| Volume | `kimi-k3-weights` → mounted at `/models/kimi-k3` |
| Serve GPU | `B300:8` (vLLM day-0 recipe); cold load can take many minutes |
| Secret | `huggingface` (`HF_TOKEN`) — same as SAM |
| Cost | Volume storage (~1.56 TB) + B300:8 runtime while the class is warm |

```powershell
$env:PYTHONUTF8='1'
npm run modal:deploy

# 1) Fetch (many hours; resume-safe — re-run if interrupted)
modal run modal_app/app.py::fetch_kimi_k3_local
modal run modal_app/app.py::kimi_k3_status_local

# 2) Court keypoint experiment (spins KimiK3Server / B300:8)
.\.venv\Scripts\python.exe -m worker.test_kimi_court .data/videos/<id>/thumb.jpg
# writes .data/kimi-k3-court-test/kimi_k3.court.keypoints.json + overlay_*.jpg

# or via Modal local entrypoint:
modal run modal_app/app.py::test_kimi_court --image-path .data/videos/<id>/thumb.jpg
```

Serve image: official `vllm/vllm-openai:kimi-k3` (day-0 KDA/MXFP4 deps; pip-only installs are not usable yet).

vLLM flags (inside `KimiK3Server`): `--tensor-parallel-size 8 --trust-remote-code --load-format fastsafetensors --enable-prefix-caching --max-model-len 32768` (+ Kimi tool/reasoning parsers).

Keep the class warm during experiments (`scaledown_window` 20 min) — reloading 1.56 TB is expensive.

### Test court keypoints

```powershell
npm run modal:deploy
# image or video — writes .data/court-test/court.keypoints.json + overlay_*.jpg
modal run modal_app/app.py::test_court --video-path .data/videos/<id>/thumb.jpg
# or after deploy, via worker client:
.\.venv\Scripts\python.exe -m worker.test_court .data/videos/<id>/work.mp4
```

### Compare court models (normalized overlays)

```powershell
modal run modal_app/app.py::fetch_court_models_local
.\.venv\Scripts\python.exe -m worker.test_court_compare .data/court-model-test/images .data/court-model-compare
```

Outputs under `.data/court-model-compare/<image>/` — one overlay + JSON per model.

### Ball models (baked into Modal image)

| Key | ONNX | Acc@5px | Use |
|---|---|---|---|
| `v1_148` (default) | `VballNetV1_seq9_grayscale_148_…` | ~87% | Best quality |
| `v1_204` | `VballNetV1_seq9_grayscale_204_…` | ~86% | Alternate |

Source: [asigatchov/fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference) (MIT)

Worker env:
- `MODAL_TRACK_BALL_FN=track_ball` (default) or `track_ball_fast`
- `VBALLNET_MODEL_KEY=v1_148` (default) or `v1_204`
- `VBALLNET_CONFIDENCE=0.5`
- `MODAL_DETECT_COURT_FN=detect_court`
- `COURT_SAMPLE_FPS=1` / `COURT_MAX_FRAMES=30` / `COURT_CONFIDENCE=0.55`

### Court model (baked into Modal image)

| File | Source | Notes |
|---|---|---|
| `yolo_court_keypoints.pt` | [Davidsv/volley-ref-ai](https://huggingface.co/Davidsv/volley-ref-ai) (MIT) | 14 keypoints, YOLOv11n-pose |

Dataset used upstream: Roboflow [volleyball-court-keypoints](https://universe.roboflow.com/volleyballcourt/volleyball-court-keypoints-k6y7r) (CC BY 4.0).
