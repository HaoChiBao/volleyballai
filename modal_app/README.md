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
