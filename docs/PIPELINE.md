# Analysis pipeline

Stages run in order. Each writes artifacts + progress. Stages should be independently re-runnable when inputs unchanged (cache by content hash + `pipeline_version`).

## Stages

| # | Stage | Runs on | Input | Output |
|---|---|---|---|---|
| 1 | `ingest` | **Cloud Run Job** | upload / (later YouTube) | `source.mp4`, metadata |
| 2 | `normalize` | **Cloud Run Job** (ffmpeg) | source | `work.mp4`, `thumb.jpg` |
| 3 | `detect_court` | Job → **Modal** YOLOv11n-pose | `work.mp4` | `court.keypoints.json` (+ overlays) |
| 4 | `track_players` | Job → **Modal** SAM 3.1 | `work.mp4` | `players.tracks.json` |
| 5 | `track_ball` | Job → **Modal** VballNet | `work.mp4` | `ball.tracks.json` |
| — | *(3–5 concurrent)* | Spawning all three Modal fns together | same `work.mp4` | wall-clock ≈ max(stage) |
| 6 | `calibrate` | Auto from keypoints (+ optional UI override) | court kpts / lines | `calibration.json` |
| 7 | `actions` | Job calls **Modal** (+ rules) | tracks | `events.json` |
| 8 | `score` | Heuristics + UI edit | events | rallies / sets |
| 9 | `project_3d` | Job CPU math (preferred) | calibration + tracks | `court3d.json` |
| 9b | `spatial_scene` (optional) | Modal Nerfstudio `splatfacto-big` | `work.mp4` (+ optional player tracks) | `spatial/scene.ply` |

Orchestration: [JOBS.md](JOBS.md). Local v0 runs stages **3–5 in parallel** on Modal
after `normalize`, then auto-calibrates and builds `court3d`. Actions/score later.

## Player tracking (SAM 3.1 on Modal)

1. Prompt concept e.g. `person` / `volleyball player` (court ROI if calibrated)
2. Instance masks + IDs; video track with Object Multiplex
3. Post-process: smooth, gap-fill, filter bench tracks
4. Optional: project feet → `court_xy` using calibration
5. Interactive repair: user click/box → Modal refine from frame `t` → patch tracks

SAM does **not** replace: court geometry, ball model, actions/score, jersey identity.

## Ball tracking (VballNet on Modal)

1. **VballNetV1_148** (TrackNet-family heatmap) — best Acc@5px (~87%) among published ONNX
2. Modal functions:
   - `track_ball` — sliding-window center-frame (default, best quality)
   - `track_ball_fast` — non-overlapping batches (iteration / cost)
3. Why not YOLO: volleyball is often tiny / blurred; temporal heatmaps beat single-frame detectors
4. Output: image-space `{t, xy, r}` per detection (+ short gap-fill for occlusions)
5. After calibration: worker / UI reprojects to `court_xyz` (unchanged contract)
6. Source tag: `vballnet` in `ball.tracks.json`

Weights baked into the Modal ball image from
[fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference) (MIT).
Alternate key `v1_204` is also baked (`VBALLNET_MODEL_KEY`).

## Court keypoints (Modal `detect_court`)

1. **YOLOv11n-pose** weights from [Davidsv/volley-ref-ai](https://huggingface.co/Davidsv/volley-ref-ai) (MIT) — 14 court keypoints
2. Runs in the analysis job after `normalize`, before player/ball tracking
3. Sample video at `COURT_SAMPLE_FPS` (default 1) up to `COURT_MAX_FRAMES`
4. Output: `court.keypoints.json` + `court.overlay_*.jpg` previews
5. Worker builds `calibration.json` (`source: auto_keypoints`) from the best
   keypoint frame → H + matched camera for 3D court/net
6. Manual line drawing in the UI sets `source: manual` and overrides auto

## Job status flow

```
queued → running → (needs_calibration?) → running → completed
                 ↘ failed (retryable flag)
```

UI polls (Local v0) or uses Supabase Realtime (later).

## Caching rules

- Changing calibration → invalidate `project_3d` (+ maybe score), **not** SAM
- Changing SAM model/version → bump `pipeline_version`, re-track
- `USE_MOCK_TRACKS=1` skips Modal and writes fixture tracks for UI work
