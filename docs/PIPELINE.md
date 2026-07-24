# Analysis pipeline

Stages run in order. Each writes artifacts + progress. Stages should be independently re-runnable when inputs unchanged (cache by content hash + `pipeline_version`).

## Stages

| # | Stage | Runs on | Input | Output |
|---|---|---|---|---|
| 1 | `ingest` | **Cloud Run Job** | upload / (later YouTube) | `source.mp4`, metadata |
| 2 | `normalize` | **Cloud Run Job** (ffmpeg) | source | `work.mp4`, `thumb.jpg` |
| 3 | `calibrate` | Next.js UI + math | frames + corners | `calibration.json` |
| 4 | `track_players` | Job calls **Modal** SAM 3.1 | `work.mp4` (+ ROI) | `players.tracks.json` |
| 5 | `track_ball` | Job calls **Modal** | `work.mp4` | `ball.tracks.json` |
| 6 | `actions` | Job calls **Modal** (+ rules) | tracks | `events.json` |
| 7 | `score` | Heuristics + UI edit | events | rallies / sets |
| 8 | `project_3d` | Job CPU math (preferred) | calibration + tracks | `court3d.json` |

Orchestration: [JOBS.md](JOBS.md). Local v0 implements 1–4 + overlays + Modal refine. 5–8 come later.

## Player tracking (SAM 3.1 on Modal)

1. Prompt concept e.g. `person` / `volleyball player` (court ROI if calibrated)
2. Instance masks + IDs; video track with Object Multiplex
3. Post-process: smooth, gap-fill, filter bench tracks
4. Optional: project feet → `court_xy` using calibration
5. Interactive repair: user click/box → Modal refine from frame `t` → patch tracks

SAM does **not** replace: court geometry, ball model, actions/score, jersey identity.

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
