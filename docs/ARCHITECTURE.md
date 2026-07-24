# Architecture

## Product goal

Upload a volleyball court video (or later a YouTube URL) → calibrate court (camera may move) → track players + ball → detect actions → score → 3D court view synced to the timeline.

## Production shape

```
Browser (Next.js)
    → Vercel API (thin) — create job, trigger execution, signed URLs
    → Supabase (Auth, Postgres, Storage, Realtime) — metadata + files
    → Google Cloud Run Job — analysis execution (CPU: ffmpeg, orchestration)
         → Modal — all AI inference (SAM, ball, pose, actions)
```

| Layer | Role |
|---|---|
| Vercel / Next.js | Control plane only — never long analysis |
| Supabase | Users, job rows, video/artifact storage |
| **Cloud Run Jobs** | Run analysis pipeline executions |
| **Modal** | All neural models / GPUs |

## Local v0 hybrid (build target)

```
Browser (Next.js @ localhost)
    → Local API
    → .data/ or cloud storage
    → Same worker image:
         • local Docker / process (UI + mocks), OR
         • Cloud Run Job execution (real path)
         → Modal for SAM / AI stages
```

| Concern | Where |
|---|---|
| Upload, library, overlays, calibration UI | Next.js |
| Job create + trigger | Next.js API → Cloud Run Jobs (`gcloud run jobs execute`) |
| Normalize video (ffmpeg) | Cloud Run Job container (or local worker image) |
| Homography / court math | App / worker (no neural net) |
| SAM track / refine | Modal (called from job container) |
| Ball / pose / actions (later) | Modal |
| Mock tracks for UI | Local flag, no Modal / no GCP required |

See [JOBS.md](JOBS.md).

## Stage ownership

```
ingest / normalize     → Cloud Run Job (ffmpeg)
calibrate (user edit)  → Next.js UI
track_players          → Modal (SAM 3.1), invoked by job
track_repair           → Modal, invoked by job or dedicated execution
track_ball             → Modal (later)
actions / score        → Modal + UI edits (later)
project_3d             → Job CPU math (preferred) or Modal if heavy
```

## Design principles

1. Thin web control plane; analysis = Cloud Run Job executions
2. AI only on Modal ([AI_POLICY.md](AI_POLICY.md))
3. Stage-cache artifacts — recalibration must not re-run SAM
4. Pipeline version on every artifact
5. Human-in-the-loop: calibration, ID repair, score corrections
6. Quality over realtime

## Moving camera

- Split timeline into camera segments (cuts / pans)
- Calibration keyframes per segment (corners → homography / PnP)
- Tracking can continue; **projection** uses active segment’s calibration
- Editing calibration re-runs geometry / 3D project, not SAM
