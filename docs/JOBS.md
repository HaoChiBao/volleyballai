# Analysis jobs — Google Cloud Run Jobs

**Decision:** Video analysis runs as **Google Cloud Run Jobs** (not inline in Next.js, not a long-lived local daemon in production).

## Roles

| Piece | Responsibility |
|---|---|
| Next.js (local / later Vercel) | Create job record, trigger Cloud Run Job execution, show progress |
| Job metadata | `jobs` store (Local v0: `jobs.json`; later Postgres) |
| **Cloud Run Job** | Run pipeline stages in a container (CPU): ffmpeg, orchestration, progress writes |
| **Modal** | All AI inference (SAM, ball, pose, actions) — called *from* the job container |
| Object storage | Video + artifacts (Local v0 disk for pure UI work; GCS/Supabase when using Cloud Run) |

```
UI creates job (queued)
    → API starts Cloud Run Job execution
        → container: ingest / normalize (ffmpeg)
        → container: call Modal track_players
        → container: write artifacts + progress
    → UI reads status until completed / failed
```

## Why Cloud Run Jobs

- Fits batch / minutes-long analysis (timeouts, retries, one execution per analysis)
- Same pattern as other projects (e.g. music-assemble style workers)
- Web tier stays thin
- CPU work (ffmpeg) stays off Modal; GPUs stay on Modal per [AI_POLICY.md](AI_POLICY.md)

## Job record (app-level)

```
Job {
  id
  video_id
  status: queued | running | needs_calibration | completed | failed
  stage: ingest | normalize | track_players | …
  progress
  error
  pipeline_version
  cloud_run_execution_name   # set when GCP execution starts
}
```

Cloud Run’s execution ID is stored on the job for reconcile / cancel / logs.

## Container contract

Suggested env when an execution starts:

- `JOB_ID`, `VIDEO_ID`
- `PIPELINE_VERSION`
- Storage credentials / bucket paths for source + artifacts
- `MODAL_*` auth to invoke AI functions
- `USE_MOCK_TRACKS` (dev only)

Entry: one command that runs the staged pipeline and exits 0/non-zero.

## Local development

Two modes:

1. **Mock / UI mode** — Next.js + `.data/` + optional in-process or local Docker run of the *same* worker image (no GCP). Use for calibration UI + mock tracks.
2. **Real job mode** — Next.js triggers `gcloud run jobs execute …` against a deployed job; artifacts in cloud storage; Modal for SAM.

Prefer keeping **one worker codebase** (`worker/` image) that runs locally *or* as the Cloud Run Job — same stages, same artifact paths.

## What does *not* go in Cloud Run Jobs

- Interactive Next.js UI
- AI model weights / GPU inference → **Modal**
- Holding HTTP requests open for the whole analysis
