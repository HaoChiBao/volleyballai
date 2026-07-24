# Local v0 — implementation start guide

**Goal:** Upload → normalize → calibrate → **Modal SAM track** (via analysis job) → overlays → track repair.

**Jobs:** Analysis runs as **Google Cloud Run Jobs** in the real path. Same worker image can run locally for UI/mocks. See [JOBS.md](JOBS.md).

## Prerequisites

- Node 20+
- Python 3.11+ (worker)
- Docker (worker image)
- ffmpeg (local or in image)
- GCP project + `gcloud` (for Cloud Run Jobs)
- Modal account (AI stages)
- Short indoor volleyball test clip (do not commit copyrighted bulk footage)

## Build order

### 1. Repo + app shell
- [x] Monorepo: `apps/web`, `worker/`, `packages/types`, `modal/`
- [x] Next.js routes: Library, Upload, Video detail
- [x] Shared types: `Video`, `Job`, `Calibration`, `PlayerTrack`
- [x] `.gitignore`, README, `.env.example`
- [x] Worker deps: **no** torch/SAM ([AI_POLICY.md](AI_POLICY.md))

### 2. Job model + trigger
- [x] Job record (`jobs.json` or DB): status, stage, progress, `cloud_run_execution_name`
- [x] API: create job, get status
- [x] **Cloud Run Job** definition + Dockerfile for `worker/` (image stub; deploy later)
- [x] API local runner fallback (`POST /api/jobs/claim` + worker poller)
- [ ] API triggers Cloud Run `jobs.execute` when `JOB_BACKEND=cloudrun`
- [x] Fake ~10s stage so UI progress works before ffmpeg/SAM

### 3. Ingest in worker (ffmpeg)
- [ ] ffprobe → metadata
- [ ] ffmpeg → `work.mp4` + `thumb.jpg`
- [ ] Artifacts readable by UI (disk or storage URL)
- [ ] Video detail plays `work.mp4` + job status

### 4. Calibration (Next.js)
- [ ] `calibration.json` schema
- [ ] Drag 4 court corners
- [ ] Homography + grid overlay
- [ ] Save/load on Video detail

### 5. Overlay UI with mocks
- [ ] `USE_MOCK_TRACKS=1` (no Modal, optional no GCP)
- [ ] Boxes synced to `currentTime`
- [ ] Repair click UX against mocks

### 6. Modal SAM from the job
- [ ] Modal SAM 3.1 function (weights on Modal only)
- [ ] Job stage `track_players` calls Modal with `work.mp4`
- [ ] Write `players.tracks.json`; UI overlays real tracks

### 7. Track repair
- [ ] Click → Modal refine (via job or lightweight execution)
- [ ] Patch tracks; refresh overlays

## Explicitly out of Local v0

- Ball, 3D, actions, score
- YouTube
- Full Supabase Auth (optional later in v0)
- Any local AI model install

## Suggested week-one focus

1–2: app shell + job record + fake progress  
3: worker Dockerfile + ffmpeg normalize (local image and/or Cloud Run Job)  
4–5: calibration + mock overlays  
6: Modal SAM wired through the job  
7: repair
