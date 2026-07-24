# Tech stack

## Locked

| Layer | Choice | Notes |
|---|---|---|
| Web app | Next.js (App Router) + TypeScript | Local first; deploy to Vercel later |
| Auth / DB (later) | Supabase | Skip for pure UI Local v0; use `.data/` or Postgres early if needed |
| Object storage | Local `.data/` for mocks; GCS or Supabase Storage for Cloud Run Jobs | Job container needs readable video paths |
| **Analysis jobs** | **Google Cloud Run Jobs** | One execution per analysis; CPU pipeline |
| Worker image | Docker (`worker/`) | Same image locally or on Cloud Run |
| AI / GPU | Modal | **All** models; called from job container |
| Player tracking | Meta SAM 3.1 | On Modal (Object Multiplex) |
| 3D | React Three Fiber / Three.js | Known FIVB court mesh |
| Video tools | ffmpeg, ffprobe | Inside Cloud Run Job container |
| YouTube (later) | yt-dlp | Inside job container or Modal CPU, not day one |

## Open (not blockers)

| Area | Direction |
|---|---|
| Ball | Hybrid detector — **on Modal** |
| Actions | Rules + pose first → learned later — **on Modal** |
| Score | Rally state machine + human edits |
| Jersey / team ID | Later (OCR optional) |

## Explicit non-goals (early)

- Realtime webcam inference
- Local GPU / AI model installs
- Running full analysis inside a Next.js request
- Multi-camera fusion
- Billing (until product polish)
