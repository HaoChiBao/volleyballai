# Volleyball AI

Offline volleyball video analysis: upload a court video, calibrate the camera, track players (SAM 3.1 on Modal), track the ball, detect actions, count score, and view a 3D court.

**Status:** greenfield planning. Implementation starts with **Local v0** (Next.js + **Cloud Run Jobs** for analysis + **Modal** for all AI).

## Hard rules (read first)

1. **No AI models on the laptop.** All model download + inference runs on **[Modal](https://modal.com)** only.
2. **Analysis = Google Cloud Run Jobs.** One execution per analysis run (ffmpeg + orchestration); the job container calls Modal for AI.
3. **Quality over realtime.** Pipelines are offline / batch.

See [docs/AI_POLICY.md](docs/AI_POLICY.md) · [docs/JOBS.md](docs/JOBS.md).

## Stack (locked)

| Layer | Choice |
|---|---|
| Web | Next.js (TypeScript) → later Vercel |
| Analysis jobs | **Google Cloud Run Jobs** |
| AI / GPU | Modal |
| Data (later) | Supabase (Auth, Postgres, Storage) |
| Player tracking | Meta SAM 3.1 (on Modal) |
| 3D viewer | React Three Fiber |
| Video | ffmpeg in job container (+ yt-dlp later) |

Details: [docs/STACK.md](docs/STACK.md) · Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Docs index

| Doc | Purpose |
|---|---|
| [docs/AI_POLICY.md](docs/AI_POLICY.md) | Modal-only AI rule |
| [docs/JOBS.md](docs/JOBS.md) | Cloud Run Jobs for analysis |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Hosting + hybrid flow |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Analysis stages |
| [docs/DATA_MODEL.md](docs/DATA_MODEL.md) | Schemas + storage layout |
| [docs/LOCAL_V0.md](docs/LOCAL_V0.md) | How to start implementing |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log |
| [docs/NOTION.md](docs/NOTION.md) | Notion wiki + tickets |

## Planned monorepo layout

```
volleyballai/
├── apps/web/              # Next.js control plane
├── packages/types/        # Shared TS types
├── packages/court-math/   # Homography / PnP (no neural nets)
├── worker/                # Cloud Run Job image (ffmpeg + Modal client)
├── modal_app/             # All AI stages on Modal (SAM, ball, …)
├── docs/
└── .data/                 # Local videos/jobs for mocks (gitignored)
```

## Run locally (Local v0 scaffold)

```powershell
# from repo root
npm install
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r worker\requirements.txt

# terminal 1 — web
npm run dev

# terminal 2 — worker (fake ~10s pipeline; polls Next API)
npm run worker
```

Open http://localhost:3000 → **Upload** a short mp4 → watch job progress on the video page.

`JOB_BACKEND=local` is the default. Cloud Run Job wiring and Modal SAM come next (see [docs/LOCAL_V0.md](docs/LOCAL_V0.md)).

## Notion

Project wiki and ticket board live in Notion — see [docs/NOTION.md](docs/NOTION.md).
