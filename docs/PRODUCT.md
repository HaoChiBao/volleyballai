# Product scope

## Vision

A coach/analyst uploads a volleyball court video (or later pastes a YouTube URL), adjusts court calibration when the camera moves, and gets:

- Player tracks (masks + stable IDs)
- Ball trajectory
- Auto actions (spike, dig, set, block, serve, …)
- Score / rally log (editable)
- 3D court with ball + players synced to the video playhead

## Local v0 (near-term ship)

Must work on localhost:

1. Upload file
2. Normalize with ffmpeg
3. Calibrate 4 court corners
4. Track players via **Modal SAM 3.1**
5. Scrub overlays
6. Click-to-repair track (Modal refine)

## Later milestones

| Milestone | Adds |
|---|---|
| Local v1 | Ball, `court_xy`, basic R3F 3D |
| Cloud deploy | Vercel + Supabase + same Modal stages |
| Later | Actions, score, YouTube, jersey OCR, billing |

## Non-goals (for now)

- Realtime live camera scoring
- Perfect auto score without human edit
- Multi-angle broadcast fusion
- Installing AI models on developer machines
