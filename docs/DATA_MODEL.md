# Data model & storage

## Local v0 disk layout

```
.data/
  jobs.json
  videos/{video_id}/
    source.mp4
    work.mp4
    thumb.jpg
    meta.json                 # duration, fps, size, …
    calibration.json
    players.tracks.json
    ball.tracks.json          # later
    events.json               # later
    court3d.json              # later
```

Gitignore `.data/` entirely.

## Core entities (conceptual)

### Video
- `id`, `created_at`
- `source_type`: `upload` | `youtube` (later)
- `paths`: source / work / thumb
- `duration_s`, `fps`, `width`, `height`

### Job
- `id`, `video_id`
- `status`: `queued` | `running` | `needs_calibration` | `completed` | `failed`
- `stage`: current pipeline stage name
- `progress`: 0–1 or percent
- `error`, `retryable`
- `pipeline_version`
- `cloud_run_execution_name` — set when a Cloud Run Job execution starts (see [JOBS.md](JOBS.md))

### Calibration
- `keyframes[]`: `{ t, image_points[], court_points_m[] }`
- `segments[]`: `{ t0, t1, keyframe_id }` (multi-segment later)
- Homography `H` and optional camera `R,t,K` derived / stored

Court model: indoor volleyball **18m × 9m** (FIVB). Net height configurable.

### PlayerTrack (in `players.tracks.json`)

```json
{
  "video_id": "uuid",
  "pipeline_version": "0.1.0",
  "players": [
    {
      "track_id": 7,
      "frames": [
        {
          "t": 12.04,
          "bbox": [x, y, w, h],
          "court_xy": [3.2, 5.1]
        }
      ]
    }
  ]
}
```

Optional later: mask refs, pose joints, team/jersey labels.

### Ball / events / rallies
- Ball: time series `t`, image xy, `court_xy`, `z`
- Events: `type`, `t_start`, `t_end`, `player_id`, `confidence`
- Rallies / sets: scoreboard state + human corrections

## Later (Supabase)

Tables: `profiles`/`teams`, `videos`, `jobs`, `calibrations`, `artifacts`, `events`, `rallies`/`sets`  
Heavy time-series stay in Storage JSON; DB holds metadata + RLS by `user_id` / `team_id`.

Artifact path convention (cloud):

```
artifacts/{video_id}/{pipeline_version}/…
```
