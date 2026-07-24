# MVP scope

Ship a usable local loop:

1. Upload video  
2. Normalize (`work.mp4` + thumb) via worker/ffmpeg  
3. Calibrate court (4 corners)  
4. Player tracks (Modal **SAM 3.1** when `USE_MOCK_TRACKS=0`; mock otherwise)  
5. Ball tracks (mock locally; Modal motion stub when real)  
6. Analysis player: custom controls, outline/ball/court-overlay toggles  
7. Free-orbit 3D court with players + ball synced to playhead  

## Out of MVP (next)

- Actions, score  
- Cloud Run Job deploy (local worker is fine for MVP)  
- YouTube ingest, auth, billing  

## Env

| Var | Meaning |
|---|---|
| `USE_MOCK_TRACKS=1` | Synthetic players + ball (default) |
| `USE_MOCK_TRACKS=0` | Modal SAM 3.1 `track_players` + `track_ball` |
| `SAM3_PROMPT` | Text prompt for SAM (default `person`) |
