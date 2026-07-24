# MVP scope

Ship a usable local loop:

1. Upload video  
2. Normalize (`work.mp4` + thumb) via worker/ffmpeg  
3. Calibrate court (4 corners)  
4. Player tracks (Modal SAM when configured; **mock tracks** until then)  
5. 2D overlays on video  
6. 3D court with player markers synced to playhead  

## Out of MVP (next)

- Ball tracking, actions, score  
- Cloud Run Job deploy (local worker is fine for MVP)  
- YouTube ingest, auth, billing  

## Env

| Var | Meaning |
|---|---|
| `USE_MOCK_TRACKS=1` | Worker writes synthetic tracks (default until Modal SAM is wired) |
| `USE_MOCK_TRACKS=0` | Worker calls Modal `track_players` |
