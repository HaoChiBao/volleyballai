# MVP scope

Ship a usable local loop:

1. Upload video  
2. Normalize (`work.mp4` + thumb) via worker/ffmpeg  
3. Auto court keypoints via Modal (YOLOv11n-pose)  
4. Player tracks via Modal **SAM 3.1** (default)  
5. Ball tracks via Modal VballNet  
6. Manual calibrate (draw lines) → project onto official 18×9m court  
7. Analysis player + free-orbit 3D with players/ball  

## Out of MVP (next)

- Seed / auto-apply calibration from court keypoints  
- Actions, score  
- Cloud Run Job deploy  
- YouTube ingest, auth, billing  

## Env

| Var | Meaning |
|---|---|
| `USE_MOCK_TRACKS=0` | Modal SAM 3.1 + ball (**default**) |
| `USE_MOCK_TRACKS=1` | Synthetic fixtures (UI-only) |
| `SAM3_PROMPT` | Text prompt for SAM (default `person`) |
