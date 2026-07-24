"""
Modal SAM 3.1 player tracking (AI runs here only — never on the laptop).

Deploy (after Modal auth + SAM 3.1 image is ready):
  modal deploy modal/track_players.py

Until then, keep USE_MOCK_TRACKS=1 in the local worker.
"""

from __future__ import annotations

import modal

app = modal.App("volleyball-ai")

# GPU image placeholder — pin SAM 3.1 deps when wiring for real.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy",
    "opencv-python-headless",
)


@app.function(image=image, gpu="A10G", timeout=60 * 60)
def track_players(video_path: str) -> dict:
    """
    TODO: load SAM 3.1, run multi-object track on video_path, return:
      { video_id?, pipeline_version, players: [{ track_id, frames: [{t,bbox,court_xy?}] }] }
    """
    raise NotImplementedError(
        "SAM 3.1 tracking not implemented yet. Use USE_MOCK_TRACKS=1 locally.",
    )
