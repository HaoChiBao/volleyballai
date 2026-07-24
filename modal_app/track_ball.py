"""
Modal ball tracking stub.

Deploy with: modal deploy modal/track_ball.py
Until a dedicated ball model is wired, USE_MOCK_TRACKS=1 generates ball tracks locally.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import modal

APP_NAME = os.environ.get("MODAL_APP_NAME", "volleyball-ai")
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "0.1.0")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("numpy", "opencv-python-headless")
)


@app.function(image=image, gpu="T4", timeout=60 * 30)
def track_ball(
    video_bytes: bytes,
    video_id: str = "",
    fps: float = 10.0,
    pipeline_version: str = PIPELINE_VERSION,
) -> dict:
    """
    Track volleyball in image space. Returns ball.tracks.json-compatible dict.

    Current implementation: lightweight motion blob detector (placeholder until
    a dedicated ball model / SAM prompt is productionized).
    """
    import cv2
    import numpy as np

    if not video_bytes:
        raise ValueError("Empty video_bytes")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "work.mp4"
        path.write_bytes(video_bytes)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video for ball tracking")

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 10.0
        frames_out: list[dict] = []
        prev_gray = None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            t = idx / max(vid_fps, 1e-3)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                contours, _ = cv2.findContours(
                    th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                best = None
                best_score = 0.0
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < 20 or area > 2500:
                        continue
                    (x, y), r = cv2.minEnclosingCircle(c)
                    circularity = area / (math.pi * r * r + 1e-6)
                    score = circularity * min(area, 400)
                    if score > best_score:
                        best_score = score
                        best = (float(x), float(y), float(r))
                if best:
                    frames_out.append(
                        {
                            "t": round(t, 3),
                            "xy": [round(best[0], 1), round(best[1], 1)],
                            "r": round(max(4.0, best[2]), 1),
                        },
                    )
            prev_gray = gray
            idx += 1
        cap.release()

    return {
        "video_id": video_id,
        "pipeline_version": pipeline_version,
        "frames": frames_out,
        "source": "modal-motion",
    }
