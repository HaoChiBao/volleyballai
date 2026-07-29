"""
SetOptics YOLO volleyball ball tracker (single-frame detect + BoT-SORT).

Weights: volleyball_yolo26n.pt from
  https://github.com/dawsonpar/SetOptics (Apache-2.0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_PATH = Path("/models/volleyball_yolo26n.pt")
MODEL_NAME = "volleyball_yolo26n"
MODEL_SOURCE = "setoptics_yolo"

RADIUS_MIN = 3.0
RADIUS_MAX = 48.0
DEFAULT_CONF = 0.3
# Match VballNet: bridge ~1.5s occlusions at 60fps.
GAP_FILL_MAX_FRAMES = 90


def _radius_from_bbox(xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    r = 0.5 * max(x2 - x1, y2 - y1)
    return float(np.clip(r, RADIUS_MIN, RADIUS_MAX))


def _gap_fill(
    frames: list[dict[str, Any] | None],
    *,
    fps: float,
    max_gap: int = GAP_FILL_MAX_FRAMES,
) -> list[dict[str, Any]]:
    """Linear fill short gaps between detections (same idea as VballNet)."""
    n = len(frames)
    out: list[dict[str, Any] | None] = list(frames)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        # Find previous and next detections
        prev = i - 1
        while prev >= 0 and out[prev] is None:
            prev -= 1
        nxt = i
        while nxt < n and out[nxt] is None:
            nxt += 1
        if prev < 0 or nxt >= n:
            i = nxt if nxt < n else n
            continue
        gap = nxt - prev - 1
        if gap <= 0 or gap > max_gap:
            i = nxt
            continue
        a = out[prev]
        b = out[nxt]
        assert a is not None and b is not None
        ax, ay = a["xy"]
        bx, by = b["xy"]
        ar = float(a.get("r") or RADIUS_MIN)
        br = float(b.get("r") or RADIUS_MIN)
        for k in range(1, gap + 1):
            u = k / (gap + 1)
            idx = prev + k
            t = round(idx / max(fps, 1e-6), 3)
            out[idx] = {
                "t": t,
                "xy": [
                    round(ax + (bx - ax) * u, 1),
                    round(ay + (by - ay) * u, 1),
                ],
                "r": round(ar + (br - ar) * u, 1),
            }
        i = nxt
    return [f for f in out if f is not None]


def track_ball_yolo(
    video_path: str | Path,
    *,
    model_path: Path | None = None,
    confidence_threshold: float = DEFAULT_CONF,
    gap_fill: bool = True,
) -> list[dict[str, Any]]:
    """
    Track ball with SetOptics YOLO + Ultralytics BoT-SORT.

    Returns detections in the same shape as VballNet: [{t, xy, r}, ...].
    """
    from ultralytics import YOLO

    path = Path(video_path)
    weights = Path(model_path) if model_path else MODEL_PATH
    if not weights.exists():
        raise FileNotFoundError(f"YOLO ball model not found: {weights}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    print(
        f"[yolo_ball] model={weights.name} conf={confidence_threshold} "
        f"fps={fps:.2f} frames≈{frame_count}",
        flush=True,
    )

    model = YOLO(str(weights))
    # stream=True keeps memory bounded; persist keeps BoT-SORT IDs across frames.
    results = model.track(
        source=str(path),
        stream=True,
        conf=confidence_threshold,
        persist=True,
        tracker="botsort.yaml",
        verbose=False,
    )

    raw: list[dict[str, Any] | None] = []
    for i, r in enumerate(results):
        t = round(i / max(fps, 1e-6), 3)
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            raw.append(None)
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = (
            boxes.conf.cpu().numpy()
            if boxes.conf is not None
            else np.ones(len(xyxy), dtype=np.float32)
        )
        best = int(np.argmax(confs))
        box = xyxy[best]
        cx = 0.5 * (float(box[0]) + float(box[2]))
        cy = 0.5 * (float(box[1]) + float(box[3]))
        raw.append(
            {
                "t": t,
                "xy": [round(cx, 1), round(cy, 1)],
                "r": round(_radius_from_bbox(box), 1),
            }
        )

    if gap_fill:
        out = _gap_fill(raw, fps=fps)
    else:
        out = [f for f in raw if f is not None]

    print(f"[yolo_ball] detections={len(out)} / frames={len(raw)}", flush=True)
    return out
