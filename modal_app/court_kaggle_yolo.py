"""
Kaggle YOLOv8x-pose volleyball court keypoints (4 corners) → normalized schema.

Weights: pythonistasamurai/yolov8x_volleyball_analysis_models
  key_points_regression_model.pt

Upstream inference uses grayscale→RGB frames at 640×640, conf≈0.1.
Corner order (radar): TL, TR, BR, BL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from court_normalize import (  # type: ignore
    normalize_from_kaggle4,
    pack_result,
)

DEFAULT_MODEL_PATH = Path("/models/key_points_regression_model.pt")
VOLUME_MODEL_PATH = Path("/vol/court-extra/key_points_regression_model.pt")
DEFAULT_CONF = 0.001


def resolve_kaggle_model_path() -> Path:
    for p in (DEFAULT_MODEL_PATH, VOLUME_MODEL_PATH):
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    return DEFAULT_MODEL_PATH


def _to_gray_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def detect_kaggle_court_image(
    frame_bgr: np.ndarray,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    confidence: float = DEFAULT_CONF,
    video_id: str = "",
    pipeline_version: str = "0.1.0",
    return_overlay: bool = True,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model_path = model_path if model_path.exists() else resolve_kaggle_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Kaggle court model not found: {model_path}. "
            "Create Modal secret `kaggle` (KAGGLE_USERNAME + KAGGLE_KEY) and "
            "run: modal run modal_app/app.py::fetch_court_models",
        )

    h, w = frame_bgr.shape[:2]
    model = YOLO(str(model_path))

    # Match upstream: resize 640. Try grayscale-as-RGB first (as trained),
    # then fall back to color if empty.
    resized = cv2.resize(frame_bgr, (640, 640))
    variants = [
        ("gray_rgb", _to_gray_rgb(resized)),
        ("bgr", resized),
    ]

    result = None
    used = ""
    for name, img in variants:
        results = model.predict(img, conf=confidence, imgsz=640, verbose=False)
        if not results:
            continue
        cand = results[0]
        has_kpts = (
            cand.keypoints is not None
            and cand.keypoints.xy is not None
            and len(cand.keypoints.xy) > 0
        )
        has_boxes = cand.boxes is not None and len(cand.boxes) > 0
        print(
            f"[kaggle] variant={name} boxes={0 if not has_boxes else len(cand.boxes)} "
            f"kpt_inst={0 if cand.keypoints is None or cand.keypoints.xy is None else len(cand.keypoints.xy)}",
        )
        if has_kpts:
            result = cand
            used = name
            break
        if result is None and has_boxes:
            result = cand
            used = name

    if result is None or result.keypoints is None or result.keypoints.xy is None:
        return pack_result(
            model_id="kaggle_yolov8x",
            model_name="key_points_regression_model",
            model_repo="pythonistasamurai/yolov8x_volleyball_analysis_models",
            video_id=video_id,
            pipeline_version=pipeline_version,
            image_size={"width": w, "height": h},
            keypoints=normalize_from_kaggle4([])[0],
            raw_keypoints=[],
            box_conf=0.0,
            frame=frame_bgr,
            return_overlay=False,
            note=f"No detection (conf={confidence})",
        )

    kxy = result.keypoints.xy.cpu().numpy()
    if len(kxy) == 0:
        return pack_result(
            model_id="kaggle_yolov8x",
            model_name="key_points_regression_model",
            model_repo="pythonistasamurai/yolov8x_volleyball_analysis_models",
            video_id=video_id,
            pipeline_version=pipeline_version,
            image_size={"width": w, "height": h},
            keypoints=normalize_from_kaggle4([])[0],
            raw_keypoints=[],
            box_conf=0.0,
            frame=frame_bgr,
            return_overlay=False,
            note=f"Empty keypoints (variant={used})",
        )

    kcf = None
    if result.keypoints.conf is not None:
        kcf = result.keypoints.conf.cpu().numpy()

    boxes = result.boxes
    confs_box = (
        boxes.conf.cpu().numpy()
        if boxes is not None and boxes.conf is not None
        else np.ones(len(kxy))
    )
    best = int(np.argmax(confs_box))
    xy = kxy[best]
    conf_row = kcf[best] if kcf is not None else np.ones(len(xy))
    box_conf = float(confs_box[best])

    # Scale 640×640 → original
    sx = w / 640.0
    sy = h / 640.0
    pts: list[list[float] | None] = []
    confs: list[float] = []
    for i in range(4):
        if i >= len(xy):
            pts.append(None)
            confs.append(0.0)
            continue
        c = float(conf_row[i]) if i < len(conf_row) else 0.0
        x, y = float(xy[i][0]), float(xy[i][1])
        if x <= 1 and y <= 1:
            pts.append(None)
            confs.append(0.0)
            continue
        pts.append([x * sx, y * sy])
        confs.append(c)

    keypoints, raw = normalize_from_kaggle4(pts, confs)
    return pack_result(
        model_id="kaggle_yolov8x",
        model_name="key_points_regression_model",
        model_repo="pythonistasamurai/yolov8x_volleyball_analysis_models",
        video_id=video_id,
        pipeline_version=pipeline_version,
        image_size={"width": w, "height": h},
        keypoints=keypoints,
        raw_keypoints=raw,
        box_conf=box_conf,
        frame=frame_bgr,
        return_overlay=return_overlay,
        note=(
            "Native output is 4 corners (TL,TR,BR,BL); "
            "net/mid derived; attack lines left empty."
        ),
    )
