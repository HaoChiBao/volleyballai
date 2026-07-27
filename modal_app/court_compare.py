"""
Multi-model court detection on one image/video frame.

Runs baseline volley-ref + Kaggle YOLOv8x + TennisCourtDetector and returns
normalized volleyball_court_v1 payloads for side-by-side comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from court_kaggle_yolo import detect_kaggle_court_image  # type: ignore
from court_normalize import (  # type: ignore
    normalize_from_volley14,
    pack_result,
)
from court_tennis_detector import detect_tennis_court_image  # type: ignore

VOLLEY_MODEL = Path("/models/yolo_court_keypoints.pt")
KAGGLE_MODEL = Path("/models/key_points_regression_model.pt")
TENNIS_MODEL = Path("/models/tennis_court_detector.pth")


def _kaggle_path() -> Path:
    from court_kaggle_yolo import resolve_kaggle_model_path  # type: ignore

    return resolve_kaggle_model_path()


def _detect_volley_ref(
    frame_bgr: Any,
    *,
    confidence: float,
    video_id: str,
    pipeline_version: str,
    return_overlay: bool,
) -> dict[str, Any]:
    from ultralytics import YOLO

    from court_keypoints import _parse_keypoints  # type: ignore

    h, w = frame_bgr.shape[:2]
    if not VOLLEY_MODEL.exists():
        raise FileNotFoundError(f"Missing {VOLLEY_MODEL}")
    model = YOLO(str(VOLLEY_MODEL))
    results = model.predict(frame_bgr, conf=confidence, verbose=False)
    parsed = _parse_keypoints(results[0], conf_min=0.25) if results else None
    if not parsed:
        empty_kp, empty_raw = normalize_from_volley14([])
        return pack_result(
            model_id="volley_ref_yolo11n",
            model_name="yolo_court_keypoints",
            model_repo="Davidsv/volley-ref-ai",
            video_id=video_id,
            pipeline_version=pipeline_version,
            image_size={"width": w, "height": h},
            keypoints=empty_kp,
            raw_keypoints=empty_raw,
            box_conf=0.0,
            frame=frame_bgr,
            return_overlay=False,
            note="No detection",
        )
    keypoints, _bbox, box_conf = parsed
    norm, raw = normalize_from_volley14(keypoints)
    return pack_result(
        model_id="volley_ref_yolo11n",
        model_name="yolo_court_keypoints",
        model_repo="Davidsv/volley-ref-ai",
        video_id=video_id,
        pipeline_version=pipeline_version,
        image_size={"width": w, "height": h},
        keypoints=norm,
        raw_keypoints=raw,
        box_conf=box_conf,
        frame=frame_bgr,
        return_overlay=return_overlay,
        note="Baseline (current production).",
    )


def compare_court_models_image(
    media_path: Path,
    *,
    video_id: str = "",
    pipeline_version: str = "0.1.0",
    volley_confidence: float = 0.55,
    kaggle_confidence: float = 0.001,
    models: tuple[str, ...] = ("volley_ref", "kaggle", "tennis"),
    return_overlays: bool = True,
) -> dict[str, Any]:
    frame = cv2.imread(str(media_path))
    if frame is None:
        raise RuntimeError(f"Failed to read image: {media_path}")

    h, w = frame.shape[:2]
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    if "volley_ref" in models:
        try:
            results["volley_ref"] = _detect_volley_ref(
                frame,
                confidence=volley_confidence,
                video_id=video_id,
                pipeline_version=pipeline_version,
                return_overlay=return_overlays,
            )
        except Exception as e:  # noqa: BLE001 — collect per-model failures
            errors["volley_ref"] = f"{type(e).__name__}: {e}"

    if "kaggle" in models:
        try:
            results["kaggle"] = detect_kaggle_court_image(
                frame,
                model_path=_kaggle_path(),
                confidence=kaggle_confidence,
                video_id=video_id,
                pipeline_version=pipeline_version,
                return_overlay=return_overlays,
            )
        except Exception as e:  # noqa: BLE001
            errors["kaggle"] = f"{type(e).__name__}: {e}"

    if "tennis" in models:
        try:
            results["tennis"] = detect_tennis_court_image(
                frame,
                model_path=TENNIS_MODEL,
                video_id=video_id,
                pipeline_version=pipeline_version,
                return_overlay=return_overlays,
            )
        except Exception as e:  # noqa: BLE001
            errors["tennis"] = f"{type(e).__name__}: {e}"

    summary = []
    for mid, payload in results.items():
        summary.append(
            {
                "model": mid,
                "detections": payload.get("detections", 0),
                "visible_keypoints": payload.get("visible_keypoints", 0),
                "box_conf": (payload.get("frames") or [{}])[0].get("box_conf"),
            },
        )

    return {
        "schema": "volleyball_court_v1",
        "video_id": video_id,
        "pipeline_version": pipeline_version,
        "image_size": {"width": w, "height": h},
        "models_requested": list(models),
        "models_ok": list(results.keys()),
        "errors": errors,
        "summary": summary,
        "results": results,
        "model_paths": {
            "volley_ref": str(VOLLEY_MODEL),
            "kaggle": str(_kaggle_path()),
            "tennis": str(TENNIS_MODEL),
            "kaggle_exists": _kaggle_path().exists(),
            "tennis_exists": TENNIS_MODEL.exists(),
            "volley_exists": VOLLEY_MODEL.exists(),
        },
    }
