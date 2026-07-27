"""
Volleyball court keypoint detection (YOLOv11n-pose).

Weights: Davidsv/volley-ref-ai → yolo_court_keypoints.pt (MIT)
Trained on Roboflow volleyball-court-keypoints (14 points).

Keypoint names / court meters follow VOLLEY-REF AI semantics with FIVB
attack lines at 6 m / 12 m (their published config incorrectly used 3 m / 15 m).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

KEYPOINT_NAMES: tuple[str, ...] = (
    "corner_top_left",
    "corner_top_right",
    "corner_bottom_left",
    "corner_bottom_right",
    "attack_top_left",
    "attack_top_right",
    "attack_bottom_left",
    "attack_bottom_right",
    "net_left",
    "net_right",
    "midline_left",
    "midline_right",
    "center_top",
    "center_bottom",
)

# Court meters (x along length, y along width). FIVB indoor 18×9.
COURT_POINTS_M: list[dict[str, float]] = [
    {"x": 0.0, "y": 0.0},  # corner_top_left
    {"x": 18.0, "y": 0.0},  # corner_top_right
    {"x": 0.0, "y": 9.0},  # corner_bottom_left
    {"x": 18.0, "y": 9.0},  # corner_bottom_right
    {"x": 6.0, "y": 0.0},  # attack_top_left
    {"x": 6.0, "y": 9.0},  # attack_top_right
    {"x": 12.0, "y": 0.0},  # attack_bottom_left
    {"x": 12.0, "y": 9.0},  # attack_bottom_right
    {"x": 9.0, "y": 0.0},  # net_left
    {"x": 9.0, "y": 9.0},  # net_right
    {"x": 0.0, "y": 4.5},  # midline_left
    {"x": 18.0, "y": 4.5},  # midline_right
    {"x": 9.0, "y": 0.0},  # center_top (same as net_left)
    {"x": 9.0, "y": 9.0},  # center_bottom (same as net_right)
]

# Outline edges for visualization (index pairs into KEYPOINT_NAMES).
SKELETON: list[tuple[int, int]] = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),  # outer rectangle
    (4, 5),
    (6, 7),  # attack lines
    (8, 9),  # net
    (10, 11),  # midlines (sideline midpoints)
]

DEFAULT_MODEL_PATH = Path("/models/yolo_court_keypoints.pt")
DEFAULT_CONF = 0.55
DEFAULT_IOU = 0.45


def court_points_for_size(
    length_m: float = 18.0,
    width_m: float = 9.0,
) -> list[dict[str, float]]:
    """Scale default FIVB keypoints to a custom court size."""
    sx = length_m / 18.0
    sy = width_m / 9.0
    return [{"x": p["x"] * sx, "y": p["y"] * sy} for p in COURT_POINTS_M]


def _valid(pt: np.ndarray, conf: float, conf_min: float) -> bool:
    return bool(pt[0] > 1 and pt[1] > 1 and conf >= conf_min)


def _parse_keypoints(
    result: Any,
    *,
    conf_min: float,
) -> tuple[list[dict[str, Any]], list[float], float] | None:
    """Return (keypoints, bbox_xywh, box_conf) for best court detection."""
    if result.keypoints is None or result.boxes is None:
        return None
    if result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return None

    kxy = result.keypoints.xy.cpu().numpy()
    kcf = None
    if result.keypoints.conf is not None:
        kcf = result.keypoints.conf.cpu().numpy()

    boxes = result.boxes
    if boxes.xywh is None or len(boxes.xywh) == 0:
        return None

    # Highest-confidence court box.
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(kxy))
    best = int(np.argmax(confs))
    xy = kxy[best]
    conf_row = kcf[best] if kcf is not None else np.ones(len(xy))
    box = boxes.xywh.cpu().numpy()[best]
    box_conf = float(confs[best])

    keypoints: list[dict[str, Any]] = []
    for i, name in enumerate(KEYPOINT_NAMES):
        if i >= len(xy):
            keypoints.append(
                {
                    "name": name,
                    "xy": None,
                    "conf": 0.0,
                    "visible": False,
                    "court_m": COURT_POINTS_M[i],
                },
            )
            continue
        c = float(conf_row[i]) if i < len(conf_row) else 0.0
        ok = _valid(xy[i], c, conf_min)
        keypoints.append(
            {
                "name": name,
                "xy": [round(float(xy[i][0]), 1), round(float(xy[i][1]), 1)]
                if ok
                else None,
                "conf": round(c, 3),
                "visible": ok,
                "court_m": COURT_POINTS_M[i],
            },
        )

    bbox = [round(float(v), 1) for v in box.tolist()]
    return keypoints, bbox, box_conf


def draw_court_overlay(
    frame: np.ndarray,
    keypoints: list[dict[str, Any]],
    *,
    bbox: list[float] | None = None,
) -> np.ndarray:
    """Draw keypoints + skeleton outline on a BGR frame."""
    out = frame.copy()
    pts: list[tuple[float, float] | None] = []
    for kp in keypoints:
        xy = kp.get("xy")
        if xy is None:
            pts.append(None)
            continue
        x, y = int(xy[0]), int(xy[1])
        pts.append((float(x), float(y)))
        cv2.circle(out, (x, y), 6, (0, 255, 120), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
        label = str(kp.get("name", ""))[:8]
        cv2.putText(
            out,
            label,
            (x + 8, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for i, j in SKELETON:
        if i >= len(pts) or j >= len(pts):
            continue
        a, b = pts[i], pts[j]
        if a is None or b is None:
            continue
        cv2.line(
            out,
            (int(a[0]), int(a[1])),
            (int(b[0]), int(b[1])),
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    if bbox and len(bbox) >= 4:
        x, y, w, h = bbox
        cv2.rectangle(
            out,
            (int(x - w / 2), int(y - h / 2)),
            (int(x + w / 2), int(y + h / 2)),
            (80, 80, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _encode_jpg_b64(frame: np.ndarray, *, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode overlay JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def detect_court_media(
    media_path: Path,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    confidence: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    sample_fps: float = 1.0,
    max_frames: int = 30,
    kpt_conf_min: float = 0.25,
    return_overlays: int = 3,
    video_id: str = "",
    pipeline_version: str = "0.1.0",
) -> dict[str, Any]:
    """
    Run court keypoint detection on an image or video.

    For video: sample at `sample_fps` (capped by `max_frames`).
    Returns court.keypoints.json-shaped payload + optional preview JPEGs (b64).
    """
    from ultralytics import YOLO

    if not model_path.exists():
        raise FileNotFoundError(f"Court model not found: {model_path}")
    if not media_path.exists():
        raise FileNotFoundError(f"Media not found: {media_path}")

    model = YOLO(str(model_path))
    suffix = media_path.suffix.lower()
    frames_out: list[dict[str, Any]] = []
    overlays_b64: list[dict[str, Any]] = []

    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        frame = cv2.imread(str(media_path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {media_path}")
        h, w = frame.shape[:2]
        results = model.predict(
            frame,
            conf=confidence,
            iou=iou,
            verbose=False,
        )
        parsed = _parse_keypoints(results[0], conf_min=kpt_conf_min) if results else None
        if parsed:
            keypoints, bbox, box_conf = parsed
            frames_out.append(
                {
                    "t": 0.0,
                    "frame_index": 0,
                    "bbox": bbox,
                    "box_conf": round(box_conf, 3),
                    "keypoints": keypoints,
                },
            )
            if return_overlays > 0:
                ov = draw_court_overlay(frame, keypoints, bbox=bbox)
                overlays_b64.append(
                    {"t": 0.0, "frame_index": 0, "jpg_b64": _encode_jpg_b64(ov)},
                )
        return {
            "video_id": video_id,
            "pipeline_version": pipeline_version,
            "source": "volley-ref-ai",
            "model": "yolo_court_keypoints",
            "model_repo": "Davidsv/volley-ref-ai",
            "keypoint_names": list(KEYPOINT_NAMES),
            "skeleton": [list(e) for e in SKELETON],
            "court_points_m": COURT_POINTS_M,
            "image_size": {"width": w, "height": h},
            "sample_fps": None,
            "frames": frames_out,
            "overlays": overlays_b64,
            "detections": len(frames_out),
        }

    # Video path
    cap = cv2.VideoCapture(str(media_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {media_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(native_fps / max(sample_fps, 0.1))))

    idx = 0
    kept = 0
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue

        t = idx / max(native_fps, 1e-3)
        results = model.predict(
            frame,
            conf=confidence,
            iou=iou,
            verbose=False,
        )
        parsed = _parse_keypoints(results[0], conf_min=kpt_conf_min) if results else None
        if parsed:
            keypoints, bbox, box_conf = parsed
            frames_out.append(
                {
                    "t": round(t, 3),
                    "frame_index": idx,
                    "bbox": bbox,
                    "box_conf": round(box_conf, 3),
                    "keypoints": keypoints,
                },
            )
            if len(overlays_b64) < return_overlays:
                ov = draw_court_overlay(frame, keypoints, bbox=bbox)
                overlays_b64.append(
                    {
                        "t": round(t, 3),
                        "frame_index": idx,
                        "jpg_b64": _encode_jpg_b64(ov),
                    },
                )
        kept += 1
        idx += 1

    cap.release()

    return {
        "video_id": video_id,
        "pipeline_version": pipeline_version,
        "source": "volley-ref-ai",
        "model": "yolo_court_keypoints",
        "model_repo": "Davidsv/volley-ref-ai",
        "keypoint_names": list(KEYPOINT_NAMES),
        "skeleton": [list(e) for e in SKELETON],
        "court_points_m": COURT_POINTS_M,
        "image_size": {"width": width, "height": height},
        "native_fps": native_fps,
        "sample_fps": sample_fps,
        "frame_step": step,
        "total_frames": total,
        "frames": frames_out,
        "overlays": overlays_b64,
        "detections": len(frames_out),
    }


MediaKind = Literal["image", "video"]
