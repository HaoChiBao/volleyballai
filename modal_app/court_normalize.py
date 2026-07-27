"""
Shared court-keypoint schema + overlays for multi-model comparison.

Canonical volleyball landmarks (14) match volley-ref-ai / FIVB 18×9m.
Adapters map Kaggle (4 corners) and TennisCourtDetector (14 tennis pts)
into this schema so JSON + overlays are comparable.
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

# Canonical volleyball keypoint names (order matters for skeleton indices).
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

COURT_POINTS_M: list[dict[str, float]] = [
    {"x": 0.0, "y": 0.0},
    {"x": 18.0, "y": 0.0},
    {"x": 0.0, "y": 9.0},
    {"x": 18.0, "y": 9.0},
    {"x": 6.0, "y": 0.0},
    {"x": 6.0, "y": 9.0},
    {"x": 12.0, "y": 0.0},
    {"x": 12.0, "y": 9.0},
    {"x": 9.0, "y": 0.0},
    {"x": 9.0, "y": 9.0},
    {"x": 0.0, "y": 4.5},
    {"x": 18.0, "y": 4.5},
    {"x": 9.0, "y": 0.0},
    {"x": 9.0, "y": 9.0},
]

SKELETON: list[tuple[int, int]] = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
]

# TennisCourtDetector raw names (14 heatmaps; index order from court_reference.key_points).
TENNIS_RAW_NAMES: tuple[str, ...] = (
    "t_baseline_top_left",
    "t_baseline_top_right",
    "t_baseline_bottom_left",
    "t_baseline_bottom_right",
    "t_left_inner_top",
    "t_left_inner_bottom",
    "t_right_inner_top",
    "t_right_inner_bottom",
    "t_top_service_left",
    "t_top_service_right",
    "t_bottom_service_left",
    "t_bottom_service_right",
    "t_center_service_top",
    "t_center_service_bottom",
)

# Kaggle YOLOv8x-pose: 4 corners in radar order TL, TR, BR, BL.
KAGGLE_RAW_NAMES: tuple[str, ...] = (
    "k_corner_tl",
    "k_corner_tr",
    "k_corner_br",
    "k_corner_bl",
)


def empty_keypoints() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "xy": None,
            "conf": 0.0,
            "visible": False,
            "court_m": COURT_POINTS_M[i],
        }
        for i, name in enumerate(KEYPOINT_NAMES)
    ]


def _xy(pt: Any) -> list[float] | None:
    if pt is None:
        return None
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        if pt[0] is None or pt[1] is None:
            return None
        return [round(float(pt[0]), 1), round(float(pt[1]), 1)]
    return None


def _mid(
    a: list[float] | None,
    b: list[float] | None,
) -> list[float] | None:
    if a is None or b is None:
        return None
    return [round((a[0] + b[0]) / 2.0, 1), round((a[1] + b[1]) / 2.0, 1)]


def set_kp(
    keypoints: list[dict[str, Any]],
    name: str,
    xy: list[float] | None,
    conf: float,
) -> None:
    for kp in keypoints:
        if kp["name"] != name:
            continue
        ok = xy is not None and conf > 0
        kp["xy"] = xy if ok else None
        kp["conf"] = round(float(conf), 3) if ok else 0.0
        kp["visible"] = bool(ok)
        return


def normalize_from_kaggle4(
    pts_xy: list[Any],
    confs: list[float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Map Kaggle 4 corners (TL, TR, BR, BL) → canonical volleyball schema.

    Only outer corners are filled; attack/net/mid are left invisible.
    """
    confs = confs or [1.0] * 4
    raw: list[dict[str, Any]] = []
    for i, name in enumerate(KAGGLE_RAW_NAMES):
        xy = _xy(pts_xy[i]) if i < len(pts_xy) else None
        c = float(confs[i]) if i < len(confs) else 0.0
        raw.append(
            {
                "name": name,
                "xy": xy,
                "conf": round(c, 3) if xy else 0.0,
                "visible": xy is not None,
            },
        )

    # Kaggle order TL,TR,BR,BL → our TL,TR,BL,BR
    tl = _xy(pts_xy[0]) if len(pts_xy) > 0 else None
    tr = _xy(pts_xy[1]) if len(pts_xy) > 1 else None
    br = _xy(pts_xy[2]) if len(pts_xy) > 2 else None
    bl = _xy(pts_xy[3]) if len(pts_xy) > 3 else None
    c0 = float(confs[0]) if len(confs) > 0 else 0.0
    c1 = float(confs[1]) if len(confs) > 1 else 0.0
    c2 = float(confs[2]) if len(confs) > 2 else 0.0
    c3 = float(confs[3]) if len(confs) > 3 else 0.0

    out = empty_keypoints()
    set_kp(out, "corner_top_left", tl, c0)
    set_kp(out, "corner_top_right", tr, c1)
    set_kp(out, "corner_bottom_left", bl, c3)
    set_kp(out, "corner_bottom_right", br, c2)
    # Derive net / mid from corners when all four exist (rough geometry).
    set_kp(out, "net_left", _mid(tl, bl), min(c0, c3) * 0.5 if tl and bl else 0.0)
    set_kp(out, "net_right", _mid(tr, br), min(c1, c2) * 0.5 if tr and br else 0.0)
    set_kp(out, "midline_left", _mid(tl, bl), min(c0, c3) * 0.4 if tl and bl else 0.0)
    set_kp(out, "midline_right", _mid(tr, br), min(c1, c2) * 0.4 if tr and br else 0.0)
    set_kp(out, "center_top", _mid(tl, tr), min(c0, c1) * 0.4 if tl and tr else 0.0)
    set_kp(out, "center_bottom", _mid(bl, br), min(c3, c2) * 0.4 if bl and br else 0.0)
    return out, raw


def normalize_from_tennis14(
    pts_xy: list[Any],
    confs: list[float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Best-effort map of tennis landmarks → volleyball schema.

    Tennis geometry ≠ volleyball; corners map 1:1, service box ≈ attack,
    sideline midpoints ≈ net. Expect imperfect alignment on volleyball photos.
    """
    confs = confs or [1.0] * 14
    raw: list[dict[str, Any]] = []
    xs: list[list[float] | None] = []
    for i, name in enumerate(TENNIS_RAW_NAMES):
        xy = _xy(pts_xy[i]) if i < len(pts_xy) else None
        c = float(confs[i]) if i < len(confs) else 0.0
        xs.append(xy)
        raw.append(
            {
                "name": name,
                "xy": xy,
                "conf": round(c, 3) if xy else 0.0,
                "visible": xy is not None,
            },
        )

    out = empty_keypoints()
    # Corners
    set_kp(out, "corner_top_left", xs[0], confs[0] if len(confs) > 0 else 0.0)
    set_kp(out, "corner_top_right", xs[1], confs[1] if len(confs) > 1 else 0.0)
    set_kp(out, "corner_bottom_left", xs[2], confs[2] if len(confs) > 2 else 0.0)
    set_kp(out, "corner_bottom_right", xs[3], confs[3] if len(confs) > 3 else 0.0)
    # Service lines → attack (rough)
    set_kp(out, "attack_top_left", xs[8], confs[8] if len(confs) > 8 else 0.0)
    set_kp(out, "attack_top_right", xs[9], confs[9] if len(confs) > 9 else 0.0)
    set_kp(out, "attack_bottom_left", xs[10], confs[10] if len(confs) > 10 else 0.0)
    set_kp(out, "attack_bottom_right", xs[11], confs[11] if len(confs) > 11 else 0.0)
    # Net ≈ mid of each sideline (tennis net sits between baselines)
    set_kp(
        out,
        "net_left",
        _mid(xs[0], xs[2]),
        min(confs[0], confs[2]) * 0.6 if xs[0] and xs[2] else 0.0,
    )
    set_kp(
        out,
        "net_right",
        _mid(xs[1], xs[3]),
        min(confs[1], confs[3]) * 0.6 if xs[1] and xs[3] else 0.0,
    )
    # Center service line → center_top / center_bottom
    set_kp(out, "center_top", xs[12], confs[12] if len(confs) > 12 else 0.0)
    set_kp(out, "center_bottom", xs[13], confs[13] if len(confs) > 13 else 0.0)
    set_kp(out, "midline_left", _mid(xs[0], xs[2]), 0.3 if xs[0] and xs[2] else 0.0)
    set_kp(out, "midline_right", _mid(xs[1], xs[3]), 0.3 if xs[1] and xs[3] else 0.0)
    return out, raw


def normalize_from_volley14(
    keypoints: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pass-through for models already in canonical order; keep raw copy."""
    raw = [
        {
            "name": kp.get("name"),
            "xy": kp.get("xy"),
            "conf": kp.get("conf", 0.0),
            "visible": bool(kp.get("visible")),
        }
        for kp in keypoints
    ]
    # Re-index onto canonical template (fill missing names).
    out = empty_keypoints()
    by_name = {kp.get("name"): kp for kp in keypoints}
    for kp in out:
        src = by_name.get(kp["name"])
        if not src:
            continue
        xy = src.get("xy")
        conf = float(src.get("conf") or 0.0)
        set_kp(out, kp["name"], xy if isinstance(xy, list) else None, conf)
    return out, raw


def bbox_from_keypoints(keypoints: list[dict[str, Any]]) -> list[float] | None:
    pts = [kp["xy"] for kp in keypoints if kp.get("xy")]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [
        round((x0 + x1) / 2.0, 1),
        round((y0 + y1) / 2.0, 1),
        round(x1 - x0, 1),
        round(y1 - y0, 1),
    ]


def draw_court_overlay(
    frame: np.ndarray,
    keypoints: list[dict[str, Any]],
    *,
    bbox: list[float] | None = None,
    title: str = "",
    color_bgr: tuple[int, int, int] = (0, 220, 255),
) -> np.ndarray:
    out = frame.copy()
    if title:
        cv2.putText(
            out,
            title[:48],
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
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
        cv2.putText(
            out,
            str(kp.get("name", ""))[:10],
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
            color_bgr,
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


def encode_jpg_b64(frame: np.ndarray, *, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode overlay JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def pack_result(
    *,
    model_id: str,
    model_name: str,
    model_repo: str,
    video_id: str,
    pipeline_version: str,
    image_size: dict[str, int],
    keypoints: list[dict[str, Any]],
    raw_keypoints: list[dict[str, Any]],
    box_conf: float,
    frame: np.ndarray | None,
    return_overlay: bool,
    note: str = "",
) -> dict[str, Any]:
    bbox = bbox_from_keypoints(keypoints)
    visible = sum(1 for k in keypoints if k.get("visible"))
    frames = []
    overlays: list[dict[str, Any]] = []
    if visible > 0:
        frames.append(
            {
                "t": 0.0,
                "frame_index": 0,
                "bbox": bbox,
                "box_conf": round(float(box_conf), 3),
                "keypoints": keypoints,
                "raw_keypoints": raw_keypoints,
            },
        )
        if return_overlay and frame is not None:
            ov = draw_court_overlay(
                frame,
                keypoints,
                bbox=bbox,
                title=f"{model_id} ({visible}/14)",
            )
            overlays.append({"t": 0.0, "frame_index": 0, "jpg_b64": encode_jpg_b64(ov)})

    return {
        "video_id": video_id,
        "pipeline_version": pipeline_version,
        "schema": "volleyball_court_v1",
        "source": model_id,
        "model": model_name,
        "model_repo": model_repo,
        "keypoint_names": list(KEYPOINT_NAMES),
        "skeleton": [list(e) for e in SKELETON],
        "court_points_m": COURT_POINTS_M,
        "image_size": image_size,
        "frames": frames,
        "overlays": overlays,
        "detections": len(frames),
        "visible_keypoints": visible,
        "note": note,
    }
