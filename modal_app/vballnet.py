"""
VballNet volleyball ball tracker (TrackNet-family heatmap).

Adapted from MIT-licensed inference in:
  https://github.com/asigatchov/fast-volleyball-tracking-inference
  Copyright (c) 2025 Alexander Sigatchov

Default weights: VballNetV1_seq9_grayscale_148 — best Acc@5px among published ONNX variants.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

# Best Acc@5px (all / visible) among published ONNX benchmarks.
MODEL_CATALOG: dict[str, dict[str, str]] = {
    "v1_148": {
        "name": "VballNetV1_seq9_grayscale_148_h288_w512",
        "filename": "VballNetV1_seq9_grayscale_148_h288_w512.onnx",
        "path": "/models/vballnet_v1_148.onnx",
        "note": "best Acc@5px (~87%)",
    },
    "v1_204": {
        "name": "VballNetV1_seq9_grayscale_204_h288_w512",
        "filename": "VballNetV1_seq9_grayscale_204_h288_w512.onnx",
        "path": "/models/vballnet_v1_204.onnx",
        "note": "2nd Acc@5px (~86%)",
    },
}

DEFAULT_MODEL_KEY = "v1_148"
_REPO_RAW = (
    "https://github.com/asigatchov/fast-volleyball-tracking-inference/raw/master/models/"
)

INPUT_WIDTH = 512
INPUT_HEIGHT = 288
SEQ_LEN = 9
HEATMAP_THRESHOLD = 0.5
RADIUS_MIN = 3.0
RADIUS_MAX = 48.0
GAP_FILL_MAX_FRAMES = 8

InferMode = Literal["quality", "fast"]


def model_url(key: str = DEFAULT_MODEL_KEY) -> str:
    meta = MODEL_CATALOG[key]
    return f"{_REPO_RAW}{meta['filename']}"


def model_path_for(key: str = DEFAULT_MODEL_KEY) -> Path:
    return Path(MODEL_CATALOG[key]["path"])


def ensure_model(
    path: Path | None = None,
    *,
    url: str | None = None,
    model_key: str = DEFAULT_MODEL_KEY,
) -> Path:
    """Download ONNX weights once into the Modal image/volume path."""
    path = path or model_path_for(model_key)
    url = url or model_url(model_key)
    if path.exists() and path.stat().st_size > 10_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    tmp = path.with_suffix(".tmp")
    print(f"[vballnet] downloading model → {path}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    if path.stat().st_size < 10_000:
        raise RuntimeError(f"Downloaded model looks empty: {path} ({path.stat().st_size} B)")
    return path


def _ort_session(model_path: Path):
    import onnxruntime as ort

    providers: list[str] = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"[vballnet] providers={session.get_providers()}")
    return session


def _preprocess_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def _decode_heatmap(
    heatmap: np.ndarray,
    *,
    threshold: float,
) -> tuple[bool, float, float, float]:
    """Return (visible, x_model, y_model, r_model) in model input pixels."""
    hm = np.asarray(heatmap, dtype=np.float32)
    _, binary = cv2.threshold(hm, threshold, 1.0, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        (binary * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return False, 0.0, 0.0, 0.0
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < 1.0:
        return False, 0.0, 0.0, 0.0
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return False, 0.0, 0.0, 0.0
    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])
    r = math.sqrt(area / math.pi)
    return True, cx, cy, r


def _gap_fill(frames: list[dict[str, Any]], max_gap: int = GAP_FILL_MAX_FRAMES) -> list[dict[str, Any]]:
    """Linear-interpolate short visibility gaps between detections."""
    if len(frames) < 2:
        return [f for f in frames if f.get("xy") is not None]
    idxs = [i for i, f in enumerate(frames) if f.get("xy") is not None]
    if len(idxs) < 2:
        return [f for f in frames if f.get("xy") is not None]
    out = [dict(f) for f in frames]
    for a, b in zip(idxs, idxs[1:]):
        gap = b - a
        if gap <= 1 or gap > max_gap + 1:
            continue
        fa, fb = out[a], out[b]
        xa, ya = fa["xy"]
        xb, yb = fb["xy"]
        ra = float(fa.get("r") or 8.0)
        rb = float(fb.get("r") or ra)
        for k in range(1, gap):
            t = k / gap
            out[a + k] = {
                "t": out[a + k]["t"],
                "xy": [
                    round(xa + (xb - xa) * t, 1),
                    round(ya + (yb - ya) * t, 1),
                ],
                "r": round(ra + (rb - ra) * t, 1),
            }
    return [f for f in out if f.get("xy") is not None]


def _emit_detection(
    *,
    t: float,
    visible: bool,
    xm: float,
    ym: float,
    rm: float,
    sx: float,
    sy: float,
    radius_hist: deque[float],
) -> dict[str, Any]:
    if not visible:
        return {"t": round(t, 3)}
    x = xm * sx
    y = ym * sy
    r = rm * ((sx + sy) * 0.5)
    r = float(np.clip(r, RADIUS_MIN, RADIUS_MAX))
    if radius_hist:
        r = 0.65 * r + 0.35 * float(np.median(radius_hist))
    radius_hist.append(r)
    return {
        "t": round(t, 3),
        "xy": [round(float(x), 1), round(float(y), 1)],
        "r": round(max(RADIUS_MIN, r), 1),
    }


def track_ball_vballnet(
    video_path: str | Path,
    *,
    model_path: Path | None = None,
    model_key: str = DEFAULT_MODEL_KEY,
    confidence_threshold: float = HEATMAP_THRESHOLD,
    gap_fill: bool = True,
    mode: InferMode = "quality",
) -> list[dict[str, Any]]:
    """
    Run VballNet on a video file.

    mode:
      - quality: per-frame sliding window, center-heatmap (TrackNet-style; best accuracy)
      - fast: non-overlapping seq batches (~9× cheaper)
    """
    model_path = ensure_model(model_path, model_key=model_key)
    session = _ort_session(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    in_shape = session.get_inputs()[0].shape
    out_shape = session.get_outputs()[0].shape
    seq_in = int(in_shape[1]) if isinstance(in_shape[1], int) else SEQ_LEN
    seq_out = int(out_shape[1]) if isinstance(out_shape[1], int) else seq_in
    center_out = seq_out // 2
    half = seq_in // 2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for ball tracking: {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or INPUT_WIDTH)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or INPUT_HEIGHT)
    vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if vid_fps <= 1e-3:
        vid_fps = 30.0
    sx = frame_w / float(INPUT_WIDTH)
    sy = frame_h / float(INPUT_HEIGHT)

    raw_frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frames.append(frame)
    cap.release()

    if not raw_frames:
        return []

    processed = [_preprocess_gray(f) for f in raw_frames]
    dense: list[dict[str, Any]] = []
    radius_hist: deque[float] = deque(maxlen=12)
    frame_count = len(raw_frames)

    print(f"[vballnet] mode={mode} frames={frame_count} seq={seq_in} model={model_path.name}")

    if mode == "quality":
        # One inference per frame; use center heatmap (matches training metrics).
        for i in range(frame_count):
            window = [
                processed[max(0, min(frame_count - 1, i - half + k))]
                for k in range(seq_in)
            ]
            tensor = np.stack(window, axis=0)[None, ...].astype(np.float32)
            preds = np.asarray(session.run([output_name], {input_name: tensor})[0][0])
            visible, xm, ym, rm = _decode_heatmap(
                preds[center_out],
                threshold=confidence_threshold,
            )
            dense.append(
                _emit_detection(
                    t=i / vid_fps,
                    visible=visible,
                    xm=xm,
                    ym=ym,
                    rm=rm,
                    sx=sx,
                    sy=sy,
                    radius_hist=radius_hist,
                ),
            )
    else:
        # Fast non-overlapping batches.
        step = max(1, min(seq_out, seq_in))
        i = 0
        while i < frame_count:
            window = list(processed[i : i + seq_in])
            while len(window) < seq_in:
                window.append(processed[-1])
            tensor = np.stack(window, axis=0)[None, ...].astype(np.float32)
            preds = np.asarray(session.run([output_name], {input_name: tensor})[0][0])
            n = min(step, frame_count - i, int(preds.shape[0]))
            for j in range(n):
                visible, xm, ym, rm = _decode_heatmap(
                    preds[j],
                    threshold=confidence_threshold,
                )
                dense.append(
                    _emit_detection(
                        t=(i + j) / vid_fps,
                        visible=visible,
                        xm=xm,
                        ym=ym,
                        rm=rm,
                        sx=sx,
                        sy=sy,
                        radius_hist=radius_hist,
                    ),
                )
            i += step

    if gap_fill:
        return _gap_fill(dense)
    return [f for f in dense if f.get("xy") is not None]
