"""
Volleyball AI — Modal app (SAM 3.1 players + ball tracker).

Deploy:
  modal deploy modal/app.py

Requires secret: `huggingface` with HF_TOKEN (access to facebook/sam3 / sam3.1).
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import modal

APP_NAME = os.environ.get("MODAL_APP_NAME", "volleyball-ai")
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "0.1.0")
DEFAULT_PROMPT = os.environ.get("SAM3_PROMPT", "person")

app = modal.App(APP_NAME)

sam_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        # sam3 pins numpy<2; keep opencv compatible with that.
        "numpy>=1.26,<2",
        "opencv-python-headless==4.10.0.84",
        "huggingface_hub",
        "einops",
        "pycocotools",
        "psutil",
        "timm",
        "ftfy==6.1.1",
        "regex",
        "iopath",
        "decord",
    )
    .run_commands(
        "pip install --no-cache-dir git+https://github.com/facebookresearch/sam3.git",
        "pip install --no-cache-dir --force-reinstall 'pycocotools>=2.0.7' psutil",
        "python -c \"import pycocotools, psutil; from sam3.model_builder import build_sam3_video_predictor; print('sam3 ready')\"",
    )
    .env({"VOLLEYBALL_SAM_IMAGE": "v6-safe-collect"})
)

ball_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("numpy", "opencv-python-headless")
)


def _mask_to_bbox_and_outline(
    mask: Any,
    *,
    max_points: int = 72,
) -> tuple[list[float], list[list[float]]] | None:
    """Return (bbox xywh, simplified outline [[x,y],...]) from a binary mask."""
    import cv2
    import numpy as np

    arr = np.asarray(mask)
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    binary = (arr > 0.5).astype(np.uint8) * 255
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    bbox = [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return bbox, []
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    eps = max(1.5, peri * 0.004)
    approx = cv2.approxPolyDP(contour, eps, True)
    pts = approx.reshape(-1, 2).astype(float)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    outline = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]
    return bbox, outline


def _is_missing(val: Any) -> bool:
    """True if val should be treated as absent (None / empty tensor/array/list)."""
    if val is None:
        return True
    try:
        import numpy as np

        if isinstance(val, np.ndarray):
            return val.size == 0
    except Exception:
        pass
    try:
        import torch

        if isinstance(val, torch.Tensor):
            return val.numel() == 0
    except Exception:
        pass
    if isinstance(val, (list, tuple, set, dict)):
        return len(val) == 0
    return False


def _first_key(outputs: dict, *keys: str) -> Any:
    """Fetch first present value; never use `a or b` with numpy/torch."""
    for key in keys:
        if key not in outputs:
            continue
        val = outputs[key]
        if _is_missing(val):
            continue
        return val
    return None


def _collect_frame_objects(
    outputs: Any,
) -> list[tuple[int, list[float], list[list[float]]]]:
    import numpy as np

    found: list[tuple[int, list[float], list[list[float]]]] = []
    try:
        if outputs is None or not isinstance(outputs, dict):
            return found

        ids = _first_key(outputs, "out_obj_ids", "obj_ids", "object_ids")
        masks = _first_key(outputs, "out_binary_masks", "masks", "pred_masks")
        boxes = _first_key(outputs, "boxes", "out_boxes", "bbox")

        if not _is_missing(masks):
            if isinstance(masks, np.ndarray):
                mask_list = [masks[i] for i in range(len(masks))]
            else:
                try:
                    import torch

                    if isinstance(masks, torch.Tensor):
                        mask_list = [masks[i] for i in range(int(masks.shape[0]))]
                    else:
                        mask_list = list(masks)
                except Exception:
                    mask_list = list(masks)
            if _is_missing(ids):
                id_list = list(range(len(mask_list)))
            else:
                id_list = [int(x) for x in list(ids)]
            for i, m in enumerate(mask_list):
                parsed = _mask_to_bbox_and_outline(m)
                if parsed is None:
                    continue
                bb, outline = parsed
                oid = id_list[i] if i < len(id_list) else i + 1
                found.append((oid, bb, outline))
            return found

        if not _is_missing(boxes):
            box_list = list(boxes)
            if _is_missing(ids):
                id_list = list(range(len(box_list)))
            else:
                id_list = [int(x) for x in list(ids)]
            for i, b in enumerate(box_list):
                arr = np.asarray(b).reshape(-1)
                if arr.size < 4:
                    continue
                x0, y0, x1, y1 = map(float, arr[:4])
                if x1 > x0 and y1 > y0:
                    bb = [x0, y0, x1 - x0, y1 - y0]
                else:
                    continue
                oid = id_list[i] if i < len(id_list) else i + 1
                outline = [
                    [bb[0], bb[1]],
                    [bb[0] + bb[2], bb[1]],
                    [bb[0] + bb[2], bb[1] + bb[3]],
                    [bb[0], bb[1] + bb[3]],
                ]
                found.append((oid, bb, outline))
    except Exception as exc:  # noqa: BLE001
        print(f"[track_players] collect_frame_objects skipped: {type(exc).__name__}: {exc}")
    return found


def _run_sam3(video_path: str, prompt: str, fps: float) -> dict[int, list[dict[str, Any]]]:
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor()
    start = predictor.handle_request(
        {"type": "start_session", "resource_path": video_path},
    )
    session_id = start["session_id"]

    predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
        },
    )

    by_id: dict[int, list[dict[str, Any]]] = {}
    try:
        stream = predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
            },
        )
        for response in stream:
            frame_index = int(response.get("frame_index", 0))
            t = frame_index / max(fps, 1e-3)
            for oid, bbox, outline in _collect_frame_objects(response.get("outputs")):
                frame: dict[str, Any] = {
                    "t": round(t, 3),
                    "bbox": [round(v, 1) for v in bbox],
                }
                if outline and len(outline) >= 3:
                    frame["outline"] = outline
                by_id.setdefault(oid, []).append(frame)
    except Exception as exc:  # noqa: BLE001
        print(f"[track_players] propagate failed: {type(exc).__name__}: {exc}")
        prompted = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
            },
        )
        for oid, bbox, outline in _collect_frame_objects(prompted.get("outputs")):
            frame = {
                "t": 0.0,
                "bbox": [round(v, 1) for v in bbox],
            }
            if outline and len(outline) >= 3:
                frame["outline"] = outline
            by_id.setdefault(oid, []).append(frame)

    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception:
        pass

    return by_id


@app.function(
    image=sam_image,
    gpu="A100",
    timeout=60 * 60,
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface")],
)
def track_players(
    video_bytes: bytes,
    video_id: str = "",
    prompt: str = DEFAULT_PROMPT,
    fps: float = 10.0,
    pipeline_version: str = PIPELINE_VERSION,
) -> dict:
    """Track players with SAM 3.1. Returns players.tracks.json payload."""
    if not video_bytes:
        raise ValueError("Empty video_bytes")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "work.mp4"
        path.write_bytes(video_bytes)
        by_id = _run_sam3(str(path), prompt=prompt or DEFAULT_PROMPT, fps=fps)

    players = [
        {"track_id": tid, "frames": frames}
        for tid, frames in sorted(by_id.items(), key=lambda kv: kv[0])
    ]
    return {
        "video_id": video_id,
        "pipeline_version": pipeline_version,
        "players": players,
        "source": "sam3.1",
        "prompt": prompt,
    }


@app.function(image=ball_image, gpu="T4", timeout=60 * 30)
def track_ball(
    video_bytes: bytes,
    video_id: str = "",
    fps: float = 10.0,
    pipeline_version: str = PIPELINE_VERSION,
) -> dict:
    """Track ball (motion blob). Returns ball.tracks.json payload."""
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


@app.local_entrypoint()
def main(video_path: str = "", prompt: str = DEFAULT_PROMPT):
    if not video_path:
        print("Deployed app volleyball-ai with track_players + track_ball")
        return
    data = Path(video_path).read_bytes()
    out = track_players.remote(video_bytes=data, video_id="local", prompt=prompt)
    print(f"players={len(out.get('players', []))} source={out.get('source')}")
