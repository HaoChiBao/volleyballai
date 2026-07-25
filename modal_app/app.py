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
    .env({"VOLLEYBALL_SAM_IMAGE": "v7-chunked-sam"})
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


def _ffmpeg_resample(src: Path, dst: Path, *, fps: float, max_width: int = 640) -> None:
    """Write a low-FPS copy so SAM does not load the full native stream into VRAM."""
    import subprocess

    dst.parent.mkdir(parents=True, exist_ok=True)
    # scale then fps — keeps court readable while cutting memory ~6× vs 30fps.
    vf = f"scale='min({max_width},iw)':-2,fps={fps:.3f}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-an",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg resample failed ({proc.returncode}): {proc.stderr[-800:]}",
        )


def _probe_duration_s(path: Path) -> float:
    import subprocess

    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float((proc.stdout or "").strip())
    except ValueError:
        return 0.0


def _ffmpeg_slice(src: Path, dst: Path, *, start_s: float, duration_s: float) -> None:
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration_s:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg slice failed ({proc.returncode}): {proc.stderr[-800:]}",
        )


def _run_sam3(
    video_path: str,
    prompt: str,
    fps: float,
    *,
    time_offset_s: float = 0.0,
) -> dict[int, list[dict[str, Any]]]:
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    # Keep VRAM from fragmenting across chunks.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
            t = time_offset_s + frame_index / max(fps, 1e-3)
            for oid, bbox, outline in _collect_frame_objects(response.get("outputs")):
                frame: dict[str, Any] = {
                    "t": round(t, 3),
                    "bbox": [round(v, 1) for v in bbox],
                }
                if outline and len(outline) >= 3:
                    frame["outline"] = outline
                by_id.setdefault(oid, []).append(frame)
    except Exception as exc:  # noqa: BLE001
        # Raise a plain RuntimeError so the local worker (no torch) can deserialize.
        msg = f"{type(exc).__name__}: {exc}"
        print(f"[track_players] propagate failed: {msg}")
        try:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(msg) from None

    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Drop predictor references before next chunk.
        del predictor

    return by_id


def _merge_track_maps(
    parts: list[dict[int, list[dict[str, Any]]]],
) -> dict[int, list[dict[str, Any]]]:
    """Concatenate chunk tracks with unique ids (chunk-local ids remapped)."""
    merged: dict[int, list[dict[str, Any]]] = {}
    next_id = 1
    for part in parts:
        remap: dict[int, int] = {}
        for oid, frames in sorted(part.items(), key=lambda kv: kv[0]):
            if not frames:
                continue
            tid = next_id
            next_id += 1
            remap[oid] = tid
            merged[tid] = frames
    return merged


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

    # SAM 3 video loads the whole clip into VRAM — never feed native 30fps.
    sam_fps = float(os.environ.get("SAM3_FPS", "5"))
    sam_fps = max(2.0, min(sam_fps, float(fps) if fps else 5.0, 8.0))
    # ~20s @ 5fps ≈ 100 frames per chunk keeps A100 comfortable.
    chunk_s = float(os.environ.get("SAM3_CHUNK_SECONDS", "20"))
    chunk_s = max(8.0, min(chunk_s, 45.0))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "source.mp4"
        low = root / "sam_input.mp4"
        src.write_bytes(video_bytes)
        print(f"[track_players] resampling to {sam_fps} fps (max_w=640)…")
        _ffmpeg_resample(src, low, fps=sam_fps, max_width=640)
        duration = _probe_duration_s(low)
        print(f"[track_players] duration={duration:.1f}s chunk={chunk_s}s")

        parts: list[dict[int, list[dict[str, Any]]]] = []
        if duration <= 0:
            parts.append(
                _run_sam3(
                    str(low),
                    prompt=prompt or DEFAULT_PROMPT,
                    fps=sam_fps,
                ),
            )
        else:
            start = 0.0
            chunk_i = 0
            while start < duration - 0.05:
                slice_path = root / f"chunk_{chunk_i:03d}.mp4"
                take = min(chunk_s, duration - start)
                print(
                    f"[track_players] chunk {chunk_i} "
                    f"start={start:.1f}s len={take:.1f}s",
                )
                _ffmpeg_slice(low, slice_path, start_s=start, duration_s=take)
                part = _run_sam3(
                    str(slice_path),
                    prompt=prompt or DEFAULT_PROMPT,
                    fps=sam_fps,
                    time_offset_s=start,
                )
                print(
                    f"[track_players] chunk {chunk_i} "
                    f"tracks={len(part)} frames="
                    f"{sum(len(v) for v in part.values())}",
                )
                parts.append(part)
                start += chunk_s
                chunk_i += 1

        by_id = _merge_track_maps(parts)

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
        "sam_fps": sam_fps,
        "chunk_seconds": chunk_s,
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
