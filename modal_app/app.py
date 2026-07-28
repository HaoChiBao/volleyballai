"""
Volleyball AI — Modal app (SAM 3.1 players + VballNet ball + court keypoints).

Deploy:
  modal deploy modal_app/app.py

Requires secret: `huggingface` with HF_TOKEN (access to facebook/sam3 / sam3.1).

Court keypoints weights (MIT) are baked into the court image from
Hugging Face `Davidsv/volley-ref-ai` (yolo_court_keypoints.pt).
"""

from __future__ import annotations

import os
import shutil
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

# Best Acc@5px + 2nd-best baked into the ball image (shared by both Modal fns).
_VBALLNET_MODELS = {
    "v1_148": {
        "url": (
            "https://github.com/asigatchov/fast-volleyball-tracking-inference/raw/master/"
            "models/VballNetV1_seq9_grayscale_148_h288_w512.onnx"
        ),
        "path": "/models/vballnet_v1_148.onnx",
        "name": "VballNetV1_seq9_grayscale_148_h288_w512",
    },
    "v1_204": {
        "url": (
            "https://github.com/asigatchov/fast-volleyball-tracking-inference/raw/master/"
            "models/VballNetV1_seq9_grayscale_204_h288_w512.onnx"
        ),
        "path": "/models/vballnet_v1_204.onnx",
        "name": "VballNetV1_seq9_grayscale_204_h288_w512",
    },
}
_VBALLNET_DEFAULT_KEY = "v1_148"
_VBALLNET_MODEL_PATH = _VBALLNET_MODELS[_VBALLNET_DEFAULT_KEY]["path"]

_ball_fetch_cmds = [
    "mkdir -p /models",
    *[
        f"curl -fsSL -o {m['path']} {m['url']} && "
        f"test $(stat -c%s {m['path']}) -gt 10000"
        for m in _VBALLNET_MODELS.values()
    ],
    "ls -la /models && "
    "python -c \"import onnxruntime, cv2, numpy; print('ball deps ok', onnxruntime.__version__)\"",
]

ball_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "curl", "ca-certificates")
    .pip_install(
        "numpy",
        "opencv-python-headless==4.10.0.84",
        "onnxruntime==1.20.1",
    )
    .env(
        {
            "VBALLNET_MODEL_PATH": _VBALLNET_MODEL_PATH,
            "VBALLNET_MODEL_KEY": _VBALLNET_DEFAULT_KEY,
        },
    )
    .run_commands(*_ball_fetch_cmds)
    # Must be last image step (Modal mounts local files at container start).
    .add_local_file(
        str(Path(__file__).parent / "vballnet.py"),
        remote_path="/root/vballnet.py",
    )
)

# Court models (all on Modal — never on the laptop):
#  1) volley-ref YOLOv11n-pose (14 kpts) — current production baseline
#  2) Kaggle YOLOv8x-pose (4 corners) — fetched via secret `kaggle` / Volume
#  3) TennisCourtDetector heatmap (14 tennis kpts) — baked via gdown
_COURT_MODEL_URL = (
    "https://huggingface.co/Davidsv/volley-ref-ai/resolve/main/yolo_court_keypoints.pt"
)
_COURT_MODEL_PATH = "/models/yolo_court_keypoints.pt"
_KAGGLE_COURT_MODEL_PATH = "/models/key_points_regression_model.pt"
_TENNIS_COURT_MODEL_PATH = "/models/tennis_court_detector.pth"
_TENNIS_GDRIVE = "https://drive.google.com/uc?id=1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"

court_models_volume = modal.Volume.from_name("court-extra-models", create_if_missing=True)

# High-quality video → Gaussian splat (Nerfstudio). Weights/train on Modal only.
spatial_volume = modal.Volume.from_name("spatial-scenes", create_if_missing=True)
_SPATIAL_MOUNT = "/spatial"

# Official-ish all-in-one image: nerfstudio + COLMAP + gsplat (CUDA 11.8).
spatial_image = (
    modal.Image.from_registry("dromni/nerfstudio:1.1.5")
    .entrypoint([])
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "numpy",
    )
    .env(
        {
            "SPATIAL_MOUNT": _SPATIAL_MOUNT,
            "NERFSTUDIO_METHOD": "splatfacto-big",
        },
    )
    .add_local_file(
        str(Path(__file__).parent / "spatial_scene.py"),
        remote_path="/root/spatial_scene.py",
    )
)

# Full moonshotai/Kimi-K3 (~1.56 TB) — Volume only, never baked into the image / laptop.
kimi_k3_volume = modal.Volume.from_name("kimi-k3-weights", create_if_missing=True)
_KIMI_MOUNT = "/models/kimi-k3"
_KIMI_REPO = "moonshotai/Kimi-K3"

kimi_fetch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "git")
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.30.0",
        "hf_transfer",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "KIMI_K3_MOUNT": _KIMI_MOUNT,
            "KIMI_K3_REPO": _KIMI_REPO,
        },
    )
    .add_local_file(
        str(Path(__file__).parent / "kimi_k3.py"),
        remote_path="/root/kimi_k3.py",
    )
)

# Day-0 Kimi K3: vLLM documents that only the official Docker image is usable
# (pre-release FlashInfer / KDA deps). See https://vllm.ai/blog/2026-07-27-k3
kimi_serve_image = (
    modal.Image.from_registry("vllm/vllm-openai:kimi-k3")
    .entrypoint([])
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "httpx",
        "numpy",
        "opencv-python-headless==4.10.0.84",
        "pillow",
    )
    .env(
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION": "1",
            "KIMI_K3_MOUNT": _KIMI_MOUNT,
            "KIMI_K3_REPO": _KIMI_REPO,
            "KIMI_VLLM_PORT": "8000",
        },
    )
    .add_local_file(
        str(Path(__file__).parent / "kimi_k3.py"),
        remote_path="/root/kimi_k3.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "court_normalize.py"),
        remote_path="/root/court_normalize.py",
    )
)

court_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "curl", "ca-certificates")
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "ultralytics>=8.3.0",
        "opencv-python-headless==4.10.0.84",
        "numpy",
        "huggingface_hub",
        "gdown",
        "kagglehub",
    )
    .env(
        {
            "COURT_MODEL_PATH": _COURT_MODEL_PATH,
            "KAGGLE_COURT_MODEL_PATH": _KAGGLE_COURT_MODEL_PATH,
            "TENNIS_COURT_MODEL_PATH": _TENNIS_COURT_MODEL_PATH,
            "COURT_MODELS_DIR": "/models",
        },
    )
    .run_commands(
        "mkdir -p /models",
        f"curl -fsSL -L -o {_COURT_MODEL_PATH} {_COURT_MODEL_URL} && "
        f"test $(stat -c%s {_COURT_MODEL_PATH}) -gt 1000000",
        "python -c \""
        "from ultralytics import YOLO; "
        f"m=YOLO('{_COURT_MODEL_PATH}'); "
        "print('volley-ref court model ready', m.task)\"",
        # Bake TennisCourtDetector weights (public Drive link from upstream README).
        "python -c \""
        "import gdown; "
        f"gdown.download('{_TENNIS_GDRIVE}', '{_TENNIS_COURT_MODEL_PATH}', quiet=False); "
        f"import os; assert os.path.getsize('{_TENNIS_COURT_MODEL_PATH}') > 1_000_000\"",
        "ls -la /models",
    )
    .add_local_file(
        str(Path(__file__).parent / "court_keypoints.py"),
        remote_path="/root/court_keypoints.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "court_normalize.py"),
        remote_path="/root/court_normalize.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "court_tennis_detector.py"),
        remote_path="/root/court_tennis_detector.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "court_kaggle_yolo.py"),
        remote_path="/root/court_kaggle_yolo.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "court_compare.py"),
        remote_path="/root/court_compare.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "fetch_court_weights.py"),
        remote_path="/root/fetch_court_weights.py",
    )
)


def _run_ball_track(
    video_bytes: bytes,
    *,
    video_id: str,
    pipeline_version: str,
    mode: str,
    model_key: str,
    label: str,
) -> dict:
    """Shared VballNet runner for quality / fast Modal functions."""
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from vballnet import MODEL_CATALOG, track_ball_vballnet  # type: ignore

    if not video_bytes:
        raise ValueError("Empty video_bytes")

    key = model_key if model_key in MODEL_CATALOG else _VBALLNET_DEFAULT_KEY
    meta = _VBALLNET_MODELS.get(key) or _VBALLNET_MODELS[_VBALLNET_DEFAULT_KEY]
    model_path = Path(os.environ.get("VBALLNET_MODEL_PATH", meta["path"]))
    # Prefer key-specific baked path when present.
    key_path = Path(meta["path"])
    if key_path.exists():
        model_path = key_path
    conf = float(os.environ.get("VBALLNET_CONFIDENCE", "0.5"))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "work.mp4"
        path.write_bytes(video_bytes)
        print(
            f"[{label}] key={key} mode={mode} model={model_path} conf={conf}",
        )
        frames_out = track_ball_vballnet(
            path,
            model_path=model_path,
            model_key=key,
            confidence_threshold=conf,
            gap_fill=True,
            mode=mode,  # type: ignore[arg-type]
        )
        print(f"[{label}] detections={len(frames_out)}")

    return {
        "video_id": video_id,
        "pipeline_version": pipeline_version,
        "frames": frames_out,
        "source": "vballnet",
        "model": meta["name"],
        "model_key": key,
        "infer_mode": mode,
    }


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
    sam_fps: float | None = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> dict:
    """Track players with SAM 3.1. Returns players.tracks.json payload."""
    if not video_bytes:
        raise ValueError("Empty video_bytes")

    # SAM 3 video loads the whole clip into VRAM — never feed native 30fps.
    # Prefer explicit sam_fps from the worker (local SAM3_FPS), else Modal env.
    requested = (
        float(sam_fps)
        if sam_fps is not None
        else float(os.environ.get("SAM3_FPS", "5"))
    )
    # Allow up to 15fps when requested; still clamp for VRAM safety.
    sam_fps_val = max(2.0, min(requested, float(fps) if fps else requested, 15.0))
    # ~20s @ 5fps ≈ 100 frames per chunk keeps A100 comfortable.
    chunk_s = float(os.environ.get("SAM3_CHUNK_SECONDS", "20"))
    # Shorter chunks at higher fps so VRAM stays bounded (~100–120 frames).
    if sam_fps_val >= 10:
        chunk_s = min(chunk_s, 12.0)
    elif sam_fps_val >= 8:
        chunk_s = min(chunk_s, 15.0)
    chunk_s = max(8.0, min(chunk_s, 45.0))
    sam_fps = sam_fps_val

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


@app.function(image=ball_image, cpu=4.0, memory=8192, timeout=60 * 60)
def track_ball(
    video_bytes: bytes,
    video_id: str = "",
    fps: float = 10.0,
    pipeline_version: str = PIPELINE_VERSION,
    model_key: str = _VBALLNET_DEFAULT_KEY,
) -> dict:
    """
    Best-quality ball track: VballNetV1_148 + sliding-window center-frame inference.

    Returns ball.tracks.json payload: frames[{t, xy, r}].
    """
    _ = fps
    return _run_ball_track(
        video_bytes,
        video_id=video_id,
        pipeline_version=pipeline_version,
        mode="quality",
        model_key=model_key or _VBALLNET_DEFAULT_KEY,
        label="track_ball",
    )


@app.function(image=ball_image, cpu=2.0, memory=4096, timeout=60 * 30)
def track_ball_fast(
    video_bytes: bytes,
    video_id: str = "",
    fps: float = 10.0,
    pipeline_version: str = PIPELINE_VERSION,
    model_key: str = _VBALLNET_DEFAULT_KEY,
) -> dict:
    """
    Faster ball track: same VballNetV1 weights, non-overlapping seq batches.

    Use when iterating quickly; prefer `track_ball` for final analysis quality.
    """
    _ = fps
    return _run_ball_track(
        video_bytes,
        video_id=video_id,
        pipeline_version=pipeline_version,
        mode="fast",
        model_key=model_key or _VBALLNET_DEFAULT_KEY,
        label="track_ball_fast",
    )


@app.function(image=court_image, cpu=2.0, memory=4096, timeout=60 * 20)
def detect_court(
    video_bytes: bytes,
    video_id: str = "",
    pipeline_version: str = PIPELINE_VERSION,
    sample_fps: float = 1.0,
    max_frames: int = 30,
    confidence: float = 0.55,
    return_overlays: int = 3,
    media_suffix: str = ".mp4",
) -> dict:
    """
    Detect 14 volleyball court keypoints (YOLOv11n-pose).

    Returns court.keypoints.json payload + optional JPEG overlays (base64).
    Weights: Davidsv/volley-ref-ai (baked into image).
    """
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from court_keypoints import detect_court_media  # type: ignore

    if not video_bytes:
        raise ValueError("Empty video_bytes")

    model_path = Path(os.environ.get("COURT_MODEL_PATH", _COURT_MODEL_PATH))
    suffix = media_suffix if media_suffix.startswith(".") else f".{media_suffix}"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"input{suffix}"
        path.write_bytes(video_bytes)
        print(
            f"[detect_court] model={model_path} sample_fps={sample_fps} "
            f"max_frames={max_frames} conf={confidence}",
        )
        out = detect_court_media(
            path,
            model_path=model_path,
            confidence=confidence,
            sample_fps=sample_fps,
            max_frames=max_frames,
            return_overlays=return_overlays,
            video_id=video_id,
            pipeline_version=pipeline_version,
        )
        print(f"[detect_court] detections={out.get('detections')} frames={len(out.get('frames', []))}")
        return out


@app.function(
    image=court_image,
    cpu=2.0,
    memory=8192,
    timeout=60 * 30,
    volumes={"/vol/court-extra": court_models_volume},
)
def fetch_court_models(
    kaggle_username: str = "",
    kaggle_key: str = "",
) -> dict:
    """
    Download TennisCourtDetector + Kaggle YOLOv8x court weights onto the
    `court-extra-models` Volume.

    Tennis weights are public. For Kaggle, pass username/key once:
      modal run modal_app/app.py::fetch_court_models_local \\
        --kaggle-username YOU --kaggle-key YOUR_KEY
    or create `modal secret create kaggle KAGGLE_USERNAME=… KAGGLE_KEY=…`
    and wire the secret onto this function later.
    """
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    if kaggle_username:
        os.environ["KAGGLE_USERNAME"] = kaggle_username
    if kaggle_key:
        os.environ["KAGGLE_KEY"] = kaggle_key

    import fetch_court_weights  # type: ignore

    code = fetch_court_weights.main()
    court_models_volume.commit()
    vol = Path("/vol/court-extra")
    return {
        "ok": code == 0,
        "exit_code": code,
        "volume_files": sorted(p.name for p in vol.iterdir()) if vol.exists() else [],
        "kaggle_username_set": bool(os.environ.get("KAGGLE_USERNAME")),
    }


@app.function(
    image=court_image,
    cpu=2.0,
    memory=8192,
    timeout=60 * 20,
    volumes={"/vol/court-extra": court_models_volume},
)
def compare_court_models(
    image_bytes: bytes,
    video_id: str = "",
    pipeline_version: str = PIPELINE_VERSION,
    media_suffix: str = ".jpg",
    models: list[str] | None = None,
    volley_confidence: float = 0.55,
    kaggle_confidence: float = 0.001,
    return_overlays: bool = True,
) -> dict:
    """
    Run volley-ref + Kaggle YOLOv8x + TennisCourtDetector on one image.

    All results use schema `volleyball_court_v1` (normalized keypoints).
    """
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from court_compare import compare_court_models_image  # type: ignore

    if not image_bytes:
        raise ValueError("Empty image_bytes")

    wanted = tuple(models or ("volley_ref", "kaggle", "tennis"))
    suffix = media_suffix if media_suffix.startswith(".") else f".{media_suffix}"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"input{suffix}"
        path.write_bytes(image_bytes)
        print(f"[compare_court] models={wanted} size={len(image_bytes)}")
        out = compare_court_models_image(
            path,
            video_id=video_id,
            pipeline_version=pipeline_version,
            volley_confidence=volley_confidence,
            kaggle_confidence=kaggle_confidence,
            models=wanted,
            return_overlays=return_overlays,
        )
        print(
            f"[compare_court] ok={out.get('models_ok')} errors={list((out.get('errors') or {}).keys())}",
        )
        return out


@app.function(
    image=kimi_fetch_image,
    volumes={_KIMI_MOUNT: kimi_k3_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60 * 12,
    cpu=8.0,
    memory=65536,
    # Staging only — full weights land on the Volume mount, not ephemeral disk.
    ephemeral_disk=200_000,
)
def fetch_kimi_k3() -> dict:
    """
    Download full moonshotai/Kimi-K3 (~1.56 TB) onto Volume `kimi-k3-weights`.

    Resume-safe. Commits the Volume every ~5 minutes during download.
    """
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from kimi_k3 import (  # type: ignore
        KIMI_MOUNT,
        fetch_snapshot,
        free_disk_gb,
        volume_stats,
    )

    # Keep HF temp/cache off the Volume so we don't double ~1.56 TB.
    os.environ.setdefault("HF_HOME", "/tmp/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf/hub")
    Path("/tmp/hf/hub").mkdir(parents=True, exist_ok=True)

    root = Path(KIMI_MOUNT)
    print(f"[fetch_kimi_k3] free_disk_gb={free_disk_gb(root)} stats={volume_stats(root)}")

    def _commit() -> None:
        kimi_k3_volume.commit()

    result = fetch_snapshot(root, commit_fn=_commit, commit_every_s=300.0)
    _commit()
    print(f"[fetch_kimi_k3] done ok={result.get('ok')} stats={result.get('stats')}")
    return result


@app.function(
    image=kimi_fetch_image,
    volumes={_KIMI_MOUNT: kimi_k3_volume},
    timeout=60 * 10,
    cpu=2.0,
    memory=4096,
)
def kimi_k3_volume_status() -> dict:
    """Report whether the Kimi-K3 Volume snapshot looks complete."""
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from kimi_k3 import KIMI_MOUNT, is_snapshot_complete, volume_stats  # type: ignore

    root = Path(KIMI_MOUNT)
    st = volume_stats(root)
    return {
        "complete": is_snapshot_complete(root),
        "stats": st,
        "repo_id": _KIMI_REPO,
        "mount": _KIMI_MOUNT,
    }


@app.cls(
    image=kimi_serve_image,
    gpu="B300:8",
    volumes={_KIMI_MOUNT: kimi_k3_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60 * 4,
    startup_timeout=60 * 60,
    scaledown_window=60 * 20,
    memory=524288,  # 512 GiB host RAM for weight staging
    cpu=32.0,
    max_containers=1,
    ephemeral_disk=100_000,
)
class KimiK3Server:
    """
    Full Kimi-K3 via vLLM on 8×B300 (tensor-parallel-size 8).

    Serves OpenAI-compatible HTTP on localhost:8000 inside the container.
    """

    @modal.enter()
    def start(self) -> None:
        import subprocess
        import sys
        import time

        import httpx

        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        from kimi_k3 import KIMI_MOUNT, is_snapshot_complete, volume_stats  # type: ignore

        root = Path(KIMI_MOUNT)
        st = volume_stats(root)
        print(f"[KimiK3Server] volume stats={st}", flush=True)
        if not is_snapshot_complete(root):
            raise RuntimeError(
                "Kimi-K3 snapshot incomplete on Volume. "
                "Run: modal run modal_app/app.py::fetch_kimi_k3_local",
            )

        port = int(os.environ.get("KIMI_VLLM_PORT", "8000"))
        # Prefer weights at mount root (config.json present).
        model_path = str(root)
        # Prefer `vllm` CLI from the official kimi-k3 image; fall back to -m.
        vllm_bin = "vllm"
        cmd = [
            vllm_bin,
            "serve",
            model_path,
            "--served-model-name",
            "kimi-k3",
            "--tensor-parallel-size",
            "8",
            "--trust-remote-code",
            "--load-format",
            "fastsafetensors",
            "--enable-prefix-caching",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "kimi_k3",
            "--reasoning-parser",
            "kimi_k3",
            "--max-model-len",
            os.environ.get("KIMI_MAX_MODEL_LEN", "32768"),
            "--gpu-memory-utilization",
            os.environ.get("KIMI_GPU_MEM_UTIL", "0.90"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        print(f"[KimiK3Server] starting: {' '.join(cmd)}", flush=True)
        self._port = port
        self._base_url = f"http://127.0.0.1:{port}/v1"
        self._proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Cold load of ~1.56 TB can take many minutes.
        deadline = time.time() + 60 * 45
        last_err = ""
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited early with code {self._proc.returncode}",
                )
            try:
                r = httpx.get(f"{self._base_url}/models", timeout=5.0)
                if r.status_code == 200:
                    print(f"[KimiK3Server] ready: {r.text[:300]}", flush=True)
                    return
                last_err = f"status={r.status_code}"
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
            time.sleep(5)
        raise TimeoutError(f"vLLM did not become ready in time ({last_err})")

    @modal.exit()
    def stop(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                proc.kill()

    @modal.method()
    def health(self) -> dict:
        import httpx

        r = httpx.get(f"{self._base_url}/models", timeout=30.0)
        return {"ok": r.status_code == 200, "body": r.json() if r.status_code == 200 else r.text}

    @modal.method()
    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Proxy to local vLLM OpenAI chat.completions."""
        import httpx

        body = dict(payload)
        body.setdefault("model", "kimi-k3")
        r = httpx.post(
            f"{self._base_url}/chat/completions",
            json=body,
            timeout=60 * 20,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"vLLM chat error {r.status_code}: {r.text[:2000]}")
        return r.json()

    @modal.method()
    def analyze_court(
        self,
        image_bytes: bytes,
        *,
        video_id: str = "",
        pipeline_version: str = PIPELINE_VERSION,
        media_suffix: str = ".jpg",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        return_overlay: bool = True,
    ) -> dict[str, Any]:
        """Vision → volleyball_court_v1 keypoints using full self-hosted Kimi-K3."""
        import sys

        import cv2
        import numpy as np

        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        from court_normalize import (  # type: ignore
            COURT_POINTS_M,
            KEYPOINT_NAMES,
            SKELETON,
            draw_court_overlay,
            encode_jpg_b64,
            bbox_from_keypoints,
        )
        from kimi_k3 import (  # type: ignore
            court_keypoint_system_prompt,
            court_keypoint_user_text,
            encode_image_data_url,
            mime_for_suffix,
            parse_keypoints_json,
        )

        if not image_bytes:
            raise ValueError("Empty image_bytes")

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode image")
        h, w = frame.shape[:2]
        mime = mime_for_suffix(media_suffix)
        data_url = encode_image_data_url(image_bytes, mime=mime)

        payload = {
            "model": "kimi-k3",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": court_keypoint_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": court_keypoint_user_text(w, h)},
                    ],
                },
            ],
        }
        print(f"[KimiK3Server.analyze_court] {w}x{h} bytes={len(image_bytes)}", flush=True)
        import httpx

        r = httpx.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            timeout=60 * 20,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"vLLM chat error {r.status_code}: {r.text[:2000]}")
        completion = r.json()
        text = (
            ((completion.get("choices") or [{}])[0].get("message") or {}).get("content")
            or ""
        )
        keypoints = parse_keypoints_json(text, width=w, height=h)
        # Attach court meters for schema compatibility with YOLO path.
        for i, kp in enumerate(keypoints):
            if i < len(COURT_POINTS_M):
                kp["court_m"] = COURT_POINTS_M[i]

        visible = sum(1 for k in keypoints if k.get("visible"))
        bbox = bbox_from_keypoints(keypoints)
        overlays: list[dict[str, Any]] = []
        frames_out: list[dict[str, Any]] = []
        if visible > 0:
            frames_out.append(
                {
                    "t": 0.0,
                    "frame_index": 0,
                    "bbox": bbox,
                    "box_conf": round(
                        float(
                            np.mean([k["conf"] for k in keypoints if k.get("visible")])
                            if visible
                            else 0.0,
                        ),
                        3,
                    ),
                    "keypoints": keypoints,
                    "raw_text": text[:8000],
                },
            )
            if return_overlay:
                ov = draw_court_overlay(
                    frame,
                    keypoints,
                    bbox=bbox,
                    title=f"kimi-k3 ({visible}/14)",
                )
                overlays.append(
                    {"t": 0.0, "frame_index": 0, "jpg_b64": encode_jpg_b64(ov)},
                )

        return {
            "video_id": video_id,
            "pipeline_version": pipeline_version,
            "schema": "volleyball_court_v1",
            "source": "kimi_k3",
            "model": "moonshotai/Kimi-K3",
            "model_repo": _KIMI_REPO,
            "keypoint_names": list(KEYPOINT_NAMES),
            "skeleton": [list(e) for e in SKELETON],
            "court_points_m": COURT_POINTS_M,
            "image_size": {"width": w, "height": h},
            "frames": frames_out,
            "overlays": overlays,
            "detections": len(frames_out),
            "visible_keypoints": visible,
            "raw_completion": text[:8000],
            "note": "Full self-hosted Kimi-K3 on Modal Volume + vLLM B300:8",
        }


@app.function(timeout=60 * 90)
def analyze_court_with_kimi_k3(
    image_bytes: bytes,
    video_id: str = "",
    pipeline_version: str = PIPELINE_VERSION,
    media_suffix: str = ".jpg",
    return_overlay: bool = True,
) -> dict:
    """
    Convenience wrapper: spawn KimiK3Server and run court keypoint analysis.

    Requires a completed `fetch_kimi_k3` Volume snapshot.
    """
    return KimiK3Server().analyze_court.remote(
        image_bytes,
        video_id=video_id,
        pipeline_version=pipeline_version,
        media_suffix=media_suffix,
        return_overlay=return_overlay,
    )


@app.function(
    image=spatial_image,
    gpu="A100-80GB",
    volumes={_SPATIAL_MOUNT: spatial_volume},
    timeout=60 * 60 * 3,
    memory=65536,
    cpu=8.0,
    ephemeral_disk=200_000,
)
def build_spatial_scene(
    video_bytes: bytes,
    video_id: str = "scene",
    *,
    method: str = "splatfacto-big",
    max_iters: int = 30_000,
    num_frames_target: int = 280,
    appearance_embedding: bool = True,
    burn_transients: bool = True,
    players_tracks_json: str | None = None,
) -> dict:
    """
    Best-quality static environment splat from video (Nerfstudio splatfacto-big).

    Writes `/spatial/{video_id}/export/scene.ply` on Volume `spatial-scenes`.
    Optionally burns player bboxes from players.tracks.json to reduce ghosts.
    """
    import sys
    import tempfile

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from spatial_scene import SPATIAL_MOUNT, build_gaussian_scene  # type: ignore

    if not video_bytes:
        raise ValueError("Empty video_bytes")

    tracks_payload = None
    if players_tracks_json:
        import json as _json

        tracks_payload = _json.loads(players_tracks_json)

    out_root = Path(SPATIAL_MOUNT) / (video_id or "scene")
    out_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spatial_in_") as td:
        video_path = Path(td) / "input.mp4"
        video_path.write_bytes(video_bytes)
        meta = build_gaussian_scene(
            video_path,
            out_root,
            method=method,
            max_iters=max_iters,
            num_frames_target=num_frames_target,
            appearance_embedding=appearance_embedding,
            tracks_payload=tracks_payload,
            burn_transients=burn_transients,
        )

    # Persist under a stable volume-relative path for download.
    ply_src = Path(meta["ply"])
    stable_dir = Path(SPATIAL_MOUNT) / (video_id or "scene") / "publish"
    stable_dir.mkdir(parents=True, exist_ok=True)
    stable_ply = stable_dir / "scene.ply"
    stable_meta = stable_dir / "meta.json"
    if ply_src.exists():
        shutil.copy2(ply_src, stable_ply)
    import json as _json

    pub = {
        **meta,
        "video_id": video_id,
        "pipeline_version": PIPELINE_VERSION,
        "volume": "spatial-scenes",
        "volume_ply": str(stable_ply),
        "volume_meta": str(stable_meta),
    }
    stable_meta.write_text(_json.dumps(pub, indent=2), encoding="utf-8")
    spatial_volume.commit()
    print(f"[build_spatial_scene] published {stable_ply} bytes={stable_ply.stat().st_size}")
    return pub


@app.function(
    image=spatial_image,
    volumes={_SPATIAL_MOUNT: spatial_volume},
    timeout=60 * 20,
    memory=8192,
    cpu=2.0,
)
def download_spatial_scene_ply(video_id: str) -> dict:
    """Return scene.ply bytes + meta from Volume (for local artifact write)."""
    import base64
    import json as _json

    pub = Path(_SPATIAL_MOUNT) / video_id / "publish"
    ply = pub / "scene.ply"
    meta_path = pub / "meta.json"
    if not ply.exists():
        raise FileNotFoundError(
            f"No published splat for video_id={video_id}. "
            "Run build_spatial_scene first.",
        )
    raw = ply.read_bytes()
    meta = {}
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "video_id": video_id,
        "ply_b64": base64.b64encode(raw).decode("ascii"),
        "ply_bytes": len(raw),
        "meta": meta,
    }


@app.local_entrypoint()
def main(video_path: str = "", prompt: str = DEFAULT_PROMPT):
    if not video_path:
        print(
            "Deployed volleyball-ai: track_players + track_ball (+fast) + "
            "detect_court + compare_court_models + fetch_kimi_k3 + KimiK3Server + "
            "build_spatial_scene",
        )
        return
    data = Path(video_path).read_bytes()
    out = track_players.remote(video_bytes=data, video_id="local", prompt=prompt)
    print(f"players={len(out.get('players', []))} source={out.get('source')}")


@app.local_entrypoint()
def build_spatial_scene_local(
    video_path: str = "",
    video_id: str = "local-spatial",
    tracks_path: str = "",
    max_iters: int = 30_000,
):
    """
    Best-quality env splat on Modal (splatfacto-big / A100-80GB).

    Example:
      modal run modal_app/app.py::build_spatial_scene_local \\
        --video-path .data/videos/<id>/work.mp4 \\
        --video-id <id> \\
        --tracks-path .data/videos/<id>/players.tracks.json
    """
    if not video_path:
        raise SystemExit("Pass --video-path path/to/work.mp4")
    media = Path(video_path)
    if not media.exists():
        raise SystemExit(f"Not found: {media}")
    tracks_json = None
    if tracks_path:
        tp = Path(tracks_path)
        if tp.exists():
            tracks_json = tp.read_text(encoding="utf-8")
    print(
        f"[build_spatial_scene_local] {media} → Modal splatfacto-big "
        f"(iters={max_iters})…",
    )
    meta = build_spatial_scene.remote(
        media.read_bytes(),
        video_id=video_id,
        max_iters=max_iters,
        players_tracks_json=tracks_json,
    )
    print(meta)


@app.local_entrypoint()
def fetch_kimi_k3_local():
    """Download full moonshotai/Kimi-K3 onto Modal Volume kimi-k3-weights."""
    print("[fetch_kimi_k3_local] starting (this can take many hours / ~1.56 TB)…")
    result = fetch_kimi_k3.remote()
    print(result)


@app.local_entrypoint()
def kimi_k3_status_local():
    """Print Volume completeness for Kimi-K3."""
    print(kimi_k3_volume_status.remote())


@app.local_entrypoint()
def test_kimi_court(image_path: str = ""):
    """
    Run full self-hosted Kimi-K3 court keypoint analysis on one image.

    Requires fetch_kimi_k3 completed and B300:8 capacity.
    """
    if not image_path:
        raise SystemExit("Pass --image-path path/to/court.jpg")
    media = Path(image_path)
    if not media.exists():
        raise SystemExit(f"Not found: {media}")
    data = media.read_bytes()
    print(f"[test_kimi_court] {media} ({len(data)} bytes) → KimiK3Server…")
    out = analyze_court_with_kimi_k3.remote(
        data,
        video_id=media.stem,
        media_suffix=media.suffix or ".jpg",
        return_overlay=True,
    )
    print(
        f"[test_kimi_court] visible={out.get('visible_keypoints')}/14 "
        f"source={out.get('source')} model={out.get('model')}",
    )
    print((out.get("raw_completion") or "")[:500])


@app.local_entrypoint()
def fetch_court_models_local(
    kaggle_username: str = "",
    kaggle_key: str = "",
):
    """Download tennis + kaggle court weights onto the Modal Volume."""
    result = fetch_court_models.remote(
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key,
    )
    print(result)


@app.local_entrypoint()
def test_court(
    video_path: str = "",
    out_dir: str = ".data/court-test",
    sample_fps: float = 1.0,
    max_frames: int = 20,
    confidence: float = 0.55,
):
    """
    Run detect_court on a local video/image and write JSON + overlay JPEGs.

    Example:
      modal run modal_app/app.py::test_court --video-path .data/videos/<id>/work.mp4
    """
    import base64
    import json

    if not video_path:
        raise SystemExit(
            "Pass --video-path to an mp4/jpg (e.g. .data/videos/<id>/work.mp4 or thumb.jpg)",
        )

    src = Path(video_path)
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    suffix = src.suffix.lower() or ".mp4"
    data = src.read_bytes()
    print(f"[test_court] uploading {src} ({len(data)} bytes)…")
    result = detect_court.remote(
        video_bytes=data,
        video_id=src.stem,
        sample_fps=sample_fps,
        max_frames=max_frames,
        confidence=confidence,
        return_overlays=3,
        media_suffix=suffix,
    )

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    json_path = dest / "court.keypoints.json"
    # Drop heavy overlays from the saved JSON copy (keep on disk as jpgs).
    payload = {k: v for k, v in result.items() if k != "overlays"}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[test_court] wrote {json_path}")
    print(
        f"[test_court] detections={result.get('detections')} "
        f"sample_fps={result.get('sample_fps')} "
        f"size={result.get('image_size')}",
    )

    for i, ov in enumerate(result.get("overlays") or []):
        raw = base64.b64decode(ov["jpg_b64"])
        jpg_path = dest / f"overlay_{i:02d}_t{ov.get('t', 0)}.jpg"
        jpg_path.write_bytes(raw)
        print(f"[test_court] wrote {jpg_path}")

    # Print first-frame keypoint summary
    frames = result.get("frames") or []
    if frames:
        vis = [k for k in frames[0].get("keypoints", []) if k.get("visible")]
        print(f"[test_court] first hit: {len(vis)}/14 visible keypoints @ t={frames[0].get('t')}")
        for k in vis[:8]:
            print(f"  - {k['name']}: {k['xy']} conf={k['conf']}")
    else:
        print("[test_court] no court detections — try a clearer frame / lower --confidence")


@app.local_entrypoint()
def test_court_compare(
    image_path: str = "",
    out_dir: str = ".data/court-model-compare",
):
    """
    Compare volley-ref / Kaggle / TennisCourtDetector on one image.

    Example:
      modal run modal_app/app.py::test_court_compare \\
        --image-path .data/court-model-test/images/05_penn_state_vb.jpg
    """
    import base64
    import json

    if not image_path:
        raise SystemExit("Pass --image-path to a jpg/png")

    src = Path(image_path)
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    data = src.read_bytes()
    print(f"[test_court_compare] uploading {src} ({len(data)} bytes)…")
    result = compare_court_models.remote(
        image_bytes=data,
        video_id=src.stem,
        media_suffix=src.suffix.lower() or ".jpg",
        return_overlays=True,
    )

    dest = Path(out_dir) / src.stem
    dest.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in result.items() if k != "results"}
    models_out = {}
    for mid, payload in (result.get("results") or {}).items():
        models_out[mid] = {k: v for k, v in payload.items() if k != "overlays"}
        for i, ov in enumerate(payload.get("overlays") or []):
            jpg = dest / f"{mid}_overlay_{i:02d}.jpg"
            jpg.write_bytes(base64.b64decode(ov["jpg_b64"]))
            print(f"[test_court_compare] wrote {jpg}")
    slim["results"] = models_out
    (dest / "compare.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"[test_court_compare] wrote {dest / 'compare.json'}")
    print("summary:", json.dumps(result.get("summary"), indent=2))
    if result.get("errors"):
        print("errors:", json.dumps(result.get("errors"), indent=2))
