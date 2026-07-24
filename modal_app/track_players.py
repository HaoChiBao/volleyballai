"""
Modal SAM 3 / 3.1 player tracking (AI runs here only — never on the laptop).

Setup:
  1. Request access to https://huggingface.co/facebook/sam3.1 (or sam3)
  2. modal secret create huggingface --HF_TOKEN=hf_...
  3. modal deploy modal/track_players.py
  4. Locally: USE_MOCK_TRACKS=0 and `pip install modal` in the worker venv

Worker calls track_players.remote(video_bytes=..., video_id=...).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import modal

APP_NAME = os.environ.get("MODAL_APP_NAME", "volleyball-ai")
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "0.1.0")
DEFAULT_PROMPT = os.environ.get("SAM3_PROMPT", "person")

app = modal.App(APP_NAME)

# Heavy GPU image — SAM 3 lives only on Modal.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy",
        "opencv-python-headless",
        "huggingface_hub",
        "torch",
        "torchvision",
    )
    .run_commands(
        "pip install --no-cache-dir git+https://github.com/facebookresearch/sam3.git",
    )
)


def _mask_to_bbox(mask: Any) -> list[float] | None:
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.squeeze()
    ys, xs = np.where(arr > 0.5)
    if len(xs) == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]


def _collect_frame_objects(outputs: Any) -> list[tuple[int, list[float]]]:
    """Best-effort parse of SAM3 frame outputs → (obj_id, bbox)."""
    import numpy as np

    found: list[tuple[int, list[float]]] = []
    if outputs is None:
        return found

    # Common shapes: dict with out_obj_ids / out_binary_masks / boxes
    if isinstance(outputs, dict):
        ids = outputs.get("out_obj_ids") or outputs.get("obj_ids") or outputs.get("object_ids")
        masks = (
            outputs.get("out_binary_masks")
            or outputs.get("masks")
            or outputs.get("pred_masks")
        )
        boxes = outputs.get("boxes") or outputs.get("out_boxes") or outputs.get("bbox")

        if masks is not None:
            mask_list = list(masks) if not isinstance(masks, np.ndarray) else [
                masks[i] for i in range(len(masks))
            ]
            id_list = list(ids) if ids is not None else list(range(len(mask_list)))
            for i, m in enumerate(mask_list):
                bb = _mask_to_bbox(m)
                if bb is None:
                    continue
                oid = int(id_list[i]) if i < len(id_list) else i + 1
                found.append((oid, bb))
            return found

        if boxes is not None:
            box_list = list(boxes)
            id_list = list(ids) if ids is not None else list(range(len(box_list)))
            for i, b in enumerate(box_list):
                arr = np.asarray(b).reshape(-1)
                if arr.size < 4:
                    continue
                # xyxy or cxcywh — assume xyxy pixels if values look large
                x0, y0, x1, y1 = map(float, arr[:4])
                if x1 > x0 and y1 > y0:
                    bb = [x0, y0, x1 - x0, y1 - y0]
                else:
                    continue
                oid = int(id_list[i]) if i < len(id_list) else i + 1
                found.append((oid, bb))
            return found

    return found


def _run_sam3(video_path: str, prompt: str, fps: float) -> dict[int, list[dict[str, Any]]]:
    """Run SAM 3 video predictor; return track_id → frames."""
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
            for oid, bbox in _collect_frame_objects(response.get("outputs")):
                by_id.setdefault(oid, []).append(
                    {"t": round(t, 3), "bbox": [round(v, 1) for v in bbox]},
                )
    except Exception:
        # Fallback: use outputs from the prompted frame only
        prompted = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
            },
        )
        for oid, bbox in _collect_frame_objects(prompted.get("outputs")):
            by_id.setdefault(oid, []).append(
                {"t": 0.0, "bbox": [round(v, 1) for v in bbox]},
            )

    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception:
        pass

    return by_id


@app.function(
    image=image,
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
    """
    Track people/players with SAM 3.1 on Modal.
    Returns players.tracks.json-compatible dict.
    """
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


@app.local_entrypoint()
def main(video_path: str, prompt: str = DEFAULT_PROMPT):
    data = Path(video_path).read_bytes()
    out = track_players.remote(video_bytes=data, video_id="local", prompt=prompt)
    print(f"tracks={len(out.get('players', []))} source={out.get('source')}")
