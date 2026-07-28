"""
Net-only detector (camera-motion aware later).

For now: given a frame, return the four corners of the volleyball net plane
in image pixels. Court geometry is intentionally out of scope.

Why net-first:
  - Net is the strongest mid-court anchor when the camera pans/zooms.
  - When the camera moves, net corners move → re-run detect (or track + refresh).
  - Court / calibration can hang off a stable net track later in the pipeline.

Output contract (per frame):
  {
    "t": 0.0,
    "image": {"width": W, "height": H},
    "net": {
      "top_left": {"x","y"}, "top_right": {"x","y"},
      "bottom_right": {"x","y"}, "bottom_left": {"x","y"}
    },
    "model": "...",
    "source": "openai_vision"
  }

Video strategy (stubbed, not fully wired yet):
  1. Detect on keyframes (or every N seconds).
  2. Estimate camera motion (optical flow / frame homography).
  3. If motion > threshold → detect again; else warp previous net or skip.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from worker.openai_court_outline import CORNER_ORDER, load_dotenv

NET_ONLY_PROMPT = """Analyze this volleyball photo and locate ONLY the volleyball NET.

Return the four corners of the net's rectangular plane (top tape × antennas/posts,
bottom tape × antennas/posts) in pixel coordinates from the image top-left (0,0).

Order (camera view):
  top_left, top_right, bottom_right, bottom_left

Infer obscured corners from posts, antennas, and visible tape. Do not invent a
court outline. Do not return court corners.

Return only JSON:
{
  "image": { "width": 0, "height": 0 },
  "net": {
    "top_left": { "x": 0, "y": 0 },
    "top_right": { "x": 0, "y": 0 },
    "bottom_right": { "x": 0, "y": 0 },
    "bottom_left": { "x": 0, "y": 0 }
  }
}
"""


def _parse_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def _as_xy(pt: Any) -> dict[str, float]:
    if not isinstance(pt, dict):
        raise ValueError(f"Point must be object, got {type(pt)}")
    return {"x": float(pt["x"]), "y": float(pt["y"])}


def normalize_net(
    data: dict[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    src = data.get("net") or {}
    if not isinstance(src, dict):
        raise ValueError("Missing net")
    net = {name: _as_xy(src[name]) for name in CORNER_ORDER}
    return {
        "image": {"width": width, "height": height},
        "net": net,
    }


def _prepare_model_image(
    im: Image.Image,
    *,
    max_side: int | None,
) -> tuple[bytes, str, int, int, float, float]:
    """
    Optionally downscale for the API. Returns:
      png/jpeg bytes, mime, model_w, model_h, scale_x_to_orig, scale_y_to_orig

    max_side=None (or <=0): no degradation — send full resolution.
    """
    import io

    orig_w, orig_h = im.size
    if max_side is None or max_side <= 0:
        work = im.convert("RGB")
        scale_x = scale_y = 1.0
    else:
        scale = min(1.0, float(max_side) / max(orig_w, orig_h))
        if scale >= 0.999:
            work = im.convert("RGB")
            scale_x = scale_y = 1.0
        else:
            nw = max(1, int(round(orig_w * scale)))
            nh = max(1, int(round(orig_h * scale)))
            work = im.convert("RGB").resize((nw, nh), Image.Resampling.LANCZOS)
            scale_x = orig_w / nw
            scale_y = orig_h / nh

    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=95)
    mw, mh = work.size
    return buf.getvalue(), "image/jpeg", mw, mh, scale_x, scale_y


def detect_net_in_image(
    image_path: Path,
    *,
    model: str | None = None,
    api_key: str | None = None,
    t: float = 0.0,
    max_side: int | None = None,
) -> dict[str, Any]:
    """
    Detect net corners in a single still. Returns pipeline-shaped frame result.

    max_side: optional longest-side cap before the vision call (degradation).
      None / 0 = full quality (default — preferred for sparse fixed-camera
      redetects). Set e.g. 512/768 for cheaper / faster experiments.
    """
    import httpx

    load_dotenv()
    key = api_key or os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    model = (
        model
        or os.environ.get("OPENAI_NET_DETECT_MODEL")
        or os.environ.get("OPENAI_COURT_OUTLINE_MODEL")
        or "gpt-5.6"
    )
    # Env override when caller leaves default None.
    if max_side is None:
        env_ms = os.environ.get("OPENAI_NET_DETECT_MAX_SIDE", "").strip()
        if env_ms:
            max_side = int(env_ms)

    im = Image.open(image_path)
    orig_w, orig_h = im.size
    jpeg_bytes, mime, mw, mh, sx, sy = _prepare_model_image(im, max_side=max_side)
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    prompt = (
        f"The attached image is exactly {mw}×{mh} pixels. "
        f"Set image.width={mw} and image.height={mh}.\n\n"
        + NET_ONLY_PROMPT
    )
    payload: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    # GPT-5.x rejects custom temperature on Chat Completions.
    if not str(model).startswith("gpt-5"):
        payload["temperature"] = 0.1

    with httpx.Client(timeout=180.0) as client:
        r = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:2000]}")
    body = r.json()
    text = (((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    parsed = normalize_net(_parse_json(text), width=mw, height=mh)

    # Scale model-space corners back to original pixels when degraded.
    net = {
        name: {
            "x": round(parsed["net"][name]["x"] * sx, 1),
            "y": round(parsed["net"][name]["y"] * sy, 1),
        }
        for name in CORNER_ORDER
    }
    usage = body.get("usage") or {}
    return {
        "t": float(t),
        "image": {"width": orig_w, "height": orig_h},
        "net": net,
        "model": model,
        "source": "openai_vision",
        "max_side": max_side,
        "model_image": {"width": mw, "height": mh},
        "degraded": bool(max_side and max_side > 0 and (mw < orig_w or mh < orig_h)),
        "usage": usage,
        "raw_text": text,
    }


def camera_motion_score(
    prev_gray: Any,
    curr_gray: Any,
) -> float:
    """
    Cheap camera-motion proxy in [0, ~1+].

    Uses mean optical-flow magnitude (Farneback). High values → re-detect net.
    Requires OpenCV. Returns 0 if frames are invalid.
    """
    import cv2
    import numpy as np

    if prev_gray is None or curr_gray is None:
        return 0.0
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    # Normalize roughly by frame diagonal.
    diag = float(np.hypot(prev_gray.shape[1], prev_gray.shape[0])) or 1.0
    return float(np.mean(mag) / (0.01 * diag))


def should_redetect_net(
    motion: float,
    *,
    threshold: float = 0.35,
    frames_since_detect: int = 0,
    max_gap_frames: int = 45,
) -> bool:
    """True when camera likely moved enough, or track is stale."""
    if frames_since_detect >= max_gap_frames:
        return True
    return motion >= threshold


def detect_net_on_video_keyframes(
    video_path: Path,
    *,
    sample_fps: float = 1.0,
    motion_threshold: float = 0.35,
    model: str | None = None,
    force_every_keyframe: bool = False,
    max_side: int | None = None,
) -> dict[str, Any]:
    """
    Sample a video and detect the net when the camera moves (or on each keyframe).

    Saves nothing — returns a tracks-like payload for later pipeline wiring.
    Default max_side=None keeps full frame quality (sparse redetects on a
    mostly-fixed camera).
    """
    import cv2
    import tempfile

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(fps / max(sample_fps, 0.1))))
    frames: list[dict[str, Any]] = []
    prev_gray = None
    last_net: dict[str, Any] | None = None
    frames_since = 10**9
    idx = 0

    with tempfile.TemporaryDirectory(prefix="net_detect_") as tmp:
        tmp_path = Path(tmp)
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue
            t = idx / fps
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            motion = camera_motion_score(prev_gray, gray) if prev_gray is not None else 999.0
            prev_gray = gray
            frames_since += step

            need = force_every_keyframe or should_redetect_net(
                motion,
                threshold=motion_threshold,
                frames_since_detect=frames_since,
            )
            if need or last_net is None:
                still = tmp_path / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(still), bgr)
                det = detect_net_in_image(
                    still, model=model, t=t, max_side=max_side
                )
                last_net = det["net"]
                frames_since = 0
                frames.append(
                    {
                        "t": t,
                        "frame_index": idx,
                        "net": det["net"],
                        "redetected": True,
                        "motion": motion,
                        "model": det["model"],
                        "max_side": max_side,
                        "degraded": det.get("degraded"),
                        "usage": det.get("usage"),
                    }
                )
            else:
                frames.append(
                    {
                        "t": t,
                        "frame_index": idx,
                        "net": last_net,
                        "redetected": False,
                        "motion": motion,
                        "model": model,
                        "max_side": max_side,
                    }
                )
            idx += 1

    cap.release()
    return {
        "video": str(video_path),
        "sample_fps": sample_fps,
        "motion_threshold": motion_threshold,
        "max_side": max_side,
        "source": "openai_vision_net",
        "frames": frames,
        "detections": sum(1 for f in frames if f.get("redetected")),
    }


def draw_net_overlay(
    source: Path | Image.Image,
    net: dict[str, dict[str, float]],
    out_path: Path,
    *,
    label_prefix: str = "n",
) -> None:
    im = (
        source.convert("RGB")
        if isinstance(source, Image.Image)
        else Image.open(source).convert("RGB")
    )
    draw = ImageDraw.Draw(im)
    xy = [(net[n]["x"], net[n]["y"]) for n in CORNER_ORDER]
    draw.line(xy + [xy[0]], fill=(0, 255, 200), width=3)
    draw.line([xy[0], xy[3]], fill=(0, 255, 200), width=3)
    draw.line([xy[1], xy[2]], fill=(0, 255, 200), width=3)
    for name, (x, y) in zip(CORNER_ORDER, xy, strict=True):
        r = 7
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0))
        draw.text((x + 9, y - 10), f"{label_prefix}_{name}", fill=(0, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=92)


def process_still(
    image_path: Path,
    out_dir: Path,
    *,
    model: str | None = None,
    max_side: int | None = None,
) -> dict[str, Any]:
    """Detect net on one still; write numbered review artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    original = out_dir / f"{stem}.01_original.jpg"
    json_path = out_dir / f"{stem}.02_net.json"
    overlay = out_dir / f"{stem}.03_overlay.jpg"

    im = Image.open(image_path).convert("RGB")
    im.save(original, quality=95)
    print(
        f"[net] detect {image_path.name} "
        f"max_side={max_side if max_side else 'full'} …",
        flush=True,
    )
    det = detect_net_in_image(image_path, model=model, max_side=max_side)
    draw_net_overlay(im, det["net"], overlay)
    payload = {
        **{k: v for k, v in det.items() if k != "raw_text"},
        "source_image": str(image_path),
        "artifacts": {
            "01_original": str(original),
            "02_net": str(json_path),
            "03_overlay": str(overlay),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[net] {stem}: degraded={det.get('degraded')} "
        f"model_image={det.get('model_image')} "
        f"overlay={overlay} usage={det.get('usage')}",
        flush=True,
    )
    return payload
