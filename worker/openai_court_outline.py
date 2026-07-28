"""
OpenAI vision → net + court quadrilaterals (8 corners) for overlay experiments.

API key: OPENAI_API_KEY in repo-root .env (never commit).
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

OUTLINE_PROMPT = """Analyze the provided volleyball image and identify two quadrilaterals in 2D image space:

net: The four corners of the rectangular plane formed by the volleyball net.
court: The four corners of the full volleyball court boundary.

Return exactly four points for each object in this order:

top_left
top_right
bottom_right
bottom_left

Use pixel coordinates measured from the image's top-left corner, where (0, 0) is the top-left of the image. If a court corner lies outside the visible frame, infer its approximate position using the visible court lines, perspective, vanishing points, and standard volleyball-court geometry. Off-screen coordinates may be negative or exceed the image width or height. Do not clamp them to the image boundaries.

The net points should outline the net's full rectangular plane, including the top edge, bottom edge, and side boundaries. Infer obscured corners when necessary.

Return only valid JSON with no Markdown, explanation, or additional text:

{
  "image": {
    "width": 0,
    "height": 0
  },
  "net": {
    "top_left": { "x": 0, "y": 0 },
    "top_right": { "x": 0, "y": 0 },
    "bottom_right": { "x": 0, "y": 0 },
    "bottom_left": { "x": 0, "y": 0 }
  },
  "court": {
    "top_left": { "x": 0, "y": 0 },
    "top_right": { "x": 0, "y": 0 },
    "bottom_right": { "x": 0, "y": 0 },
    "bottom_left": { "x": 0, "y": 0 }
  }
}

All coordinates must be numeric. Preserve the specified point order and do not return normalized coordinates.
"""

CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


def load_dotenv(repo_root: Path | None = None) -> None:
    root = repo_root or Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def mime_for_path(path: Path) -> str:
    s = path.suffix.lower()
    if s == ".png":
        return "image/png"
    if s == ".webp":
        return "image/webp"
    return "image/jpeg"


def parse_outline_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Outline JSON root must be an object")
    return data


def _as_xy(pt: Any) -> dict[str, float]:
    if not isinstance(pt, dict):
        raise ValueError(f"Point must be object, got {type(pt)}")
    return {"x": float(pt["x"]), "y": float(pt["y"])}


def normalize_outline(
    data: dict[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Validate corner names/order; fill image size from actual pixels if needed."""
    out: dict[str, Any] = {
        "image": {
            "width": int((data.get("image") or {}).get("width") or width),
            "height": int((data.get("image") or {}).get("height") or height),
        },
        "net": {},
        "court": {},
    }
    # Always trust the real decoded image size (models often invent 1024×768).
    out["image"]["width"] = width
    out["image"]["height"] = height

    for group in ("net", "court"):
        src = data.get(group) or {}
        if not isinstance(src, dict):
            raise ValueError(f"Missing {group}")
        for name in CORNER_ORDER:
            if name not in src:
                raise ValueError(f"Missing {group}.{name}")
            out[group][name] = _as_xy(src[name])
    return out


def call_openai_outline(
    image_bytes: bytes,
    *,
    mime: str = "image/jpeg",
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.1,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Call OpenAI Chat Completions (vision). Returns (parsed_outline, raw_text).
    """
    import httpx

    key = api_key or os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY missing. Add it to repo-root .env and re-run.",
        )
    model = model or os.environ.get("OPENAI_COURT_OUTLINE_MODEL", "gpt-5.6")

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    prompt = OUTLINE_PROMPT
    if image_width and image_height:
        prompt = (
            f"The attached image is exactly {image_width}×{image_height} pixels. "
            f'Set image.width={image_width} and image.height={image_height} in the JSON.\n\n'
            + OUTLINE_PROMPT
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
            },
        ],
    }
    # GPT-5.x reasoning models reject custom temperature on Chat Completions.
    if not str(model).startswith("gpt-5"):
        payload["temperature"] = temperature
    with httpx.Client(timeout=180.0) as client:
        r = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:2000]}")
    body = r.json()
    text = (
        ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
        or ""
    )
    parsed = parse_outline_json(text)
    return parsed, text


def draw_outline_overlay(
    image_path: Path,
    outline: dict[str, Any],
    out_path: Path,
) -> None:
    """Draw court (cyan) + net (orange) quads for manual review."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    styles = {
        "court": {"outline": (0, 200, 255), "fill": (0, 200, 255, 40)},
        "net": {"outline": (255, 140, 0), "fill": (255, 140, 0, 50)},
    }

    for group, style in styles.items():
        pts = outline.get(group) or {}
        xy = [
            (pts[n]["x"], pts[n]["y"])
            for n in CORNER_ORDER
            if n in pts
        ]
        if len(xy) != 4:
            continue
        # Closed polygon
        draw.line(xy + [xy[0]], fill=style["outline"], width=3)
        for name, (x, y) in zip(CORNER_ORDER, xy, strict=True):
            r = 5
            draw.ellipse((x - r, y - r, x + r, y + r), fill=style["outline"])
            label = f"{group[0]}_{name}"
            if font:
                draw.text((x + 6, y - 10), label, fill=style["outline"], font=font)
            else:
                draw.text((x + 6, y - 10), label, fill=style["outline"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=92)


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # width, height
