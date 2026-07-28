"""
ImageGen + FIVB geometry experiment:

1) gpt-image-2 places four lime-green markers on the court boundary corners.
2) We extract those corners, scale back to the original image, and solve
   image↔court homography (18×9 m rectangle, right angles).
3) Net corners are derived in math (center line + net height + camera pose),
   not labeled by ImageGen.
4) Final overlay draws the full ground court (boundary, attack lines, center)
   plus the derived net plane.

Same aspect ratio is preserved via letterboxing into the API canvas size.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from worker.court_calib import (
    apply_homography,
    compute_homography,
    estimate_camera_from_H,
    invert_homography,
    project_world_to_image,
)
from worker.openai_court_outline import CORNER_ORDER, load_dotenv

# FIVB indoor defaults (meters). Net height: women 2.24, men 2.43.
DEFAULT_LENGTH_M = 18.0
DEFAULT_WIDTH_M = 9.0
DEFAULT_NET_HEIGHT_M = 2.24

LABEL_PROMPT = """Edit this volleyball photo. Do NOT restyle, redraw, or change the scene.
Keep the original photo pixels everywhere except the four markers you add.

CRITICAL: Mark the PLAYING COURT FLOOR corners only — the four corners of the
white boundary rectangle painted on the ground. The net will be computed later.

DO NOT place markers on: the net tape, net mesh, antennas, net posts, referee
stand, scoreboard, or players. FLOOR corners only.

Add exactly four solid filled circles, pure lime green RGB(0,255,0), diameter
about 30–44px. No text, no lines, no other colors, no other overlays.

Corner meanings (camera view):
- top_left: farthest-from-camera × leftmost court corner on the floor
- top_right: farthest-from-camera × rightmost court corner on the floor
- bottom_right: nearest-to-camera × rightmost court corner on the floor
- bottom_left: nearest-to-camera × leftmost court corner on the floor

These are the four intersections of the two endlines with the two sidelines.
If a corner is off-screen, clamp the marker just inside the image edge along
that corner's direction. If lines are faint, infer using the net as mid-court,
player positions, attack lines, and perspective.
"""


def _meter_mappings(
    length_m: float,
    width_m: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    FIVB assignments for camera-relative TL/TR/BR/BL.

    Covers endline vs sideline camera sides and which end/side is "far".
    Also includes near-half variants (ImageGen often stops at the net/center).
    """
    L, W = length_m, width_m
    mid = L / 2.0
    mid_w = W / 2.0
    return {
        # Camera behind near endline (X=L), looking toward X=0
        "endline_far_x0": {
            "top_left": {"x": 0.0, "y": W},
            "top_right": {"x": 0.0, "y": 0.0},
            "bottom_right": {"x": L, "y": 0.0},
            "bottom_left": {"x": L, "y": W},
        },
        # Camera behind near endline (X=0), looking toward X=L
        "endline_far_xL": {
            "top_left": {"x": L, "y": W},
            "top_right": {"x": L, "y": 0.0},
            "bottom_right": {"x": 0.0, "y": 0.0},
            "bottom_left": {"x": 0.0, "y": W},
        },
        # Camera on near sideline (Y=0), looking toward Y=W
        "sideline_far_yW": {
            "top_left": {"x": 0.0, "y": W},
            "top_right": {"x": L, "y": W},
            "bottom_right": {"x": L, "y": 0.0},
            "bottom_left": {"x": 0.0, "y": 0.0},
        },
        # Camera on near sideline (Y=W), looking toward Y=0
        "sideline_far_y0": {
            "top_left": {"x": L, "y": 0.0},
            "top_right": {"x": 0.0, "y": 0.0},
            "bottom_right": {"x": 0.0, "y": W},
            "bottom_left": {"x": L, "y": W},
        },
        # ImageGen often marks near-half only: near endline + center/net line.
        "half_near_xL": {
            "top_left": {"x": mid, "y": W},
            "top_right": {"x": mid, "y": 0.0},
            "bottom_right": {"x": L, "y": 0.0},
            "bottom_left": {"x": L, "y": W},
        },
        "half_near_x0": {
            "top_left": {"x": mid, "y": W},
            "top_right": {"x": mid, "y": 0.0},
            "bottom_right": {"x": 0.0, "y": 0.0},
            "bottom_left": {"x": 0.0, "y": W},
        },
        "half_near_y0": {
            "top_left": {"x": 0.0, "y": mid_w},
            "top_right": {"x": L, "y": mid_w},
            "bottom_right": {"x": L, "y": 0.0},
            "bottom_left": {"x": 0.0, "y": 0.0},
        },
        "half_near_yW": {
            "top_left": {"x": L, "y": mid_w},
            "top_right": {"x": 0.0, "y": mid_w},
            "bottom_right": {"x": 0.0, "y": W},
            "bottom_left": {"x": L, "y": W},
        },
    }


def _score_geometry(
    camera: dict[str, Any],
    ground_lines: dict[str, Any],
    net_image: dict[str, dict[str, float]],
    *,
    image_width: int,
    image_height: int,
) -> float:
    """Higher is better — prefers sane camera height and on-frame court."""
    score = 0.0
    z = float(camera["position"][2])
    if 2.5 <= z <= 35.0:
        score += 10.0 - abs(z - 12.0) * 0.15
    else:
        score -= 50.0

    # Court boundary should stay near the frame (not shooting to infinity).
    for x, y in ground_lines.get("boundary") or []:
        if -0.25 * image_width <= x <= 1.25 * image_width:
            score += 1.0
        else:
            score -= 8.0
        if -0.25 * image_height <= y <= 1.25 * image_height:
            score += 1.0
        else:
            score -= 8.0
        # Ground features shouldn't sit in the extreme top sky band.
        if y < 0.05 * image_height:
            score -= 6.0

    # Net top should be above net bottom in image space (y increases downward).
    try:
        if net_image["top_left"]["y"] < net_image["bottom_left"]["y"]:
            score += 4.0
        else:
            score -= 8.0
        if net_image["top_right"]["y"] < net_image["bottom_right"]["y"]:
            score += 4.0
        else:
            score -= 8.0
        # Net should have some vertical extent (pixels), but not cover the whole frame.
        h_l = abs(net_image["bottom_left"]["y"] - net_image["top_left"]["y"])
        h_r = abs(net_image["bottom_right"]["y"] - net_image["top_right"]["y"])
        net_h = max(h_l, h_r)
        if 12 <= net_h <= 0.55 * image_height:
            score += 6.0
        elif net_h > 8:
            score += 2.0
        else:
            score -= 10.0

        # Net width should be meaningful.
        net_w = abs(net_image["top_right"]["x"] - net_image["top_left"]["x"])
        if net_w > 0.08 * image_width:
            score += 4.0
        else:
            score -= 8.0
    except (KeyError, TypeError):
        score -= 20.0

    return score


def choose_api_size(width: int, height: int) -> tuple[int, int]:
    """Pick an OpenAI GPT-Image canvas close to the image aspect."""
    r = width / max(height, 1)
    if r >= 1.25:
        return 1536, 1024
    if r <= 0.8:
        return 1024, 1536
    return 1024, 1024


def downscale_keep_aspect(im: Image.Image, max_side: int) -> Image.Image:
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale >= 0.999:
        return im.copy()
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def letterbox(
    im: Image.Image,
    canvas_w: int,
    canvas_h: int,
    *,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, dict[str, float]]:
    """
    Center `im` on a canvas. Returns (canvas, meta) where meta has
    offset_x/y and content_w/h for mapping dots back.
    """
    w, h = im.size
    scale = min(canvas_w / w, canvas_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), fill)
    ox = (canvas_w - nw) // 2
    oy = (canvas_h - nh) // 2
    canvas.paste(resized, (ox, oy))
    meta = {
        "offset_x": float(ox),
        "offset_y": float(oy),
        "content_w": float(nw),
        "content_h": float(nh),
        "canvas_w": float(canvas_w),
        "canvas_h": float(canvas_h),
        "work_w": float(w),
        "work_h": float(h),
    }
    return canvas, meta


def call_image_edit(
    png_bytes: bytes,
    *,
    prompt: str = LABEL_PROMPT,
    model: str | None = None,
    size: str = "1024x1024",
    api_key: str | None = None,
    quality: str = "low",
) -> tuple[bytes, dict[str, Any]]:
    """POST /v1/images/edits — returns (png_bytes, usage_dict)."""
    import httpx

    key = api_key or os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    model = model or os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")

    files = {
        "image": ("input.png", png_bytes, "image/png"),
    }
    data = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": "1",
    }
    # quality supported on gpt-image-*; low = cheaper for experiments
    if model.startswith("gpt-image"):
        data["quality"] = quality
        # high fidelity helps keep geometry; unsupported on some mini variants
        if "mini" not in model:
            data["input_fidelity"] = "high"

    with httpx.Client(timeout=300.0) as client:
        r = client.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {key}"},
            data=data,
            files=files,
        )
    if r.status_code >= 400:
        # Retry once without optional fidelity/quality if the model rejects them.
        if "input_fidelity" in data or "quality" in data:
            data.pop("input_fidelity", None)
            data.pop("quality", None)
            with httpx.Client(timeout=300.0) as client:
                r = client.post(
                    "https://api.openai.com/v1/images/edits",
                    headers={"Authorization": f"Bearer {key}"},
                    data=data,
                    files=files,
                )
        if r.status_code >= 400:
            raise RuntimeError(f"Image edit error {r.status_code}: {r.text[:2500]}")
    body = r.json()
    b64 = ((body.get("data") or [{}])[0]).get("b64_json")
    if not b64:
        raise RuntimeError(f"No b64_json in image edit response: {str(body)[:500]}")
    usage = body.get("usage") or {}
    return base64.b64decode(b64), usage


def _mask_rgb(
    arr: np.ndarray,
    target: tuple[int, int, int],
    *,
    tol: int = 55,
) -> np.ndarray:
    t = np.array(target, dtype=np.int16)
    d = np.abs(arr.astype(np.int16) - t).sum(axis=2)
    return d <= tol


def _centroids_from_mask(
    mask: np.ndarray,
    *,
    min_pixels: int = 12,
) -> list[tuple[float, float, int]]:
    """Connected-component centroids (4-connected). Returns (x, y, area)."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    cents: list[tuple[float, float, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            ys: list[int] = []
            xs: list[int] = []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            area = len(xs)
            if area >= min_pixels:
                cents.append((float(np.mean(xs)), float(np.mean(ys)), area))
    return cents


def order_quad(
    points: list[tuple[float, float]] | list[tuple[float, float, int]],
) -> dict[str, dict[str, float]]:
    """Order 4 points into TL, TR, BR, BL (image coords, y down)."""
    if len(points) < 4:
        raise ValueError(f"Need 4 points, got {len(points)}")
    pts2 = [(float(p[0]), float(p[1])) for p in points[:4]]
    by_y = sorted(pts2, key=lambda p: p[1])
    top = sorted(by_y[:2], key=lambda p: p[0])
    bot = sorted(by_y[2:], key=lambda p: p[0])
    tl, tr = top[0], top[1]
    bl, br = bot[0], bot[1]
    return {
        "top_left": {"x": tl[0], "y": tl[1]},
        "top_right": {"x": tr[0], "y": tr[1]},
        "bottom_right": {"x": br[0], "y": br[1]},
        "bottom_left": {"x": bl[0], "y": bl[1]},
    }


def extract_green_markers(edited: Image.Image) -> dict[str, Any]:
    """Extract the four lime-green court-corner markers from an ImageGen edit."""
    arr = np.asarray(edited.convert("RGB"))
    # Pure lime green (high G, low R/B) — ignore cyan leftovers from older runs.
    court_mask = (
        (arr[:, :, 1] >= 180)
        & (arr[:, :, 0] <= 90)
        & (arr[:, :, 2] <= 90)
    )
    court_c = _centroids_from_mask(court_mask)

    if len(court_c) < 4:
        any_green = (
            (arr[:, :, 1] >= 170)
            & (arr[:, :, 0] <= 110)
            & (arr[:, :, 2] <= 120)
        )
        court_c = _centroids_from_mask(any_green, min_pixels=8)

    raw_count = len(court_c)
    # Prefer the four largest marker blobs (ImageGen dots beat tiny noise).
    if len(court_c) > 4:
        court_c = sorted(court_c, key=lambda p: p[2], reverse=True)[:4]

    return {
        "court_raw_count": raw_count,
        "court": order_quad(court_c) if len(court_c) >= 4 else None,
    }


def map_point_to_original(
    x: float,
    y: float,
    *,
    letterbox_meta: dict[str, float],
    orig_w: int,
    orig_h: int,
) -> dict[str, float]:
    """Canvas → work (downscaled) → original pixels."""
    ox = letterbox_meta["offset_x"]
    oy = letterbox_meta["offset_y"]
    cw = letterbox_meta["content_w"]
    ch = letterbox_meta["content_h"]
    ww = letterbox_meta["work_w"]
    wh = letterbox_meta["work_h"]
    # canvas → content/work
    wx = (x - ox) * (ww / cw)
    wy = (y - oy) * (wh / ch)
    # work → original
    sx = orig_w / ww
    sy = orig_h / wh
    return {"x": round(wx * sx, 1), "y": round(wy * sy, 1)}


def map_outline_to_original(
    markers: dict[str, Any],
    *,
    letterbox_meta: dict[str, float],
    orig_w: int,
    orig_h: int,
) -> dict[str, Any]:
    court = markers.get("court")
    return {
        "image": {"width": orig_w, "height": orig_h},
        "court": (
            {
                name: map_point_to_original(
                    court[name]["x"],
                    court[name]["y"],
                    letterbox_meta=letterbox_meta,
                    orig_w=orig_w,
                    orig_h=orig_h,
                )
                for name in CORNER_ORDER
            }
            if court
            else None
        ),
    }


def _build_geometry_for_mapping(
    court_image: dict[str, dict[str, float]],
    meters: dict[str, dict[str, float]],
    *,
    image_width: int,
    image_height: int,
    length_m: float,
    width_m: float,
    net_height_m: float,
    mapping_name: str,
) -> dict[str, Any]:
    image_pts = [court_image[n] for n in CORNER_ORDER]
    court_pts = [meters[n] for n in CORNER_ORDER]
    H = compute_homography(image_pts, court_pts)
    H_c2i = invert_homography(H)
    camera = estimate_camera_from_H(H, image_width, image_height, length_m, width_m)

    mid = length_m / 2.0
    # Net plane: left/right follow +Y = camera-left for endline_view; for
    # sideline_view, +Y is far — still use Y=W as "left" label consistently
    # with CORNER_ORDER (image left ≈ higher Y in endline_view).
    net_world = {
        "top_left": (mid, width_m, net_height_m),
        "top_right": (mid, 0.0, net_height_m),
        "bottom_right": (mid, 0.0, 0.0),
        "bottom_left": (mid, width_m, 0.0),
    }
    net_image: dict[str, dict[str, float]] = {}
    for name, (X, Y, Z) in net_world.items():
        proj = project_world_to_image(camera, X, Y, Z)
        if proj is None:
            g = apply_homography(H_c2i, {"x": X, "y": Y})
            if Z > 0:
                g = {"x": g["x"], "y": g["y"] - abs(Z) * 40.0}
            proj = g
        net_image[name] = {"x": round(proj["x"], 1), "y": round(proj["y"], 1)}

    def pt(X: float, Y: float) -> list[float]:
        p = apply_homography(H_c2i, {"x": X, "y": Y})
        return [round(p["x"], 1), round(p["y"], 1)]

    def line_xy(a: tuple[float, float], b: tuple[float, float]) -> list[list[float]]:
        return [pt(a[0], a[1]), pt(b[0], b[1])]

    attack_a = length_m / 3.0
    attack_b = 2.0 * length_m / 3.0
    ground_lines = {
        "boundary": [
            pt(0.0, 0.0),
            pt(length_m, 0.0),
            pt(length_m, width_m),
            pt(0.0, width_m),
        ],
        "center": line_xy((mid, 0.0), (mid, width_m)),
        "attack_a": line_xy((attack_a, 0.0), (attack_a, width_m)),
        "attack_b": line_xy((attack_b, 0.0), (attack_b, width_m)),
        "sidelines": [
            line_xy((0.0, 0.0), (length_m, 0.0)),
            line_xy((0.0, width_m), (length_m, width_m)),
        ],
        "endlines": [
            line_xy((0.0, 0.0), (0.0, width_m)),
            line_xy((length_m, 0.0), (length_m, width_m)),
        ],
    }
    reprojected = {
        name: {
            "x": round(apply_homography(H_c2i, meters[name])["x"], 1),
            "y": round(apply_homography(H_c2i, meters[name])["y"], 1),
        }
        for name in CORNER_ORDER
    }
    score = _score_geometry(
        camera,
        ground_lines,
        net_image,
        image_width=image_width,
        image_height=image_height,
    )
    return {
        "mapping": mapping_name,
        "score": score,
        "H_image_to_court": H,
        "H_court_to_image": H_c2i,
        "camera": camera,
        "court_meters": meters,
        "court_reprojected": reprojected,
        "net": net_image,
        "net_world_m": {
            k: {"x": v[0], "y": v[1], "z": v[2]} for k, v in net_world.items()
        },
        "ground_lines": ground_lines,
        "court": {
            "length_m": length_m,
            "width_m": width_m,
            "net_height_m": net_height_m,
        },
    }


def derive_court_and_net(
    court_image: dict[str, dict[str, float]],
    *,
    image_width: int,
    image_height: int,
    length_m: float = DEFAULT_LENGTH_M,
    width_m: float = DEFAULT_WIDTH_M,
    net_height_m: float = DEFAULT_NET_HEIGHT_M,
) -> dict[str, Any]:
    """
    From four labeled court corners (image), build H + camera and derive net.

    Tries endline-view and sideline-view meter assignments and keeps the
    higher-scoring pose (sane camera height, net upright, court on-frame).
    """
    candidates: list[dict[str, Any]] = []
    for name, meters in _meter_mappings(length_m, width_m).items():
        try:
            candidates.append(
                _build_geometry_for_mapping(
                    court_image,
                    meters,
                    image_width=image_width,
                    image_height=image_height,
                    length_m=length_m,
                    width_m=width_m,
                    net_height_m=net_height_m,
                    mapping_name=name,
                )
            )
        except Exception as e:  # noqa: BLE001
            candidates.append(
                {
                    "mapping": name,
                    "score": -1e9,
                    "error": str(e),
                }
            )

    valid = [c for c in candidates if "H_image_to_court" in c]
    if not valid:
        raise RuntimeError(
            "No valid court→net geometry; "
            + ", ".join(
                f"{c.get('mapping')}: {c.get('error', 'low score')}" for c in candidates
            )
        )

    # If ImageGen only marked the near field (top edge low in the frame),
    # prefer half-court meter assignments when scores are close.
    top_y = (
        court_image["top_left"]["y"] + court_image["top_right"]["y"]
    ) / 2.0
    near_field = top_y > 0.55 * image_height

    def rank(c: dict[str, Any]) -> tuple[float, int]:
        score = float(c.get("score") or -1e9)
        is_half = str(c.get("mapping") or "").startswith("half_")
        # Tie-break: nudge half mappings when labels look near-field-only.
        if near_field and is_half:
            score += 1.5
        return (score, 1 if is_half and near_field else 0)

    best = max(valid, key=rank)
    best["candidates"] = [
        {"mapping": c.get("mapping"), "score": c.get("score"), "error": c.get("error")}
        for c in candidates
    ]
    best["near_field_labels"] = near_field
    return best


def draw_court_geometry_overlay(
    source: Image.Image | Path,
    *,
    court_labeled: dict[str, dict[str, float]] | None,
    geometry: dict[str, Any] | None,
    out_path: Path,
) -> None:
    """Draw ground court + derived net on the original photo."""
    im = (
        source.convert("RGB")
        if isinstance(source, Image.Image)
        else Image.open(source).convert("RGB")
    )
    base = im.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    if geometry and geometry.get("ground_lines"):
        gl = geometry["ground_lines"]
        boundary = [tuple(p) for p in gl["boundary"]]
        if len(boundary) >= 4:
            draw.polygon(boundary, fill=(40, 200, 90, 55))
            draw.line(boundary + [boundary[0]], fill=(0, 255, 80, 230), width=4)

        for key, color, width in (
            ("center", (255, 255, 255, 210), 3),
            ("attack_a", (255, 220, 80, 200), 2),
            ("attack_b", (255, 220, 80, 200), 2),
        ):
            seg = gl.get(key)
            if seg and len(seg) == 2:
                draw.line(
                    [tuple(seg[0]), tuple(seg[1])],
                    fill=color,
                    width=width,
                )

    if geometry and geometry.get("net"):
        net = geometry["net"]
        nxy = [(net[n]["x"], net[n]["y"]) for n in CORNER_ORDER]
        draw.polygon(nxy, fill=(0, 220, 255, 45))
        draw.line(nxy + [nxy[0]], fill=(0, 255, 200, 240), width=3)
        # Vertical posts at antennas
        draw.line(
            [nxy[0], nxy[3]],
            fill=(0, 255, 200, 240),
            width=3,
        )
        draw.line(
            [nxy[1], nxy[2]],
            fill=(0, 255, 200, 240),
            width=3,
        )
        for x, y in nxy:
            r = 5
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 200, 255))

    # Labeled ImageGen corners (verification dots)
    if court_labeled:
        for name in CORNER_ORDER:
            p = court_labeled[name]
            x, y = p["x"], p["y"]
            r = 7
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0, 255))
            draw.text((x + 9, y - 10), f"c_{name}", fill=(0, 255, 0, 255))

    out = Image.alpha_composite(base, layer).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=92)


def recompute_from_outline(
    json_path: Path,
    *,
    length_m: float = DEFAULT_LENGTH_M,
    width_m: float = DEFAULT_WIDTH_M,
    net_height_m: float | None = None,
) -> dict[str, Any]:
    """Re-run FIVB math + overlay from an existing *.03_outline.json (no ImageGen)."""
    load_dotenv()
    net_h = (
        net_height_m
        if net_height_m is not None
        else float(os.environ.get("NET_HEIGHT_M", DEFAULT_NET_HEIGHT_M))
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    court = payload.get("court_labeled") or payload.get("court")
    if not court:
        raise RuntimeError(f"No court_labeled in {json_path}")

    image_info = payload.get("image") or {}
    orig_w = int(image_info.get("width") or 0)
    orig_h = int(image_info.get("height") or 0)
    source = Path(payload.get("source_image") or "")
    artifacts = payload.get("artifacts") or {}
    original_path = Path(artifacts.get("01_original") or "")
    overlay_path = Path(artifacts.get("04_overlay") or json_path.with_name(
        json_path.name.replace(".03_outline.json", ".04_overlay.jpg")
    ))

    if original_path.exists():
        orig = Image.open(original_path).convert("RGB")
    elif source.exists():
        orig = Image.open(source).convert("RGB")
    else:
        raise RuntimeError(f"No original image for {json_path}")
    if not orig_w or not orig_h:
        orig_w, orig_h = orig.size

    geometry = derive_court_and_net(
        court,
        image_width=orig_w,
        image_height=orig_h,
        length_m=length_m,
        width_m=width_m,
        net_height_m=net_h,
    )
    draw_court_geometry_overlay(
        orig,
        court_labeled=court,
        geometry=geometry,
        out_path=overlay_path,
    )

    payload.update(
        {
            "court_labeled": court,
            "net_derived": geometry.get("net"),
            "ground_lines": geometry.get("ground_lines"),
            "H_image_to_court": geometry.get("H_image_to_court"),
            "camera": geometry.get("camera"),
            "court_spec": geometry.get("court"),
            "net_world_m": geometry.get("net_world_m"),
            "mapping": geometry.get("mapping"),
            "mapping_scores": geometry.get("candidates"),
            "geometry_error": None,
            "method": "image_edit_court_markers_plus_fivb_net",
        }
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[green] recomputed {json_path.name}: mapping={geometry.get('mapping')} "
        f"score={geometry.get('score')} overlay={overlay_path}",
        flush=True,
    )
    return payload


def process_image(
    image_path: Path,
    out_dir: Path,
    *,
    max_side: int = 512,
    model: str | None = None,
    quality: str = "low",
    length_m: float = DEFAULT_LENGTH_M,
    width_m: float = DEFAULT_WIDTH_M,
    net_height_m: float | None = None,
) -> dict[str, Any]:
    load_dotenv()
    net_h = (
        net_height_m
        if net_height_m is not None
        else float(os.environ.get("NET_HEIGHT_M", DEFAULT_NET_HEIGHT_M))
    )
    orig = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig.size
    work = downscale_keep_aspect(orig, max_side)
    api_w, api_h = choose_api_size(*work.size)
    canvas, lb = letterbox(work, api_w, api_h)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    # Numbered so folder sort shows the review sequence.
    original_path = out_dir / f"{stem}.01_original.jpg"
    labeled_path = out_dir / f"{stem}.02_labeled.png"
    json_path = out_dir / f"{stem}.03_outline.json"
    overlay_path = out_dir / f"{stem}.04_overlay.jpg"

    # 1) Original unprocessed source (what we start from before labeling).
    orig.save(original_path, quality=95)

    print(
        f"[green] {image_path.name} orig={orig_w}x{orig_h} "
        f"work={work.size[0]}x{work.size[1]} canvas={api_w}x{api_h} "
        f"max_side={max_side}",
        flush=True,
    )
    edited_bytes, usage = call_image_edit(
        png_bytes,
        model=model,
        size=f"{api_w}x{api_h}",
        quality=quality,
    )
    edited = Image.open(io.BytesIO(edited_bytes)).convert("RGB")
    # If API returns unexpected size, letterbox meta still assumes api_w×api_h
    if edited.size != (api_w, api_h):
        edited = edited.resize((api_w, api_h), Image.Resampling.LANCZOS)

    markers_canvas = extract_green_markers(edited)
    outline = map_outline_to_original(
        markers_canvas,
        letterbox_meta=lb,
        orig_w=orig_w,
        orig_h=orig_h,
    )

    # 2) Labeled ImageGen output
    edited.save(labeled_path)

    geometry: dict[str, Any] | None = None
    geom_error: str | None = None
    if outline.get("court"):
        try:
            geometry = derive_court_and_net(
                outline["court"],
                image_width=orig_w,
                image_height=orig_h,
                length_m=length_m,
                width_m=width_m,
                net_height_m=net_h,
            )
        except Exception as e:  # noqa: BLE001
            geom_error = str(e)
            print(f"[green] geometry failed for {stem}: {e}", flush=True)

    # 4) Overlay: ground court + math-derived net on the original
    draw_court_geometry_overlay(
        orig,
        court_labeled=outline.get("court"),
        geometry=geometry,
        out_path=overlay_path,
    )

    # 3) Outline JSON
    payload = {
        "image": outline.get("image"),
        "court_labeled": outline.get("court"),
        "net_derived": geometry.get("net") if geometry else None,
        "ground_lines": geometry.get("ground_lines") if geometry else None,
        "H_image_to_court": geometry.get("H_image_to_court") if geometry else None,
        "camera": geometry.get("camera") if geometry else None,
        "court_spec": geometry.get("court") if geometry else None,
        "net_world_m": geometry.get("net_world_m") if geometry else None,
        "mapping": geometry.get("mapping") if geometry else None,
        "mapping_scores": geometry.get("candidates") if geometry else None,
        "geometry_error": geom_error,
        "artifacts": {
            "01_original": str(original_path),
            "02_labeled": str(labeled_path),
            "03_outline": str(json_path),
            "04_overlay": str(overlay_path),
        },
        "source_image": str(image_path),
        "method": "image_edit_court_markers_plus_fivb_net",
        "model": model or os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2"),
        "max_side": max_side,
        "api_size": f"{api_w}x{api_h}",
        "letterbox": lb,
        "canvas_marker_counts": {
            "court": markers_canvas.get("court_raw_count"),
        },
        "usage": usage,
        "camera_angle_notes": (
            "Homography assumes a planar 18×9 m rectangle. Extreme wide-angle, "
            "strong lens distortion, or off-axis broadcast views can make the "
            "court look wider/taller than metric; deferred fixes include "
            "radial distortion, multi-line DLT, or a few extra anchors (attack "
            "line / net posts)."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[green] artifacts for {stem}:\n"
        f"  1 original  {original_path}\n"
        f"  2 labeled   {labeled_path}\n"
        f"  3 outline   {json_path}\n"
        f"  4 overlay   {overlay_path}\n"
        f"  court={bool(outline.get('court'))} "
        f"net_derived={bool(geometry and geometry.get('net'))} "
        f"mapping={geometry.get('mapping') if geometry else None} "
        f"score={geometry.get('score') if geometry else None} "
        f"usage={usage}",
        flush=True,
    )
    return payload
