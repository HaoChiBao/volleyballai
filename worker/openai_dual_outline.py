"""
Dual ImageGen outline (court + net) with FIVB ratio-locked court.

Idea:
  - Label BOTH net (cyan) and court (lime) corners.
  - Net is the metric ruler: antennas span court width W=9 m.
  - Court size is NOT free — it is locked by FIVB ratios:
      length = 2 × net_width,  each half = 9×9 m,  net on center line.
  - Off-screen court corners from ImageGen are unreliable; replace them with
    ratio-derived projections from the net pose.
  - On-screen court labels only steer chirality / which end is "near".
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from worker.court_calib import project_world_to_image
from worker.fivb_ratios import DEFAULT_FIVB, FivbIndoor
from worker.openai_court_outline import CORNER_ORDER, load_dotenv
from worker.openai_green_markers import (
    _centroids_from_mask,
    call_image_edit,
    choose_api_size,
    downscale_keep_aspect,
    letterbox,
    map_point_to_original,
    order_quad,
)
from worker.openai_net_to_court import (
    _build_ground_from_pose,
    _pose_from_pnp,
)

LABEL_PROMPT = """Edit this volleyball photo. Do NOT restyle, redraw, or change the scene.
Keep the original photo pixels everywhere except the markers you add.

Add ONLY bright filled circular markers (no text, no connecting lines):

1) COURT floor corners — pure lime green RGB(0,255,0), diameter ~28–40px:
   - top_left / top_right: far endline corners (farther from camera)
   - bottom_right / bottom_left: near endline corners (closer to camera)
   Left/right from the camera. These are the four floor intersections of the
   two endlines with the two sidelines (the 18×9 m playing rectangle).
   If a corner is off-screen, clamp the marker just inside the image edge
   along that corner's direction (do not invent markers far outside the frame).

2) NET rectangle corners — pure cyan RGB(0,255,180), diameter ~24–36px:
   - top_left / top_right: top tape at left/right antenna
   - bottom_right / bottom_left: bottom tape at right/left antenna
   Infer obscured net corners from posts, antennas, and visible tape.

Use ONLY those two colors. No labels, no lines, no extra dots.
The net spans the full court WIDTH; court LENGTH is twice the net width.
"""


def extract_dual_markers(edited: Image.Image) -> dict[str, Any]:
    """Extract lime court + cyan net marker quads from an ImageGen edit."""
    arr = np.asarray(edited.convert("RGB"))
    court_mask = (
        (arr[:, :, 1] >= 180)
        & (arr[:, :, 0] <= 90)
        & (arr[:, :, 2] <= 90)
    )
    net_mask = (
        (arr[:, :, 1] >= 170)
        & (arr[:, :, 2] >= 120)
        & (arr[:, :, 0] <= 100)
        & (arr[:, :, 2] > arr[:, :, 0] + 20)
    )
    net_mask = net_mask & ~court_mask

    court_c = _centroids_from_mask(court_mask)
    net_c = _centroids_from_mask(net_mask)

    if len(court_c) < 4 or len(net_c) < 4:
        any_green = (
            (arr[:, :, 1] >= 170)
            & (arr[:, :, 0] <= 110)
            & (arr[:, :, 2] <= 200)
        )
        all_c = _centroids_from_mask(any_green, min_pixels=8)
        if len(court_c) < 4 and len(all_c) >= 4:
            court_c = sorted(all_c, key=lambda p: p[2], reverse=True)[:4]
        if len(net_c) < 4 and len(all_c) >= 4:
            # Prefer higher blobs for net when color split fails.
            net_c = sorted(all_c, key=lambda p: p[1])[:4]

    if len(court_c) > 4:
        court_c = sorted(court_c, key=lambda p: p[2], reverse=True)[:4]
    if len(net_c) > 4:
        net_c = sorted(net_c, key=lambda p: p[2], reverse=True)[:4]

    return {
        "court_raw_count": len(court_c),
        "net_raw_count": len(net_c),
        "court": order_quad(court_c) if len(court_c) >= 4 else None,
        "net": order_quad(net_c) if len(net_c) >= 4 else None,
    }


def _map_group(
    group: dict[str, dict[str, float]] | None,
    *,
    letterbox_meta: dict[str, float],
    orig_w: int,
    orig_h: int,
) -> dict[str, dict[str, float]] | None:
    if not group:
        return None
    return {
        name: map_point_to_original(
            group[name]["x"],
            group[name]["y"],
            letterbox_meta=letterbox_meta,
            orig_w=orig_w,
            orig_h=orig_h,
        )
        for name in CORNER_ORDER
    }


def _edge_clamped(
    p: dict[str, float],
    *,
    image_width: int,
    image_height: int,
    margin_px: float = 18.0,
) -> bool:
    """True when a marker sits on the frame edge (likely off-screen approx)."""
    return (
        p["x"] <= margin_px
        or p["y"] <= margin_px
        or p["x"] >= image_width - margin_px
        or p["y"] >= image_height - margin_px
    )


def _net_world(
    fivb: FivbIndoor,
    *,
    y_left: float,
    floor_bottom: bool,
) -> dict[str, tuple[float, float, float]]:
    mid = fivb.mid_m
    y_right = 0.0 if y_left == fivb.width_m else fivb.width_m
    top_z = fivb.net_height_m
    bot_z = 0.0 if floor_bottom else max(0.05, fivb.net_height_m - fivb.net_depth_m)
    return {
        "top_left": (mid, y_left, top_z),
        "top_right": (mid, y_right, top_z),
        "bottom_right": (mid, y_right, bot_z),
        "bottom_left": (mid, y_left, bot_z),
    }


def _court_agreement(
    derived_boundary: list[list[float]],
    court_labeled: dict[str, dict[str, float]] | None,
    *,
    image_width: int,
    image_height: int,
) -> float:
    """
    Score how well FIVB-derived court corners match on-screen ImageGen court
    labels. Edge-clamped labels are ignored (off-screen guesses).
    """
    if not court_labeled:
        return 0.0
    # derived_boundary order: (0,0), (L,0), (L,W), (0,W)
    # Image CORNER_ORDER is camera-relative; compare each labeled point to the
    # nearest derived corner (rotation/chirality already chosen upstream).
    derived = [tuple(p) for p in derived_boundary]
    score = 0.0
    used = 0
    for name in CORNER_ORDER:
        p = court_labeled[name]
        if _edge_clamped(p, image_width=image_width, image_height=image_height):
            continue
        dmin = min(
            ((p["x"] - dx) ** 2 + (p["y"] - dy) ** 2) ** 0.5 for dx, dy in derived
        )
        score += max(0.0, 40.0 - dmin)
        used += 1
    if used == 0:
        return 0.0
    return score / used


def _near_side_bonus(
    ground_lines: dict[str, Any],
    court_labeled: dict[str, dict[str, float]] | None,
) -> float:
    """Prefer the chirality where labeled near (bottom) corners sit near near-endline."""
    if not court_labeled or not ground_lines.get("boundary"):
        return 0.0
    b = ground_lines["boundary"]
    # Near endline candidates: higher average image-y among opposite edges.
    # boundary = [(0,0),(L,0),(L,W),(0,W)]
    edge_x0 = (b[0], b[3])  # X=0
    edge_xL = (b[1], b[2])  # X=L
    y0 = (edge_x0[0][1] + edge_x0[1][1]) / 2.0
    yL = (edge_xL[0][1] + edge_xL[1][1]) / 2.0
    bot = court_labeled["bottom_left"]["y"] + court_labeled["bottom_right"]["y"]
    top = court_labeled["top_left"]["y"] + court_labeled["top_right"]["y"]
    # Bottom labels should be near the higher-y endline (closer to camera).
    if yL >= y0:
        near_y, far_y = yL, y0
    else:
        near_y, far_y = y0, yL
    bot_mid = bot / 2.0
    top_mid = top / 2.0
    score = 0.0
    score += max(0.0, 25.0 - abs(bot_mid - near_y) * 0.05)
    score += max(0.0, 25.0 - abs(top_mid - far_y) * 0.05)
    return score


def derive_ratio_locked_court(
    net_labeled: dict[str, dict[str, float]],
    court_labeled: dict[str, dict[str, float]] | None,
    *,
    image_width: int,
    image_height: int,
    fivb: FivbIndoor = DEFAULT_FIVB,
) -> dict[str, Any]:
    """
    Net → metric pose; court corners = FIVB projection (ratio-locked).
    Court labels only choose orientation and soft-score agreement.
    """
    fx0 = image_width * 0.95
    fx_values = [fx0 * s for s in (0.7, 0.85, 1.0, 1.2, 1.45, 1.75)]
    candidates: list[dict[str, Any]] = []

    for y_left, yname in ((fivb.width_m, "yleft_W"), (0.0, "yleft_0")):
        for floor_bottom, fname in ((False, "tape"), (True, "floor")):
            net_world = _net_world(fivb, y_left=y_left, floor_bottom=floor_bottom)
            for fx in fx_values:
                tag = f"{yname}/{fname}/fx={fx:.0f}"
                try:
                    pose, err = _pose_from_pnp(
                        net_labeled,
                        net_world,
                        image_width=image_width,
                        image_height=image_height,
                        fx=fx,
                    )
                except Exception as e:  # noqa: BLE001
                    candidates.append({"mapping": tag, "score": -1e9, "error": str(e)})
                    continue

                ground = _build_ground_from_pose(
                    pose, length_m=fivb.length_m, width_m=fivb.width_m
                )
                if ground is None:
                    candidates.append(
                        {
                            "mapping": tag,
                            "score": -1e9,
                            "error": "court behind camera",
                        }
                    )
                    continue

                score = 50.0 - min(err, 40.0)
                z = float(pose["position"][2])
                if 2.0 <= z <= 35.0:
                    score += 10.0
                else:
                    score -= 25.0

                # Center line should sit under net bottom.
                center = ground["center"]
                nx = (
                    net_labeled["bottom_left"]["x"] + net_labeled["bottom_right"]["x"]
                ) / 2.0
                ny = (
                    net_labeled["bottom_left"]["y"] + net_labeled["bottom_right"]["y"]
                ) / 2.0
                cx = (center[0][0] + center[1][0]) / 2.0
                cy = (center[0][1] + center[1][1]) / 2.0
                score += max(
                    0.0,
                    20.0
                    - (((cx - nx) ** 2 + (cy - ny) ** 2) ** 0.5)
                    / max(image_width, 1)
                    * 50.0,
                )

                # Full court must extend both sides of the net in image space.
                ys = sorted(p[1] for p in ground["boundary"])
                far_y = (ys[0] + ys[1]) / 2.0
                near_y = (ys[2] + ys[3]) / 2.0
                net_top_y = (
                    net_labeled["top_left"]["y"] + net_labeled["top_right"]["y"]
                ) / 2.0
                if far_y < net_top_y - 20:
                    score += 15.0
                else:
                    score -= 20.0
                if near_y > ny + 15:
                    score += 8.0

                score += _court_agreement(
                    ground["boundary"],
                    court_labeled,
                    image_width=image_width,
                    image_height=image_height,
                )
                score += _near_side_bonus(ground, court_labeled)

                # Final court corners: ratio-locked projection. Replace any
                # edge-clamped ImageGen court labels with these.
                court_final = {
                    "top_left": {
                        "x": ground["boundary"][3][0],
                        "y": ground["boundary"][3][1],
                    },
                    "top_right": {
                        "x": ground["boundary"][2][0],
                        "y": ground["boundary"][2][1],
                    },
                    "bottom_right": {
                        "x": ground["boundary"][1][0],
                        "y": ground["boundary"][1][1],
                    },
                    "bottom_left": {
                        "x": ground["boundary"][0][0],
                        "y": ground["boundary"][0][1],
                    },
                }
                # Remap camera-relative names using labeled near/far if present.
                if court_labeled:
                    # Keep derived geometry; only rename for overlay clarity via
                    # sorting by image y (far = smaller y).
                    pts = [tuple(p) for p in ground["boundary"]]
                    by_y = sorted(pts, key=lambda p: p[1])
                    top = sorted(by_y[:2], key=lambda p: p[0])
                    bot = sorted(by_y[2:], key=lambda p: p[0])
                    court_final = {
                        "top_left": {"x": top[0][0], "y": top[0][1]},
                        "top_right": {"x": top[1][0], "y": top[1][1]},
                        "bottom_right": {"x": bot[1][0], "y": bot[1][1]},
                        "bottom_left": {"x": bot[0][0], "y": bot[0][1]},
                    }

                offscreen = {}
                if court_labeled:
                    offscreen = {
                        n: _edge_clamped(
                            court_labeled[n],
                            image_width=image_width,
                            image_height=image_height,
                        )
                        for n in CORNER_ORDER
                    }

                candidates.append(
                    {
                        "mapping": tag,
                        "score": score,
                        "reproj_err_px": err,
                        "camera": pose,
                        "net_world_m": {
                            k: {"x": v[0], "y": v[1], "z": v[2]}
                            for k, v in net_world.items()
                        },
                        "ground_lines": ground,
                        "court_final": court_final,
                        "court_label_offscreen": offscreen,
                        "fivb_ratios": fivb.ratios(),
                    }
                )

    valid = [c for c in candidates if "camera" in c]
    if not valid:
        raise RuntimeError("No valid ratio-locked net→court solution")
    best = max(valid, key=lambda c: float(c["score"]))
    ranked = sorted(valid, key=lambda c: float(c["score"]), reverse=True)
    best["candidates"] = [
        {
            "mapping": c["mapping"],
            "score": c["score"],
            "reproj_err_px": c.get("reproj_err_px"),
        }
        for c in ranked[:8]
    ]
    return best


def draw_dual_overlay(
    source: Image.Image | Path,
    *,
    net_labeled: dict[str, dict[str, float]] | None,
    court_labeled: dict[str, dict[str, float]] | None,
    geometry: dict[str, Any] | None,
    out_path: Path,
) -> None:
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
            ("center", (255, 255, 255, 220), 3),
            ("attack_a", (255, 220, 80, 200), 2),
            ("attack_b", (255, 220, 80, 200), 2),
        ):
            seg = gl.get(key)
            if seg and len(seg) == 2:
                draw.line([tuple(seg[0]), tuple(seg[1])], fill=color, width=width)

    if net_labeled:
        nxy = [(net_labeled[n]["x"], net_labeled[n]["y"]) for n in CORNER_ORDER]
        draw.polygon(nxy, fill=(0, 220, 255, 50))
        draw.line(nxy + [nxy[0]], fill=(0, 255, 200, 255), width=3)
        for name, (x, y) in zip(CORNER_ORDER, nxy, strict=True):
            r = 6
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 200, 255))
            draw.text((x + 8, y - 10), f"n_{name}", fill=(0, 255, 200, 255))

    # Ghost ImageGen court labels (may be edge-clamped)
    if court_labeled:
        off = (geometry or {}).get("court_label_offscreen") or {}
        for name in CORNER_ORDER:
            p = court_labeled[name]
            x, y = p["x"], p["y"]
            r = 5
            color = (255, 160, 0, 220) if off.get(name) else (0, 255, 0, 200)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
            draw.text((x + 8, y + 4), f"c_{name}", fill=color)

    # Final ratio-locked court corners
    if geometry and geometry.get("court_final"):
        cf = geometry["court_final"]
        for name in CORNER_ORDER:
            p = cf[name]
            x, y = p["x"], p["y"]
            r = 7
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0, 255))
            draw.text((x + 8, y - 12), f"f_{name}", fill=(0, 255, 0, 255))

    out = Image.alpha_composite(base, layer).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=92)


def process_image(
    image_path: Path,
    out_dir: Path,
    *,
    max_side: int = 512,
    model: str | None = None,
    quality: str = "medium",
    fivb: FivbIndoor | None = None,
) -> dict[str, Any]:
    load_dotenv()
    fivb = fivb or DEFAULT_FIVB
    orig = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig.size
    work = downscale_keep_aspect(orig, max_side)
    api_w, api_h = choose_api_size(*work.size)
    canvas, lb = letterbox(work, api_w, api_h)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    original_path = out_dir / f"{stem}.01_original.jpg"
    labeled_path = out_dir / f"{stem}.02_labeled.png"
    json_path = out_dir / f"{stem}.03_outline.json"
    overlay_path = out_dir / f"{stem}.04_overlay.jpg"

    orig.save(original_path, quality=95)
    print(
        f"[fivb] {image_path.name} orig={orig_w}x{orig_h} "
        f"work={work.size[0]}x{work.size[1]} canvas={api_w}x{api_h}",
        flush=True,
    )

    edited_bytes, usage = call_image_edit(
        buf.getvalue(),
        prompt=LABEL_PROMPT,
        model=model,
        size=f"{api_w}x{api_h}",
        quality=quality,
    )
    edited = Image.open(io.BytesIO(edited_bytes)).convert("RGB")
    if edited.size != (api_w, api_h):
        edited = edited.resize((api_w, api_h), Image.Resampling.LANCZOS)

    markers = extract_dual_markers(edited)
    court_labeled = _map_group(
        markers.get("court"), letterbox_meta=lb, orig_w=orig_w, orig_h=orig_h
    )
    net_labeled = _map_group(
        markers.get("net"), letterbox_meta=lb, orig_w=orig_w, orig_h=orig_h
    )
    edited.save(labeled_path)

    geometry: dict[str, Any] | None = None
    geom_error: str | None = None
    if net_labeled:
        try:
            geometry = derive_ratio_locked_court(
                net_labeled,
                court_labeled,
                image_width=orig_w,
                image_height=orig_h,
                fivb=fivb,
            )
        except Exception as e:  # noqa: BLE001
            geom_error = str(e)
            print(f"[fivb] geometry failed for {stem}: {e}", flush=True)
    else:
        geom_error = "net markers not found"

    draw_dual_overlay(
        orig,
        net_labeled=net_labeled,
        court_labeled=court_labeled,
        geometry=geometry,
        out_path=overlay_path,
    )

    payload = {
        "image": {"width": orig_w, "height": orig_h},
        "net_labeled": net_labeled,
        "court_labeled": court_labeled,
        "court_final": geometry.get("court_final") if geometry else None,
        "court_label_offscreen": geometry.get("court_label_offscreen")
        if geometry
        else None,
        "ground_lines": geometry.get("ground_lines") if geometry else None,
        "camera": geometry.get("camera") if geometry else None,
        "fivb_ratios": (geometry.get("fivb_ratios") if geometry else fivb.ratios()),
        "mapping": geometry.get("mapping") if geometry else None,
        "mapping_scores": geometry.get("candidates") if geometry else None,
        "reproj_err_px": geometry.get("reproj_err_px") if geometry else None,
        "geometry_error": geom_error,
        "artifacts": {
            "01_original": str(original_path),
            "02_labeled": str(labeled_path),
            "03_outline": str(json_path),
            "04_overlay": str(overlay_path),
        },
        "source_image": str(image_path),
        "method": "dual_outline_fivb_ratio_locked",
        "model": model or os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2"),
        "max_side": max_side,
        "api_size": f"{api_w}x{api_h}",
        "letterbox": lb,
        "canvas_marker_counts": {
            "court": markers.get("court_raw_count"),
            "net": markers.get("net_raw_count"),
        },
        "usage": usage,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[fivb] artifacts for {stem}:\n"
        f"  1 original  {original_path}\n"
        f"  2 labeled   {labeled_path}\n"
        f"  3 outline   {json_path}\n"
        f"  4 overlay   {overlay_path}\n"
        f"  net={bool(net_labeled)} court_lbl={bool(court_labeled)} "
        f"mapping={geometry.get('mapping') if geometry else None} "
        f"reproj={geometry.get('reproj_err_px') if geometry else None}",
        flush=True,
    )
    return payload
