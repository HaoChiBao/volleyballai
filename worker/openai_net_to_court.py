"""
Net-from-ImageGen → court-from-FIVB-math.

1) gpt-image-2 places four lime-green markers on the net rectangle corners.
2) Extract markers, scale back to the original image.
3) Solve camera pose from the known 9×1 m net plane (center line, net height)
   via OpenCV PnP, then project the full 18×9 m ground court.
4) Overlay: labeled net + derived ground court (boundary, attack, center).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from worker.court_calib import project_world_to_image
from worker.openai_court_outline import CORNER_ORDER, load_dotenv
from worker.openai_green_markers import (
    call_image_edit,
    choose_api_size,
    downscale_keep_aspect,
    extract_green_markers,
    letterbox,
    map_point_to_original,
)

DEFAULT_LENGTH_M = 18.0
DEFAULT_WIDTH_M = 9.0
DEFAULT_NET_HEIGHT_M = 2.24
DEFAULT_NET_DEPTH_M = 1.0  # FIVB net vertical extent (top tape → bottom tape)

LABEL_PROMPT = """Edit this volleyball photo. Do NOT restyle, redraw, or change the scene.
Keep the original photo pixels everywhere except the four markers you add.

CRITICAL: Mark the NET rectangle corners only — the four corners of the net's
vertical plane (top tape × antennas/posts, bottom tape × antennas/posts).
The court floor will be computed later from these points.

DO NOT place markers on: court floor corners, endlines, sidelines, attack lines,
players, scoreboard, or logos. NET corners only.

Add exactly four solid filled circles, pure lime green RGB(0,255,0), diameter
about 28–40px. No text, no lines, no other colors, no other overlays.

Corner meanings (camera view of the net plane):
- top_left: top tape at the left antenna/post
- top_right: top tape at the right antenna/post
- bottom_right: bottom tape at the right antenna/post
- bottom_left: bottom tape at the left antenna/post

Left/right are from the camera's view. Infer obscured corners from posts,
antennas, and the visible top/bottom tape. If a corner is barely off-screen,
clamp just inside the image edge.
"""


def _make_K(image_width: int, image_height: int, fx: float) -> np.ndarray:
    fy = fx
    cx = image_width / 2.0
    cy = image_height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _net_world_variants(
    *,
    length_m: float,
    width_m: float,
    net_height_m: float,
    net_depth_m: float,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Known metric net corners for CORNER_ORDER; left/right chirality variants."""
    mid = length_m / 2.0
    top_z = net_height_m
    bot_z = max(0.05, net_height_m - net_depth_m)
    W = width_m
    out: dict[str, dict[str, tuple[float, float, float]]] = {
        # Image-left ↔ court Y=W
        "yleft_W": {
            "top_left": (mid, W, top_z),
            "top_right": (mid, 0.0, top_z),
            "bottom_right": (mid, 0.0, bot_z),
            "bottom_left": (mid, W, bot_z),
        },
        # Image-left ↔ court Y=0
        "yleft_0": {
            "top_left": (mid, 0.0, top_z),
            "top_right": (mid, W, top_z),
            "bottom_right": (mid, W, bot_z),
            "bottom_left": (mid, 0.0, bot_z),
        },
    }
    # Fallback: treat bottom markers as antenna feet on the court (Z=0).
    # Helps when ImageGen marks low on the posts / floor junction.
    if bot_z > 0.05:
        out["yleft_W_floor"] = {
            "top_left": (mid, W, top_z),
            "top_right": (mid, 0.0, top_z),
            "bottom_right": (mid, 0.0, 0.0),
            "bottom_left": (mid, W, 0.0),
        }
        out["yleft_0_floor"] = {
            "top_left": (mid, 0.0, top_z),
            "top_right": (mid, W, top_z),
            "bottom_right": (mid, W, 0.0),
            "bottom_left": (mid, 0.0, 0.0),
        }
    return out


def _camera_from_rt(
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    C = (-R.T @ t).tolist()
    return {
        "position": [float(C[0]), float(C[1]), float(C[2])],
        "R": R.astype(float).reshape(-1).tolist(),
        "t": [float(t[0]), float(t[1]), float(t[2])],
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "image_width": image_width,
        "image_height": image_height,
        "fov_y_deg": float(
            (2 * np.arctan(image_height / (2 * K[1, 1])) * 180) / np.pi
        ),
    }


def _depth_of_point(pose: dict[str, Any], X: float, Y: float, Z: float) -> float:
    R = pose["R"]
    t = pose["t"]
    return float(R[6] * X + R[7] * Y + R[8] * Z + t[2])


def _pose_from_pnp(
    net_image: dict[str, dict[str, float]],
    net_world: dict[str, tuple[float, float, float]],
    *,
    image_width: int,
    image_height: int,
    fx: float,
) -> tuple[dict[str, Any], float]:
    """Return (camera_pose_dict, mean_reproj_error_px) for a given focal length."""
    obj = np.array([net_world[n] for n in CORNER_ORDER], dtype=np.float64)
    img = np.array(
        [[net_image[n]["x"], net_image[n]["y"]] for n in CORNER_ORDER],
        dtype=np.float64,
    )
    K = _make_K(image_width, image_height, fx)
    dist = np.zeros(5, dtype=np.float64)

    best: tuple[dict[str, Any], float] | None = None
    for flags in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flags)
        except cv2.error:
            continue
        if not ok:
            continue
        ok2, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            K,
            dist,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok2:
            continue

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)

        # Try pose and its chirality flip; keep the one looking at court center.
        for R_try, t_try in ((R, t), (-R, -t)):
            pose = _camera_from_rt(
                R_try, t_try, K, image_width=image_width, image_height=image_height
            )
            # Court center and net center must be in front of the camera.
            if _depth_of_point(pose, 9.0, 4.5, 0.0) <= 0.2:
                continue
            if _depth_of_point(pose, 9.0, 4.5, 1.5) <= 0.2:
                continue
            # Prefer cameras above the floor.
            if pose["position"][2] < 0.5:
                continue

            rvec_try, _ = cv2.Rodrigues(R_try)
            proj, _ = cv2.projectPoints(obj, rvec_try, t_try.reshape(3, 1), K, dist)
            proj = proj.reshape(-1, 2)
            err = float(np.mean(np.linalg.norm(proj - img, axis=1)))
            if best is None or err < best[1]:
                best = (pose, err)

    if best is None:
        raise RuntimeError("solvePnP failed for all flags")
    return best


def _score_pose(
    pose: dict[str, Any],
    ground_lines: dict[str, Any],
    net_image: dict[str, dict[str, float]],
    *,
    image_width: int,
    image_height: int,
    reproj_err: float,
) -> float:
    score = 40.0 - min(reproj_err, 40.0)
    z = float(pose["position"][2])
    if 2.0 <= z <= 40.0:
        score += 12.0 - abs(z - 10.0) * 0.2
    else:
        score -= 40.0

    for x, y in ground_lines.get("boundary") or []:
        if -0.3 * image_width <= x <= 1.3 * image_width:
            score += 1.0
        else:
            score -= 6.0
        if -0.3 * image_height <= y <= 1.3 * image_height:
            score += 1.0
        else:
            score -= 6.0

    # Court polygon should have meaningful area in the image.
    b = ground_lines.get("boundary") or []
    if len(b) >= 4:
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        area = abs(
            sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i] for i in range(4))
        ) / 2.0
        frame_area = image_width * image_height
        if 0.02 * frame_area <= area <= 0.95 * frame_area:
            score += 8.0
        else:
            score -= 12.0

    # Center line on the ground should sit under the labeled net (near bottom tape).
    center = ground_lines.get("center")
    if center and len(center) == 2:
        cx = (center[0][0] + center[1][0]) / 2.0
        cy = (center[0][1] + center[1][1]) / 2.0
        nx = (
            net_image["bottom_left"]["x"] + net_image["bottom_right"]["x"]
        ) / 2.0
        ny = (
            net_image["bottom_left"]["y"] + net_image["bottom_right"]["y"]
        ) / 2.0
        # Ground center should be near / slightly below net bottom midline.
        dist = ((cx - nx) ** 2 + (cy - ny) ** 2) ** 0.5
        score += max(0.0, 15.0 - dist / max(image_width, 1) * 40.0)
        if cy + 5 < ny:
            # Center line above the net bottom in image = flipped / floating.
            score -= 20.0

        # Antenna feet (center-line endpoints) should hug bottom net corners.
        # Match either chirality: pick best pairing.
        d_a = (
            (center[0][0] - net_image["bottom_left"]["x"]) ** 2
            + (center[0][1] - net_image["bottom_left"]["y"]) ** 2
        ) ** 0.5 + (
            (center[1][0] - net_image["bottom_right"]["x"]) ** 2
            + (center[1][1] - net_image["bottom_right"]["y"]) ** 2
        ) ** 0.5
        d_b = (
            (center[0][0] - net_image["bottom_right"]["x"]) ** 2
            + (center[0][1] - net_image["bottom_right"]["y"]) ** 2
        ) ** 0.5 + (
            (center[1][0] - net_image["bottom_left"]["x"]) ** 2
            + (center[1][1] - net_image["bottom_left"]["y"]) ** 2
        ) ** 0.5
        foot_dist = min(d_a, d_b)
        score += max(0.0, 20.0 - foot_dist / max(image_width, 1) * 50.0)

    # Prefer courts that keep both endlines in a plausible vertical band.
    b = ground_lines.get("boundary") or []
    if len(b) >= 4:
        ys = sorted(p[1] for p in b)
        far_y = (ys[0] + ys[1]) / 2.0
        near_y = (ys[2] + ys[3]) / 2.0
        if near_y > far_y + 20:
            score += 6.0
        # Near endline shouldn't fall far below the frame; far shouldn't be above top.
        if near_y <= image_height * 1.2:
            score += 4.0
        else:
            score -= 15.0
        if far_y >= -0.05 * image_height:
            score += 3.0

        # Full court must extend past the net on BOTH sides in the image.
        # If the "far" endline sits on the net, the far half collapsed.
        net_top_y = (
            net_image["top_left"]["y"] + net_image["top_right"]["y"]
        ) / 2.0
        net_bot_y = (
            net_image["bottom_left"]["y"] + net_image["bottom_right"]["y"]
        ) / 2.0
        if far_y < net_top_y - 25:
            score += 18.0
        elif far_y < net_bot_y - 10:
            score += 6.0
        else:
            score -= 30.0
        if near_y > net_bot_y + 25:
            score += 10.0
        else:
            score -= 12.0

    return score


def _build_ground_from_pose(
    pose: dict[str, Any],
    *,
    length_m: float,
    width_m: float,
) -> dict[str, Any] | None:
    def proj(X: float, Y: float, Z: float = 0.0) -> list[float] | None:
        p = project_world_to_image(pose, X, Y, Z)
        if p is None:
            return None
        return [round(p["x"], 1), round(p["y"], 1)]

    mid = length_m / 2.0
    attack_a = length_m / 3.0
    attack_b = 2.0 * length_m / 3.0
    corners = [
        proj(0.0, 0.0),
        proj(length_m, 0.0),
        proj(length_m, width_m),
        proj(0.0, width_m),
    ]
    if any(c is None for c in corners):
        return None

    def seg(a: tuple[float, float], b: tuple[float, float]) -> list[list[float]]:
        pa = proj(a[0], a[1])
        pb = proj(b[0], b[1])
        assert pa is not None and pb is not None
        return [pa, pb]

    return {
        "boundary": corners,
        "center": seg((mid, 0.0), (mid, width_m)),
        "attack_a": seg((attack_a, 0.0), (attack_a, width_m)),
        "attack_b": seg((attack_b, 0.0), (attack_b, width_m)),
    }


def derive_court_from_net(
    net_image: dict[str, dict[str, float]],
    *,
    image_width: int,
    image_height: int,
    length_m: float = DEFAULT_LENGTH_M,
    width_m: float = DEFAULT_WIDTH_M,
    net_height_m: float = DEFAULT_NET_HEIGHT_M,
    net_depth_m: float = DEFAULT_NET_DEPTH_M,
) -> dict[str, Any]:
    """PnP from labeled net corners → project full FIVB ground court."""
    # Sweep focal length + net depth. ImageGen often under-marks vertical net
    # extent; forcing a full 1.0 m depth warps the ground plane.
    fx0 = image_width * 0.95
    fx_values = [fx0 * s for s in (0.55, 0.7, 0.85, 1.0, 1.15, 1.35, 1.6, 1.9)]
    depth_values = sorted(
        {net_depth_m, 0.35, 0.5, 0.65, 0.8, 1.0}
    )

    candidates: list[dict[str, Any]] = []

    for depth in depth_values:
        variants = _net_world_variants(
            length_m=length_m,
            width_m=width_m,
            net_height_m=net_height_m,
            net_depth_m=depth,
        )
        for name, net_world in variants.items():
            for fx in fx_values:
                tag = f"{name}/fx={fx:.0f}/d={depth:.2f}"
                try:
                    pose, err = _pose_from_pnp(
                        net_image,
                        net_world,
                        image_width=image_width,
                        image_height=image_height,
                        fx=fx,
                    )
                except Exception as e:  # noqa: BLE001
                    candidates.append({"mapping": tag, "score": -1e9, "error": str(e)})
                    continue

                ground_lines = _build_ground_from_pose(
                    pose, length_m=length_m, width_m=width_m
                )
                if ground_lines is None:
                    candidates.append(
                        {
                            "mapping": tag,
                            "score": -1e9,
                            "error": "court corner behind camera",
                        }
                    )
                    continue

                score = _score_pose(
                    pose,
                    ground_lines,
                    net_image,
                    image_width=image_width,
                    image_height=image_height,
                    reproj_err=err,
                )
                net_reproj = {}
                for n, (X, Y, Z) in net_world.items():
                    p = project_world_to_image(pose, X, Y, Z)
                    if p:
                        net_reproj[n] = {"x": round(p["x"], 1), "y": round(p["y"], 1)}

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
                        "net_reprojected": net_reproj,
                        "ground_lines": ground_lines,
                        "court": {
                            "length_m": length_m,
                            "width_m": width_m,
                            "net_height_m": net_height_m,
                            "net_depth_m": depth,
                        },
                    }
                )

    valid = [c for c in candidates if "camera" in c]
    if not valid:
        raise RuntimeError(
            "No valid net→court pose; "
            + ", ".join(
                f"{c.get('mapping')}: {c.get('error')}"
                for c in candidates
                if c.get("error")
            )[:500]
        )
    best = max(valid, key=lambda c: float(c.get("score") or -1e9))
    # Keep a compact scoreboard (top few + errors).
    ranked = sorted(valid, key=lambda c: float(c.get("score") or -1e9), reverse=True)
    best["candidates"] = [
        {
            "mapping": c.get("mapping"),
            "score": c.get("score"),
            "reproj_err_px": c.get("reproj_err_px"),
        }
        for c in ranked[:8]
    ]
    return best


def draw_net_court_overlay(
    source: Image.Image | Path,
    *,
    net_labeled: dict[str, dict[str, float]] | None,
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

    # Labeled net (source of truth)
    if net_labeled:
        nxy = [(net_labeled[n]["x"], net_labeled[n]["y"]) for n in CORNER_ORDER]
        draw.polygon(nxy, fill=(0, 220, 255, 50))
        draw.line(nxy + [nxy[0]], fill=(0, 255, 200, 255), width=3)
        draw.line([nxy[0], nxy[3]], fill=(0, 255, 200, 255), width=3)
        draw.line([nxy[1], nxy[2]], fill=(0, 255, 200, 255), width=3)
        for name, (x, y) in zip(CORNER_ORDER, nxy, strict=True):
            r = 7
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0, 255))
            draw.text((x + 9, y - 10), f"n_{name}", fill=(0, 255, 0, 255))

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
    length_m: float = DEFAULT_LENGTH_M,
    width_m: float = DEFAULT_WIDTH_M,
    net_height_m: float | None = None,
    net_depth_m: float = DEFAULT_NET_DEPTH_M,
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
    original_path = out_dir / f"{stem}.01_original.jpg"
    labeled_path = out_dir / f"{stem}.02_labeled.png"
    json_path = out_dir / f"{stem}.03_outline.json"
    overlay_path = out_dir / f"{stem}.04_overlay.jpg"

    orig.save(original_path, quality=95)
    print(
        f"[net→court] {image_path.name} orig={orig_w}x{orig_h} "
        f"work={work.size[0]}x{work.size[1]} canvas={api_w}x{api_h}",
        flush=True,
    )

    edited_bytes, usage = call_image_edit(
        png_bytes,
        prompt=LABEL_PROMPT,
        model=model,
        size=f"{api_w}x{api_h}",
        quality=quality,
    )
    edited = Image.open(io.BytesIO(edited_bytes)).convert("RGB")
    if edited.size != (api_w, api_h):
        edited = edited.resize((api_w, api_h), Image.Resampling.LANCZOS)

    markers = extract_green_markers(edited)
    net_canvas = markers.get("court")  # extractor key; these are net blobs
    net_labeled = None
    if net_canvas:
        net_labeled = {
            name: map_point_to_original(
                net_canvas[name]["x"],
                net_canvas[name]["y"],
                letterbox_meta=lb,
                orig_w=orig_w,
                orig_h=orig_h,
            )
            for name in CORNER_ORDER
        }

    edited.save(labeled_path)

    geometry: dict[str, Any] | None = None
    geom_error: str | None = None
    if net_labeled:
        try:
            geometry = derive_court_from_net(
                net_labeled,
                image_width=orig_w,
                image_height=orig_h,
                length_m=length_m,
                width_m=width_m,
                net_height_m=net_h,
                net_depth_m=net_depth_m,
            )
        except Exception as e:  # noqa: BLE001
            geom_error = str(e)
            print(f"[net→court] geometry failed for {stem}: {e}", flush=True)

    draw_net_court_overlay(
        orig,
        net_labeled=net_labeled,
        geometry=geometry,
        out_path=overlay_path,
    )

    payload = {
        "image": {"width": orig_w, "height": orig_h},
        "net_labeled": net_labeled,
        "court_derived": geometry.get("ground_lines") if geometry else None,
        "camera": geometry.get("camera") if geometry else None,
        "net_world_m": geometry.get("net_world_m") if geometry else None,
        "net_reprojected": geometry.get("net_reprojected") if geometry else None,
        "court_spec": geometry.get("court") if geometry else None,
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
        "method": "image_edit_net_markers_plus_fivb_court",
        "model": model or os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2"),
        "max_side": max_side,
        "api_size": f"{api_w}x{api_h}",
        "letterbox": lb,
        "canvas_marker_counts": {"net": markers.get("court_raw_count")},
        "usage": usage,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[net→court] artifacts for {stem}:\n"
        f"  1 original  {original_path}\n"
        f"  2 labeled   {labeled_path}\n"
        f"  3 outline   {json_path}\n"
        f"  4 overlay   {overlay_path}\n"
        f"  net={bool(net_labeled)} court_derived={bool(geometry)} "
        f"mapping={geometry.get('mapping') if geometry else None} "
        f"reproj={geometry.get('reproj_err_px') if geometry else None} "
        f"usage={usage}",
        flush=True,
    )
    return payload


def recompute_from_outline(
    json_path: Path,
    *,
    length_m: float = DEFAULT_LENGTH_M,
    width_m: float = DEFAULT_WIDTH_M,
    net_height_m: float | None = None,
    net_depth_m: float = DEFAULT_NET_DEPTH_M,
) -> dict[str, Any]:
    load_dotenv()
    net_h = (
        net_height_m
        if net_height_m is not None
        else float(os.environ.get("NET_HEIGHT_M", DEFAULT_NET_HEIGHT_M))
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    net = payload.get("net_labeled")
    if not net:
        raise RuntimeError(f"No net_labeled in {json_path}")

    image_info = payload.get("image") or {}
    orig_w = int(image_info.get("width") or 0)
    orig_h = int(image_info.get("height") or 0)
    artifacts = payload.get("artifacts") or {}
    original_path = Path(artifacts.get("01_original") or "")
    source = Path(payload.get("source_image") or "")
    overlay_path = Path(
        artifacts.get("04_overlay")
        or json_path.with_name(json_path.name.replace(".03_outline.json", ".04_overlay.jpg"))
    )
    if original_path.exists():
        orig = Image.open(original_path).convert("RGB")
    elif source.exists():
        orig = Image.open(source).convert("RGB")
    else:
        raise RuntimeError(f"No original image for {json_path}")
    if not orig_w or not orig_h:
        orig_w, orig_h = orig.size

    geometry = derive_court_from_net(
        net,
        image_width=orig_w,
        image_height=orig_h,
        length_m=length_m,
        width_m=width_m,
        net_height_m=net_h,
        net_depth_m=net_depth_m,
    )
    draw_net_court_overlay(
        orig, net_labeled=net, geometry=geometry, out_path=overlay_path
    )
    payload.update(
        {
            "court_derived": geometry.get("ground_lines"),
            "camera": geometry.get("camera"),
            "net_world_m": geometry.get("net_world_m"),
            "net_reprojected": geometry.get("net_reprojected"),
            "court_spec": geometry.get("court"),
            "mapping": geometry.get("mapping"),
            "mapping_scores": geometry.get("candidates"),
            "reproj_err_px": geometry.get("reproj_err_px"),
            "geometry_error": None,
            "method": "image_edit_net_markers_plus_fivb_court",
        }
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[net→court] recomputed {json_path.name}: mapping={geometry.get('mapping')} "
        f"reproj={geometry.get('reproj_err_px')} overlay={overlay_path}",
        flush=True,
    )
    return payload
