from __future__ import annotations

import math
from typing import Any

# SAM track_players historically resampled with ffmpeg max_width=640.
# Full-res is the default now; this constant remains for repairing old tracks.
SAM_MAX_WIDTH = 640


def _sam_resample_size(native_w: int, native_h: int) -> tuple[int, int]:
    """Legacy SAM max-640 size (used only to repair older track files)."""
    native_w = max(1, int(native_w))
    native_h = max(1, int(native_h))
    if native_w <= SAM_MAX_WIDTH:
        return native_w, native_h - (native_h % 2)
    sam_w = SAM_MAX_WIDTH
    sam_h = int(round(native_h * (sam_w / native_w)))
    sam_h = max(2, sam_h - (sam_h % 2))
    return sam_w, sam_h


def _track_bbox_extent(tracks: dict[str, Any]) -> tuple[float, float]:
    max_r = 0.0
    max_b = 0.0
    for p in tracks.get("players") or []:
        for f in p.get("frames") or []:
            bbox = f.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            max_r = max(max_r, float(bbox[0]) + float(bbox[2]))
            max_b = max(max_b, float(bbox[1]) + float(bbox[3]))
            outline = f.get("outline") or []
            for pt in outline:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    max_r = max(max_r, float(pt[0]))
                    max_b = max(max_b, float(pt[1]))
    return max_r, max_b


def scale_player_tracks_to_native(
    tracks: dict[str, Any],
    native_w: int,
    native_h: int,
) -> dict[str, Any]:
    """
    Ensure player bbox/outline are in native work.mp4 pixels.

    No-op when tracks are already native (current Modal default). Still
    upscales legacy SAM max-640 track files before H projection / overlay.
    """
    native_w = max(1, int(native_w))
    native_h = max(1, int(native_h))

    src_w = tracks.get("image_width") or tracks.get("sam_width")
    src_h = tracks.get("image_height") or tracks.get("sam_height")
    if src_w and src_h:
        src_w, src_h = int(src_w), int(src_h)
    else:
        # Already native (or mock): extents cover most of the frame.
        max_r, max_b = _track_bbox_extent(tracks)
        if max_r >= native_w * 0.65 and max_b >= native_h * 0.55:
            out = {**tracks, "image_width": native_w, "image_height": native_h}
            return out
        src_w, src_h = _sam_resample_size(native_w, native_h)

    if src_w <= 0 or src_h <= 0:
        return tracks
    if src_w == native_w and src_h == native_h:
        out = {**tracks, "image_width": native_w, "image_height": native_h}
        return out

    sx = native_w / float(src_w)
    sy = native_h / float(src_h)

    def scale_bbox(bbox: list[float]) -> list[float]:
        return [
            round(float(bbox[0]) * sx, 2),
            round(float(bbox[1]) * sy, 2),
            round(float(bbox[2]) * sx, 2),
            round(float(bbox[3]) * sy, 2),
        ]

    def scale_outline(outline: list[Any] | None) -> list[list[float]] | None:
        if not outline:
            return outline if outline is not None else None
        out_pts: list[list[float]] = []
        for pt in outline:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                out_pts.append([round(float(pt[0]) * sx, 1), round(float(pt[1]) * sy, 1)])
        return out_pts

    players = []
    for p in tracks.get("players") or []:
        frames = []
        for f in p.get("frames") or []:
            nf = {**f}
            if f.get("bbox") and len(f["bbox"]) >= 4:
                nf["bbox"] = scale_bbox(f["bbox"])
            if "outline" in f:
                nf["outline"] = scale_outline(f.get("outline"))
            # court_xy was projected in the wrong space — drop so caller reprojects.
            nf.pop("court_xy", None)
            frames.append(nf)
        players.append({**p, "frames": frames})

    return {
        **tracks,
        "players": players,
        "sam_width": src_w,
        "sam_height": src_h,
        "image_width": native_w,
        "image_height": native_h,
        "coord_space": "native",
    }


def _capsule_outline(x: float, y: float, w: float, h: float, n: int = 24) -> list[list[float]]:
    """Rough body silhouette (stadium / capsule) inside a bbox for mock overlays."""
    cx = x + w / 2
    cy = y + h / 2
    rx = w * 0.42
    ry = h * 0.48
    pts: list[list[float]] = []
    for i in range(n):
        a = (2 * math.pi * i) / n
        # Slightly pinched waist for a body-like look
        waist = 0.85 + 0.15 * abs(math.sin(a))
        px = cx + math.cos(a) * rx * (0.75 if abs(math.sin(a)) < 0.35 else waist)
        py = cy + math.sin(a) * ry
        pts.append([round(px, 1), round(py, 1)])
    return pts


def generate_mock_players(
    *,
    video_id: str,
    pipeline_version: str,
    duration_s: float,
    width: int,
    height: int,
    fps: float = 10.0,
) -> dict[str, Any]:
    """Synthetic players for UI/3D until Modal SAM is wired."""
    duration_s = max(duration_s or 5.0, 1.0)
    width = width or 1280
    height = height or 720
    dt = 1.0 / fps
    n = max(int(duration_s * fps), 5)

    players = []
    configs = [
        (1, 0.25, 0.55, 0.08, 0.04),
        (2, 0.55, 0.50, -0.06, 0.03),
        (3, 0.40, 0.35, 0.05, -0.02),
        (4, 0.70, 0.60, -0.04, -0.03),
    ]
    bw, bh = width * 0.06, height * 0.16

    for track_id, x0, y0, vx, vy in configs:
        frames = []
        for i in range(n):
            t = i * dt
            if t > duration_s:
                break
            cx = (x0 + vx * t) % 0.85
            cy = 0.25 + ((y0 + vy * t) % 0.55)
            x = cx * width - bw / 2
            y = cy * height - bh / 2
            court_x = (cx) * 18.0
            court_y = (cy - 0.2) / 0.6 * 9.0
            court_y = max(0.0, min(9.0, court_y))
            frames.append(
                {
                    "t": round(t, 3),
                    "bbox": [round(x, 1), round(y, 1), round(bw, 1), round(bh, 1)],
                    "outline": _capsule_outline(x, y, bw, bh),
                    "court_xy": [round(court_x, 2), round(court_y, 2)],
                },
            )
        players.append({"track_id": track_id, "frames": frames})

    return {
        "video_id": video_id,
        "pipeline_version": pipeline_version,
        "players": players,
        "source": "mock",
    }


def generate_mock_ball(
    *,
    video_id: str,
    pipeline_version: str,
    duration_s: float,
    width: int,
    height: int,
    fps: float = 10.0,
) -> dict[str, Any]:
    """Parabolic ball path across the net for 2D/3D overlays."""
    duration_s = max(duration_s or 5.0, 1.0)
    width = width or 1280
    height = height or 720
    dt = 1.0 / fps
    n = max(int(duration_s * fps), 5)
    frames = []

    for i in range(n):
        t = i * dt
        if t > duration_s:
            break
        # Court: left → right with bounce-like arcs
        phase = (t / max(duration_s, 1e-3)) * math.pi * 2
        court_x = 2.0 + (t / duration_s) * 14.0
        court_y = 4.5 + math.sin(phase * 0.7) * 1.8
        z = abs(math.sin(phase)) * 3.2 + 0.15
        # Rough image projection (refined after calibration via reproject)
        ix = (court_x / 18.0) * width * 0.7 + width * 0.15
        iy = height * (0.75 - z * 0.08 - (court_y / 9.0) * 0.25)
        frames.append(
            {
                "t": round(t, 3),
                "xy": [round(ix, 1), round(iy, 1)],
                "r": 8.0,
                "court_xyz": [round(court_x, 2), round(court_y, 2), round(z, 2)],
            },
        )

    return {
        "video_id": video_id,
        "pipeline_version": pipeline_version,
        "frames": frames,
        "source": "mock",
    }


def project_tracks_with_homography(
    tracks: dict[str, Any],
    H: list[float],
) -> dict[str, Any]:
    """Apply 3x3 row-major H (image→court) to bbox foot points."""
    if len(H) != 9:
        return tracks

    def apply(x: float, y: float) -> tuple[float, float]:
        denom = H[6] * x + H[7] * y + H[8]
        if abs(denom) < 1e-9:
            return x, y
        return (H[0] * x + H[1] * y + H[2]) / denom, (
            H[3] * x + H[4] * y + H[5]
        ) / denom

    out_players = []
    for p in tracks.get("players", []):
        frames = []
        for f in p.get("frames", []):
            bbox = f["bbox"]
            fx = bbox[0] + bbox[2] / 2
            fy = bbox[1] + bbox[3]
            cx, cy = apply(fx, fy)
            frames.append({**f, "court_xy": [round(cx, 2), round(cy, 2)]})
        out_players.append({**p, "frames": frames})
    return {**tracks, "players": out_players}


def project_ball_with_homography(
    ball: dict[str, Any],
    H: list[float],
) -> dict[str, Any]:
    """Project ball image xy → court; keep z from track or default."""
    if len(H) != 9:
        return ball

    def apply(x: float, y: float) -> tuple[float, float]:
        denom = H[6] * x + H[7] * y + H[8]
        if abs(denom) < 1e-9:
            return x, y
        return (H[0] * x + H[1] * y + H[2]) / denom, (
            H[3] * x + H[4] * y + H[5]
        ) / denom

    frames = []
    for f in ball.get("frames", []):
        xy = f.get("xy")
        z = (f.get("court_xyz") or [0, 0, 1.5])[2]
        if not xy:
            frames.append(f)
            continue
        cx, cy = apply(float(xy[0]), float(xy[1]))
        frames.append(
            {
                **f,
                "court_xyz": [round(cx, 2), round(cy, 2), round(float(z), 2)],
            },
        )
    return {**ball, "frames": frames}


def build_court3d(
    *,
    video_id: str,
    pipeline_version: str,
    tracks: dict[str, Any],
    ball: dict[str, Any] | None = None,
    sample_hz: float = 10.0,
) -> dict[str, Any]:
    """Sample player + ball court positions for the 3D viewer."""
    samples: list[dict[str, Any]] = []
    players = tracks.get("players") or []
    ball_frames = (ball or {}).get("frames") or []

    times: set[float] = set()
    for p in players:
        for f in p.get("frames", []):
            times.add(float(f["t"]))
    for f in ball_frames:
        times.add(float(f["t"]))

    ordered = sorted(times)
    if not ordered:
        return {
            "video_id": video_id,
            "pipeline_version": pipeline_version,
            "court": {"length_m": 18, "width_m": 9},
            "samples": [],
        }

    step = max(1, int(round((1.0 / sample_hz) / max(ordered[1] - ordered[0], 1e-3))))
    for t in ordered[::step]:
        markers = []
        for p in players:
            best = min(p["frames"], key=lambda f: abs(float(f["t"]) - t))
            if abs(float(best["t"]) - t) > 0.25:
                continue
            xy = best.get("court_xy")
            if not xy:
                continue
            markers.append(
                {
                    "track_id": p["track_id"],
                    "x": xy[0],
                    "y": xy[1],
                    "z": 0.0,
                },
            )

        ball_xyz = None
        if ball_frames:
            best_b = min(ball_frames, key=lambda f: abs(float(f["t"]) - t))
            if abs(float(best_b["t"]) - t) <= 0.25:
                xyz = best_b.get("court_xyz")
                if xyz and len(xyz) >= 3:
                    ball_xyz = {
                        "x": xyz[0],
                        "y": xyz[1],
                        "z": xyz[2],
                    }

        samples.append({"t": t, "players": markers, "ball": ball_xyz})

    return {
        "video_id": video_id,
        "pipeline_version": pipeline_version,
        "court": {"length_m": 18, "width_m": 9},
        "samples": samples,
    }
