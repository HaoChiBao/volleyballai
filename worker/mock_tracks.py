from __future__ import annotations

from typing import Any


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
            # Court meters if we assume full-frame mapping (refined after calibration in UI/project)
            court_x = (cx) * 18.0
            court_y = (cy - 0.2) / 0.6 * 9.0
            court_y = max(0.0, min(9.0, court_y))
            frames.append(
                {
                    "t": round(t, 3),
                    "bbox": [round(x, 1), round(y, 1), round(bw, 1), round(bh, 1)],
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


def build_court3d(
    *,
    video_id: str,
    pipeline_version: str,
    tracks: dict[str, Any],
    sample_hz: float = 10.0,
) -> dict[str, Any]:
    """Sample player court positions for the 3D viewer."""
    samples: list[dict[str, Any]] = []
    players = tracks.get("players") or []
    if not players:
        return {
            "video_id": video_id,
            "pipeline_version": pipeline_version,
            "court": {"length_m": 18, "width_m": 9},
            "samples": [],
        }

    # Collect all timestamps
    times: set[float] = set()
    for p in players:
        for f in p.get("frames", []):
            times.add(float(f["t"]))
    ordered = sorted(times)
    if not ordered:
        return {
            "video_id": video_id,
            "pipeline_version": pipeline_version,
            "court": {"length_m": 18, "width_m": 9},
            "samples": [],
        }

    # Downsample
    step = max(1, int(round((1.0 / sample_hz) / max(ordered[1] - ordered[0], 1e-3))))
    for t in ordered[::step]:
        markers = []
        for p in players:
            # nearest frame
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
        samples.append({"t": t, "players": markers})

    return {
        "video_id": video_id,
        "pipeline_version": pipeline_version,
        "court": {"length_m": 18, "width_m": 9},
        "samples": samples,
    }
