from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from worker.ffmpeg_util import normalize, probe
from worker.mock_tracks import (
    build_court3d,
    generate_mock_ball,
    generate_mock_players,
    project_ball_with_homography,
    project_tracks_with_homography,
)
from worker.paths import video_dir

ProgressFn = Callable[[str, float], None]
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "0.1.0")


def use_mock_tracks() -> bool:
    """Read at call time (after .env load). Default = real Modal SAM."""
    return os.environ.get("USE_MOCK_TRACKS", "0") == "1"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_pipeline(video_id: str, on_progress: ProgressFn) -> dict[str, Any]:
    """
    Full analysis pipeline:
      ingest → normalize → SAM3 players → ball → project_3d → done

    Court positions use calibration H when present. If missing, tracks are still
    written in image space; saving calibration in the UI reprojects without
    re-running SAM.
    """
    mock = use_mock_tracks()
    vdir = video_dir(video_id)
    source = vdir / "source.mp4"
    work = vdir / "work.mp4"
    thumb = vdir / "thumb.jpg"
    meta_path = vdir / "meta.json"
    cal_path = vdir / "calibration.json"
    tracks_path = vdir / "players.tracks.json"
    ball_path = vdir / "ball.tracks.json"
    court3d_path = vdir / "court3d.json"

    if not source.exists():
        raise FileNotFoundError(f"Missing source.mp4 for video {video_id}")

    on_progress("ingest", 0.05)
    meta = probe(source)

    on_progress("normalize", 0.15)
    norm = normalize(source, work, thumb)
    meta.update({k: v for k, v in norm.items() if v is not None})

    stored = _read_json(meta_path) or {"id": video_id}
    stored["meta"] = {
        **(stored.get("meta") or {}),
        "duration_s": meta.get("duration_s"),
        "fps": meta.get("fps"),
        "width": meta.get("width"),
        "height": meta.get("height"),
    }
    stored["has_work"] = True
    stored["has_thumb"] = True
    _write_json(meta_path, stored)

    duration_s = float(meta.get("duration_s") or 8.0)
    width = int(meta.get("width") or 1280)
    height = int(meta.get("height") or 720)
    # Keep sampling reasonable for Modal cost/latency; still covers the clip.
    fps = min(float(meta.get("fps") or 10.0), 12.0)

    on_progress("track_players", 0.35)
    if mock:
        print("[worker] WARNING: USE_MOCK_TRACKS=1 — synthetic players")
        tracks = generate_mock_players(
            video_id=video_id,
            pipeline_version=PIPELINE_VERSION,
            duration_s=duration_s,
            width=width,
            height=height,
            fps=fps,
        )
    else:
        print("[worker] Modal SAM 3.1 track_players…")
        from worker.modal_bridge import track_players_modal

        tracks = track_players_modal(work, video_id=video_id, fps=fps)
        tracks.setdefault("source", "sam3.1")
        tracks["video_id"] = video_id
        tracks["pipeline_version"] = PIPELINE_VERSION

    on_progress("track_ball", 0.60)
    if mock:
        print("[worker] WARNING: USE_MOCK_TRACKS=1 — synthetic ball")
        ball = generate_mock_ball(
            video_id=video_id,
            pipeline_version=PIPELINE_VERSION,
            duration_s=duration_s,
            width=width,
            height=height,
            fps=fps,
        )
    else:
        print("[worker] Modal track_ball…")
        from worker.modal_bridge import track_ball_modal

        ball = track_ball_modal(work, video_id=video_id, fps=fps)
        ball.setdefault("source", "modal-motion")
        ball["video_id"] = video_id
        ball["pipeline_version"] = PIPELINE_VERSION

    cal = _read_json(cal_path)
    projected = False
    if cal and isinstance(cal.get("H"), list) and len(cal["H"]) == 9:
        tracks = project_tracks_with_homography(tracks, cal["H"])
        ball = project_ball_with_homography(ball, cal["H"])
        projected = True
        print("[worker] Projected players/ball onto court via calibration H")
    else:
        print(
            "[worker] No calibration yet — image tracks saved; "
            "calibrate corners in UI to map onto the 18×9m court",
        )

    _write_json(tracks_path, tracks)
    _write_json(ball_path, ball)

    on_progress("project_3d", 0.85)
    court3d = build_court3d(
        video_id=video_id,
        pipeline_version=PIPELINE_VERSION,
        tracks=tracks,
        ball=ball,
    )
    if cal and isinstance(cal.get("court"), dict):
        court3d["court"] = cal["court"]
    else:
        # Official FIVB indoor court
        court3d["court"] = {"length_m": 18, "width_m": 9}
    _write_json(court3d_path, court3d)

    on_progress("done", 1.0)
    return {
        "meta": stored["meta"],
        "tracks": tracks_path.name,
        "ball": ball_path.name,
        "court3d": court3d_path.name,
        "mock": mock,
        "projected": projected,
        "player_source": tracks.get("source"),
        "ball_source": ball.get("source"),
        "player_count": len(tracks.get("players") or []),
        "ball_frames": len(ball.get("frames") or []),
    }
