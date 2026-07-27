from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
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
from worker.court_calib import calibration_from_keypoints
from worker.paths import (
    latest_run_pointer_path,
    run_dir,
    run_id_from_iso,
    video_dir,
)

ProgressFn = Callable[[str, float], None]
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "0.1.0")


def use_mock_tracks() -> bool:
    """Read at call time (after .env load). Default = real Modal SAM."""
    return os.environ.get("USE_MOCK_TRACKS", "0") == "1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _stamp_run(
    payload: dict[str, Any],
    *,
    run: dict[str, Any],
) -> dict[str, Any]:
    """Attach run metadata to a tracks/ball artifact payload."""
    out = dict(payload)
    out["run"] = run
    return out


def _duration_s(started_at: str, finished_at: str) -> float:
    try:
        a = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        b = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        return max(0.0, round((b - a).total_seconds(), 3))
    except ValueError:
        return 0.0


def _save_court_overlays(
    out_dir: Path,
    overlays: list[dict[str, Any]] | None,
) -> int:
    """Decode Modal overlay JPEGs onto disk; return count written."""
    import base64

    if not overlays:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, ov in enumerate(overlays):
        b64 = ov.get("jpg_b64")
        if not b64:
            continue
        try:
            raw = base64.b64decode(b64)
        except Exception:  # noqa: BLE001
            continue
        t = ov.get("t", i)
        path = out_dir / f"court.overlay_{i:02d}_t{t}.jpg"
        path.write_bytes(raw)
        n += 1
    return n


def _mirror_json(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def run_pipeline(video_id: str, on_progress: ProgressFn) -> dict[str, Any]:
    """
    Full analysis pipeline:
      ingest → normalize → (court ‖ players ‖ ball on Modal) → project_3d → done

    The three Modal AI stages spawn concurrently (separate containers).
    Analysis artifacts are written under:
      videos/{id}/runs/{YYYY-MM-DD_HH-MM-SSZ}/
    plus mirrored copies at the video root for the current/latest set.
    """
    mock = use_mock_tracks()
    started_at = _utc_now_iso()
    run_id = run_id_from_iso(started_at)
    relative_dir = f"runs/{run_id}"

    vdir = video_dir(video_id)
    rdir = run_dir(video_id, run_id)
    rdir.mkdir(parents=True, exist_ok=True)

    source = vdir / "source.mp4"
    work = vdir / "work.mp4"
    thumb = vdir / "thumb.jpg"
    meta_path = vdir / "meta.json"
    cal_path = vdir / "calibration.json"

    # Per-run artifact paths (authoritative history)
    court_path = rdir / "court.keypoints.json"
    tracks_path = rdir / "players.tracks.json"
    ball_path = rdir / "ball.tracks.json"
    court3d_path = rdir / "court3d.json"
    analysis_path = rdir / "analysis.run.json"

    if not source.exists():
        raise FileNotFoundError(f"Missing source.mp4 for video {video_id}")

    print(f"[worker] run_id={run_id} dir={relative_dir}", flush=True)

    on_progress("ingest", 0.05)
    meta = probe(source)

    on_progress("normalize", 0.12)
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
    stored["latest_run_id"] = run_id
    _write_json(meta_path, stored)

    duration_s = float(meta.get("duration_s") or 8.0)
    width = int(meta.get("width") or 1280)
    height = int(meta.get("height") or 720)
    # Keep sampling reasonable for Modal cost/latency; still covers the clip.
    fps = min(float(meta.get("fps") or 10.0), 12.0)

    court_detections = 0
    if mock:
        on_progress("detect_court", 0.22)
        print("[worker] WARNING: USE_MOCK_TRACKS=1 — synthetic court/players/ball")
        court = {
            "video_id": video_id,
            "pipeline_version": PIPELINE_VERSION,
            "source": "mock",
            "model": "mock",
            "keypoint_names": [],
            "skeleton": [],
            "court_points_m": [],
            "image_size": {"width": width, "height": height},
            "frames": [],
            "detections": 0,
        }
        court_model = "mock"
        on_progress("track_players", 0.40)
        tracks = generate_mock_players(
            video_id=video_id,
            pipeline_version=PIPELINE_VERSION,
            duration_s=duration_s,
            width=width,
            height=height,
            fps=fps,
        )
        player_model = "mock"
        players_fps = fps
        on_progress("track_ball", 0.65)
        ball = generate_mock_ball(
            video_id=video_id,
            pipeline_version=PIPELINE_VERSION,
            duration_s=duration_s,
            width=width,
            height=height,
            fps=fps,
        )
        ball_model = "mock"
        ball_infer_mode = None
        ball_model_key = None
    else:
        # Spawn court + SAM + ball together (separate Modal containers).
        on_progress("detect_court", 0.22)
        print(
            "[worker] Modal AI in parallel: detect_court ‖ track_players ‖ track_ball",
            flush=True,
        )
        from worker.modal_bridge import run_modal_ai_parallel

        ai = run_modal_ai_parallel(
            work,
            video_id=video_id,
            fps=fps,
            stages=("court", "players", "ball"),
            court_return_overlays=2,
        )
        on_progress("track_players", 0.55)
        on_progress("track_ball", 0.70)

        court_raw = dict(ai["court"])
        overlays = court_raw.pop("overlays", None)
        n_ov = _save_court_overlays(
            rdir,
            overlays if isinstance(overlays, list) else None,
        )
        if n_ov:
            print(f"[worker] Wrote {n_ov} court overlay preview(s) → {relative_dir}")
        court = court_raw
        court["video_id"] = video_id
        court["pipeline_version"] = PIPELINE_VERSION
        court.setdefault("source", "volley-ref-ai")
        court_detections = int(
            court.get("detections") or len(court.get("frames") or []),
        )
        court["detections"] = court_detections
        court_model = str(
            court.get("model")
            or court.get("source")
            or "yolo_court_keypoints",
        )
        print(f"[worker] Court keypoints: {court_detections} frame hit(s)")

        tracks = dict(ai["players"])
        tracks.setdefault("source", "sam3.1")
        tracks["video_id"] = video_id
        tracks["pipeline_version"] = PIPELINE_VERSION
        player_model = str(tracks.get("source") or "sam3.1")
        if tracks.get("prompt"):
            player_model = f"{player_model} ({tracks.get('prompt')})"
        players_fps = tracks.get("sam_fps")
        if players_fps is None:
            players_fps = float(os.environ.get("SAM3_FPS", "8"))

        ball = dict(ai["ball"])
        ball.setdefault("source", "vballnet")
        ball["video_id"] = video_id
        ball["pipeline_version"] = PIPELINE_VERSION
        ball_model = str(
            ball.get("model")
            or ball.get("source")
            or "vballnet",
        )
        ball_infer_mode = ball.get("infer_mode")
        ball_model_key = ball.get("model_key")

    _write_json(court_path, court)

    cal = _read_json(cal_path)
    # Prefer YOLO court keypoints for H/camera unless user saved a manual cal.
    auto_cal = None
    if (not cal) or cal.get("source") != "manual":
        auto_cal = calibration_from_keypoints(
            court,
            video_id=video_id,
            pipeline_version=PIPELINE_VERSION,
            length_m=float((cal or {}).get("court", {}).get("length_m") or 18),
            width_m=float((cal or {}).get("court", {}).get("width_m") or 9),
            image_width=width,
            image_height=height,
            from_run_id=run_id,
        )
        if auto_cal and isinstance(auto_cal.get("H"), list):
            cal = auto_cal
            _write_json(cal_path, cal)
            print(
                "[worker] Auto-calibration from court keypoints "
                f"(t={cal['keyframes'][0].get('t')}, "
                f"names={cal.get('auto_keypoint_names')})",
                flush=True,
            )
        else:
            print(
                "[worker] Court keypoints present but could not solve H "
                "(need ≥4 landmarks)",
                flush=True,
            )

    projected = False
    if cal and isinstance(cal.get("H"), list) and len(cal["H"]) == 9:
        tracks = project_tracks_with_homography(tracks, cal["H"])
        ball = project_ball_with_homography(ball, cal["H"])
        projected = True
        src = cal.get("source") or "unknown"
        print(f"[worker] Projected players/ball onto court via H (source={src})")
    else:
        print(
            "[worker] No calibration yet — image tracks saved; "
            "manual calibrate in UI if auto keypoints failed",
        )

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
        court3d["court"] = {"length_m": 18, "width_m": 9}
    if cal and isinstance(cal.get("camera"), dict):
        court3d["camera"] = cal["camera"]

    finished_at = _utc_now_iso()
    run_duration_s = _duration_s(started_at, finished_at)
    run_info: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": run_duration_s,
        "relative_dir": relative_dir,
        "pipeline_version": PIPELINE_VERSION,
        "mock": mock,
        "models": {
            "players": player_model,
            "ball": ball_model,
            "court": court_model,
            "players_fps": players_fps,
            "ball_infer_mode": ball_infer_mode,
            "ball_model_key": ball_model_key,
            "court_detections": court_detections,
        },
    }
    tracks = _stamp_run(tracks, run=run_info)
    ball = _stamp_run(ball, run=run_info)
    court = _stamp_run(court, run=run_info)
    court3d["run"] = run_info

    _write_json(tracks_path, tracks)
    _write_json(ball_path, ball)
    _write_json(court_path, court)
    _write_json(court3d_path, court3d)
    _write_json(analysis_path, run_info)

    # Mirror current set at video root for legacy readers / calibration reproject.
    for name in (
        "players.tracks.json",
        "ball.tracks.json",
        "court.keypoints.json",
        "court3d.json",
        "analysis.run.json",
    ):
        _mirror_json(rdir / name, vdir / name)

    pointer = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": run_duration_s,
        "relative_dir": relative_dir,
        "pipeline_version": PIPELINE_VERSION,
    }
    _write_json(latest_run_pointer_path(video_id), pointer)

    print(
        f"[worker] run {run_id} {started_at} → {finished_at} "
        f"({run_duration_s}s) "
        f"court={court_model}({court_detections}) "
        f"players={player_model}@{players_fps}fps ball={ball_model}",
        flush=True,
    )

    on_progress("done", 1.0)
    return {
        "meta": stored["meta"],
        "run_id": run_id,
        "relative_dir": relative_dir,
        "tracks": f"{relative_dir}/players.tracks.json",
        "ball": f"{relative_dir}/ball.tracks.json",
        "court": f"{relative_dir}/court.keypoints.json",
        "court3d": f"{relative_dir}/court3d.json",
        "mock": mock,
        "projected": projected,
        "player_source": tracks.get("source"),
        "ball_source": ball.get("source"),
        "court_source": court.get("source"),
        "player_count": len(tracks.get("players") or []),
        "ball_frames": len(ball.get("frames") or []),
        "court_detections": court_detections,
        "run": run_info,
    }
