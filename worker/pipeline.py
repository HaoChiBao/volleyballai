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
    scale_player_tracks_to_native,
)
from worker.stitch_player_tracks import stitch_players_file
from worker.court_calib import calibration_from_keypoints
from worker.paths import (
    latest_run_pointer_path,
    run_dir,
    run_id_from_iso,
    video_dir,
)

ProgressFn = Callable[[str, float], None]
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "0.1.0")

# Modal AI stage ids (match packages/types PipelineStageTarget).
MODAL_STAGES = ("court", "players", "ball", "ball_yolo", "ball_wasb")
ALL_STAGE_TARGETS = ("normalize",) + MODAL_STAGES

ARTIFACT_NAMES = (
    "players.tracks.json",
    "ball.tracks.json",
    "ball.tracks.yolo.json",
    "ball.tracks.wasb.json",
    "court.keypoints.json",
    "court3d.json",
    "analysis.run.json",
)


def _normalize_stages(
    stages: list[str] | tuple[str, ...] | None,
) -> set[str] | None:
    """None = full pipeline. Otherwise a non-empty set of stage targets."""
    if stages is None:
        return None
    allowed = set(ALL_STAGE_TARGETS)
    cleaned = {str(s).strip() for s in stages if str(s).strip() in allowed}
    if not cleaned:
        raise ValueError(
            f"No valid stages (allowed: {', '.join(ALL_STAGE_TARGETS)})",
        )
    return cleaned


def _load_existing_artifact(vdir: Path, name: str) -> dict[str, Any] | None:
    data = _read_json(vdir / name)
    return data if isinstance(data, dict) else None


def _seed_run_dir_from_latest(vdir: Path, rdir: Path) -> None:
    """Copy current root artifacts into a new run folder as a baseline."""
    for name in ARTIFACT_NAMES:
        src = vdir / name
        if src.exists():
            _mirror_json(src, rdir / name)


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


def run_pipeline(
    video_id: str,
    on_progress: ProgressFn,
    stages: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Analysis pipeline (full or partial).

    stages=None → full run (normalize + all Modal AI + stitch + project_3d).
    stages=["ball_wasb"] → refresh only WASB (reuse work.mp4 + other artifacts).
    """
    mock = use_mock_tracks()
    wanted = _normalize_stages(stages)
    partial = wanted is not None
    modal_wanted = (
        tuple(s for s in MODAL_STAGES if s in (wanted or set()))
        if partial
        else MODAL_STAGES
    )
    force_normalize = (not partial) or ("normalize" in (wanted or set()))

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

    court_path = rdir / "court.keypoints.json"
    tracks_path = rdir / "players.tracks.json"
    ball_path = rdir / "ball.tracks.json"
    ball_yolo_path = rdir / "ball.tracks.yolo.json"
    ball_wasb_path = rdir / "ball.tracks.wasb.json"
    court3d_path = rdir / "court3d.json"
    analysis_path = rdir / "analysis.run.json"

    if not source.exists():
        raise FileNotFoundError(f"Missing source.mp4 for video {video_id}")

    if partial:
        _seed_run_dir_from_latest(vdir, rdir)
        print(
            f"[worker] PARTIAL run stages={sorted(wanted or [])} run_id={run_id}",
            flush=True,
        )
    else:
        print(f"[worker] FULL run run_id={run_id} dir={relative_dir}", flush=True)

    on_progress("ingest", 0.05)
    stored = _read_json(meta_path) or {"id": video_id}
    meta = dict(stored.get("meta") or {})

    need_normalize = force_normalize or (not work.exists())
    if need_normalize:
        meta = probe(source)
        on_progress("normalize", 0.12)
        norm = normalize(source, work, thumb)
        meta.update({k: v for k, v in norm.items() if v is not None})
        stored["has_work"] = True
        stored["has_thumb"] = True
    else:
        on_progress("normalize", 0.12)
        print("[worker] Reusing existing work.mp4 (skip normalize)", flush=True)
        if not meta.get("fps") or not meta.get("width"):
            meta = {**meta, **(probe(work) or {})}

    if partial and not work.exists():
        raise FileNotFoundError(
            "Missing work.mp4 for partial run — queue a full analysis first",
        )

    stored["meta"] = {
        **(stored.get("meta") or {}),
        "duration_s": meta.get("duration_s"),
        "fps": meta.get("fps"),
        "width": meta.get("width"),
        "height": meta.get("height"),
    }
    stored["latest_run_id"] = run_id
    _write_json(meta_path, stored)

    duration_s = float(meta.get("duration_s") or 8.0)
    width = int(meta.get("width") or 1280)
    height = int(meta.get("height") or 720)
    fps = float(meta.get("fps") or 30.0)

    court = _load_existing_artifact(vdir, "court.keypoints.json") or {}
    tracks = _load_existing_artifact(vdir, "players.tracks.json") or {}
    ball = _load_existing_artifact(vdir, "ball.tracks.json") or {}
    ball_yolo = _load_existing_artifact(vdir, "ball.tracks.yolo.json")
    ball_wasb = _load_existing_artifact(vdir, "ball.tracks.wasb.json")

    court_detections = int(
        court.get("detections") or len(court.get("frames") or []),
    )
    court_model = str(court.get("model") or court.get("source") or "cached")
    player_model = str(tracks.get("source") or "cached")
    players_fps = tracks.get("sam_fps")
    if players_fps is None:
        players_fps = float(os.environ.get("SAM3_FPS", "8"))
    ball_model = str(ball.get("model") or ball.get("source") or "cached")
    ball_infer_mode = ball.get("infer_mode")
    ball_model_key = ball.get("model_key")
    ball_yolo_model = (
        str(ball_yolo.get("model") or ball_yolo.get("source"))
        if isinstance(ball_yolo, dict)
        else None
    )
    ball_wasb_model = (
        str(ball_wasb.get("model") or ball_wasb.get("source"))
        if isinstance(ball_wasb, dict)
        else None
    )

    refreshed: set[str] = set()
    if need_normalize:
        refreshed.add("normalize")

    def _mock_ball_base() -> dict[str, Any]:
        if ball.get("frames"):
            return ball
        return generate_mock_ball(
            video_id=video_id,
            pipeline_version=PIPELINE_VERSION,
            duration_s=duration_s,
            width=width,
            height=height,
            fps=fps,
        )

    if mock:
        do_all = not partial
        on_progress("detect_court", 0.22)
        print("[worker] WARNING: USE_MOCK_TRACKS=1 — synthetic court/players/ball")
        if do_all or "court" in (wanted or set()):
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
            court_detections = 0
            refreshed.add("court")
        on_progress("track_players", 0.40)
        if do_all or "players" in (wanted or set()):
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
            refreshed.add("players")
        on_progress("track_ball", 0.65)
        if do_all or "ball" in (wanted or set()):
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
            refreshed.add("ball")
        if do_all or "ball_yolo" in (wanted or set()):
            base = _mock_ball_base()
            ball_yolo = {
                **base,
                "source": "setoptics_yolo",
                "model": "mock_yolo",
                "model_key": "yolo26n",
                "infer_mode": "mock",
                "frames": [
                    {
                        **f,
                        "xy": (
                            [f["xy"][0] + 18.0, f["xy"][1] - 12.0]
                            if f.get("xy")
                            else f.get("xy")
                        ),
                    }
                    for f in (base.get("frames") or [])
                ],
            }
            ball_yolo_model = "mock_yolo"
            refreshed.add("ball_yolo")
        if do_all or "ball_wasb" in (wanted or set()):
            base = _mock_ball_base()
            ball_wasb = {
                **base,
                "source": "wasb_sbdt",
                "model": "mock_wasb",
                "model_key": "wasb_volleyball",
                "infer_mode": "mock",
                "frames": [
                    {
                        **f,
                        "xy": (
                            [f["xy"][0] - 14.0, f["xy"][1] + 10.0]
                            if f.get("xy")
                            else f.get("xy")
                        ),
                    }
                    for f in (base.get("frames") or [])
                ],
            }
            ball_wasb_model = "mock_wasb"
            refreshed.add("ball_wasb")
    elif modal_wanted:
        on_progress("detect_court", 0.22)
        print(
            "[worker] Modal AI in parallel: " + " | ".join(modal_wanted),
            flush=True,
        )
        from worker.modal_bridge import run_modal_ai_parallel

        ai = run_modal_ai_parallel(
            work,
            video_id=video_id,
            fps=fps,
            stages=modal_wanted,
            court_return_overlays=2 if "court" in modal_wanted else 0,
        )
        on_progress("track_players", 0.55)
        on_progress("track_ball", 0.70)

        if "court" in ai:
            court_raw = dict(ai["court"])
            overlays = court_raw.pop("overlays", None)
            n_ov = _save_court_overlays(
                rdir,
                overlays if isinstance(overlays, list) else None,
            )
            if n_ov:
                print(
                    f"[worker] Wrote {n_ov} court overlay preview(s) → {relative_dir}",
                )
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
            refreshed.add("court")

        if "players" in ai:
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
            refreshed.add("players")

        if "ball" in ai:
            ball = dict(ai["ball"])
            ball.setdefault("source", "vballnet")
            ball["video_id"] = video_id
            ball["pipeline_version"] = PIPELINE_VERSION
            ball_model = str(
                ball.get("model") or ball.get("source") or "vballnet",
            )
            ball_infer_mode = ball.get("infer_mode")
            ball_model_key = ball.get("model_key")
            refreshed.add("ball")

        if isinstance(ai.get("ball_yolo"), dict) and "frames" in ai["ball_yolo"]:
            ball_yolo = dict(ai["ball_yolo"])
            ball_yolo.setdefault("source", "setoptics_yolo")
            ball_yolo["video_id"] = video_id
            ball_yolo["pipeline_version"] = PIPELINE_VERSION
            ball_yolo_model = str(
                ball_yolo.get("model")
                or ball_yolo.get("source")
                or "setoptics_yolo",
            )
            print(
                f"[worker] SetOptics YOLO ball: "
                f"{len(ball_yolo.get('frames') or [])} detection(s)",
                flush=True,
            )
            refreshed.add("ball_yolo")
        elif "ball_yolo" in modal_wanted:
            print(
                "[worker] SetOptics YOLO ball tracks unavailable this run",
                flush=True,
            )

        if isinstance(ai.get("ball_wasb"), dict) and "frames" in ai["ball_wasb"]:
            ball_wasb = dict(ai["ball_wasb"])
            ball_wasb.setdefault("source", "wasb_sbdt")
            ball_wasb["video_id"] = video_id
            ball_wasb["pipeline_version"] = PIPELINE_VERSION
            ball_wasb_model = str(
                ball_wasb.get("model")
                or ball_wasb.get("source")
                or "wasb_sbdt",
            )
            print(
                f"[worker] WASB ball: "
                f"{len(ball_wasb.get('frames') or [])} detection(s)",
                flush=True,
            )
            refreshed.add("ball_wasb")
        elif "ball_wasb" in modal_wanted:
            print(
                "[worker] WASB ball tracks unavailable this run",
                flush=True,
            )
    else:
        on_progress("detect_court", 0.22)
        on_progress("track_players", 0.55)
        on_progress("track_ball", 0.70)
        print("[worker] No Modal stages requested — postprocess only", flush=True)

    if tracks.get("players") and ("players" in refreshed or not partial):
        before_img = (tracks.get("image_width"), tracks.get("image_height"))
        tracks = scale_player_tracks_to_native(tracks, width, height)
        if (
            (tracks.get("image_width"), tracks.get("image_height")) != before_img
            or tracks.get("sam_width")
        ):
            print(
                f"[worker] Player coords → native {width}x{height} "
                f"(from {tracks.get('sam_width')}x{tracks.get('sam_height')})",
                flush=True,
            )

    if court and ("court" in refreshed or not partial):
        _write_json(court_path, court)

    cal = _read_json(cal_path)
    if (
        court.get("frames")
        and ((not cal) or cal.get("source") != "manual")
        and ("court" in refreshed or not partial)
    ):
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
        if tracks.get("players") and ("players" in refreshed or not partial):
            tracks = project_tracks_with_homography(tracks, cal["H"])
        if ball.get("frames") and ("ball" in refreshed or not partial):
            ball = project_ball_with_homography(ball, cal["H"])
        if ball_yolo is not None and ("ball_yolo" in refreshed or not partial):
            ball_yolo = project_ball_with_homography(ball_yolo, cal["H"])
        if ball_wasb is not None and ("ball_wasb" in refreshed or not partial):
            ball_wasb = project_ball_with_homography(ball_wasb, cal["H"])
        projected = True
        src = cal.get("source") or "unknown"
        print(f"[worker] Projected players/ball onto court via H (source={src})")
    else:
        print(
            "[worker] No calibration yet — image tracks saved; "
            "manual calibrate in UI if auto keypoints failed",
        )

    if tracks.get("players") and ("players" in refreshed or not partial):
        before_n = len(tracks.get("players") or [])
        tracks = stitch_players_file(tracks)
        st = tracks.get("stitch") or {}
        print(
            f"[worker] Stitched player tracks: {before_n} -> {st.get('tracks_after')} "
            f"(merges={st.get('merges')}, dwell_merges={st.get('dwell_merges')}, "
            f"dropped_short={st.get('dropped_short')}, "
            f"dropped_off_court={st.get('dropped_off_court')})",
            flush=True,
        )

    on_progress("project_3d", 0.85)
    rebuild_3d = (not partial) or bool(
        refreshed & {"players", "ball", "court", "ball_yolo", "ball_wasb"},
    )
    if rebuild_3d and (tracks.get("players") or ball.get("frames")):
        court3d = build_court3d(
            video_id=video_id,
            pipeline_version=PIPELINE_VERSION,
            tracks=tracks if tracks.get("players") else {"players": []},
            ball=ball if ball.get("frames") else {"frames": []},
        )
        if cal and isinstance(cal.get("court"), dict):
            court3d["court"] = cal["court"]
        else:
            court3d["court"] = {"length_m": 18, "width_m": 9}
        if cal and isinstance(cal.get("camera"), dict):
            court3d["camera"] = cal["camera"]
    else:
        court3d = _load_existing_artifact(vdir, "court3d.json") or {
            "video_id": video_id,
            "pipeline_version": PIPELINE_VERSION,
            "court": {"length_m": 18, "width_m": 9},
            "samples": [],
        }

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
        "stages": sorted(wanted) if partial else None,
        "models": {
            "players": player_model,
            "ball": ball_model,
            "ball_yolo": ball_yolo_model,
            "ball_wasb": ball_wasb_model,
            "court": court_model,
            "players_fps": players_fps,
            "ball_infer_mode": ball_infer_mode,
            "ball_model_key": ball_model_key,
            "court_detections": court_detections,
        },
    }

    # Always write the working set into this run folder (seeded + refreshed).
    if tracks.get("players"):
        tracks = _stamp_run(tracks, run=run_info)
        _write_json(tracks_path, tracks)
    if ball.get("frames"):
        ball = _stamp_run(ball, run=run_info)
        _write_json(ball_path, ball)
    if ball_yolo is not None:
        ball_yolo = _stamp_run(ball_yolo, run=run_info)
        _write_json(ball_yolo_path, ball_yolo)
    if ball_wasb is not None:
        ball_wasb = _stamp_run(ball_wasb, run=run_info)
        _write_json(ball_wasb_path, ball_wasb)
    if court:
        court = _stamp_run(court, run=run_info)
        _write_json(court_path, court)

    court3d["run"] = run_info
    _write_json(court3d_path, court3d)
    _write_json(analysis_path, run_info)

    for name in ARTIFACT_NAMES:
        src = rdir / name
        if src.exists():
            _mirror_json(src, vdir / name)

    pointer = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": run_duration_s,
        "relative_dir": relative_dir,
        "pipeline_version": PIPELINE_VERSION,
        "stages": sorted(wanted) if partial else None,
    }
    _write_json(latest_run_pointer_path(video_id), pointer)

    print(
        f"[worker] run {run_id} {started_at} → {finished_at} "
        f"({run_duration_s}s) refreshed={sorted(refreshed) or ['(none)']} "
        f"court={court_model}({court_detections}) "
        f"players={player_model}@{players_fps}fps ball={ball_model}"
        + (f" ball_yolo={ball_yolo_model}" if ball_yolo_model else "")
        + (f" ball_wasb={ball_wasb_model}" if ball_wasb_model else ""),
        flush=True,
    )

    on_progress("done", 1.0)
    return {
        "meta": stored["meta"],
        "run_id": run_id,
        "relative_dir": relative_dir,
        "tracks": f"{relative_dir}/players.tracks.json",
        "ball": f"{relative_dir}/ball.tracks.json",
        "ball_yolo": (
            f"{relative_dir}/ball.tracks.yolo.json" if ball_yolo is not None else None
        ),
        "ball_wasb": (
            f"{relative_dir}/ball.tracks.wasb.json" if ball_wasb is not None else None
        ),
        "court": f"{relative_dir}/court.keypoints.json",
        "court3d": f"{relative_dir}/court3d.json",
        "mock": mock,
        "projected": projected,
        "stages": sorted(wanted) if partial else None,
        "refreshed": sorted(refreshed),
        "player_source": tracks.get("source"),
        "ball_source": ball.get("source"),
        "ball_yolo_source": (ball_yolo or {}).get("source") if ball_yolo else None,
        "ball_wasb_source": (ball_wasb or {}).get("source") if ball_wasb else None,
        "court_source": court.get("source"),
        "player_count": len(tracks.get("players") or []),
        "ball_frames": len(ball.get("frames") or []),
        "ball_yolo_frames": len((ball_yolo or {}).get("frames") or []),
        "ball_wasb_frames": len((ball_wasb or {}).get("frames") or []),
        "court_detections": court_detections,
        "run": run_info,
    }
