from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _modal_fn(app_name: str, fn_name: str):
    try:
        import modal  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Modal package is not installed in the worker venv. "
            "Run: .venv\\Scripts\\pip install modal",
        ) from exc
    return modal.Function.from_name(app_name, fn_name)


def _app_name() -> str:
    return os.environ.get("MODAL_APP_NAME", "volleyball-ai")


def track_players_modal(
    work_mp4: Path,
    *,
    video_id: str,
    fps: float = 10.0,
) -> dict[str, Any]:
    """Call Modal SAM 3.1 track_players (single stage)."""
    return run_modal_ai_parallel(
        work_mp4,
        video_id=video_id,
        fps=fps,
        stages=("players",),
    )["players"]


def track_ball_modal(
    work_mp4: Path,
    *,
    video_id: str,
    fps: float = 10.0,
) -> dict[str, Any]:
    """Call Modal ball tracker (single stage)."""
    return run_modal_ai_parallel(
        work_mp4,
        video_id=video_id,
        fps=fps,
        stages=("ball",),
    )["ball"]


def detect_court_modal(
    media_path: Path,
    *,
    video_id: str,
    sample_fps: float | None = None,
    max_frames: int | None = None,
    confidence: float | None = None,
    return_overlays: int = 3,
) -> dict[str, Any]:
    """Call Modal court keypoint detector (single stage)."""
    return run_modal_ai_parallel(
        media_path,
        video_id=video_id,
        fps=10.0,
        stages=("court",),
        court_sample_fps=sample_fps,
        court_max_frames=max_frames,
        court_confidence=confidence,
        court_return_overlays=return_overlays,
    )["court"]


def run_modal_ai_parallel(
    media_path: Path,
    *,
    video_id: str,
    fps: float = 10.0,
    stages: tuple[str, ...] = ("court", "players", "ball"),
    court_sample_fps: float | None = None,
    court_max_frames: int | None = None,
    court_confidence: float | None = None,
    court_return_overlays: int = 2,
) -> dict[str, dict[str, Any]]:
    """
    Spawn court / players / ball on Modal at the same time, then wait for all.

    Each stage is a separate Modal function (own container):
      - detect_court  → CPU YOLO pose
      - track_players → A100 SAM 3.1
      - track_ball    → CPU VballNet

    Wall-clock ≈ max(stage times), not the sum.
    """
    app = _app_name()
    pipeline_version = os.environ.get("PIPELINE_VERSION", "0.1.0")
    video_bytes = media_path.read_bytes()
    suffix = media_path.suffix.lower() or ".mp4"
    wanted = set(stages)

    calls: dict[str, Any] = {}
    t0 = time.perf_counter()

    if "court" in wanted:
        court_fn = _modal_fn(
            app,
            os.environ.get("MODAL_DETECT_COURT_FN", "detect_court"),
        )
        sample_fps = float(
            court_sample_fps
            if court_sample_fps is not None
            else os.environ.get("COURT_SAMPLE_FPS", "1"),
        )
        max_frames = int(
            court_max_frames
            if court_max_frames is not None
            else os.environ.get("COURT_MAX_FRAMES", "30"),
        )
        conf = float(
            court_confidence
            if court_confidence is not None
            else os.environ.get("COURT_CONFIDENCE", "0.55"),
        )
        calls["court"] = court_fn.spawn(
            video_bytes=video_bytes,
            video_id=video_id,
            pipeline_version=pipeline_version,
            sample_fps=sample_fps,
            max_frames=max_frames,
            confidence=conf,
            return_overlays=court_return_overlays,
            media_suffix=suffix,
        )
        print(
            f"[modal] spawned detect_court sample_fps={sample_fps} "
            f"max_frames={max_frames}",
            flush=True,
        )

    if "players" in wanted:
        players_fn = _modal_fn(
            app,
            os.environ.get("MODAL_TRACK_PLAYERS_FN", "track_players"),
        )
        prompt = os.environ.get("SAM3_PROMPT", "person")
        sam_fps = float(os.environ.get("SAM3_FPS", "8"))
        calls["players"] = players_fn.spawn(
            video_bytes=video_bytes,
            video_id=video_id,
            prompt=prompt,
            fps=fps,
            sam_fps=sam_fps,
            pipeline_version=pipeline_version,
        )
        print(
            f"[modal] spawned track_players sam_fps={sam_fps} prompt={prompt!r}",
            flush=True,
        )

    if "ball" in wanted:
        ball_fn = _modal_fn(
            app,
            os.environ.get("MODAL_TRACK_BALL_FN", "track_ball"),
        )
        model_key = os.environ.get("VBALLNET_MODEL_KEY", "v1_148")
        calls["ball"] = ball_fn.spawn(
            video_bytes=video_bytes,
            video_id=video_id,
            fps=fps,
            pipeline_version=pipeline_version,
            model_key=model_key,
        )
        print(
            f"[modal] spawned track_ball model_key={model_key}",
            flush=True,
        )

    if not calls:
        return {}

    print(
        f"[modal] waiting in parallel for: {', '.join(sorted(calls))} …",
        flush=True,
    )

    results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    def _await(name: str, call: Any) -> tuple[str, dict[str, Any]]:
        try:
            result = call.get()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{name}: {exc}") from None
        if not isinstance(result, dict):
            raise RuntimeError(f"{name}: unexpected non-dict payload")
        return name, result

    with ThreadPoolExecutor(max_workers=max(1, len(calls))) as pool:
        futures = {
            pool.submit(_await, name, call): name for name, call in calls.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                finished_name, result = fut.result()
                results[finished_name] = result
                print(
                    f"[modal] {finished_name} done @ "
                    f"{time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                print(f"[modal] {name} FAILED: {exc}", flush=True)

    total = time.perf_counter() - t0
    print(
        f"[modal] parallel AI finished in {total:.1f}s "
        f"({', '.join(sorted(results))})",
        flush=True,
    )

    if errors:
        raise RuntimeError(
            "Modal parallel AI failed: " + "; ".join(errors),
        ) from None

    if "court" in wanted and "frames" not in results.get("court", {}):
        raise RuntimeError("Modal detect_court returned unexpected payload")
    if "players" in wanted and "players" not in results.get("players", {}):
        raise RuntimeError("Modal track_players returned unexpected payload")
    if "ball" in wanted and "frames" not in results.get("ball", {}):
        raise RuntimeError("Modal track_ball returned unexpected payload")

    return results
