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
    stages: tuple[str, ...] = (
        "court",
        "players",
        "ball",
        "ball_yolo",
        "ball_wasb",
    ),
    court_sample_fps: float | None = None,
    court_max_frames: int | None = None,
    court_confidence: float | None = None,
    court_return_overlays: int = 2,
) -> dict[str, dict[str, Any]]:
    """
    Spawn court / players / ball trackers on Modal at the same time, then wait.

    Each stage is a separate Modal function (own container):
      - detect_court    → CPU YOLO pose
      - track_players   → A100 SAM 3.1
      - track_ball      → CPU VballNet
      - track_ball_yolo → CPU SetOptics YOLO (soft-fail if undeployed)
      - track_ball_wasb → GPU WASB HRNet (soft-fail if undeployed)

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
        try:
            sam_max_width = int(os.environ.get("SAM3_MAX_WIDTH", "0") or "0")
        except ValueError:
            sam_max_width = 0
        calls["players"] = players_fn.spawn(
            video_bytes=video_bytes,
            video_id=video_id,
            prompt=prompt,
            fps=fps,
            sam_fps=sam_fps,
            sam_max_width=sam_max_width,
            pipeline_version=pipeline_version,
        )
        mw = sam_max_width if sam_max_width > 0 else "full"
        print(
            f"[modal] spawned track_players sam_fps={sam_fps} "
            f"max_w={mw} prompt={prompt!r}",
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

    if "ball_yolo" in wanted and os.environ.get("DISABLE_BALL_YOLO", "0") != "1":
        yolo_fn_name = os.environ.get("MODAL_TRACK_BALL_YOLO_FN", "track_ball_yolo")
        try:
            yolo_fn = _modal_fn(app, yolo_fn_name)
            calls["ball_yolo"] = yolo_fn.spawn(
                video_bytes=video_bytes,
                video_id=video_id,
                fps=fps,
                pipeline_version=pipeline_version,
            )
            print("[modal] spawned track_ball_yolo (SetOptics)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[modal] track_ball_yolo unavailable (skipping): {exc}",
                flush=True,
            )

    if "ball_wasb" in wanted and os.environ.get("DISABLE_BALL_WASB", "0") != "1":
        wasb_fn_name = os.environ.get("MODAL_TRACK_BALL_WASB_FN", "track_ball_wasb")
        try:
            wasb_fn = _modal_fn(app, wasb_fn_name)
            wasb_step = int(os.environ.get("WASB_STEP", "1") or "1")
            calls["ball_wasb"] = wasb_fn.spawn(
                video_bytes=video_bytes,
                video_id=video_id,
                fps=fps,
                pipeline_version=pipeline_version,
                step=wasb_step,
            )
            print(
                f"[modal] spawned track_ball_wasb (WASB step={wasb_step})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[modal] track_ball_wasb unavailable (skipping): {exc}",
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
    soft_fail = {"ball_yolo", "ball_wasb"}

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
                if name in soft_fail:
                    print(f"[modal] {name} FAILED (soft): {exc}", flush=True)
                else:
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
    if "ball_yolo" in results and "frames" not in results["ball_yolo"]:
        print("[modal] ball_yolo payload missing frames — dropping", flush=True)
        results.pop("ball_yolo", None)
    if "ball_wasb" in results and "frames" not in results["ball_wasb"]:
        print("[modal] ball_wasb payload missing frames — dropping", flush=True)
        results.pop("ball_wasb", None)

    return results
