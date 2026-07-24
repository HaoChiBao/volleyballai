from __future__ import annotations

import os
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


def track_players_modal(
    work_mp4: Path,
    *,
    video_id: str,
    fps: float = 10.0,
) -> dict[str, Any]:
    """
    Call Modal SAM 3.1 track_players with video bytes (no local model weights).
    Requires `modal deploy modal_app/app.py` and Hugging Face secret.
    """
    app_name = os.environ.get("MODAL_APP_NAME", "volleyball-ai")
    fn_name = os.environ.get("MODAL_TRACK_PLAYERS_FN", "track_players")
    prompt = os.environ.get("SAM3_PROMPT", "person")
    pipeline_version = os.environ.get("PIPELINE_VERSION", "0.1.0")

    track_fn = _modal_fn(app_name, fn_name)
    result = track_fn.remote(
        video_bytes=work_mp4.read_bytes(),
        video_id=video_id,
        prompt=prompt,
        fps=fps,
        pipeline_version=pipeline_version,
    )
    if not isinstance(result, dict) or "players" not in result:
        raise RuntimeError("Modal track_players returned unexpected payload")
    return result


def track_ball_modal(
    work_mp4: Path,
    *,
    video_id: str,
    fps: float = 10.0,
) -> dict[str, Any]:
    """Call Modal track_ball with video bytes."""
    app_name = os.environ.get("MODAL_APP_NAME", "volleyball-ai")
    fn_name = os.environ.get("MODAL_TRACK_BALL_FN", "track_ball")
    pipeline_version = os.environ.get("PIPELINE_VERSION", "0.1.0")

    track_fn = _modal_fn(app_name, fn_name)
    result = track_fn.remote(
        video_bytes=work_mp4.read_bytes(),
        video_id=video_id,
        fps=fps,
        pipeline_version=pipeline_version,
    )
    if not isinstance(result, dict) or "frames" not in result:
        raise RuntimeError("Modal track_ball returned unexpected payload")
    return result
