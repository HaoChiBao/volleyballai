from __future__ import annotations

from pathlib import Path
from typing import Any


def track_players_modal(work_mp4: Path) -> dict[str, Any]:
    """
    Call Modal SAM 3.1 track_players.
    Requires `modal` CLI auth and a deployed function — see modal/track_players.py.
    """
    try:
        import modal  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "USE_MOCK_TRACKS=0 but modal package is not installed. "
            "pip install modal, or set USE_MOCK_TRACKS=1",
        ) from exc

    app_name = __import__("os").environ.get("MODAL_APP_NAME", "volleyball-ai")
    fn_name = __import__("os").environ.get("MODAL_TRACK_PLAYERS_FN", "track_players")

    track_fn = modal.Function.from_name(app_name, fn_name)
    result = track_fn.remote(str(work_mp4))
    if not isinstance(result, dict) or "players" not in result:
        raise RuntimeError("Modal track_players returned unexpected payload")
    return result
