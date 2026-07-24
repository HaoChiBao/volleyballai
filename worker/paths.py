from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    raw = os.environ.get("DATA_DIR", ".data")
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def video_dir(video_id: str) -> Path:
    return data_root() / "videos" / video_id
