from __future__ import annotations

import os
import re
from datetime import datetime, timezone
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


def run_id_from_iso(iso: str) -> str:
    """
    Filesystem-safe run id from an ISO timestamp.
    e.g. 2026-07-27T04:30:59.083Z → 2026-07-27_04-30-59Z
    """
    s = (iso or "").strip()
    if not s:
        s = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Normalize common ISO forms
    s = s.replace("+00:00", "Z")
    m = re.match(
        r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2}):(\d{2})",
        s,
    )
    if m:
        date, hh, mm, ss = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"{date}_{hh}-{mm}-{ss}Z"
    safe = re.sub(r"[^\w.\-]+", "_", s)
    return safe[:64] or "run"


def run_dir(video_id: str, run_id: str) -> Path:
    return video_dir(video_id) / "runs" / run_id


def latest_run_pointer_path(video_id: str) -> Path:
    return video_dir(video_id) / "latest_run.json"
