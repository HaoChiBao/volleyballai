"""
Full moonshotai/Kimi-K3 helpers (download verify + court keypoint LLM prompt).

Weights live on Modal Volume `kimi-k3-weights` only — never on the laptop.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

# Exact open-weight checkpoint (native vision, ~1.56 TB).
KIMI_REPO_ID = "moonshotai/Kimi-K3"
KIMI_MOUNT = "/models/kimi-k3"
KIMI_READY_MARKER = ".kimi_k3_ready.json"
# Lower bound for a complete snapshot (HF reports ~1.56 TB).
KIMI_MIN_BYTES = 1_400_000_000_000

COURT_KEYPOINT_NAMES: tuple[str, ...] = (
    "corner_top_left",
    "corner_top_right",
    "corner_bottom_left",
    "corner_bottom_right",
    "attack_top_left",
    "attack_top_right",
    "attack_bottom_left",
    "attack_bottom_right",
    "net_left",
    "net_right",
    "midline_left",
    "midline_right",
    "center_top",
    "center_bottom",
)


def volume_stats(root: Path) -> dict[str, Any]:
    total = 0
    n_files = 0
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                n_files += 1
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    ready_path = root / KIMI_READY_MARKER
    ready = False
    marker: dict[str, Any] = {}
    if ready_path.exists():
        try:
            marker = json.loads(ready_path.read_text(encoding="utf-8"))
            ready = bool(marker.get("ok"))
        except Exception:  # noqa: BLE001
            ready = False
    return {
        "root": str(root),
        "files": n_files,
        "bytes": total,
        "tb": round(total / 1e12, 3),
        "ready": ready,
        "marker": marker,
        "has_config": (root / "config.json").exists(),
    }


def is_snapshot_complete(root: Path) -> bool:
    st = volume_stats(root)
    return bool(st["has_config"] and st["bytes"] >= KIMI_MIN_BYTES)


def mark_ready(root: Path, stats: dict[str, Any]) -> None:
    payload = {
        "ok": True,
        "repo_id": KIMI_REPO_ID,
        "files": stats.get("files"),
        "bytes": stats.get("bytes"),
        "tb": stats.get("tb"),
    }
    (root / KIMI_READY_MARKER).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetch_snapshot(
    root: Path,
    *,
    commit_fn: Callable[[], None] | None = None,
    commit_every_s: float = 300.0,
) -> dict[str, Any]:
    """
    Resume-safe HF snapshot into `root` (Modal Volume mount).

    Uses hf_transfer when HF_HUB_ENABLE_HF_TRANSFER=1.
    """
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def _committer() -> None:
        while not stop.wait(commit_every_s):
            if commit_fn:
                try:
                    commit_fn()
                    print("[kimi-k3] volume checkpoint commit", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[kimi-k3] commit warn: {e}", flush=True)

    t = threading.Thread(target=_committer, daemon=True)
    if commit_fn:
        t.start()

    print(f"[kimi-k3] snapshot_download {KIMI_REPO_ID} → {root}", flush=True)
    try:
        path = snapshot_download(
            repo_id=KIMI_REPO_ID,
            local_dir=str(root),
            # Materialize files on the Volume mount (resume-safe by default).
            max_workers=8,
        )
        print(f"[kimi-k3] snapshot path={path}", flush=True)
    finally:
        stop.set()
        if commit_fn:
            try:
                commit_fn()
            except Exception as e:  # noqa: BLE001
                print(f"[kimi-k3] final commit warn: {e}", flush=True)

    st = volume_stats(root)
    complete = is_snapshot_complete(root)
    if complete:
        mark_ready(root, st)
        st = volume_stats(root)
    return {"ok": complete, "stats": st, "repo_id": KIMI_REPO_ID}


def court_keypoint_system_prompt() -> str:
    names = ", ".join(COURT_KEYPOINT_NAMES)
    return (
        "You are a precise volleyball court geometry annotator. "
        "Given one image of a volleyball court, locate the standard FIVB indoor "
        "court landmarks in pixel coordinates (origin top-left of the image).\n\n"
        f"Return ONLY a JSON object (no markdown) with this shape:\n"
        "{\n"
        '  "image_size": {"width": <int>, "height": <int>},\n'
        '  "keypoints": [\n'
        '    {"name": "<name>", "xy": [<x>, <y>]|null, "visible": <bool>, "conf": <0..1>}\n'
        "  ]\n"
        "}\n\n"
        f"You must include exactly these 14 names in order: {names}.\n"
        "If a point is occluded or off-frame, set xy=null, visible=false, conf=0.\n"
        "Coordinates must be in the original image pixel space."
    )


def court_keypoint_user_text(width: int, height: int) -> str:
    return (
        f"Image size is {width}x{height} pixels. "
        "Identify all 14 volleyball court keypoints and return JSON only."
    )


def parse_keypoints_json(
    text: str,
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Parse model JSON into volleyball_court_v1 keypoint list."""
    raw = text.strip()
    # Strip ```json fences if the model ignores instructions.
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    # Grab outermost object.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    data = json.loads(raw)
    items = data.get("keypoints") if isinstance(data, dict) else None
    by_name: dict[str, dict[str, Any]] = {}
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            name = str(it.get("name") or "")
            if name:
                by_name[name] = it

    out: list[dict[str, Any]] = []
    for name in COURT_KEYPOINT_NAMES:
        it = by_name.get(name) or {}
        xy = it.get("xy")
        conf = float(it.get("conf") or 0.0)
        visible = bool(it.get("visible"))
        if (
            isinstance(xy, (list, tuple))
            and len(xy) >= 2
            and xy[0] is not None
            and xy[1] is not None
        ):
            x = float(xy[0])
            y = float(xy[1])
            # Clamp to image bounds when clearly in-frame.
            if 0 <= x <= width and 0 <= y <= height and visible:
                out.append(
                    {
                        "name": name,
                        "xy": [round(x, 1), round(y, 1)],
                        "conf": round(max(0.0, min(1.0, conf if conf else 0.7)), 3),
                        "visible": True,
                    },
                )
                continue
        out.append({"name": name, "xy": None, "conf": 0.0, "visible": False})
    return out


def encode_image_data_url(image_bytes: bytes, *, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def mime_for_suffix(suffix: str) -> str:
    s = suffix.lower()
    if s in {".png"}:
        return "image/png"
    if s in {".webp"}:
        return "image/webp"
    return "image/jpeg"


def free_disk_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        return round(usage.free / (1024**3), 1)
    except OSError:
        return None
