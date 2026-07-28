"""
High-quality video → Gaussian splat environment (Nerfstudio Splatfacto).

Runs only on Modal. Hybrid product path: static gym splat + live player/ball
tracks from the volleyball pipeline (dynamic subjects are not the splat target).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


SPATIAL_MOUNT = "/spatial"
DEFAULT_METHOD = "splatfacto-big"
DEFAULT_MAX_ITERS = 30_000
DEFAULT_NUM_FRAMES = 280


def _run(cmd: list[str], *, cwd: str | None = None) -> None:
    print(f"[spatial] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def _find_latest_config(outputs_root: Path) -> Path | None:
    configs = sorted(outputs_root.rglob("config.yml"), key=lambda p: p.stat().st_mtime)
    return configs[-1] if configs else None


def _find_export_ply(export_dir: Path) -> Path | None:
    ply_files = sorted(export_dir.rglob("*.ply"), key=lambda p: p.stat().st_size, reverse=True)
    return ply_files[0] if ply_files else None


def burn_transient_bboxes(
    images_dir: Path,
    tracks_payload: dict[str, Any] | None,
    *,
    fps_hint: float = 30.0,
) -> int:
    """
    Black out player bboxes on extracted frames to reduce ghosting.

    Expects volleyball `players.tracks.json` shape: {players:[{frames:[{t,bbox}]}]}
    """
    if not tracks_payload:
        return 0
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[spatial] opencv missing; skip transient burn", flush=True)
        return 0

    players = tracks_payload.get("players") or []
    if not players:
        return 0

    # Build per-second bbox lists (image coords).
    by_t: dict[float, list[list[float]]] = {}
    for p in players:
        for fr in p.get("frames") or []:
            bbox = fr.get("bbox")
            t = fr.get("t")
            if bbox is None or t is None:
                continue
            key = round(float(t), 2)
            by_t.setdefault(key, []).append([float(x) for x in bbox])

    if not by_t:
        return 0

    images = sorted(
        [
            *images_dir.glob("*.jpg"),
            *images_dir.glob("*.png"),
            *images_dir.glob("*.jpeg"),
        ],
    )
    if not images:
        return 0

    # Nerfstudio frame names are often frame_00001.jpg — assume uniform spacing.
    n = len(images)
    duration = max(by_t.keys()) if by_t else (n / max(fps_hint, 1.0))
    painted = 0
    for i, img_path in enumerate(images):
        t = (i / max(n - 1, 1)) * float(duration)
        # nearest track time
        nearest = min(by_t.keys(), key=lambda k: abs(k - t))
        if abs(nearest - t) > 0.35:
            continue
        boxes = by_t[nearest]
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        for b in boxes:
            if len(b) < 4:
                continue
            x1, y1, x2, y2 = b[:4]
            # bbox may be xyxy or xywh — heuristic
            if x2 <= 2 and y2 <= 2:
                # normalized xywh
                bw, bh = x2 * w, y2 * h
                x1, y1 = x1 * w, y1 * h
                x2, y2 = x1 + bw, y1 + bh
            elif x2 < x1 or y2 < y1:
                continue
            xa, ya = max(0, int(x1)), max(0, int(y1))
            xb, yb = min(w, int(x2)), min(h, int(y2))
            if xb > xa and yb > ya:
                im[ya:yb, xa:xb] = 0
                painted += 1
        cv2.imwrite(str(img_path), im)
    print(f"[spatial] burned transient boxes on frames touches={painted}", flush=True)
    return painted


def build_gaussian_scene(
    video_path: Path,
    work_dir: Path,
    *,
    method: str = DEFAULT_METHOD,
    max_iters: int = DEFAULT_MAX_ITERS,
    num_frames_target: int = DEFAULT_NUM_FRAMES,
    appearance_embedding: bool = True,
    tracks_payload: dict[str, Any] | None = None,
    burn_transients: bool = True,
) -> dict[str, Any]:
    """
    Full quality pipeline: video → COLMAP → splatfacto-big → .ply

    Returns meta including paths to config + ply under work_dir.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    processed = work_dir / "processed"
    outputs = work_dir / "outputs"
    export_dir = work_dir / "export"
    for d in (processed, outputs, export_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _run(
        [
            "ns-process-data",
            "video",
            "--data",
            str(video_path),
            "--output-dir",
            str(processed),
            "--num-frames-target",
            str(int(num_frames_target)),
        ],
    )

    images_dir = processed / "images"
    if not images_dir.exists():
        # some versions use `images_2` etc.
        candidates = list(processed.glob("images*"))
        images_dir = candidates[0] if candidates else images_dir

    burned = 0
    if burn_transients and tracks_payload:
        burned = burn_transient_bboxes(images_dir, tracks_payload)

    train_cmd = [
        "ns-train",
        method,
        "--data",
        str(processed),
        "--output-dir",
        str(outputs),
        "--max-num-iterations",
        str(int(max_iters)),
        "--pipeline.model.cull-alpha-thresh",
        "0.005",
        "--pipeline.model.continue-cull-post-densification",
        "False",
        "--viewer.quit-on-train-completion",
        "True",
        "--vis",
        "tensorboard",
    ]
    if appearance_embedding:
        train_cmd += ["--pipeline.model.use-appearance-embedding", "True"]

    _run(train_cmd)

    config = _find_latest_config(outputs)
    if config is None:
        raise RuntimeError("Training finished but no config.yml found")

    _run(
        [
            "ns-export",
            "gaussian-splat",
            "--load-config",
            str(config),
            "--output-dir",
            str(export_dir),
        ],
    )
    ply = _find_export_ply(export_dir)
    if ply is None:
        raise RuntimeError("Export finished but no .ply found")

    # Normalize name for consumers.
    final_ply = export_dir / "scene.ply"
    if ply.resolve() != final_ply.resolve():
        shutil.copy2(ply, final_ply)

    meta = {
        "ok": True,
        "method": method,
        "max_iters": max_iters,
        "num_frames_target": num_frames_target,
        "appearance_embedding": appearance_embedding,
        "transient_burn_touches": burned,
        "config": str(config),
        "ply": str(final_ply),
        "ply_bytes": final_ply.stat().st_size,
        "elapsed_s": round(time.time() - t0, 1),
        "note": (
            "Static environment splat (best quality: splatfacto-big). "
            "Live players/ball should come from volleyball tracks, not the splat."
        ),
    }
    (export_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[spatial] done {meta}", flush=True)
    return meta


def free_disk_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        return round(usage.free / (1024**3), 1)
    except OSError:
        return None
