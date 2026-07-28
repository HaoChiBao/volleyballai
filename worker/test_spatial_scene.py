"""
Build a best-quality Gaussian splat env on Modal and save under the video folder.

Usage (from repo root):
  .\\.venv\\Scripts\\python.exe -m worker.test_spatial_scene .data/videos/<id>/work.mp4
  .\\.venv\\Scripts\\python.exe -m worker.test_spatial_scene .data/videos/<id>/work.mp4 --video-id <id>

Writes:
  .data/videos/<id>/spatial/scene.ply
  .data/videos/<id>/spatial/meta.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Modal splatfacto-big spatial scene")
    p.add_argument("video", type=Path, help="Path to work.mp4 (or any mp4)")
    p.add_argument("--video-id", default="", help="Artifact id (default: parent folder name)")
    p.add_argument("--tracks", type=Path, default=None, help="players.tracks.json for transient burn")
    p.add_argument("--max-iters", type=int, default=30_000)
    p.add_argument("--out-dir", type=Path, default=None, help="Override output directory")
    args = p.parse_args(argv)

    media: Path = args.video
    if not media.exists():
        print(f"Not found: {media}", file=sys.stderr)
        return 1

    video_id = args.video_id or media.parent.name or media.stem
    out_dir = args.out_dir or (media.parent / "spatial")
    tracks_path = args.tracks
    if tracks_path is None:
        # Prefer run artifact, then flat layout.
        candidates = [
            media.parent / "players.tracks.json",
            *sorted(media.parent.glob("runs/*/players.tracks.json"), reverse=True),
        ]
        for c in candidates:
            if c.exists():
                tracks_path = c
                break

    try:
        import modal  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Modal package missing. Run: .venv\\Scripts\\pip install modal",
        ) from exc

    app_name = os.environ.get("MODAL_APP_NAME", "volleyball-ai")
    build_fn = modal.Function.from_name(app_name, "build_spatial_scene")
    dl_fn = modal.Function.from_name(app_name, "download_spatial_scene_ply")

    tracks_json = None
    if tracks_path and tracks_path.exists():
        tracks_json = tracks_path.read_text(encoding="utf-8")
        print(f"[spatial] transient burn from {tracks_path}")

    print(
        f"[spatial] build_spatial_scene on Modal (splatfacto-big / A100-80GB) "
        f"video_id={video_id} iters={args.max_iters}…",
    )
    meta = build_fn.remote(
        media.read_bytes(),
        video_id=video_id,
        max_iters=args.max_iters,
        players_tracks_json=tracks_json,
    )
    print(f"[spatial] train meta: ok={meta.get('ok')} elapsed_s={meta.get('elapsed_s')} "
          f"ply_bytes={meta.get('ply_bytes')}")

    print("[spatial] downloading scene.ply from Volume…")
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "scene.ply"
    meta_path = out_dir / "meta.json"
    raw: bytes | None = None
    # Prefer Volume streaming (avoids Modal return-size limits on large .ply).
    try:
        vol = modal.Volume.from_name("spatial-scenes")
        rel = f"{video_id}/publish/scene.ply"
        chunks: list[bytes] = []
        for chunk in vol.read_file(rel):
            chunks.append(chunk)
        raw = b"".join(chunks)
        print(f"[spatial] volume read {rel} ({len(raw)} bytes)")
    except Exception as e:  # noqa: BLE001
        print(f"[spatial] volume read failed ({e}); falling back to fn download")
        packed = dl_fn.remote(video_id)
        raw = base64.b64decode(packed["ply_b64"])
        meta = {**(packed.get("meta") or meta)}

    assert raw is not None
    ply_path.write_bytes(raw)
    payload = {
        **meta,
        "local_ply": str(ply_path),
        "downloaded_bytes": len(raw),
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[spatial] wrote {ply_path} ({len(raw)} bytes)")
    print(f"[spatial] wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
