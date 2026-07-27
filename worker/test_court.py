"""
Local helper: call Modal detect_court and write artifacts under .data/court-test.

Usage (from repo root, with Modal auth + deployed app):
  .\\.venv\\Scripts\\python.exe -m worker.test_court .data/videos/<id>/work.mp4
  .\\.venv\\Scripts\\python.exe -m worker.test_court .data/videos/<id>/thumb.jpg
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "Usage: python -m worker.test_court <video_or_image> [out_dir]",
            file=sys.stderr,
        )
        return 2

    media = Path(args[0])
    out_dir = Path(args[1] if len(args) > 1 else ".data/court-test")
    if not media.exists():
        print(f"Not found: {media}", file=sys.stderr)
        return 1

    from worker.modal_bridge import detect_court_modal

    print(f"[test_court] calling Modal detect_court on {media}…")
    result = detect_court_modal(
        media,
        video_id=media.parent.name if media.parent.name else media.stem,
        return_overlays=3,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k != "overlays"}
    json_path = out_dir / "court.keypoints.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[test_court] wrote {json_path}")
    print(
        f"[test_court] detections={result.get('detections')} "
        f"size={result.get('image_size')}",
    )

    for i, ov in enumerate(result.get("overlays") or []):
        raw = base64.b64decode(ov["jpg_b64"])
        jpg_path = out_dir / f"overlay_{i:02d}_t{ov.get('t', 0)}.jpg"
        jpg_path.write_bytes(raw)
        print(f"[test_court] wrote {jpg_path}")

    frames = result.get("frames") or []
    if frames:
        vis = [k for k in frames[0].get("keypoints", []) if k.get("visible")]
        print(f"[test_court] first hit: {len(vis)}/14 visible @ t={frames[0].get('t')}")
    else:
        print("[test_court] no detections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
