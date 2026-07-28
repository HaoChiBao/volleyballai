"""
Local helper: call Modal analyze_court_with_kimi_k3 and write artifacts.

Requires:
  - Full moonshotai/Kimi-K3 on Volume (modal run …::fetch_kimi_k3_local)
  - Deployed volleyball-ai app with B300:8 capacity
  - Modal auth in this venv

Usage (from repo root):
  .\\.venv\\Scripts\\python.exe -m worker.test_kimi_court .data/videos/<id>/thumb.jpg
  .\\.venv\\Scripts\\python.exe -m worker.test_kimi_court path/to/court.png .data/kimi-k3-court-test
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "Usage: python -m worker.test_kimi_court <image> [out_dir]",
            file=sys.stderr,
        )
        return 2

    media = Path(args[0])
    out_dir = Path(args[1] if len(args) > 1 else ".data/kimi-k3-court-test")
    if not media.exists():
        print(f"Not found: {media}", file=sys.stderr)
        return 1

    try:
        import modal  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Modal package is not installed. Run: .venv\\Scripts\\pip install modal",
        ) from exc

    app_name = os.environ.get("MODAL_APP_NAME", "volleyball-ai")
    fn = modal.Function.from_name(app_name, "analyze_court_with_kimi_k3")
    print(f"[test_kimi_court] calling {app_name}/analyze_court_with_kimi_k3 on {media}…")
    result = fn.remote(
        media.read_bytes(),
        video_id=media.stem,
        media_suffix=media.suffix or ".jpg",
        return_overlay=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k != "overlays"}
    json_path = out_dir / "kimi_k3.court.keypoints.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[test_kimi_court] wrote {json_path}")
    print(
        f"[test_kimi_court] visible={result.get('visible_keypoints')}/14 "
        f"source={result.get('source')} model={result.get('model')}",
    )

    for i, ov in enumerate(result.get("overlays") or []):
        raw = base64.b64decode(ov["jpg_b64"])
        jpg_path = out_dir / f"overlay_{i:02d}_t{ov.get('t', 0)}.jpg"
        jpg_path.write_bytes(raw)
        print(f"[test_kimi_court] wrote {jpg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
