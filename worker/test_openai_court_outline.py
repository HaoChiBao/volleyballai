"""
Call OpenAI vision for net + court 8-corner outline; write JSON + overlay.

Usage (from repo root):
  1) Put OPENAI_API_KEY=sk-... in .env
  2) .\\.venv\\Scripts\\python.exe -m worker.test_openai_court_outline path/to/court.jpg
  3) Optional: --out-dir .data/openai-court-outline --model gpt-4o

Writes:
  <out_dir>/<stem>.outline.json
  <out_dir>/<stem>.overlay.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="OpenAI court/net quadrilateral outline")
    p.add_argument("image", type=Path, help="Court photo or video frame (.jpg/.png)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".data/openai-court-outline"),
        help="Output directory",
    )
    p.add_argument(
        "--model",
        default=None,
        help="OpenAI vision model (default: OPENAI_COURT_OUTLINE_MODEL or gpt-5.6)",
    )
    args = p.parse_args(argv)

    from worker.openai_court_outline import (
        call_openai_outline,
        draw_outline_overlay,
        image_size,
        load_dotenv,
        mime_for_path,
        normalize_outline,
    )

    load_dotenv()

    image: Path = args.image
    if not image.exists():
        print(f"Not found: {image}", file=sys.stderr)
        return 1

    w, h = image_size(image)
    print(f"[outline] {image} {w}x{h} → OpenAI…")
    raw_parsed, raw_text = call_openai_outline(
        image.read_bytes(),
        mime=mime_for_path(image),
        model=args.model,
        image_width=w,
        image_height=h,
    )
    outline = normalize_outline(raw_parsed, width=w, height=h)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image.stem
    json_path = out_dir / f"{stem}.outline.json"
    overlay_path = out_dir / f"{stem}.overlay.jpg"
    payload = {
        **outline,
        "source_image": str(image),
        "model": args.model
        or __import__("os").environ.get("OPENAI_COURT_OUTLINE_MODEL", "gpt-5.6"),
        "raw_text": raw_text[:8000],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    draw_outline_overlay(image, outline, overlay_path)

    print(f"[outline] wrote {json_path}")
    print(f"[outline] wrote {overlay_path}")
    print(
        "[outline] court TL/TR/BR/BL =",
        [
            (round(outline["court"][n]["x"], 1), round(outline["court"][n]["y"], 1))
            for n in ("top_left", "top_right", "bottom_right", "bottom_left")
        ],
    )
    print(
        "[outline] net TL/TR/BR/BL =",
        [
            (round(outline["net"][n]["x"], 1), round(outline["net"][n]["y"], 1))
            for n in ("top_left", "top_right", "bottom_right", "bottom_left")
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
