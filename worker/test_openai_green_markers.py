"""
Court-from-ImageGen + FIVB net math test.

For each image, writes (sorted for review):
  <stem>.01_original.jpg  — unprocessed source
  <stem>.02_labeled.png   — ImageGen court-corner markers only
  <stem>.03_outline.json  — labeled court + derived net / H / camera
  <stem>.04_overlay.jpg   — ground court + net drawn on original

Usage:
  .\\.venv\\Scripts\\python.exe -m worker.test_openai_green_markers \\
    .data/court-model-test/images/05_penn_state_vb.jpg \\
    --max-side 512 --model gpt-image-2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path(".data/openai-green-markers"))
    p.add_argument("--max-side", type=int, default=512, help="Downscale longest side before edit")
    p.add_argument("--model", default=None, help="Default: OPENAI_IMAGE_MODEL or gpt-image-2")
    p.add_argument("--quality", default="low", choices=["low", "medium", "high", "auto"])
    p.add_argument("--net-height-m", type=float, default=None, help="Default 2.24 (women)")
    p.add_argument(
        "--recompute-only",
        action="store_true",
        help="Skip ImageGen; recompute math/overlay from existing *.03_outline.json",
    )
    args = p.parse_args(argv)

    from worker.openai_green_markers import process_image, recompute_from_outline

    fails = 0
    for img in args.images:
        if args.recompute_only:
            stem = img.stem
            # Allow passing either source image or outline json path.
            json_path = (
                img
                if img.suffix.lower() == ".json"
                else args.out_dir / f"{stem}.03_outline.json"
            )
            if not json_path.exists() and stem.endswith(".03_outline"):
                json_path = img
            if not json_path.exists():
                print(f"missing outline: {json_path}", file=sys.stderr)
                fails += 1
                continue
            try:
                recompute_from_outline(json_path, net_height_m=args.net_height_m)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAILED {json_path}: {e}", file=sys.stderr)
            continue

        if not img.exists():
            print(f"missing: {img}", file=sys.stderr)
            fails += 1
            continue
        try:
            process_image(
                img,
                args.out_dir,
                max_side=args.max_side,
                model=args.model,
                quality=args.quality,
                net_height_m=args.net_height_m,
            )
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAILED {img}: {e}", file=sys.stderr)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
