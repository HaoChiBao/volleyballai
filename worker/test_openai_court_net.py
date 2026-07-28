"""
Court-from-ImageGen + FIVB net math — dedicated 3-image fixture.

Images live in:
  .data/openai-court-net/images/
    01_penn_state_vb.jpg
    02_enc_volleyball.jpg
    03_video_thumb.jpg

Outputs (per image) go to:
  .data/openai-court-net/out/
    <stem>.01_original.jpg
    <stem>.02_labeled.png
    <stem>.03_outline.json
    <stem>.04_overlay.jpg

Usage:
  .\\.venv\\Scripts\\python.exe -m worker.test_openai_court_net
  .\\.venv\\Scripts\\python.exe -m worker.test_openai_court_net --recompute-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / ".data" / "openai-court-net"
IMAGES_DIR = FIXTURE_DIR / "images"
OUT_DIR = FIXTURE_DIR / "out"

DEFAULT_IMAGES = (
    "01_penn_state_vb.jpg",
    "02_enc_volleyball.jpg",
    "03_video_thumb.jpg",
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--images-dir",
        type=Path,
        default=IMAGES_DIR,
        help="Folder with the 3 fixture images",
    )
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--model", default=None)
    p.add_argument("--quality", default="medium", choices=["low", "medium", "high", "auto"])
    p.add_argument("--net-height-m", type=float, default=None)
    p.add_argument(
        "--recompute-only",
        action="store_true",
        help="Skip ImageGen; recompute from existing *.03_outline.json in out-dir",
    )
    args = p.parse_args(argv)

    from worker.openai_green_markers import process_image, recompute_from_outline

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fails = 0

    if args.recompute_only:
        outlines = sorted(args.out_dir.glob("*.03_outline.json"))
        if not outlines:
            print(f"no outlines in {args.out_dir}", file=sys.stderr)
            return 1
        for json_path in outlines:
            try:
                recompute_from_outline(json_path, net_height_m=args.net_height_m)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAILED {json_path}: {e}", file=sys.stderr)
        return 1 if fails else 0

    for name in DEFAULT_IMAGES:
        img = args.images_dir / name
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
