"""
Dual court+net ImageGen outline with FIVB ratio-locked court.

Images:
  .data/openai-fivb-ratios/images/
    01_penn_state_vb.jpg
    02_enc_volleyball.jpg
    03_video_thumb.jpg

Outputs:
  .data/openai-fivb-ratios/out/

Usage:
  .\\.venv\\Scripts\\python.exe -m worker.test_openai_fivb_ratios
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / ".data" / "openai-fivb-ratios"
IMAGES_DIR = FIXTURE_DIR / "images"
OUT_DIR = FIXTURE_DIR / "out"

DEFAULT_IMAGES = (
    "01_penn_state_vb.jpg",
    "02_enc_volleyball.jpg",
    "03_video_thumb.jpg",
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--model", default=None)
    p.add_argument("--quality", default="medium", choices=["low", "medium", "high", "auto"])
    args = p.parse_args(argv)

    from worker.openai_dual_outline import process_image

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fails = 0
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
            )
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAILED {img}: {e}", file=sys.stderr)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
