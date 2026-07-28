"""
Net-only detector test (3 stills).

Default: degraded vision input (--max-side 512). Use --max-side 0 for full quality.

Optional: --max-side 512 (default) or OPENAI_NET_DETECT_MAX_SIDE.

Images:
  .data/openai-net-detect/images/
Outputs (full-quality run):
  .data/openai-net-detect/out-full/

Usage:
  .\\.venv\\Scripts\\python.exe -m worker.test_net_detect
  .\\.venv\\Scripts\\python.exe -m worker.test_net_detect --max-side 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / ".data" / "openai-net-detect"
IMAGES_DIR = FIXTURE_DIR / "images"
OUT_DIR_FULL = FIXTURE_DIR / "out-full"
OUT_DIR = FIXTURE_DIR / "out"

DEFAULT_IMAGES = (
    "01_penn_state_vb.jpg",
    "02_enc_volleyball.jpg",
    "03_video_thumb.jpg",
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: out-full when max_side unset, else out",
    )
    p.add_argument("--model", default=None, help="Default: OPENAI_NET_DETECT_MODEL or gpt-5.6")
    p.add_argument(
        "--max-side",
        type=int,
        default=512,
        help="Longest-side cap before vision call. 0 = full quality. Default 512.",
    )
    args = p.parse_args(argv)

    from worker.net_detect import process_still

    max_side = args.max_side if args.max_side and args.max_side > 0 else None
    out_dir = args.out_dir or (OUT_DIR_FULL if max_side is None else OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[net] test max_side={max_side or 'full'} out={out_dir}",
        flush=True,
    )

    fails = 0
    for name in DEFAULT_IMAGES:
        img = args.images_dir / name
        if not img.exists():
            print(f"missing: {img}", file=sys.stderr)
            fails += 1
            continue
        try:
            process_still(img, out_dir, model=args.model, max_side=max_side)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAILED {img}: {e}", file=sys.stderr)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
