"""
Run settle → net detect → FIVB PnP on an existing video (no full pipeline).

Default: sample video with camera_motion.json already present.

Usage:
  .\\.venv\\Scripts\\python.exe -m worker.test_net_settle
  .\\.venv\\Scripts\\python.exe -m worker.test_net_settle --limit 1
  .\\.venv\\Scripts\\python.exe -m worker.test_net_settle --max-side 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = (
    ROOT / ".data" / "videos" / "0437fa6a-1f14-4147-9315-26a813f7ce42"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N settle points (faster smoke test)",
    )
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args(argv)

    from worker.net_settle_pipeline import run_net_settle_on_video

    tracks = run_net_settle_on_video(
        args.video_dir,
        max_side=args.max_side,
        model=args.model,
        settle_limit=args.limit,
    )
    print()
    print(
        f"Done: {tracks['summary']['num_settles']} settles · "
        f"primary_t={tracks['summary']['primary_t']} · "
        f"score={tracks['summary'].get('primary_score')} · "
        f"reproj={tracks['summary'].get('primary_reproj_err_px')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
