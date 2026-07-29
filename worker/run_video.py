"""One-shot pipeline run for a video id (full or selected stages).

Examples:
  python -m worker.run_video <video_id>
  python -m worker.run_video <video_id> --stages ball_wasb
  python -m worker.run_video <video_id> --stages court,players,ball
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Load .env before importing pipeline (same as worker.__main__).
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        import os

        if key and key not in os.environ:
            os.environ[key] = value

from worker.pipeline import ALL_STAGE_TARGETS, run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run analysis pipeline for a video")
    parser.add_argument(
        "video_id",
        nargs="?",
        default="0437fa6a-1f14-4147-9315-26a813f7ce42",
        help="Video id under .data/videos/",
    )
    parser.add_argument(
        "--stages",
        default=None,
        help=(
            "Comma-separated stage targets for a partial run "
            f"(allowed: {', '.join(ALL_STAGE_TARGETS)}). "
            "Omit for a full pipeline run."
        ),
    )
    args = parser.parse_args()
    video_id = args.video_id
    stages: list[str] | None = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        if not stages:
            print("[run] --stages was empty", file=sys.stderr, flush=True)
            return 2

    def on_progress(stage: str, progress: float) -> None:
        print(f"[run] {stage} {progress:.0%}", flush=True)

    label = ",".join(stages) if stages else "full"
    print(f"[run] starting pipeline for {video_id} stages={label}", flush=True)
    result = run_pipeline(video_id, on_progress, stages=stages)
    print("[run] done", flush=True)
    print(
        {
            "run_id": (result.get("run") or {}).get("run_id"),
            "stages": (result.get("run") or {}).get("stages"),
            "projected": result.get("projected"),
            "court_detections": result.get("court_detections"),
            "player_count": result.get("player_count"),
            "ball_frames": result.get("ball_frames"),
            "models": (result.get("run") or {}).get("models"),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
