"""
Camera-motion test on a sample volleyball video.

Compares scorers, writes motion start/peak/end timestamps, and a score CSV.

Default video: .data/videos/0437…/work.mp4
Outputs: .data/camera-motion/

Usage:
  .\\.venv\\Scripts\\python.exe -m worker.test_camera_motion
  .\\.venv\\Scripts\\python.exe -m worker.test_camera_motion --video path\\to\\work.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = (
    ROOT
    / ".data"
    / "videos"
    / "0437fa6a-1f14-4147-9315-26a813f7ce42"
    / "work.mp4"
)
OUT_DIR = ROOT / ".data" / "camera-motion"


def _video_id_from_path(video: Path) -> str | None:
    """If path is …/videos/{uuid}/work.mp4 (or source), return the uuid."""
    parts = video.resolve().parts
    try:
        i = parts.index("videos")
    except ValueError:
        return None
    if i + 1 >= len(parts):
        return None
    return parts[i + 1]


def _write_plot(samples: list[dict], segments: list[dict], out_path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ts = [s["t"] for s in samples]
    sc = [s["score"] for s in samples]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(ts, sc, color="#1f77b4", linewidth=1.0, label="motion score")
    for seg in segments:
        ax.axvspan(seg["start_t"], seg["end_t"], color="#ff7f0e", alpha=0.25)
        ax.axvline(seg["start_t"], color="#d62728", linestyle="--", linewidth=0.8)
        ax.axvline(seg["end_t"], color="#2ca02c", linestyle="--", linewidth=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--sample-fps", type=float, default=5.0)
    p.add_argument("--analyze-max-side", type=int, default=480)
    p.add_argument(
        "--merge-gap",
        type=float,
        default=None,
        help="Merge motion segments whose quiet gap is ≤ this many seconds (default 1.0)",
    )
    p.add_argument(
        "--start-unsettled",
        type=float,
        default=None,
        help="If first motion starts within this many seconds, treat t=0 as unsettled (default 0.5)",
    )
    p.add_argument(
        "--skip-compare",
        action="store_true",
        help="Only run the recommended global_affine scorer (faster)",
    )
    args = p.parse_args(argv)

    if not args.video.exists():
        print(f"missing video: {args.video}", file=sys.stderr)
        return 1

    from worker.camera_motion import (
        DEFAULT_MERGE_GAP_S,
        DEFAULT_START_UNSETTLED_S,
        analyze_camera_motion,
        compare_methods,
    )

    merge_gap_s = (
        DEFAULT_MERGE_GAP_S if args.merge_gap is None else float(args.merge_gap)
    )
    start_unsettled_s = (
        DEFAULT_START_UNSETTLED_S
        if args.start_unsettled is None
        else float(args.start_unsettled)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[cam] analyzing {args.video} sample_fps={args.sample_fps} "
        f"max_side={args.analyze_max_side} merge_gap_s={merge_gap_s} "
        f"start_unsettled_s={start_unsettled_s}",
        flush=True,
    )

    if args.skip_compare:
        recommended = "global_affine"
        cmp = None
    else:
        cmp = compare_methods(
            args.video,
            sample_fps=args.sample_fps,
            analyze_max_side=args.analyze_max_side,
            merge_gap_s=merge_gap_s,
            start_unsettled_s=start_unsettled_s,
        )
        recommended = cmp["recommended_method"]
        cmp_path = args.out_dir / "compare_methods.json"
        slim = {
            "video": cmp["video"],
            "recommended_method": cmp["recommended_method"],
            "reason": cmp["reason"],
            "methods": cmp["methods"],
        }
        cmp_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        print(f"[cam] wrote {cmp_path}", flush=True)

    # Full run for recommended method (includes samples for CSV/plot).
    primary = analyze_camera_motion(
        args.video,
        method=recommended,  # type: ignore[arg-type]
        sample_fps=args.sample_fps,
        analyze_max_side=args.analyze_max_side,
        merge_gap_s=merge_gap_s,
        start_unsettled_s=start_unsettled_s,
    )
    primary_path = args.out_dir / "motion_global_affine.json"
    primary_path.write_text(json.dumps(primary, indent=2), encoding="utf-8")

    csv_path = args.out_dir / "motion_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["t", "frame_index", "score", "tx", "ty", "angle_deg", "moving"],
        )
        w.writeheader()
        for s in primary["samples"]:
            w.writerow(s)

    plot_path = args.out_dir / "motion_timeline.png"
    _write_plot(
        primary["samples"],
        primary["segments"],
        plot_path,
        title=f"Camera motion ({primary['method']}) — {args.video.name}",
    )

    # Human-readable event log
    events_path = args.out_dir / "motion_events.txt"
    lines = [
        f"video: {args.video}",
        f"duration_s: {primary.get('duration_s')}",
        f"method: {primary['method']}",
        f"thresholds: {primary['thresholds']}",
        f"segments_raw: {primary['summary'].get('num_segments_raw')}",
        f"segments_merged: {primary['summary']['num_segments']}",
        f"settle_points: {primary['summary'].get('num_settle_points')}",
        f"net_samples: {primary['summary'].get('num_net_samples')}",
        f"time_moving_s: {primary['summary']['time_moving_s']}",
        f"starts_unsettled: {primary['summary'].get('starts_unsettled')}",
        f"settle_policy: {primary.get('settle_policy')}",
        "",
        "SETTLE POINTS (camera set)",
        "-" * 48,
    ]
    for sp in primary.get("settle_points") or []:
        lines.append(
            f"  SETTLE t={sp['t']:.2f}s  frame={sp.get('frame_index')}  kind={sp.get('kind')}"
        )
    lines.extend(["", "NET SAMPLE POINTS (settles + static refreshes)", "-" * 48])
    for sp in primary.get("net_sample_points") or []:
        lines.append(
            f"  SAMPLE t={sp['t']:.2f}s  frame={sp.get('frame_index')}  kind={sp.get('kind')}"
        )
    lines.extend(["", "EVENTS", "-" * 48])
    for ev in primary["events"]:
        if ev["type"] == "motion_start":
            lines.append(f"  START  t={ev['t']:.2f}s  frame={ev['frame_index']}")
        elif ev["type"] == "motion_peak":
            lines.append(f"  PEAK   t={ev['t']:.2f}s  score={ev['score']:.3f}")
        elif ev["type"] == "motion_end":
            lines.append(
                f"  END    t={ev['t']:.2f}s  frame={ev['frame_index']}  → settle"
            )
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Pipeline-ready artifact on the video (for the web player timeline ticks).
    # Keep samples out of this file — they are large and only needed for plots.
    video_id = _video_id_from_path(args.video)
    if video_id:
        artifact = {
            "video_id": video_id,
            "pipeline_version": "0.1.0",
            "source": "camera_motion_test",
            "method": primary["method"],
            "duration_s": primary.get("duration_s"),
            "fps": primary.get("fps"),
            "sample_fps": primary.get("sample_fps"),
            "analyze_max_side": primary.get("analyze_max_side"),
            "thresholds": primary.get("thresholds"),
            "segments": primary["segments"],
            "events": primary["events"],
            "settle_points": primary.get("settle_points") or [],
            "net_sample_points": primary.get("net_sample_points") or [],
            "settle_policy": primary.get("settle_policy"),
            "summary": primary.get("summary"),
        }
        artifact_path = args.video.parent / "camera_motion.json"
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"[cam] wrote {artifact_path}", flush=True)

    print(f"[cam] wrote {primary_path}", flush=True)
    print(f"[cam] wrote {csv_path}", flush=True)
    print(f"[cam] wrote {plot_path}", flush=True)
    print(f"[cam] wrote {events_path}", flush=True)
    print()
    print(events_path.read_text(encoding="utf-8"))

    if cmp is not None:
        print("Method comparison (segment counts):")
        for name, block in cmp["methods"].items():
            print(
                f"  {name:18} segments={block['summary']['num_segments']}  "
                f"moving={block['summary']['time_moving_s']}s  "
                f"enter={block['thresholds']['enter']}"
            )
        print(f"\nRecommended: {cmp['recommended_method']}")
        print(cmp["reason"])
    else:
        print(f"\nUsed method: {recommended} (--skip-compare)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
