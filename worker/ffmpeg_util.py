from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}",
        )


def probe(source: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(source),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    data = json.loads(proc.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}

    fps = None
    rate = stream.get("r_frame_rate") or "0/1"
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            fps = float(num) / float(den) if float(den) else None
        except ValueError:
            fps = None

    duration = stream.get("duration") or fmt.get("duration")
    return {
        "width": int(stream["width"]) if stream.get("width") else None,
        "height": int(stream["height"]) if stream.get("height") else None,
        "fps": fps,
        "duration_s": float(duration) if duration is not None else None,
    }


def normalize(source: Path, work: Path, thumb: Path) -> dict:
    """Create work.mp4 at native resolution + jpeg thumbnail.

    No spatial downscale — models must see full source resolution.
    """
    work.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            # Keep native WxH. Light re-encode for web-safe H.264 + faststart.
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(work),
        ],
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "1",
            "-i",
            str(work),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(thumb),
        ],
    )
    return probe(work)
