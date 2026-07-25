"""Entry: python -m worker"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path


def _load_dotenv() -> None:
    """Load repo-root .env into os.environ before importing pipeline."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Do not clobber vars already set in the shell (standard dotenv behavior).
        if key and key not in os.environ:
            os.environ[key] = value


# MUST run before importing worker.pipeline (env defaults / Modal flags).
_load_dotenv()

import httpx

from worker.pipeline import run_pipeline, use_mock_tracks

API_BASE = os.environ.get("WORKER_API_BASE", "http://127.0.0.1:3000").rstrip("/")
POLL_SECONDS = float(os.environ.get("WORKER_POLL_SECONDS", "2"))
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "0.1.0")


def claim_job(client: httpx.Client) -> dict | None:
    res = client.post(f"{API_BASE}/api/jobs/claim")
    res.raise_for_status()
    return res.json().get("job")


def patch_job(client: httpx.Client, job_id: str, **fields: object) -> None:
    res = client.patch(f"{API_BASE}/api/jobs/{job_id}", json=fields)
    res.raise_for_status()


def run_job(client: httpx.Client, job: dict) -> None:
    job_id = job["id"]
    video_id = job["video_id"]

    def on_progress(stage: str, progress: float) -> None:
        patch_job(
            client,
            job_id,
            status="running",
            stage=stage,
            progress=max(0.0, min(1.0, progress)),
            error=None,
        )
        print(f"[worker] {job_id} {stage} {progress:.0%}", flush=True)

    try:
        result = run_pipeline(video_id, on_progress)
        # If tracks exist but no court projection yet, surface calibration need.
        status = "completed"
        if not result.get("projected"):
            status = "needs_calibration"

        patch_job(
            client,
            job_id,
            status=status,
            stage="done",
            progress=1.0,
            error=None,
        )
        print(
            f"[worker] finished job {job_id} status={status} "
            f"mock={result.get('mock')} players={result.get('player_count')} "
            f"ball_frames={result.get('ball_frames')} "
            f"src={result.get('player_source')}/{result.get('ball_source')}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] failed job {job_id}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        patch_job(
            client,
            job_id,
            status="failed",
            stage="done",
            error=str(exc)[:2000],
            retryable=True,
        )


def main() -> int:
    mock = use_mock_tracks()
    print(f"[worker] polling {API_BASE} every {POLL_SECONDS}s", flush=True)
    print(
        f"[worker] USE_MOCK_TRACKS={'1' if mock else '0'} "
        f"({'SYNTHETIC' if mock else 'Modal SAM 3.1 + ball'})",
        flush=True,
    )
    if mock:
        print(
            "[worker] Set USE_MOCK_TRACKS=0 in .env for real tracking",
            flush=True,
        )
    print("[worker] AI models are NOT loaded here — Modal only", flush=True)

    # Long timeout: jobs wait on Modal GPU for many minutes.
    with httpx.Client(timeout=httpx.Timeout(30.0, read=120.0)) as client:
        while True:
            try:
                job = claim_job(client)
                if job is None:
                    time.sleep(POLL_SECONDS)
                    continue
                print(
                    f"[worker] claimed job {job['id']} video={job['video_id']}",
                    flush=True,
                )
                run_job(client, job)
            except httpx.HTTPError as exc:
                print(f"[worker] HTTP error: {exc}", file=sys.stderr, flush=True)
                time.sleep(POLL_SECONDS)
            except KeyboardInterrupt:
                print("[worker] stopped", flush=True)
                return 0
            except Exception as exc:  # noqa: BLE001
                print(f"[worker] error: {exc}", file=sys.stderr, flush=True)
                time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
