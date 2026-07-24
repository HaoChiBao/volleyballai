"""Entry: python -m worker"""

from __future__ import annotations

import os
import sys
import time
import traceback

import httpx

from worker.pipeline import run_pipeline

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
        print(f"[worker] {job_id} {stage} {progress:.0%}")

    try:
        result = run_pipeline(video_id, on_progress)
        patch_job(
            client,
            job_id,
            status="completed",
            stage="done",
            progress=1.0,
            error=None,
        )
        print(
            f"[worker] completed job {job_id} "
            f"(mock={result.get('mock')} pipeline={PIPELINE_VERSION})",
        )
    except Exception as exc:  # noqa: BLE001
        err = f"{exc}\n{traceback.format_exc()}"
        print(f"[worker] failed job {job_id}: {exc}", file=sys.stderr)
        patch_job(
            client,
            job_id,
            status="failed",
            stage="done",
            error=str(exc)[:2000],
            retryable=True,
        )


def main() -> int:
    mock = os.environ.get("USE_MOCK_TRACKS", "1") != "0"
    print(f"[worker] polling {API_BASE} every {POLL_SECONDS}s")
    print(f"[worker] USE_MOCK_TRACKS={'1' if mock else '0'}")
    print("[worker] AI models are NOT loaded here - Modal only")
    with httpx.Client(timeout=60.0) as client:
        while True:
            try:
                job = claim_job(client)
                if job is None:
                    time.sleep(POLL_SECONDS)
                    continue
                print(f"[worker] claimed job {job['id']} video={job['video_id']}")
                run_job(client, job)
            except httpx.HTTPError as exc:
                print(f"[worker] HTTP error: {exc}", file=sys.stderr)
                time.sleep(POLL_SECONDS)
            except KeyboardInterrupt:
                print("[worker] stopped")
                return 0
            except Exception as exc:  # noqa: BLE001
                print(f"[worker] error: {exc}", file=sys.stderr)
                time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
